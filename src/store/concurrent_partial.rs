use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex, Weak},
};


/// Pre-spawned threads that do nothing but block in a storage read.
///
/// Deliberately not rayon: rayon is a CPU work-stealing scheduler, blocking in
/// it is against its contract, and `ThreadPool::install` from another pool's
/// worker runs unrelated jobs on the caller instead of parking it. Reads want
/// the opposite of work stealing -- a thread that parks in `pread` and costs
/// nothing until its bytes arrive.
///
/// Spawned once per distinct size, so a batch never pays thread creation.
/// Sized by `vindex_fetch_threads`. The bound is about bytes in flight and the
/// storage service rate, not cores, because these threads are almost always
/// parked -- so it wants to be far larger than the core count.
pub struct VindexFetchPool {
    tx: crossbeam_channel::Sender<Box<dyn FnOnce() + Send + 'static>>,
}

impl VindexFetchPool {
    fn new(threads: usize) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded::<Box<dyn FnOnce() + Send + 'static>>();
        for index in 0..threads {
            let rx = rx.clone();
            std::thread::Builder::new()
                .name(format!("zarrs-vindex-fetch-{index}"))
                // They only wait and move bytes; no codec recursion.
                .stack_size(256 * 1024)
                .spawn(move || {
                    while let Ok(job) = rx.recv() {
                        job();
                    }
                })
                .expect("failed to spawn vindex fetch thread");
        }
        Self { tx }
    }

    pub fn submit(&self, job: impl FnOnce() + Send + 'static) {
        // The workers hold `rx` for as long as this pool is alive, so a send
        // can only fail once every worker is gone, i.e. after the last `Arc`
        // to the pool has dropped -- which cannot happen while `self` exists.
        let _ = self.tx.send(Box::new(job));
    }
}

/// Default fetch depth when unset: these threads park in `pread`, so this is a
/// bytes-in-flight bound rather than a CPU one, and it is deliberately a large
/// multiple of the core count.
pub fn default_vindex_fetch_threads() -> usize {
    std::thread::available_parallelism().map_or(64, |n| (n.get() * 8).clamp(64, 1024))
}

static VINDEX_FETCH_POOLS: LazyLock<Mutex<HashMap<usize, Weak<VindexFetchPool>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Share one fetch pool per distinct size, so many pipelines over the same
/// config do not each spawn their own hundreds of threads. Dropping the last
/// pipeline drops the sender, which lets the workers exit.
pub fn shared_vindex_fetch_pool(threads: usize) -> Arc<VindexFetchPool> {
    let mut pools = VINDEX_FETCH_POOLS.lock().unwrap();
    if let Some(pool) = pools.get(&threads).and_then(Weak::upgrade) {
        return pool;
    }
    let pool = Arc::new(VindexFetchPool::new(threads));
    pools.insert(threads, Arc::downgrade(&pool));
    pool
}
