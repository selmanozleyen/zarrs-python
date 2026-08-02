//! Threads that exist only to block in a storage read.
//!
//! The codec pipeline runs its chunk work on rayon, and rayon sizes itself to
//! the core count because it is a CPU scheduler. Issuing a blocking store read
//! from a rayon worker therefore ties the number of reads that can be in
//! flight to the number of cores, which is the wrong bound: a read costs no
//! CPU while it waits. On a local filesystem that is invisible, but on an
//! object store or a network filesystem a read is milliseconds, and a
//! sixteen-core machine will keep sixteen requests outstanding against a store
//! that would happily serve hundreds.
//!
//! This pool decouples the two. Reads are submitted here and their results
//! come back over a channel, so I/O depth is set by `fetch_threads` while CPU
//! parallelism stays with rayon.
//!
//! Deliberately not a rayon pool. Rayon documents `ThreadPool::install` as
//! *not* parking the caller -- it "will try to keep busy while the op
//! completes in its target pool", so a worker waiting on a read steals more
//! work, which blocks on more reads. Nesting compounds and no configured
//! target bounds anything. These threads have no work to steal: they block,
//! send, and go back to waiting.

use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex, Weak},
};

/// Threads are only ever parked in a read, so a small stack is plenty.
const FETCH_THREAD_STACK_BYTES: usize = 256 * 1024;

pub struct FetchPool {
    tx: crossbeam_channel::Sender<Box<dyn FnOnce() + Send + 'static>>,
}

impl FetchPool {
    fn new(threads: usize) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded::<Box<dyn FnOnce() + Send + 'static>>();
        for index in 0..threads {
            let rx = rx.clone();
            std::thread::Builder::new()
                .name(format!("zarrs-fetch-{index}"))
                .stack_size(FETCH_THREAD_STACK_BYTES)
                .spawn(move || {
                    while let Ok(job) = rx.recv() {
                        job();
                    }
                })
                .expect("failed to spawn fetch thread");
        }
        Self { tx }
    }

    /// Queue a read. Never blocks: the queue is unbounded, so submitting all
    /// of a batch's reads up front is what puts them in flight together.
    pub fn submit(&self, job: impl FnOnce() + Send + 'static) {
        // Workers hold `rx` for as long as the pool is alive, so this can only
        // fail after the last `Arc` has dropped, which cannot happen here.
        let _ = self.tx.send(Box::new(job));
    }
}

/// Depth to use when unset.
///
/// A multiple of the core count rather than the core count itself, because
/// these threads are parked rather than running. The ceiling keeps the thread
/// count and the bytes in flight bounded on very large machines; the floor
/// keeps small ones from serialising against a high-latency store.
pub fn default_fetch_threads() -> usize {
    std::thread::available_parallelism().map_or(64, |n| (n.get() * 8).clamp(64, 1024))
}

static FETCH_POOLS: LazyLock<Mutex<HashMap<usize, Weak<FetchPool>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// One pool per distinct size, shared by every pipeline that asks for it, so
/// opening many arrays does not spawn many hundreds of threads. The workers
/// exit once the last pipeline holding the pool drops it.
pub fn shared_fetch_pool(threads: usize) -> Arc<FetchPool> {
    let mut pools = FETCH_POOLS.lock().unwrap();
    if let Some(pool) = pools.get(&threads).and_then(Weak::upgrade) {
        return pool;
    }
    let pool = Arc::new(FetchPool::new(threads));
    pools.insert(threads, Arc::downgrade(&pool));
    pool
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn runs_submitted_jobs() {
        let pool = shared_fetch_pool(4);
        let seen = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = crossbeam_channel::unbounded();
        for _ in 0..32 {
            let seen = seen.clone();
            let tx = tx.clone();
            pool.submit(move || {
                seen.fetch_add(1, Ordering::Relaxed);
                let _ = tx.send(());
            });
        }
        drop(tx);
        assert_eq!(rx.iter().count(), 32);
        assert_eq!(seen.load(Ordering::Relaxed), 32);
    }

    /// The point of the pool: more reads in flight than there are threads
    /// driving them, which is impossible if the submitter blocks per read.
    #[test]
    fn keeps_more_reads_in_flight_than_submitters() {
        let pool = shared_fetch_pool(16);
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = crossbeam_channel::unbounded();
        for _ in 0..16 {
            let (in_flight, peak, tx) = (in_flight.clone(), peak.clone(), tx.clone());
            pool.submit(move || {
                let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(50));
                in_flight.fetch_sub(1, Ordering::SeqCst);
                let _ = tx.send(());
            });
        }
        drop(tx);
        assert_eq!(rx.iter().count(), 16);
        assert!(
            peak.load(Ordering::SeqCst) > 1,
            "reads did not overlap; peak in flight was {}",
            peak.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn shares_one_pool_per_size() {
        let a = shared_fetch_pool(3);
        let b = shared_fetch_pool(3);
        let c = shared_fetch_pool(5);
        assert!(Arc::ptr_eq(&a, &b));
        assert!(!Arc::ptr_eq(&a, &c));
    }
}
