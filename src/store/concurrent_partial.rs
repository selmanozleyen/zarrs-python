use std::{
    collections::HashMap,
    sync::{
        Arc, LazyLock, Mutex, Weak,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use zarrs::storage::{
    Bytes, ListableStorageTraits, MaybeBytes, MaybeBytesIterator, OffsetBytesIterator,
    ReadableStorageTraits, ReadableWritableListableStorage, StorageError, StoreKey, StoreKeys,
    StoreKeysPrefixes, StorePrefix, WritableStorageTraits, byte_range::ByteRangeIterator,
};

use crate::vindex_stats;

static ACTIVE_READS: AtomicUsize = AtomicUsize::new(0);
static MAX_ACTIVE_READS: AtomicUsize = AtomicUsize::new(0);

fn as_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

struct ActiveReadGuard;

impl ActiveReadGuard {
    fn new() -> Self {
        let active = ACTIVE_READS.fetch_add(1, Ordering::Relaxed) + 1;
        MAX_ACTIVE_READS.fetch_max(active, Ordering::Relaxed);
        Self
    }
}

impl Drop for ActiveReadGuard {
    fn drop(&mut self) {
        ACTIVE_READS.fetch_sub(1, Ordering::Relaxed);
    }
}

pub fn partial_read_max_active() -> usize {
    MAX_ACTIVE_READS.load(Ordering::Relaxed)
}

/// A storage adapter that measures logical multi-range reads.
///
/// Measurement only: it records how many reads are in flight and how long they
/// take, and dispatches nothing. Reads are handed to [`VindexFetchPool`]
/// explicitly at the call site.
///
/// # Why dispatch does not happen here
///
/// An earlier design ran each read on a shared *rayon* I/O pool via
/// `ThreadPool::install`, called from a rayon decode worker. Rayon documents
/// that as explicitly *not* parking the caller: it "will try to keep busy
/// while the op completes in its target pool ... it may potentially schedule
/// other tasks to run on the current thread in the meantime". The waiting
/// decode worker therefore stole further shard tasks, which blocked on their
/// own reads and stole more. Measured with `ZARRS_VINDEX_STATS`: nesting 31
/// deep on a *single* decode thread, with 31 concurrent reads even when the
/// decode target was 1. No configured target bounded concurrency, summed task
/// timings double-counted nested work, and stack depth grew without limit.
///
/// The fix was not a second *rayon* pool, which has the same defect for the
/// same reason -- `install` never parks. It was dedicated non-rayon threads
/// that block in `pread` and return bytes over a channel, so a waiting read
/// occupies one cheap 256 KiB thread and steals nothing. See
/// [`VindexFetchPool`].
struct MeasuredStorage {
    storage: ReadableWritableListableStorage,
}

impl MeasuredStorage {
    fn new(storage: ReadableWritableListableStorage) -> Self {
        Self { storage }
    }
}

impl ReadableStorageTraits for MeasuredStorage {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.storage.get(key)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        if !vindex_stats::enabled() {
            let _active = ActiveReadGuard::new();
            return self.storage.get_partial_many(key, byte_ranges);
        }

        let byte_ranges = byte_ranges.collect::<Vec<_>>();
        let num_ranges = as_u64(byte_ranges.len());
        let _inflight = vindex_stats::InflightGuard::new();

        let started = Instant::now();
        let results = {
            let _active = ActiveReadGuard::new();
            self.storage
                .get_partial_many(key, Box::new(byte_ranges.into_iter()))?
                .map(std::iter::Iterator::collect::<Result<Vec<_>, StorageError>>)
                .transpose()?
        };
        let call = started.elapsed();

        let bytes = results.as_ref().map_or(0, |results| {
            results.iter().map(|bytes| as_u64(bytes.len())).sum()
        });
        // Reads are inline now, so there is no queueing and no handoff: the
        // whole wait is the storage call itself.
        vindex_stats::record_io(num_ranges, bytes, Duration::ZERO, call, call);

        let Some(results) = results else {
            return Ok(None);
        };
        Ok(Some(Box::new(results.into_iter().map(Ok))))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.storage.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.storage.supports_get_partial()
    }
}

impl WritableStorageTraits for MeasuredStorage {
    fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
        self.storage.set(key, value)
    }

    fn set_partial_many(
        &self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        self.storage.set_partial_many(key, offset_values)
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        self.storage.erase(key)
    }

    fn erase_many(&self, keys: &[StoreKey]) -> Result<(), StorageError> {
        self.storage.erase_many(keys)
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        self.storage.erase_prefix(prefix)
    }

    fn supports_set_partial(&self) -> bool {
        self.storage.supports_set_partial()
    }
}

impl ListableStorageTraits for MeasuredStorage {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        self.storage.list()
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.storage.list_prefix(prefix)
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        self.storage.list_dir(prefix)
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.storage.size_prefix(prefix)
    }

    fn size(&self) -> Result<u64, StorageError> {
        self.storage.size()
    }
}

/// Wrap `storage` so the scattered path's logical multi-range reads are
/// counted and timed. Reads execute inline on the calling thread.
pub fn with_io_measurement(
    storage: ReadableWritableListableStorage,
) -> ReadableWritableListableStorage {
    Arc::new(MeasuredStorage::new(storage))
}

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
