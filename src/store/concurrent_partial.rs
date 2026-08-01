use std::{
    sync::{
        Arc,
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
/// # Why this no longer dispatches to a separate I/O pool
///
/// It previously ran each read on a shared rayon I/O pool via
/// `ThreadPool::install`, called from a rayon decode worker. That is
/// documented rayon behaviour to *not* park the caller: it "will try to keep
/// busy while the op completes in its target pool ... it may potentially
/// schedule other tasks to run on the current thread in the meantime". The
/// waiting decode worker therefore stole further shard tasks, which blocked on
/// their own reads and stole more. Measured with `ZARRS_VINDEX_STATS`: nesting
/// 31 deep on a *single* decode thread, with 31 concurrent reads even when the
/// decode target was 1.
///
/// Consequences were that no configured target bounded concurrency, summed
/// task timings double-counted nested work, and stack depth grew without limit.
///
/// A two-pool split cannot fix this, because a worker waiting on a read still
/// occupies a thread either way: outstanding reads are capped by the number of
/// waiting workers regardless. Splitting only pays once a read can be
/// outstanding *without* holding a thread, which needs completion-based I/O
/// (`io_uring`), not a second pool.
///
/// So reads now run inline on the calling shard-task thread, and concurrency
/// comes from the size of the single shared pool. Getting more outstanding
/// reads means more threads in that pool, which is cheap: they are parked in
/// `pread`, and codec CPU is a small fraction of this workload.
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
