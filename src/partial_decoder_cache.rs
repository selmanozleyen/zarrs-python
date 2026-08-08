use std::sync::{Arc, Mutex};

use lru::LruCache;
use zarrs::array::ArrayPartialDecoderTraits;
use zarrs::storage::StoreKey;

type Decoder = Arc<dyn ArrayPartialDecoderTraits>;

/// Partial decoders retained across reads, bounded by the bytes they hold.
///
/// Constructing a partial decoder can read from storage and keep the result: for
/// a sharded chunk that is the decoded shard index, and for a compressed chunk it
/// is the decoded chunk. [`ArrayPartialDecoderTraits::size_held`] reports exactly
/// what was kept, so it is both the reason to cache a decoder and the cost of
/// doing so — a decoder holding nothing read nothing, and is not admitted.
pub(crate) struct PartialDecoderCache {
    inner: Mutex<Inner>,
    budget: usize,
}

struct Inner {
    lru: LruCache<StoreKey, Decoder>,
    held: usize,
}

impl PartialDecoderCache {
    /// A cache bounded to `budget` bytes, or [`None`] to disable caching.
    pub(crate) fn new(budget: usize) -> Option<Self> {
        (budget > 0).then(|| Self {
            inner: Mutex::new(Inner {
                // Eviction is by bytes held, not by entry count.
                lru: LruCache::unbounded(),
                held: 0,
            }),
            budget,
        })
    }

    pub(crate) fn get(&self, key: &StoreKey) -> Option<Decoder> {
        self.inner.lock().unwrap().lru.get(key).cloned()
    }

    pub(crate) fn put(&self, key: StoreKey, decoder: &Decoder) {
        // `size_held` is fixed at construction — a sharding partial decoder reads
        // its index in `new` and never revises it — so the running total stays in
        // step with what the cache actually holds.
        let size = decoder.size_held();
        if size == 0 {
            return;
        }
        let mut inner = self.inner.lock().unwrap();
        if let Some(replaced) = inner.lru.put(key, decoder.clone()) {
            inner.held -= replaced.size_held();
        }
        inner.held += size;
        while inner.held > self.budget {
            // Evicts what was just inserted if it alone exceeds the budget.
            let Some((_, evicted)) = inner.lru.pop_lru() else {
                break;
            };
            inner.held -= evicted.size_held();
        }
    }

    pub(crate) fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.lru.clear();
        inner.held = 0;
    }

    pub(crate) fn held(&self) -> usize {
        self.inner.lock().unwrap().held
    }
}
