//! Opt-in stage counters for the sparse 1-D vindex read path.
//!
//! Enabled by setting `ZARRS_VINDEX_STATS`. When disabled every hook is a
//! single relaxed load of a cached boolean, so the instrumented paths can be
//! left in place unconditionally.
//!
//! The previous instrumentation reported one summed `partial_decode_task_ms`
//! that mixed storage wait, shard-index work and codec CPU together. The stats
//! here keep the same summed-over-tasks convention but split each stage, so a
//! profile can attribute the remaining time. Two facts about the numbers:
//!
//! * All `*_ms` values except the `wall` ones are sums over concurrently
//!   running tasks. They measure work, not elapsed time, and legitimately
//!   exceed the wall time.
//! * I/O is attributed to the thread that blocks on it, via a thread-local
//!   [`scope`]. Storage calls made from a thread with no active scope still
//!   reach the global counters but cannot be split by phase.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

/// Whether `ZARRS_VINDEX_STATS` was set. Read once per process.
pub fn enabled() -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("ZARRS_VINDEX_STATS").is_some());
    *ENABLED
}

/// Which part of a vindex read a storage call belongs to.
///
/// Shard index reads happen while constructing partial decoders and are
/// cacheable across batches; payload reads happen inside `partial_decode_subsets`
/// and are not. Keeping them apart is what makes the shard-index cache's
/// contribution measurable.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Index = 0,
    Payload = 1,
}

const NUM_PHASES: usize = 2;
/// Log2-microsecond buckets, saturating at ~1s in the final bucket.
const NUM_BUCKETS: usize = 21;

fn add_ns(counter: &AtomicU64, elapsed: Duration) {
    counter.fetch_add(
        u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
}

fn ms(counter: &AtomicU64) -> f64 {
    duration_ms(Duration::from_nanos(counter.load(Ordering::Relaxed)))
}

pub fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

/// Counters for one storage phase.
#[derive(Default)]
struct IoStage {
    /// Logical multi-range operations, i.e. `get_partial_many` calls.
    ops: AtomicU64,
    /// Byte ranges across all operations. For a filesystem store this is the
    /// number of `pread`s the store is asked to perform.
    ranges: AtomicU64,
    ranges_max: AtomicU64,
    bytes: AtomicU64,
    /// Submitted until the I/O pool starts running the operation.
    queue_ns: AtomicU64,
    /// Inside the underlying storage call.
    call_ns: AtomicU64,
    /// Submitted until the submitting thread resumes. Includes queue, call and
    /// the handoff back, and is the figure to subtract from decode task time.
    wait_ns: AtomicU64,
    /// Log2-microsecond histogram of `call_ns`.
    call_hist: [AtomicU64; NUM_BUCKETS],
}

impl IoStage {
    fn record(&self, ranges: u64, bytes: u64, queue: Duration, call: Duration, wait: Duration) {
        self.ops.fetch_add(1, Ordering::Relaxed);
        self.ranges.fetch_add(ranges, Ordering::Relaxed);
        self.ranges_max.fetch_max(ranges, Ordering::Relaxed);
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        add_ns(&self.queue_ns, queue);
        add_ns(&self.call_ns, call);
        add_ns(&self.wait_ns, wait);
        let micros = u64::try_from(call.as_micros()).unwrap_or(u64::MAX);
        let bucket = usize::try_from(u64::BITS - micros.leading_zeros())
            .unwrap_or(NUM_BUCKETS - 1)
            .min(NUM_BUCKETS - 1);
        self.call_hist[bucket].fetch_add(1, Ordering::Relaxed);
    }

    fn is_empty(&self) -> bool {
        self.ops.load(Ordering::Relaxed) == 0
    }

    /// Non-empty log2-microsecond buckets as `lower_bound_us:count` pairs.
    fn histogram(&self) -> String {
        let mut parts = Vec::new();
        for (bucket, count) in self.call_hist.iter().enumerate() {
            let count = count.load(Ordering::Relaxed);
            if count == 0 {
                continue;
            }
            let lower = if bucket == 0 { 0 } else { 1u64 << (bucket - 1) };
            parts.push(format!("{lower}:{count}"));
        }
        if parts.is_empty() {
            "-".to_string()
        } else {
            parts.join(",")
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn report(&self, label: &str, out: &mut String) {
        use std::fmt::Write as _;
        let ops = self.ops.load(Ordering::Relaxed);
        let ranges = self.ranges.load(Ordering::Relaxed);
        // `wait` covers submit until the submitting thread resumes. What is
        // left after queueing and the storage call is NOT pure handoff: while
        // blocked in `install`, rayon runs other pending jobs from the
        // caller's own pool on that thread, and their whole duration lands
        // here. Read it as "caller was not making progress on this op", and
        // only as scheduler cost when nest_depth_max is 1.
        let blocked = (ms(&self.wait_ns) - ms(&self.queue_ns) - ms(&self.call_ns)).max(0.0);
        let _ = writeln!(
            out,
            "  io[{label}]: ops={ops} ranges={ranges} ranges_per_op_avg={:.1} ranges_per_op_max={} bytes={} queue_ms={:.3} call_ms={:.3} blocked_ms={blocked:.3} wait_ms={:.3} call_us_hist={}",
            if ops == 0 { 0.0 } else { ranges as f64 / ops as f64 },
            self.ranges_max.load(Ordering::Relaxed),
            self.bytes.load(Ordering::Relaxed),
            ms(&self.queue_ns),
            ms(&self.call_ns),
            ms(&self.wait_ns),
            self.histogram(),
        );
    }
}

/// Counters for one `retrieve_chunks_and_apply_index` call.
#[derive(Default)]
pub struct VindexStats {
    // Planning.
    pub chunk_items: AtomicU64,
    pub sparse_subsets: AtomicU64,
    pub shard_tasks: AtomicU64,
    pub subsets_per_task_max: AtomicU64,
    plan_ns: AtomicU64,

    // Shard decoder / index construction.
    pub decoder_cache_hits: AtomicU64,
    pub decoder_cache_misses: AtomicU64,
    decoder_build_ns: AtomicU64,

    // Decode and scatter, summed over tasks.
    execute_ns: AtomicU64,
    partial_decode_ns: AtomicU64,
    scatter_ns: AtomicU64,
    pub scatter_bytes: AtomicU64,

    // Storage, split by phase and attributed to the blocking thread.
    io: [IoStage; NUM_PHASES],
    io_inflight: AtomicUsize,
    io_inflight_max: AtomicUsize,
    io_nest_depth_max: AtomicUsize,
}

thread_local! {
    static CURRENT: RefCell<Option<(Arc<VindexStats>, Phase)>> = const { RefCell::new(None) };
}

/// Restores the enclosing scope, if any, on drop.
pub struct ScopeGuard(Option<(Arc<VindexStats>, Phase)>);

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        CURRENT.with(|current| *current.borrow_mut() = self.0.take());
    }
}

/// Attribute this thread's storage calls to `stats` under `phase` until the
/// returned guard is dropped.
#[must_use]
pub fn scope(stats: &Arc<VindexStats>, phase: Phase) -> ScopeGuard {
    let previous = CURRENT.with(|current| current.borrow_mut().replace((stats.clone(), phase)));
    ScopeGuard(previous)
}

/// Record a completed storage operation against the calling thread's scope.
///
/// Cheap and safe to call from anywhere: it is a no-op unless stats are
/// enabled, and falls back to an unattributed counter outside a scope.
pub fn record_io(ranges: u64, bytes: u64, queue: Duration, call: Duration, wait: Duration) {
    if !enabled() {
        return;
    }
    CURRENT.with(|current| {
        if let Some((stats, phase)) = current.borrow().as_ref() {
            stats.io[*phase as usize].record(ranges, bytes, queue, call, wait);
        }
    });
}

thread_local! {
    /// How many storage operations this thread is nested inside.
    ///
    /// Blocking on `rayon::ThreadPool::install` does not idle the calling
    /// worker: rayon runs other pending jobs from the caller's own pool on it
    /// while it waits. Those jobs issue their own storage operations, so one
    /// thread can sit inside many at once. Anything above 1 means the summed
    /// task timings double-count nested work and cannot be read as wall time.
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Track an in-flight storage operation for the calling thread's scope.
pub struct InflightGuard(Option<Arc<VindexStats>>);

impl InflightGuard {
    #[must_use]
    pub fn new() -> Self {
        if !enabled() {
            return Self(None);
        }
        let stats = CURRENT.with(|current| current.borrow().as_ref().map(|(s, _)| s.clone()));
        if let Some(stats) = &stats {
            let inflight = stats.io_inflight.fetch_add(1, Ordering::Relaxed) + 1;
            stats.io_inflight_max.fetch_max(inflight, Ordering::Relaxed);
            let depth = DEPTH.with(|depth| {
                let next = depth.get() + 1;
                depth.set(next);
                next
            });
            stats.io_nest_depth_max.fetch_max(depth, Ordering::Relaxed);
        } else {
            UNSCOPED.fetch_add(1, Ordering::Relaxed);
        }
        Self(stats)
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        if let Some(stats) = &self.0 {
            stats.io_inflight.fetch_sub(1, Ordering::Relaxed);
            DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
        }
    }
}

/// Storage operations issued with no active scope, process-wide. Reported so
/// that I/O the phase split cannot see is at least visible as a count.
static UNSCOPED: AtomicU64 = AtomicU64::new(0);

impl VindexStats {
    /// Record one payload read directly, without relying on a thread-local
    /// scope. Reads now happen on dedicated I/O threads that never enter a
    /// scope, so the submitter hands the stats object to the closure instead.
    pub fn record_payload_read(&self, bytes: u64, call: Duration) {
        self.io[Phase::Payload as usize].record(1, bytes, Duration::ZERO, call, call);
    }

    pub fn record_plan(&self, elapsed: Duration) {
        add_ns(&self.plan_ns, elapsed);
    }

    pub fn record_decoder_build(&self, elapsed: Duration) {
        add_ns(&self.decoder_build_ns, elapsed);
    }

    pub fn record_execute(&self, elapsed: Duration) {
        add_ns(&self.execute_ns, elapsed);
    }

    pub fn record_partial_decode(&self, elapsed: Duration) {
        add_ns(&self.partial_decode_ns, elapsed);
    }

    pub fn record_scatter(&self, elapsed: Duration) {
        add_ns(&self.scatter_ns, elapsed);
    }

    /// Codec CPU. The decode task no longer performs any I/O -- the caller
    /// fetches, and decode runs on already-resident bytes -- so its time is
    /// codec work and nothing else. It used to subtract payload wait, which
    /// is meaningless now that reads happen on separate threads.
    fn codec_ms(&self) -> f64 {
        ms(&self.partial_decode_ns)
    }

    #[allow(clippy::cast_precision_loss)]
    pub fn report(&self, context: &str) {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(out, "zarrs vindex stats [{context}]");
        let _ = writeln!(
            out,
            "  plan: chunk_items={} sparse_subsets={} shard_tasks={} subsets_per_task_max={} plan_ms={:.3}",
            self.chunk_items.load(Ordering::Relaxed),
            self.sparse_subsets.load(Ordering::Relaxed),
            self.shard_tasks.load(Ordering::Relaxed),
            self.subsets_per_task_max.load(Ordering::Relaxed),
            ms(&self.plan_ns),
        );
        let _ = writeln!(
            out,
            "  decoders: cache_hits={} cache_misses={} build_ms={:.3}",
            self.decoder_cache_hits.load(Ordering::Relaxed),
            self.decoder_cache_misses.load(Ordering::Relaxed),
            ms(&self.decoder_build_ns),
        );
        for (phase, label) in [(Phase::Index, "index"), (Phase::Payload, "payload")] {
            let stage = &self.io[phase as usize];
            if !stage.is_empty() {
                stage.report(label, &mut out);
            }
        }
        let nest_depth = self.io_nest_depth_max.load(Ordering::Relaxed);
        let _ = writeln!(
            out,
            "  io: inflight_max={} nest_depth_max={nest_depth} unscoped_ops={}",
            self.io_inflight_max.load(Ordering::Relaxed),
            UNSCOPED.load(Ordering::Relaxed),
        );
        let _ = writeln!(
            out,
            "  decode: execute_wall_ms={:.3} partial_decode_task_ms={:.3} codec_est_ms={:.3} scatter_ms={:.3} scatter_bytes={}",
            ms(&self.execute_ns),
            ms(&self.partial_decode_ns),
            self.codec_ms(),
            ms(&self.scatter_ns),
            self.scatter_bytes.load(Ordering::Relaxed),
        );
        if nest_depth > 1 {
            let _ = writeln!(
                out,
                "  WARNING: storage operations nested {nest_depth} deep on a single thread. \
                 partial_decode_task_ms, codec_est_ms, blocked_ms and wait_ms double-count \
                 nested work and are NOT wall time. Trust execute_wall_ms, queue_ms, call_ms, \
                 scatter_ms and the counts. Concurrency is also not bounded by the configured \
                 decode target.",
            );
        }
        eprint!("{out}");
    }
}
