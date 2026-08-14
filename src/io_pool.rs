//! The fetch side of a planned read: get bytes off the storage and hand each buffer to the decode
//! pool the moment it lands.
//!
//! Deliberately **not** a rayon pool. rayon's workers exist for CPU-bound work and this side does
//! the opposite: a thread parked in a read cannot steal, and a fetch blocked on the byte budget
//! holds a worker that only a decode can release. Sharing one pool therefore lets the fetch side
//! starve the decode side of the very threads that would unblock it. Plain OS threads are allowed
//! to block, so the two pools stay independent and every core stays free to decode.
//!
//! Two backends answer the same question -- how many reads can be outstanding at once -- at
//! different cost:
//!
//! * [`for_each_blocking`] pays one blocked thread per outstanding read, so queue depth costs
//!   stacks and scheduler pressure. It goes through the store, so it works for every backing store.
//! * [`uring_read_all`] pays one thread in total; the ring holds the concurrency, so depth is
//!   decoupled from thread count. It needs a real file to read, so it is filesystem-only.
//!
//! Which is faster is a property of the filesystem, not of this code: io_uring only avoids blocking
//! where the filesystem implements async reads, and where it does not the kernel services the ring
//! from its own worker pool and only syscall batching is left. Hence [`Backend::Auto`] resolves at
//! runtime and the resolved choice is reported rather than assumed.

use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

/// How the fetch side issues its reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    /// The ring where this process can actually have one, blocking reads otherwise.
    Auto,
    /// One blocking read per outstanding request, on plain OS threads.
    Threads,
    /// One ring, one thread, `depth` reads outstanding.
    Uring,
}

impl Backend {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "auto" => Some(Self::Auto),
            "threads" => Some(Self::Threads),
            "uring" => Some(Self::Uring),
            _ => None,
        }
    }

    /// What will actually run, having asked the kernel rather than trusting the target triple.
    ///
    /// `direct` says whether the caller can supply real file paths; a ring cannot read through a
    /// store abstraction, so an HTTP or object store resolves to threads however it was configured.
    ///
    /// The choice is recorded here rather than at the call site, so that resolving without
    /// reporting is not expressible. [`uring_available`] answers a narrower question -- whether a
    /// ring *could* be made -- and is true in exactly the case that still falls back for want of
    /// file paths, which is the case worth catching.
    pub fn resolve(self, direct: bool) -> Self {
        let resolved = match self {
            Self::Threads => Self::Threads,
            Self::Auto | Self::Uring if direct && uring_usable() => Self::Uring,
            _ => Self::Threads,
        };
        LAST_RESOLVED.store(
            if resolved == Self::Uring { RESOLVED_URING } else { RESOLVED_THREADS },
            Ordering::Relaxed,
        );
        resolved
    }
}

const RESOLVED_NONE: usize = 0;
const RESOLVED_THREADS: usize = 1;
const RESOLVED_URING: usize = 2;

/// What the last [`Backend::resolve`] settled on, process-wide.
///
/// A benchmark that cannot read this cannot tell a backend comparison from two runs of the same
/// backend: asking for `"uring"` and silently getting threads returns correct data at the speed of
/// the thing you were trying to measure against.
static LAST_RESOLVED: AtomicUsize = AtomicUsize::new(RESOLVED_NONE);

/// Hints submitted and hints the kernel rejected, since the ring was created.
///
/// Without these, a null result is unreadable: "GPFS ignored the advice", "the pacing never issued
/// any", and "every one failed with EINVAL" all look identical from the outside, and the first is
/// the only one that says anything about the filesystem.
static HINTS_SUBMITTED: AtomicUsize = AtomicUsize::new(0);
static HINTS_FAILED: AtomicUsize = AtomicUsize::new(0);

/// The most reads this process ever had outstanding at once, and how many rings it has built.
///
/// Depth is a request, not an outcome: the submit loop stops early on a full queue, on refused
/// credit, or simply on running out of reads, so a plan of 600 units spread over 14 arrays may
/// never reach a depth of 256 even once. Ring count exposes the other half -- a ring is built per
/// CALL, so fourteen arrays read one after another build fourteen of them and none of their reads
/// ever overlap.
static MAX_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
static RINGS_BUILT: AtomicUsize = AtomicUsize::new(0);

/// `(max_in_flight, rings_built)` since this process started.
pub fn ring_stats() -> (usize, usize) {
    (
        MAX_IN_FLIGHT.load(Ordering::Relaxed),
        RINGS_BUILT.load(Ordering::Relaxed),
    )
}

/// `(submitted, failed)` hints so far.
pub fn hint_stats() -> (usize, usize) {
    (
        HINTS_SUBMITTED.load(Ordering::Relaxed),
        HINTS_FAILED.load(Ordering::Relaxed),
    )
}

/// `"threads"`, `"uring"`, or `"none"` when no planned read has resolved a backend yet.
pub fn last_resolved() -> &'static str {
    match LAST_RESOLVED.load(Ordering::Relaxed) {
        RESOLVED_THREADS => "threads",
        RESOLVED_URING => "uring",
        _ => "none",
    }
}

/// Whether this process can create a ring.
///
/// Probed once, by creating one, because every cheaper test lies: the syscall exists on any modern
/// kernel, the symbols sit in `/proc/kallsyms` even when `io_uring_disabled=2`, and the sysctl can
/// permit only a group this process is not in. Only `io_uring_setup` handing back a ring proves it.
#[cfg(target_os = "linux")]
pub fn uring_usable() -> bool {
    use std::sync::OnceLock;
    static USABLE: OnceLock<bool> = OnceLock::new();
    *USABLE.get_or_init(|| io_uring::IoUring::new(8).is_ok())
}

#[cfg(not(target_os = "linux"))]
pub fn uring_usable() -> bool {
    false
}

/// Run `f(index)` for every index below `count`, on `threads` plain OS threads.
///
/// The work list is known up front and fixed, so an atomic cursor does what a job queue would, without
/// the lock and the allocation.
pub fn for_each_blocking<F>(count: usize, threads: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    if count == 0 {
        return;
    }
    let threads = threads.max(1).min(count);
    let next = AtomicUsize::new(0);
    // ponytail: threads are spawned per call -- about a millisecond for 32, immaterial beside a
    // read measured in seconds. A persistent pool would need a channel of boxed jobs to carry the
    // borrows this scope gets for free; build it only if a profile asks.
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= count {
                        break;
                    }
                    f(index);
                }
            });
        }
    });
}

/// One read to issue through the ring.
pub struct UringRead {
    pub path: PathBuf,
    pub offset: u64,
    pub length: u64,
}

/// Page size, asked once. Hints are applied by the kernel at page granularity, so a range that does
/// not cover whole pages hints less than it names.
#[cfg(target_os = "linux")]
pub fn page_size() -> u64 {
    use std::sync::OnceLock;
    static PAGE: OnceLock<u64> = OnceLock::new();
    *PAGE.get_or_init(|| {
        // SAFETY: sysconf with a valid name and no out-parameters.
        let raw = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if raw > 0 { raw as u64 } else { 4096 }
    })
}

/// Nothing hints off Linux, so this only has to be a sane number for `page_align`'s arithmetic.
#[cfg(not(target_os = "linux"))]
pub fn page_size() -> u64 {
    4096
}

/// Widen a byte range to the pages containing it.
///
/// Rounding the start down matters more than it looks: `fadvise` on a range starting mid-page still
/// pulls that page, but a caller that hints `[offset, offset+len)` and then reads the same range
/// has hinted one page fewer than it reads whenever the start is unaligned.
pub fn page_align(offset: u64, length: u64) -> (u64, u64) {
    let page = page_size();
    let start = offset - (offset % page);
    let end = offset.saturating_add(length).div_ceil(page).saturating_mul(page);
    (start, end.saturating_sub(start))
}

/// Issue every read through one ring, calling `on_ready` with each buffer as it lands.
///
/// `on_ready` runs on the ring thread and must not decode: decoding there would put CPU work on the
/// IO thread and stop completions being reaped while it ran. Hand the buffer to the decode pool and
/// return. Completions do not arrive in submission order, so the index says which read finished.
///
/// A read whose file is missing yields `Ok(None)`, matching a store that reports a key as absent.
#[cfg(target_os = "linux")]
pub fn uring_read_all<C, A, F>(
    reads: &[UringRead],
    depth: usize,
    lookahead: usize,
    mut admit: A,
    on_ready: F,
) -> io::Result<()>
where
    A: FnMut(usize, bool) -> Option<C>,
    F: Fn(usize, io::Result<Option<Vec<u8>>>, C),
{
    use io_uring::{IoUring, opcode, types};
    use std::collections::HashMap;
    use std::fs::File;
    use std::os::unix::io::AsRawFd;

    if reads.is_empty() {
        return Ok(());
    }
    let depth = depth.clamp(1, 4096).min(reads.len());
    // The queue holds reads AND hints. Sizing it for `depth` alone made every hint cost a read its
    // slot, which is the opposite of the point: the first pass spent 64 of 256 entries on hints and
    // then had no room left to hint again, so the rolling window did not roll.
    let hint_cap = lookahead.min(4096);
    let entries = (depth.saturating_add(hint_cap)).clamp(1, 32768) as u32;
    let mut ring = IoUring::new(entries.next_power_of_two())?;
    RINGS_BUILT.fetch_add(1, Ordering::Relaxed);

    // One fd per file rather than per read: a shard serves many ranges, and reopening it for each
    // would put back exactly the open/close pair the file handle cache exists to remove.
    let mut files: HashMap<&std::path::Path, Option<File>> = HashMap::new();
    for read in reads {
        files.entry(read.path.as_path()).or_insert_with(|| File::open(&read.path).ok());
    }

    // A slot is one in-flight read. `user_data` carries the slot index back, since that is all the
    // kernel returns, and `slot_read` maps it to the read it serves.
    // Reads and hints share one ring, so a completion has to say which it was. Reads carry their
    // slot index; a hint carries this bit and nothing else, because there is nothing to do with a
    // hint completion except stop counting it.
    const HINT_TAG: u64 = 1 << 63;

    let mut buffers: Vec<Vec<u8>> = (0..depth).map(|_| Vec::new()).collect();
    let mut slot_read: Vec<usize> = vec![usize::MAX; depth];
    // Credit is taken when a read is SUBMITTED and travels with it to `on_ready`, so this thread
    // never waits on the budget while completions are outstanding.
    let mut slot_credit: Vec<Option<C>> = (0..depth).map(|_| None).collect();
    let mut free: Vec<usize> = (0..depth).collect();
    let mut next = 0usize;
    let mut in_flight = 0usize;
    // How far ahead of `next` hints have been issued, and how many are outstanding. Hints are
    // pushed as ordinary SQEs, so they occupy queue entries and must be counted like any other.
    let mut hint_next = 0usize;
    let mut hints_in_flight = 0usize;

    while next < reads.len() || in_flight > 0 {
        // READS FIRST. Hints exist to help these; letting them take queue entries the reads
        // wanted is strictly backwards, and sizing the ring for depth alone made that unavoidable.
        while in_flight < depth && next < reads.len() {
            // Admission gates SUBMISSION, and never blocks while anything is in flight: a
            // completion is the thing that frees credit, so a thread parked here is a thread not
            // reaping the completion that would wake it. When nothing is in flight there is
            // nothing to reap, so blocking is simply waiting for a decode and costs nothing.
            let Some(credit) = admit(next, in_flight == 0) else {
                break; // out of credit -- go reap, come back with room
            };
            let read = &reads[next];
            let Some(Some(file)) = files.get(read.path.as_path()) else {
                on_ready(next, Ok(None), credit); // no such shard, so it is all fill value
                next += 1;
                continue;
            };
            if read.length == 0 {
                on_ready(next, Ok(None), credit);
                next += 1;
                continue;
            }
            let slot = free.pop().expect("in_flight < depth leaves a slot free");
            let length = usize::try_from(read.length).unwrap_or(0);
            buffers[slot].clear();
            buffers[slot].resize(length, 0);
            slot_read[slot] = next;
            let entry = opcode::Read::new(
                types::Fd(file.as_raw_fd()),
                buffers[slot].as_mut_ptr(),
                length as u32,
            )
            .offset(read.offset)
            .build()
            .user_data(slot as u64);
            // SAFETY: the buffer lives in `buffers` for the whole loop, and its slot is not reused
            // until this read's completion has been reaped below.
            if unsafe { ring.submission().push(&entry) }.is_err() {
                slot_read[slot] = usize::MAX;
                free.push(slot);
                // Credit drops with `credit` here, so the retry takes it again rather than leaking
                // the ceiling one read at a time.
                break; // the queue is full; reap before pushing more
            }
            slot_credit[slot] = Some(credit);
            next += 1;
            in_flight += 1;
            MAX_IN_FLIGHT.fetch_max(in_flight, Ordering::Relaxed);
        }

        // THEN the hint phase, with its own capacity rather than the reads' -- the ring was sized
        // for `depth + hint_cap`, so a hint here is never a read that did not get issued.
        //
        // FADVISE(WILLNEED) asks the kernel to start readahead and returns without waiting for it,
        // so a read issued later against pages already resident is a cache hit -- and a cache hit is
        // the only buffered read io_uring can complete without punting to an io-wq worker.
        //
        // The window ROLLS: hints run ahead of the read cursor and are topped up on every pass, i.e.
        // as earlier reads land. Hinting the whole plan up front instead would ask the kernel to
        // pull all of it into page cache at once and evict the head before the reads arrived.
        while hint_cap > 0
            && hint_next < reads.len()
            && hints_in_flight < hint_cap
            && hint_next < next.saturating_add(hint_cap)
        {
            let hint = &reads[hint_next];
            let Some(Some(file)) = files.get(hint.path.as_path()) else {
                hint_next += 1;
                continue;
            };
            if hint.length == 0 {
                hint_next += 1;
                continue;
            }
            let (offset, length) = page_align(hint.offset, hint.length);
            // `len` is an `off_t`, so i64 rather than the u32 a read's length is.
            let entry = opcode::Fadvise::new(
                types::Fd(file.as_raw_fd()),
                i64::try_from(length).unwrap_or(i64::MAX),
                libc::POSIX_FADV_WILLNEED,
            )
            .offset(offset)
            .build()
            .user_data(HINT_TAG);
            // SAFETY: fadvise borrows no user buffer, so the entry owns nothing that must outlive
            // the submission.
            if unsafe { ring.submission().push(&entry) }.is_err() {
                break; // queue full: reap, then resume hinting
            }
            HINTS_SUBMITTED.fetch_add(1, Ordering::Relaxed);
            hint_next += 1;
            hints_in_flight += 1;
        }

        if in_flight == 0 {
            if hints_in_flight > 0 {
                // Hints are queued but no read is: submit them so the readahead is already running
                // by the time the next read is admitted, and reap them so they stop occupying the
                // ring. Waiting on a hint is fine -- it completes as soon as the kernel has queued
                // the readahead, not when the data arrives.
                ring.submit_and_wait(1)?;
                let reaped = ring.completion().count();
                hints_in_flight = hints_in_flight.saturating_sub(reaped);
            }
            continue;
        }
        ring.submit_and_wait(1)?;
        // Collected before dispatch so the completion queue is not borrowed while `on_ready` runs.
        let reaped: Vec<(usize, i32)> = ring
            .completion()
            .map(|cqe| (cqe.user_data() as usize, cqe.result()))
            .collect();
        for (slot, result) in reaped {
            if slot as u64 & HINT_TAG != 0 {
                // A hint completion says the readahead was QUEUED, not that it finished. A failure
                // is still not an error -- advice the kernel declined is advice -- but it is counted,
                // because "the filesystem ignored the hint" and "every hint was rejected" are
                // different findings and silently discarding the result conflates them.
                if result < 0 {
                    HINTS_FAILED.fetch_add(1, Ordering::Relaxed);
                }
                hints_in_flight = hints_in_flight.saturating_sub(1);
                continue;
            }
            let index = std::mem::replace(&mut slot_read[slot], usize::MAX);
            let credit = slot_credit[slot]
                .take()
                .expect("a slot in flight was submitted, and submission takes credit");
            let wanted = buffers[slot].len();
            if result < 0 {
                on_ready(index, Err(io::Error::from_raw_os_error(-result)), credit);
            } else if (result as usize) < wanted {
                // A short read is not an error to io_uring. It is one here: the decoder would
                // otherwise be handed a buffer whose tail is still the zeros it was sized with.
                on_ready(
                    index,
                    Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!("short read: {result} of {wanted} bytes"),
                    )),
                    credit,
                );
            } else {
                on_ready(index, Ok(Some(std::mem::take(&mut buffers[slot]))), credit);
            }
            free.push(slot);
            in_flight -= 1;
        }
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn uring_read_all<C, A, F>(
    _reads: &[UringRead],
    _depth: usize,
    _lookahead: usize,
    _admit: A,
    _on_ready: F,
) -> io::Result<()>
where
    A: FnMut(usize, bool) -> Option<C>,
    F: Fn(usize, io::Result<Option<Vec<u8>>>, C),
{
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "io_uring is Linux-only",
    ))
}
