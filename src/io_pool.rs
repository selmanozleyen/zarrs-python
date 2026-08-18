//! Issuing the reads of a planned read.
//!
//! One way only: a pool of blocking threads. There was an io_uring backend and an
//! FADVISE hint phase here; both are gone. The ring never beat threads on either site
//! measured, could not be created at all where `kernel.io_uring_disabled=2`, and forced
//! every caller to carry a backend choice, a ring depth and a hint lookahead that did
//! nothing on the path they actually took.

use std::sync::atomic::{AtomicUsize, Ordering};

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
