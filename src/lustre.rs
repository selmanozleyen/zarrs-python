//! How many reads the Lustre pool a file lives on can take in flight from this client, found the
//! way `available_parallelism` finds cores: read off the client's own files, with no `lctl`, `lfs`
//! or liblustreapi. Anything missing or unexpected is `None`, and the caller keeps its default.

use std::path::Path;

const LUSTRE_SUPER_MAGIC: u64 = 0x0BD0_0BD0;
const LOV_MAGIC_V1: u32 = 0x0BD1_0BD0;
const LOV_MAGIC_V3: u32 = 0x0BD3_0BD0;
const LOV_MAGIC_COMP_V1: u32 = 0x0BD6_0BD0;

/// The pool's OSTs times each one's `max_rpcs_in_flight`, clamped to [64, 1024]: past that many,
/// further reads only queue in the client.
pub(crate) fn read_capacity(file: &Path) -> Option<usize> {
    let file = std::fs::canonicalize(file).ok()?;
    if !is_lustre(&file) {
        return None;
    }
    let fsname = fsname(&std::fs::read_to_string("/proc/self/mountinfo").ok()?, &file)?;
    let osts: Vec<String> = match pool(&xattr(&file, "lustre.lov")?)? {
        Some(pool) => std::fs::read_to_string(lov_dir(&fsname)?.join("pools").join(pool))
            .ok()?
            .lines()
            .filter_map(|l| l.trim().strip_suffix("_UUID").map(String::from))
            .collect(),
        None => all_osts(&fsname)?,
    };
    let total = osts.iter().map(|o| rpcs_in_flight(o)).sum::<Option<usize>>()?;
    (total > 0).then(|| total.clamp(64, 1024))
}

#[cfg(target_os = "linux")]
fn is_lustre(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(c) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
        return false;
    };
    let mut st = std::mem::MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: `c` is a nul-terminated path, and `st` is read only after statfs has filled it.
    unsafe {
        libc::statfs(c.as_ptr(), st.as_mut_ptr()) == 0
            && st.assume_init().f_type as u64 == LUSTRE_SUPER_MAGIC
    }
}

#[cfg(not(target_os = "linux"))]
fn is_lustre(_: &Path) -> bool {
    false
}

#[cfg(target_os = "linux")]
fn xattr(path: &Path, name: &str) -> Option<Vec<u8>> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    let name = std::ffi::CString::new(name).ok()?;
    let mut buf = vec![0u8; 1 << 16];
    // SAFETY: `buf` is valid for `buf.len()` bytes; the return is the length written, or -1.
    let len = unsafe { libc::getxattr(path.as_ptr(), name.as_ptr(), buf.as_mut_ptr().cast(), buf.len()) };
    (len > 0).then(|| {
        buf.truncate(len as usize);
        buf
    })
}

#[cfg(not(target_os = "linux"))]
fn xattr(_: &Path, _: &str) -> Option<Vec<u8>> {
    None
}

/// The filesystem name of the longest Lustre mount covering `path`: its source ends in `:/<fsname>`.
fn fsname(mountinfo: &str, path: &Path) -> Option<String> {
    mountinfo
        .lines()
        .filter_map(|line| {
            let (pre, post) = line.split_once(" - ")?;
            let mut post = post.split_whitespace();
            if post.next()? != "lustre" {
                return None;
            }
            let (_, fs) = post.next()?.rsplit_once(":/")?;
            let mount = pre.split_whitespace().nth(4)?.replace("\\040", " ");
            path.starts_with(&mount).then(|| (mount.len(), fs.to_string()))
        })
        .max_by_key(|(len, _)| *len)
        .map(|(_, fs)| fs)
}

/// The pool a layout names: `Some(None)` for none (all OSTs), `None` when it cannot be told --
/// an unknown layout, or components that disagree.
fn pool(lov: &[u8]) -> Option<Option<String>> {
    let u32at = |at: usize| lov.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    match u32at(0)? {
        LOV_MAGIC_V1 => Some(None),
        LOV_MAGIC_V3 => {
            let name = lov.get(32..48)?.split(|b| *b == 0).next()?;
            Some((!name.is_empty()).then(|| String::from_utf8_lossy(name).into_owned()))
        }
        LOV_MAGIC_COMP_V1 => {
            let count = lov.get(14..16).map(|b| u16::from_le_bytes(b.try_into().unwrap()))? as usize;
            let mut pools = (0..count).map(|i| pool(lov.get(u32at(32 + 48 * i + 24)? as usize..)?));
            let first = pools.next()??;
            pools.all(|p| p.as_ref() == Some(&first)).then_some(first)
        }
        _ => None,
    }
}

fn lov_dir(fsname: &str) -> Option<std::path::PathBuf> {
    let prefix = format!("{fsname}-clilov-");
    std::fs::read_dir("/proc/fs/lustre/lov")
        .ok()?
        .flatten()
        .find(|e| e.file_name().to_string_lossy().starts_with(&prefix))
        .map(|e| e.path())
}

/// Every OST of the filesystem, named `<fsname>-OSTxxxx` like the pool files name them.
fn all_osts(fsname: &str) -> Option<Vec<String>> {
    let prefix = format!("{fsname}-OST");
    let osts: std::collections::BTreeSet<String> = std::fs::read_dir("/sys/fs/lustre/osc")
        .ok()?
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.starts_with(&prefix).then(|| name.split_once("-osc-").map(|(o, _)| o.to_string()))?
        })
        .collect();
    (!osts.is_empty()).then(|| osts.into_iter().collect())
}

fn rpcs_in_flight(ost: &str) -> Option<usize> {
    let prefix = format!("{ost}-osc-");
    let dir = std::fs::read_dir("/sys/fs/lustre/osc")
        .ok()?
        .flatten()
        .find(|e| e.file_name().to_string_lossy().starts_with(&prefix))?;
    std::fs::read_to_string(dir.path().join("max_rpcs_in_flight")).ok()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v3(pool: &str) -> Vec<u8> {
        let mut b = vec![0u8; 48];
        b[..4].copy_from_slice(&LOV_MAGIC_V3.to_le_bytes());
        b[32..32 + pool.len()].copy_from_slice(pool.as_bytes());
        b
    }

    fn comp(blobs: &[Vec<u8>]) -> Vec<u8> {
        let mut b = vec![0u8; 32 + 48 * blobs.len()];
        b[..4].copy_from_slice(&LOV_MAGIC_COMP_V1.to_le_bytes());
        b[14..16].copy_from_slice(&(blobs.len() as u16).to_le_bytes());
        for (i, blob) in blobs.iter().enumerate() {
            let at = b.len() as u32;
            b[32 + 48 * i + 24..32 + 48 * i + 28].copy_from_slice(&at.to_le_bytes());
            b.extend_from_slice(blob);
        }
        b
    }

    /// The three shapes seen on the HMGU client, and the ones that must fall back.
    #[test]
    fn pool_reads_plain_and_composite_layouts() {
        assert_eq!(pool(&v3("ddn_hdd")), Some(Some("ddn_hdd".into())));
        let ssd = comp(&[v3("ddn_ssd"), v3("ddn_ssd"), v3("ddn_ssd")]);
        assert_eq!(pool(&ssd), Some(Some("ddn_ssd".into())));
        assert_eq!(pool(&LOV_MAGIC_V1.to_le_bytes()), Some(None), "no pool: every OST");
        assert_eq!(pool(&comp(&[v3("ddn_ssd"), v3("ddn_hdd")])), None, "components disagree");
        assert_eq!(pool(&[1, 2, 3, 4]), None, "unknown magic");
        assert_eq!(pool(&comp(&[])), None, "no components");
        let mut short = v3("ddn_ssd");
        short.truncate(20);
        assert_eq!(pool(&short), None, "truncated");
    }

    #[test]
    fn fsname_takes_the_longest_lustre_mount_covering_the_path() {
        let mountinfo = "\
22 1 8:1 / / rw - ext4 /dev/sda1 rw
263 75 2389:231328 / /ictstr01 rw shared:521 - lustre 10.11.1.16@o2ib,10.11.1.17@o2ib:10.11.1.14@o2ib:/ictstr01 rw,flock
264 75 2389:231329 / /scratch rw - lustre 10.0.0.1@o2ib:/scr rw";
        let got = |p: &str| fsname(mountinfo, Path::new(p));
        assert_eq!(got("/ictstr01/boost_ai/users/x/a.zarr/c/0").as_deref(), Some("ictstr01"));
        assert_eq!(got("/scratch/y").as_deref(), Some("scr"));
        assert_eq!(got("/home/z"), None, "ext4 is not lustre");
        assert_eq!(got("/ictstr01x/a"), None, "a prefix of the name is not a parent directory");
    }
}
