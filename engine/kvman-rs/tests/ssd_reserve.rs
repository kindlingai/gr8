//! The SSD block file is allocated, not just sized, and preserves its contents across reopen.
use g6kvman::tier::{AlignedBuf, SlotFile, ALIGN};
use std::os::unix::fs::MetadataExt;

#[test]
fn a_reserved_slot_file_excludes_other_allocations() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("c21-reserved-blocks-{}", std::process::id()));
    let bytes = 12 * 1024 * 1024u64;
    let slots = bytes / ALIGN as u64;
    let store = SlotFile::create(&path, ALIGN, slots, true).unwrap();
    let meta = std::fs::metadata(&path).unwrap();
    assert_eq!(meta.len(), bytes);
    assert!(meta.blocks() * 512 >= bytes, "file is sparse: {} of {bytes} bytes allocated", meta.blocks() * 512);

    let mut bytes = AlignedBuf::new(ALIGN);
    bytes.as_mut_slice().fill(0xa5);
    store.write(slots - 1, bytes.as_slice()).unwrap();
    bytes.as_mut_slice().fill(0);
    store.read(slots - 1, bytes.as_mut_slice()).unwrap();
    assert!(
        bytes.as_slice().iter().all(|&b| b == 0xa5),
        "reserved last slot is writable"
    );
    drop(store);
    let reopened = SlotFile::create(&path, ALIGN, slots, true).unwrap();
    bytes.as_mut_slice().fill(0);
    reopened.read(slots - 1, bytes.as_mut_slice()).unwrap();
    assert!(bytes.as_slice().iter().all(|&b| b == 0xa5), "preallocation must preserve a saved block across restart");
    drop(reopened);
    std::fs::remove_file(&path).unwrap();
}
