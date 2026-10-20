//! KVMAN_TRANSPORT reaches a group lease's open too (blocks.rs transport), not only a plain one's: a bad value
//! refuses before any connection. Its own test binary: it sets the process environment.
use g6kvman::blocks::Blocks;
use g6kvman::remote::GroupOpts;

#[test]
fn a_bad_kvman_transport_refuses_plain_and_group_opens() {
    std::env::set_var("KVMAN_TRANSPORT", "carrier-pigeon");
    let g = GroupOpts { group: "g".into(), nr: 1, rank: 0, floor_bytes: 1 << 20, cap_bytes: 0, weight: 0 };
    let e = Blocks::open_group("remote:127.0.0.1:1", "l", 4096, g).err().expect("refused");
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput, "{e}");
    let e = Blocks::open("remote:127.0.0.1:1", "l", 4096, 4).err().expect("refused");
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput, "{e}");
    std::env::remove_var("KVMAN_TRANSPORT");
}
