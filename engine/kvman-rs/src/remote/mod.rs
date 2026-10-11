//! Remote KV, a proof of concept: a node that serves fixed-size page blocks from its own locked memory (server, the
//! kvman-remote binary), and a client store over a lease on it (client::RemoteStore, a tier::BlockStore). RDMA (RoCE v2
//! RC, one-sided reads and writes into the node's registered memory) when both ends' addresses on the connection have
//! a RoCE v2 GID, TCP otherwise. libibverbs is loaded at run time, so nothing here needs rdma-core to build.
//! REMOTE-TIER.md has the design and the measurements.

pub mod client;
pub mod proto;
pub mod server;
pub mod sys;
pub mod verbs;

pub use client::{Opts, RemoteStore, Transport, Want};
pub use server::{Config, Node};
