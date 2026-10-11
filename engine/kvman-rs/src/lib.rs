//! g6 S2 KV page allocator (the KV manager). Host only. One pool per rank. Every call is at an all-rank event fence
//! (engine/runtime/s2/service.h), and every rank makes the same calls in the same order: the head reserves (or takes a
//! match hit) at the event, and each worker repeats that call at its grant and requires the same Run, list hash
//! included. Nothing here depends on time, floats or hash-map order, so the same calls give the same pages and digest.
//!
//! A session's run is a list of pages, the lowest free ones first. A reservation of more than `below` rows (a session
//! past R, CP decode) is exclusive. It is admitted only into an EMPTY pool, takes pages 0, 1, .., n - 1 (the CP map as
//! built) and, while resident, holds the whole pool (room 0, nothing else admits).
//! Two sizes on the CP binding: `pages` = the logical capacity (Config::pool_rows / 64), `place` = the physical pages
//! every rank has for runs below R (S2Model::pool_pages, the fewest of main and draft). Runs below R take pages in
//! [0, place).
//! Prefix reuse (keep): a retired run's list stays, with the session's finished PrefixReuse state, and its pages count
//! as free. A reservation takes pages no run holds or retains. When those are too few, it evicts whole retained runs,
//! the least recently retired first, so a miss destroys the oldest retained runs and keeps the fresh ones. match finds
//! the retained run whose cached prefix a new prompt continues. take adopts that run's first ceil(skip / 64) pages, lets
//! the rest go and adds pages as reserve does. Rows skip.. of the last adopted page are written before any step reads
//! them.
//!
//! The C interface (g6kvman.h, src/ffi.rs) is what engine/runtime/s2/pool.h wraps for g6m4.
//!
//! src/tier.rs is a proof of concept of an SSD store for retained runs (SSD-TIER.md). The engine does not use it.

pub mod ffi;
pub mod pool;
pub mod reuse;
pub mod tier;

pub use pool::{Error, Hit, Kept, Pool, Run, PAGE};
pub use reuse::{Finished, Rules, Why};
