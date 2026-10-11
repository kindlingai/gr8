//! gr8 S2 KV page allocator (the KV manager). Host only. One pool per rank. Every call is at an all-rank event fence
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
//! Growth on demand: a run below R starts with the pages its request needs now (the engine: prompt + 1 + guard rows)
//! and grows (grow) to at most its worst case (prompt + max_tokens + guard), its new pages at the end of its list. The
//! pool only admits a request (fits) or grants growth when the state stays safe in the Banker's sense: with the live
//! runs ordered by remaining need, each can reach its worst case from the free pages plus what the runs before it give
//! back. Retained runs count as free. A refused growth leaves the session out of a round (Driver::step) instead of
//! taking pages back, since a re-prefilled session would not emit the same bytes. From a safe state some session can
//! always go on, so the engine never deadlocks. An exclusive run takes its worst case at once.
//!
//! The C interface (g6kvman.h, src/ffi.rs) is what engine/runtime/s2/pool.h wraps for g6m4 and m4sim.
//!
//! src/tier.rs is a proof of concept of an SSD store for retained runs (SSD-TIER.md), and src/remote a KV node that
//! serves the same blocks from another box's locked memory over RDMA or TCP (REMOTE-TIER.md). The engine uses
//! neither.

pub mod ffi;
pub mod pool;
pub mod remote;
pub mod reuse;
pub mod tier;

pub use pool::{Error, Hit, Kept, Pool, Run, PAGE};
pub use reuse::{Finished, Rules, Why};
