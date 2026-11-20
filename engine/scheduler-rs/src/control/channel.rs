//! The event channels over the control links. Every rank's compute thread meets events at its gate (`boundary`); the
//! head's control thread announces each one, collects every rank's ACK and sends one commit while every rank waits
//! (`event`); a worker's control thread reads the head's frames (`receive_events`). A barrier between rounds (`sync`)
//! lines the ranks up in time. No socket activity in a round without an event or a barrier.
//!
//! The links are the C++ ControlIO's (`Io`), borrowed for the endpoint's life: the facade's links outlive its channel,
//! as before. Who touches a link: on the head, the control thread sends events and the compute thread sends barrier
//! releases, under one send lock with the watcher's quit. Whichever of the control thread (ACKs) and the compute
//! thread (barrier arrivals) waits reads a worker's link and files what it gets. On a worker, the control thread is
//! the link's only reader and the compute thread its only sender.

use super::gate::{lock, Agreement, Gate};
use super::io::Io;
use super::wire::{self, Ack, Commit, Notice, ADMIT};
use super::{refused, require, R};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

/// Wall clock in microseconds, for the event log's stamps only (comparable across ranks up to their clock skew).
pub fn wall_us() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_micros() as i64).unwrap_or(0)
}

/// A borrowed link (the facade's ControlIO, which outlives the endpoint).
#[derive(Clone, Copy)]
pub struct Link(*const Io);
// SAFETY: Io's calls take &self and are safe from any thread. The facades keep the pointee alive longer than the
// endpoint.
unsafe impl Send for Link {}
unsafe impl Sync for Link {}
impl Link {
    /// Borrows `io`, which must outlive every endpoint that holds the link.
    pub fn new(io: *const Io) -> Link {
        Link(io)
    }
    fn io(&self) -> &Io {
        // SAFETY: see the type's contract
        unsafe { &*self.0 }
    }
}

struct State {
    notice: Option<Notice>,
    commit: Option<Commit>,
    ack: Option<Ack>,
    failed: bool,
    // when the current notice was installed and when this rank arrived at its boundary (wall_us): its real margin
    installed_us: i64,
    arrived_us: i64,
    // head: each worker's ACK and barrier arrivals, and whether a thread is reading its link
    acks: Vec<Option<Ack>>,
    synced: Vec<u64>,
    reading: Vec<bool>,
    // worker: the latest bytes of each announced request, and the last barrier released
    requests: BTreeMap<u64, Vec<u8>>,
    go: u64,
}

enum Role {
    // order = ranks 1..NR-1; send serialises the control thread's events, compute's releases and the watcher's quit
    Head { peers: Vec<Link>, send: Mutex<()> },
    Worker { io: Link, rank: u32 },
}

/// One rank's end of the event channel: the head's or a worker's.
pub struct Endpoint {
    gate: Gate,
    epoch: u64,
    mu: Mutex<State>,
    cv: Condvar,
    role: Role,
    // head: the least room of the last event's ACKs, read inside make_commit
    agreed_room: AtomicU32,
    // worker: the head's orderly quit arrived
    quit_received: AtomicBool,
}

impl Endpoint {
    fn new(epoch: u64, rank: u32, n: usize, role: Role) -> R<Endpoint> {
        let gate = Gate::new(epoch, rank)?;
        let state = State {
            notice: None,
            commit: None,
            ack: None,
            failed: false,
            installed_us: 0,
            arrived_us: 0,
            acks: vec![None; n],
            synced: vec![0; n],
            reading: vec![false; n],
            requests: BTreeMap::new(),
            go: 0,
        };
        Ok(Endpoint {
            gate,
            epoch,
            mu: Mutex::new(state),
            cv: Condvar::new(),
            role,
            agreed_room: AtomicU32::new(u32::MAX),
            quit_received: AtomicBool::new(false),
        })
    }
    /// The head over its links to ranks 1.. (none at TP1: it agrees with itself).
    pub fn head(epoch: u64, peers: Vec<Link>) -> R<Endpoint> {
        let n = peers.len();
        let e = Endpoint::new(epoch, 0, n, Role::Head { peers, send: Mutex::new(()) })?;
        require(n <= 5, "expected TP1 to TP6 peers")?;
        Ok(e)
    }
    /// Worker `rank` over its link to the head.
    pub fn worker(epoch: u64, rank: u32, io: Link) -> R<Endpoint> {
        let e = Endpoint::new(epoch, rank, 0, Role::Worker { io, rank })?;
        require(rank > 0, "worker rank zero")?;
        Ok(e)
    }
    fn state(&self) -> MutexGuard<'_, State> {
        lock(&self.mu)
    }
    fn wait<'a>(&self, g: MutexGuard<'a, State>, until: impl Fn(&State) -> bool) -> MutexGuard<'a, State> {
        self.cv.wait_while(g, |s| !until(s)).unwrap_or_else(|e| e.into_inner())
    }
    /// Fails the epoch: wakes every waiter and shuts the links down.
    pub fn fail(&self) {
        self.state().failed = true;
        self.gate.fail();
        self.cv.notify_all();
        match &self.role {
            Role::Head { peers, .. } => peers.iter().for_each(|p| p.io().shutdown()),
            Role::Worker { io, .. } => io.io().shutdown(),
        }
    }
    fn failing<T>(&self, r: R<T>) -> R<T> {
        if r.is_err() {
            self.fail();
        }
        r
    }
    fn install(&self, n: &Notice) -> R<()> {
        let g = self.state();
        let mut g = self.wait(g, |s| s.notice.is_none() || s.failed);
        require(!g.failed, "control epoch failed")?;
        self.gate.announce(n)?;
        g.notice = Some(*n);
        g.commit = None;
        g.ack = None;
        g.installed_us = wall_us();
        g.arrived_us = 0;
        self.cv.notify_all();
        Ok(())
    }
    fn deliver(&self, n: &Notice, c: &Commit) -> R<()> {
        let mut g = self.state();
        require(!g.failed && g.notice == Some(*n), "unexpected event commit")?;
        wire::check_match(n, c)?;
        g.commit = Some(*c);
        self.gate.commit(n)
    }
    fn send_ack(&self, a: &Ack) -> R<()> {
        match &self.role {
            Role::Head { .. } => {
                self.state().ack = Some(*a);
                self.cv.notify_all();
                Ok(())
            }
            Role::Worker { io, .. } => io.io().send(&wire::encode_ack(a)),
        }
    }
    /// Before departing a live boundary: None, with no digest, lock, allocation or I/O, when no event is due. Else
    /// `digest` gives this rank's state digest and room, the rank acknowledges and waits for the commit. Idle engines
    /// call wait_notice and do not depart.
    pub fn boundary(&self, b: u64, digest: impl FnOnce() -> R<(u64, u32)>) -> R<Option<Commit>> {
        let r = (|| {
            if !self.gate.before_work(b)? {
                return Ok(None);
            }
            let (d, room) = digest()?;
            let a = self.gate.arrive(b, d, room)?;
            self.state().arrived_us = wall_us();
            self.send_ack(&a)?;
            self.gate.wait_commit()?;
            let c = self.state().commit;
            c.map(Some).ok_or_else(|| refused("missing commit payload"))
        })();
        self.failing(r)
    }
    /// The pending notice, once one is installed.
    pub fn wait_notice(&self) -> R<Notice> {
        let g = self.state();
        let g = self.wait(g, |s| s.notice.is_some() || s.failed);
        require(!g.failed, "control epoch failed")?;
        Ok(g.notice.unwrap_or_default())
    }
    /// The current event's installed and arrived stamps (wall_us); between boundary and applied.
    pub fn stamps(&self) -> (i64, i64) {
        let g = self.state();
        (g.installed_us, g.arrived_us)
    }
    /// The commit is applied: the gate moves on (or holds) and the head's event returns.
    pub fn applied(&self) -> R<()> {
        let mut g = self.state();
        let (n, c) = match (g.notice, g.commit) {
            (Some(n), Some(c)) if !g.failed => (n, c),
            _ => return Err(refused("apply without event commit")),
        };
        self.gate.applied(&n, c.hold)?;
        g.notice = None;
        g.commit = None;
        g.ack = None;
        self.cv.notify_all();
        Ok(())
    }
    /// The compute thread, every rank at the same point with the same seq (1, 2, ...): returns once every rank has
    /// arrived.
    pub fn sync(&self, seq: u64) -> R<()> {
        let r = match &self.role {
            Role::Head { peers, .. } => self.head_sync(peers.len(), seq),
            Role::Worker { io, rank } => self.worker_sync(io, *rank, seq),
        };
        self.failing(r)
    }
    fn head_sync(&self, n: usize, seq: u64) -> R<()> {
        for i in 0..n {
            self.await_link(i, |s| s.synced[i] >= seq)?;
        }
        require(self.state().synced.iter().all(|&s| s == seq), "barrier arrival out of order")?;
        self.send_all(&wire::encode_go(self.epoch, seq))
    }
    fn worker_sync(&self, io: &Link, rank: u32, seq: u64) -> R<()> {
        io.io().send(&wire::encode_sync(self.epoch, rank, seq))?;
        let g = self.state();
        let g = self.wait(g, |s| s.go >= seq || s.failed);
        require(!g.failed, "control epoch failed")?;
        require(g.go == seq, "barrier release out of order")
    }

    fn peers(&self) -> R<&[Link]> {
        match &self.role {
            Role::Head { peers, .. } => Ok(peers),
            Role::Worker { .. } => Err(refused("a worker's channel has no peers")),
        }
    }
    fn send_all(&self, frame: &[u8]) -> R<()> {
        if let Role::Head { peers, send } = &self.role {
            let _l = lock(send);
            for p in peers {
                p.io().send(frame)?;
            }
        }
        Ok(())
    }
    // A worker's frames are ACKs (the control thread waits for them in event) and barrier arrivals (compute waits in
    // sync), in the worker's order on one link: whichever thread waits reads the link and files what it gets.
    fn await_link(&self, i: usize, done: impl Fn(&State) -> bool) -> R<()> {
        let peers = self.peers()?;
        let mut g = self.state();
        loop {
            g = self.wait(g, |s| s.failed || done(s) || !s.reading[i]);
            require(!g.failed, "control epoch failed")?;
            if done(&g) {
                return Ok(());
            }
            g.reading[i] = true;
            drop(g);
            let r = peers[i].io().receive(false);
            g = self.state();
            g.reading[i] = false;
            self.cv.notify_all();
            let b = r?;
            let d = wire::Dec::new(&b)?;
            if d.tag == wire::ACK {
                let a = wire::decode_ack(&b)?;
                require(a.rank as usize == i + 1, "ACK on wrong rank link")?;
                require(g.acks[i].is_none(), "second ACK before the commit")?;
                g.acks[i] = Some(a);
            } else if d.tag == wire::SYNC {
                let (epoch, rank, seq) = wire::decode_sync(&b)?;
                require(
                    epoch == self.epoch && rank as usize == i + 1 && seq == g.synced[i].wrapping_add(1),
                    "barrier arrival out of order",
                )?;
                g.synced[i] = seq;
            } else {
                return Err(refused("head received a frame that is not an ACK or a barrier arrival"));
            }
        }
    }
    /// The head's control thread: announce an event at a future boundary, collect every rank's ACK, then make one
    /// commit (`make_commit` with the agreed digest) while compute is paused, send it and wait until this rank has
    /// applied it. payload: an ADMIT's request bytes, sent to every worker just before the notice.
    pub fn event(&self, n: &Notice, make_commit: impl FnOnce(u64) -> R<Commit>, payload: Option<&[u8]>) -> R<()> {
        let r = self.event_inner(n, make_commit, payload);
        self.failing(r)
    }
    fn event_inner(&self, n: &Notice, make_commit: impl FnOnce(u64) -> R<Commit>, payload: Option<&[u8]>) -> R<()> {
        let peers = self.peers()?;
        require((n.kind == ADMIT) == payload.is_some(), "an ADMIT notice carries its request bytes")?;
        self.install(n)?;
        if let Some(p) = payload {
            self.send_all(&wire::encode_request(n.epoch, n.request, p))?;
        }
        self.send_all(&wire::encode_notice(n))?;
        let mut proof = Agreement::new(peers.len() as u32 + 1, *n)?;
        {
            let g = self.state();
            let g = self.wait(g, |s| s.ack.is_some() || s.failed);
            require(!g.failed, "failed while waiting local ACK")?;
            proof.ack(&g.ack.unwrap_or_default())?;
        }
        for i in 0..peers.len() {
            self.await_link(i, |s| s.acks[i].is_some())?;
            let a = self.state().acks[i].take().unwrap_or_default();
            proof.ack(&a)?;
        }
        self.agreed_room.store(proof.room()?, Ordering::Relaxed);
        let hash = proof.agreed_hash()?;
        let c = make_commit(hash)?;
        wire::check_match(n, &c)?;
        require(c.c.before_hash == hash, "commit not based on all-rank state")?;
        self.send_all(&wire::encode_commit(n, &c)?)?;
        self.deliver(n, &c)?;
        let g = self.state();
        let g = self.wait(g, |s| s.notice.is_none() || s.failed);
        require(!g.failed, "event apply failed")
    }
    /// Inside make_commit: the least room every rank's ACK of this event reported (u32::MAX: no rank has a chunk map).
    pub fn agreed_room(&self) -> u32 {
        self.agreed_room.load(Ordering::Relaxed)
    }
    /// The head passes its orderly quit to every worker. A send to a worker already gone is ignored.
    pub fn quit(&self) {
        if let Role::Head { peers, send } = &self.role {
            let _l = lock(send);
            let q = wire::encode_quit(self.epoch);
            for p in peers {
                let _ = p.io().send(&q);
            }
        }
    }
    /// Worker: the head's orderly quit arrived on the link.
    pub fn quit_received(&self) -> bool {
        self.quit_received.load(Ordering::SeqCst)
    }
    /// Worker, once it is stopping after the head's quit: closes its send side, which tells the head this rank stops.
    /// The head's frames are still read until the head closes the link.
    pub fn close_send(&self) {
        if let Role::Worker { io, .. } = &self.role {
            io.io().close_send();
        }
    }
    /// After an orderly quit, true once no rank needs this one's links open any more. Head: every worker closed its
    /// send side (close_send) or its link failed. Worker: the head closed the link, which failed the epoch.
    pub fn quit_settled(&self) -> bool {
        match &self.role {
            Role::Head { peers, .. } => peers.iter().all(|p| p.io().peer_closed()),
            Role::Worker { .. } => self.state().failed,
        }
    }
    /// A worker's control thread on its one link: reads the head's frames until the link fails or closes, idle
    /// between events. `on_quit` runs on the head's orderly quit.
    pub fn receive_events(&self, mut on_quit: impl FnMut() -> R<()>) -> R<()> {
        let r = (|| {
            let Role::Worker { io, .. } = &self.role else {
                return Err(refused("the head's channel has no event reader"));
            };
            loop {
                let b = io.io().receive(true)?;
                let d = wire::Dec::new(&b)?;
                match d.tag {
                    wire::NOTICE => self.install(&wire::decode_notice(&b)?)?,
                    wire::COMMIT => {
                        let (n, c) = wire::decode_commit(&b)?;
                        self.deliver(&n, &c)?;
                    }
                    wire::REQUEST => {
                        let (epoch, request, bytes) = wire::decode_request(&b)?;
                        require(epoch == self.epoch, "request from another epoch")?;
                        self.state().requests.insert(request, bytes.to_vec());
                    }
                    wire::QUIT => {
                        require(wire::decode_quit(&b)? == self.epoch, "quit from another epoch")?;
                        self.quit_received.store(true, Ordering::SeqCst);
                        on_quit()?;
                    }
                    wire::GO => {
                        let (epoch, seq) = wire::decode_go(&b)?;
                        require(epoch == self.epoch, "barrier from another epoch")?;
                        let mut g = self.state();
                        require(seq == g.go.wrapping_add(1), "barrier release out of order")?;
                        g.go = seq;
                        self.cv.notify_all();
                    }
                    _ => return Err(refused("worker received non-event frame")),
                }
            }
        })();
        self.failing(r)
    }
    /// The bytes the head sent for this request with its latest ADMIT notice (the caller checks their hash).
    pub fn request(&self, id: u64) -> R<Vec<u8>> {
        let g = self.state();
        g.requests.get(&id).cloned().ok_or_else(|| refused("ADMIT notice without its request bytes"))
    }
    /// Admission is FIFO: once a request is admitted, it and every earlier one are done with.
    pub fn admitted(&self, id: u64) {
        let mut g = self.state();
        g.requests = match id.checked_add(1) {
            Some(next) => g.requests.split_off(&next),
            None => BTreeMap::new(),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::IntoRawFd;
    use std::os::unix::net::UnixStream;

    fn pair() -> (Io, Io) {
        let (a, b) = UnixStream::pair().unwrap();
        (Io::new(a.into_raw_fd()).unwrap(), Io::new(b.into_raw_fd()).unwrap())
    }

    // TP2 over a socket pair: a held admission, a release at the same boundary, a barrier and the head's quit
    #[test]
    fn head_and_worker_agree() {
        let (hi, wi) = pair();
        let head = Endpoint::head(1, vec![Link::new(&hi)]).unwrap();
        let worker = Endpoint::worker(1, 1, Link::new(&wi)).unwrap();
        let quits = std::sync::atomic::AtomicU32::new(0);
        let compute = |e: &Endpoint, worker: bool| {
            let mut b = 1;
            let mut seen = Vec::new();
            for _ in 0..2 {
                e.wait_notice().unwrap();
                loop {
                    let mut req = Vec::new();
                    let c = e
                        .boundary(b, || {
                            let n = e.wait_notice()?;
                            if worker {
                                req = e.request(n.request)?;
                            }
                            Ok((42, if worker { 3 } else { u32::MAX }))
                        })
                        .unwrap();
                    if let Some(c) = c {
                        seen.push((c.c.admit.req.id, c.hold, b, req));
                        e.applied().unwrap();
                        break;
                    }
                    b += 1;
                }
            }
            e.sync(1).unwrap();
            seen
        };
        std::thread::scope(|s| {
            let reader = s.spawn(|| {
                worker.receive_events(|| {
                    quits.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                })
            });
            let h = s.spawn(|| compute(&head, false));
            let w = s.spawn(|| compute(&worker, true));
            let mut rooms = Vec::new();
            for (i, hold) in [(0u64, true), (1, false)] {
                let n = Notice { epoch: 1, event: i + 1, boundary: 20, request: 100 + i, payload_hash: 0, kind: ADMIT };
                head.event(
                    &n,
                    |hash| {
                        rooms.push(head.agreed_room());
                        let mut c = Commit::zero();
                        (c.c.epoch, c.c.event, c.c.boundary, c.c.before_hash) = (1, n.event, 20, hash);
                        c.c.has_admit = true;
                        c.c.admit.req.id = n.request;
                        c.hold = hold;
                        Ok(c)
                    },
                    Some(format!("req {i}").as_bytes()),
                )
                .unwrap();
            }
            let (hs, ws) = (h.join().unwrap(), w.join().unwrap());
            assert_eq!(rooms, [3, 3]);
            assert_eq!(hs.iter().map(|x| (x.0, x.1, x.2)).collect::<Vec<_>>(), [(100, true, 20), (101, false, 20)]);
            assert!(ws[0].3 == b"req 0" && ws[1].3 == b"req 1");
            head.quit();
            while quits.load(Ordering::Relaxed) == 0 {
                std::thread::yield_now();
            }
            head.fail();
            assert!(reader.join().unwrap().is_err());
        });
    }

    // the orderly quit settles both ways: the head waits for every worker's close_send, a worker for the head's close
    #[test]
    fn an_orderly_quit_settles_on_both_sides() {
        let ((h1, w1), (h2, w2)) = (pair(), pair());
        let head = Endpoint::head(1, vec![Link::new(&h1), Link::new(&h2)]).unwrap();
        let workers =
            [Endpoint::worker(1, 1, Link::new(&w1)).unwrap(), Endpoint::worker(1, 2, Link::new(&w2)).unwrap()];
        std::thread::scope(|s| {
            let readers: Vec<_> = workers.iter().map(|w| s.spawn(move || w.receive_events(|| Ok(())))).collect();
            assert!(!head.quit_settled() && workers.iter().all(|w| !w.quit_received() && !w.quit_settled()));
            head.quit();
            while !workers.iter().all(|w| w.quit_received()) {
                std::thread::yield_now();
            }
            // one worker closed: the head still waits for the other
            workers[0].close_send();
            let t = std::time::Instant::now();
            while t.elapsed() < std::time::Duration::from_millis(50) {
                assert!(!head.quit_settled());
            }
            workers[1].close_send();
            while !head.quit_settled() {
                std::thread::yield_now();
            }
            // the workers still read the open link until the head closes it
            assert!(workers.iter().all(|w| !w.quit_settled()));
            head.fail();
            for r in readers {
                assert!(r.join().unwrap().is_err());
            }
            assert!(workers.iter().all(|w| w.quit_settled()));
        });
    }

    // an unread frame hides the worker's close (the monitor's linger bounds that case); once read, the close shows
    #[test]
    fn a_waiting_frame_hides_the_close() {
        let (h, w) = pair();
        let head = Endpoint::head(1, vec![Link::new(&h)]).unwrap();
        w.send(b"x").unwrap();
        assert!(!head.quit_settled());
        drop(w);
        assert!(!head.quit_settled());
        assert_eq!(h.receive(false).unwrap(), b"x");
        assert!(head.quit_settled());
    }

    // a callback that fails ends the epoch: the gate refuses afterwards and the links are shut down
    #[test]
    fn a_failed_digest_fails_the_epoch() {
        let head = Endpoint::head(1, vec![]).unwrap();
        std::thread::scope(|s| {
            let h = s.spawn(|| {
                head.wait_notice().unwrap();
                let mut b = 1;
                loop {
                    match head.boundary(b, || Err(super::super::Fail::Hook)) {
                        Ok(None) => b += 1,
                        r => return r.map(|_| ()),
                    }
                }
            });
            let n = Notice { epoch: 1, event: 1, boundary: 5, request: 7, payload_hash: 0, kind: wire::CANCEL };
            let r = head.event(&n, |_| unreachable!(), None);
            assert_eq!(h.join().unwrap(), Err(super::super::Fail::Hook));
            assert!(r.is_err());
            assert_eq!(head.wait_notice().unwrap_err().message(), b"control epoch failed");
        });
    }
}
