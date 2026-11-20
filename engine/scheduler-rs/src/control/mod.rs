//! The S2 control links between ranks: the rank bootstrap over TCP and its agreements (`bootstrap`), the framed socket
//! (`io`), the frames (`wire`), the event boundary gate (`gate`) and the head's and workers' event channels
//! (`channel`). The C++ headers s2/bootstrap.h, control_io.h, control_protocol.h, control_channel.h and event_gate.h
//! are facades over these through `g6sched.h` (`ffi`). The bytes on the links, the refusals' messages, the timeouts and
//! the order of sends and receives are the C++'s.
//!
//! No thread is started here: the C++ threads call in. On the head the control thread runs `Endpoint::event`, the
//! compute thread `boundary`, `applied`, `wait_notice` and `sync`, and the watcher `quit` and `fail`. On a worker the
//! control thread runs `receive_events`, the compute thread the same calls as the head's, and the watcher `fail`.

pub mod bootstrap;
pub mod channel;
pub mod ffi;
pub mod gate;
pub mod io;
mod sys;
pub mod wire;

use std::borrow::Cow;

/// Why a control call failed: refused with the message the C++ threw, or a C callback threw (the facade holds its
/// exception).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fail {
    /// the refusal's message (bytes: an agreement's words are copied in as they came)
    Refused(Cow<'static, [u8]>),
    /// a callback returned nonzero
    Hook,
}

/// A control call's result.
pub type R<T> = Result<T, Fail>;

/// A refusal with a fixed message.
pub fn refused(what: &'static str) -> Fail {
    Fail::Refused(Cow::Borrowed(what.as_bytes()))
}

/// `Ok(())`, or the refusal `what` when `ok` is false.
pub fn require(ok: bool, what: &'static str) -> R<()> {
    if ok {
        Ok(())
    } else {
        Err(refused(what))
    }
}

impl Fail {
    /// The refusal's message, or empty for a callback's failure.
    pub fn message(&self) -> &[u8] {
        match self {
            Fail::Refused(m) => m,
            Fail::Hook => b"",
        }
    }
}
