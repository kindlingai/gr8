//! The S2 service's file side: the shim's command and request files (`files::Ingress`), ready.s2 and the bootstrap
//! agreement (`files::capability`, `files::agreement`), the files the head publishes (`files::atomic_file`), status.s2
//! (`status`) and the Chrome trace (`trace`), and the head's control thread (`control`) with its admission decisions
//! (`admit`). The C++ headers s2/ingress.h, trace.h, bootstrap.h, admit.h and service.h are facades over these through
//! `g6sched.h` (`ffi`). The bytes in the files, the log lines and the refusals' messages are the C++'s.
//!
//! The shim and the front end read these files, and every rank compares the agreement at the bootstrap, so a changed
//! byte is a protocol change.

pub mod admit;
pub mod control;
pub mod ffi;
pub mod files;
pub mod status;
pub mod trace;
