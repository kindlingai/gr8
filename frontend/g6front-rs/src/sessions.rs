//! Logical conversation handles. KV ownership remains with the engine.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use crate::api::ApiError;
use crate::util::new_id;

#[derive(Clone)]
struct Handle {
    messages: Vec<Value>,
    prompt: Vec<u32>,
    model: String,
    render: Map<String, Value>,
    ready: bool,
    rev: u64,
    busy: bool,
    last: Instant,
}

pub struct Sessions {
    handles: Mutex<HashMap<String, Handle>>,
    idle: Duration,
}

impl Sessions {
    pub fn new(idle: Duration) -> Self {
        Self { handles: Mutex::new(HashMap::new()), idle }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Handle>> {
        let mut map = self.handles.lock().unwrap();
        map.retain(|_, h| h.last.elapsed() < self.idle);
        map
    }

    pub fn create(&self, messages: Vec<Value>, prompt: Vec<u32>, model: String, render: Map<String, Value>) -> String {
        let id = new_id("sess");
        self.lock().insert(
            id.clone(),
            Handle { messages, prompt, model, render, ready: false, rev: 0, busy: true, last: Instant::now() },
        );
        id
    }

    pub fn begin(
        &self,
        id: &str,
        append: &[Value],
        model: Option<&str>,
    ) -> Result<(Vec<Value>, Map<String, Value>), ApiError> {
        let mut map = self.lock();
        let h = map.get_mut(id).ok_or_else(|| ApiError::new(404, "unknown session", Some("session_id")))?;
        if model.is_some_and(|m| h.model != m) {
            return Err(ApiError::bad("session model differs", Some("model")));
        }
        if h.busy {
            return Err(ApiError::new(409, "session has a request in progress", Some("session_id")));
        }
        h.busy = true;
        h.last = Instant::now();
        let mut messages = h.messages.clone();
        messages.extend_from_slice(append);
        Ok((messages, h.render.clone()))
    }

    pub fn ready(&self, id: &str) {
        if let Some(h) = self.lock().get_mut(id) {
            h.ready = true;
            h.last = Instant::now();
        }
    }

    pub fn finish(&self, id: &str, append: &[Value], assistant: Value, prompt: Vec<u32>, render: Map<String, Value>) {
        if let Some(h) = self.lock().get_mut(id) {
            h.messages.extend_from_slice(append);
            h.messages.push(assistant);
            h.prompt = prompt;
            h.render = render;
            h.ready = true;
            h.rev += 1;
            h.busy = false;
            h.last = Instant::now();
        }
    }

    pub fn abort(&self, id: &str, created: bool) {
        let mut map = self.lock();
        if created {
            map.remove(id);
        } else if let Some(h) = map.get_mut(id) {
            h.busy = false;
        }
    }

    pub fn snapshot(&self, id: &str) -> Result<(Vec<Value>, Vec<u32>, Map<String, Value>, u64), ApiError> {
        let mut map = self.lock();
        let h = map.get_mut(id).ok_or_else(|| ApiError::new(404, "unknown session", Some("session_id")))?;
        if !h.ready {
            return Err(ApiError::new(409, "session is not ready to fork", Some("session_id")));
        }
        h.last = Instant::now();
        Ok((h.messages.clone(), h.prompt.clone(), h.render.clone(), h.rev))
    }

    pub fn fork(&self, parent: &str, count: usize, prefix_tokens: usize, expected_rev: u64) -> Result<Value, ApiError> {
        if !(1..=16).contains(&count) {
            return Err(ApiError::bad("count must be between 1 and 16", Some("count")));
        }
        let mut map = self.lock();
        let h = map.get(parent).ok_or_else(|| ApiError::new(404, "unknown session", Some("session_id")))?;
        if !h.ready {
            return Err(ApiError::new(409, "session is not ready to fork", Some("session_id")));
        }
        if h.rev != expected_rev {
            return Err(ApiError::new(409, "session changed during fork", Some("session_id")));
        }
        let snapshot = h.clone();
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            let id = new_id("sess");
            let mut child = snapshot.clone();
            child.busy = false;
            child.last = Instant::now();
            map.insert(id.clone(), child);
            ids.push(id);
        }
        Ok(json!({"parent_id": parent, "fork_ids": ids, "prefix_tokens": prefix_tokens}))
    }

    pub fn delete(&self, id: &str) -> Result<(), ApiError> {
        self.lock().remove(id).map(|_| ()).ok_or_else(|| ApiError::new(404, "unknown session", Some("session_id")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn msg(s: &str) -> Value {
        json!({"role": "user", "content": s})
    }
    fn answer(s: &str) -> Value {
        json!({"role": "assistant", "content": s})
    }

    #[test]
    fn lifecycle_fork_snapshot_and_parent_delete() {
        let store = Sessions::new(Duration::from_secs(60));
        let parent = store.create(vec![msg("base")], vec![1, 2, 3], "model".into(), Map::new());
        assert_eq!(store.fork(&parent, 3, 3, 0).unwrap_err().status, 409);
        store.finish(&parent, &[], answer("parent"), vec![1, 2, 3], Map::new());
        assert_eq!(store.fork(&parent, 1, 3, 0).unwrap_err().status, 409);
        let f = store.fork(&parent, 3, 3, 1).unwrap();
        assert_eq!(f["prefix_tokens"], 3);
        let ids: Vec<String> =
            f["fork_ids"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_owned()).collect();
        assert_eq!(ids.len(), 3);
        let branch = store.begin(&ids[0], &[msg("branch")], Some("model")).unwrap().0;
        assert_eq!(branch, vec![msg("base"), answer("parent"), msg("branch")]);
        store.finish(&ids[0], &[msg("branch")], answer("done"), vec![1, 2, 3, 4], Map::new());
        assert_eq!(
            store.begin(&ids[1], &[msg("other")], Some("model")).unwrap().0,
            vec![msg("base"), answer("parent"), msg("other")]
        );
        store.abort(&ids[1], false);
        store.delete(&parent).unwrap();
        assert_eq!(
            store.begin(&ids[2], &[msg("still works")], Some("model")).unwrap().0,
            vec![msg("base"), answer("parent"), msg("still works")]
        );
        store.abort(&ids[2], false);
        assert_eq!(store.delete(&parent).unwrap_err().status, 404);
    }

    #[test]
    fn bounds_unknown_busy_and_expiry() {
        let store = Sessions::new(Duration::from_secs(60));
        assert_eq!(store.fork("missing", 1, 0, 0).unwrap_err().status, 404);
        let id = store.create(vec![msg("base")], vec![1], "model".into(), Map::new());
        assert_eq!(store.fork(&id, 0, 0, 0).unwrap_err().status, 400);
        assert_eq!(store.fork(&id, 17, 0, 0).unwrap_err().status, 400);
        assert_eq!(store.begin(&id, &[], Some("model")).unwrap_err().status, 409);
        store.abort(&id, false);
        assert!(store.begin(&id, &[], Some("model")).is_ok());
        let expired = Sessions::new(Duration::ZERO);
        let eid = expired.create(vec![msg("x")], vec![], "model".into(), Map::new());
        assert_eq!(expired.delete(&eid).unwrap_err().status, 404);
    }
}
