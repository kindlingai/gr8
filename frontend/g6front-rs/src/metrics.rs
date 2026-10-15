//! Engine `metrics` snapshot -> Prometheus text with vLLM's metric names.

use serde_json::Value;

use crate::pyjson;

fn num(v: Option<&Value>) -> f64 {
    v.and_then(Value::as_f64).unwrap_or(0.0)
}

pub fn render(m: &Value, model_name: &str) -> String {
    let lab = format!("engine=\"0\",model_name=\"{model_name}\"");
    let mut out = String::with_capacity(4096);
    let mut metric = |name: &str, kind: &str, help: &str, samples: &[(String, f64)]| {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} {kind}\n"));
        for (extra, v) in samples {
            let labels = if extra.is_empty() { lab.clone() } else { format!("{lab},{extra}") };
            out.push_str(&format!("{name}{{{labels}}} {}\n", pyjson::float_string(*v)));
        }
    };
    let empty = Value::Object(Default::default());
    let g = |k: &str| m.get(k).filter(|v| !v.is_null()).unwrap_or(&empty);
    let (spec, pc, rq, tk) = (g("spec"), g("prefix_cache"), g("requests"), g("tokens"));
    let one = |v: Option<&Value>| vec![(String::new(), num(v))];
    metric(
        "vllm:num_requests_running",
        "gauge",
        "Number of requests in model execution batches.",
        &one(rq.get("running")),
    );
    metric(
        "vllm:num_requests_waiting",
        "gauge",
        "Number of requests waiting to be processed.",
        &one(rq.get("waiting")),
    );
    metric(
        "vllm:prefix_cache_queries_total",
        "counter",
        "Prefix cache queries, in terms of number of queried tokens.",
        &one(pc.get("queries")),
    );
    metric(
        "vllm:prefix_cache_hits_total",
        "counter",
        "Prefix cache hits, in terms of number of cached tokens.",
        &one(pc.get("hits")),
    );
    metric("vllm:prompt_tokens_total", "counter", "Number of prefill tokens processed.", &one(tk.get("prompt")));
    metric(
        "vllm:generation_tokens_total",
        "counter",
        "Number of generation tokens processed.",
        &one(tk.get("generation")),
    );
    let fin = rq.get("finished").filter(|v| !v.is_null()).unwrap_or(&empty);
    let fins: Vec<(String, f64)> = ["stop", "length", "abort", "error"]
        .iter()
        .map(|r| (format!("finished_reason=\"{r}\""), num(fin.get(*r))))
        .collect();
    metric("vllm:request_success_total", "counter", "Count of successfully processed requests.", &fins);
    metric("vllm:spec_decode_num_drafts_total", "counter", "Number of spec decoding drafts.", &one(spec.get("drafts")));
    metric(
        "vllm:spec_decode_num_draft_tokens_total",
        "counter",
        "Number of draft tokens.",
        &one(spec.get("draft_tokens")),
    );
    metric(
        "vllm:spec_decode_num_accepted_tokens_total",
        "counter",
        "Number of accepted tokens.",
        &one(spec.get("accepted")),
    );
    let per_pos: Vec<(String, f64)> = spec
        .get("accepted_per_pos")
        .and_then(Value::as_array)
        .map(|a| a.iter().enumerate().map(|(i, v)| (format!("position=\"{i}\""), num(Some(v)))).collect())
        .unwrap_or_default();
    metric(
        "vllm:spec_decode_num_accepted_tokens_per_pos_total",
        "counter",
        "Accepted tokens per draft position.",
        &per_pos,
    );
    // S2-only native allocator/scheduler snapshot; legacy exposition is unchanged.
    if let Some(s2) = m.get("scheduler").filter(|v| !v.is_null()) {
        for key in [
            "resident",
            "queued",
            "retiring",
            "reserved_pages",
            "free_pages",
            "largest_free_run",
            "held_pages",
            "worst_pages",
        ] {
            if let Some(v) = s2.get(key) {
                metric(
                    &format!("g6:s2_{key}"),
                    "gauge",
                    &format!("S2 native {}.", key.replace('_', " ")),
                    &one(Some(v)),
                );
            }
        }
        for key in ["admission_stalls", "fragmentation_stalls", "grown", "stalled_rounds"] {
            if let Some(v) = s2.get(key) {
                metric(
                    &format!("g6:s2_{key}_total"),
                    "counter",
                    &format!("S2 native {}.", key.replace('_', " ")),
                    &one(Some(v)),
                );
            }
        }
        // the KV stores (--kv-store), one series per store
        let stores = s2.get("tiers").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]);
        let per = |key: &str| -> Vec<(String, f64)> {
            stores.iter().enumerate().map(|(i, t)| (format!("store=\"{i}\""), num(t.get(key)))).collect()
        };
        let has = |key: &str| stores.first().is_some_and(|t| t.get(key).is_some());
        for key in KV_STORE_GAUGES {
            if has(key) {
                metric(&format!("g6:kv_store_{key}"), "gauge", &format!("KV store {}.", key.replace('_', " ")), &per(key));
            }
        }
        for key in KV_STORE_COUNTERS {
            if has(key) {
                metric(
                    &format!("g6:kv_store_{key}_total"),
                    "counter",
                    &format!("KV store {}.", key.replace('_', " ")),
                    &per(key),
                );
            }
        }
    }
    out
}

const KV_STORE_GAUGES: [&str; 4] = ["slots", "free_slots", "entries", "shared"];
const KV_STORE_COUNTERS: [&str; 10] = [
    "spilled_bytes",
    "moved_in_bytes",
    "restored_bytes",
    "hits",
    "moved_out",
    "dropped",
    "dropped_pages",
    "refused",
    "refused_pages",
    "full",
];
