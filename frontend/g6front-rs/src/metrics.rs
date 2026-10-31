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
    // measured prefix-cache counters only: an engine that does not report them leaves the series out (not a 0)
    for (key, name, help) in [
        ("queries", "vllm:prefix_cache_queries_total", "Prefix cache queries, in terms of number of queried tokens."),
        ("hits", "vllm:prefix_cache_hits_total", "Prefix cache hits, in terms of number of cached tokens."),
    ] {
        if let Some(value) = pc.get(key).and_then(Value::as_f64) {
            metric(name, "counter", help, &[(String::new(), value)]);
        }
    }
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
    // Export only values reported by the engine; missing or nonnumeric fields
    // must not become fabricated zero-valued samples.
    if let Some(kv) = m.get("kv") {
        for (key, name, help) in [
            ("size_tokens", "vllm:kv_cache_size_tokens", "KV cache pool capacity, in tokens."),
            ("active_tokens", "vllm:kv_active_tokens", "KV cache tokens held by running requests (page-rounded)."),
            (
                "reserved_tokens",
                "g6:s2_reserved_tokens",
                "KV cache tokens reserved for running requests, including unfilled room.",
            ),
        ] {
            if let Some(value) = kv.get(key).and_then(Value::as_f64) {
                metric(name, "gauge", help, &[(String::new(), value)]);
            }
        }
    }
    if let Some(cached) = tk.get("cached").and_then(Value::as_f64) {
        metric(
            "vllm:prompt_tokens_cached_total",
            "counter",
            "Number of prompt tokens served from the prefix cache.",
            &[(String::new(), cached)],
        );
        if let Some(prompt) = tk.get("prompt").and_then(Value::as_f64) {
            metric(
                "vllm:prompt_tokens_by_source_total",
                "counter",
                "Prompt tokens by source.",
                &[
                    ("source=\"local_compute\"".to_string(), (prompt - cached).max(0.0)),
                    ("source=\"local_cache_hit\"".to_string(), cached),
                ],
            );
        }
    }
    if let Some(latency) = m.get("latency") {
        // Sums and counts over finished requests (a histogram's _sum/_count, without the buckets).
        for (key, name, help) in [
            (
                "ttft_sum",
                "vllm:time_to_first_token_seconds_sum",
                "Seconds from request arrival to its first token, summed over requests.",
            ),
            ("ttft_count", "vllm:time_to_first_token_seconds_count", "Requests that produced a first token."),
            (
                "itl_sum",
                "vllm:inter_token_latency_seconds_sum",
                "Seconds between a request's token batches after its first, summed.",
            ),
            (
                "itl_count",
                "vllm:inter_token_latency_seconds_count",
                "Tokens a request emitted after its first (the tokens the ITL sum covers).",
            ),
            (
                "prefill_sum",
                "vllm:request_prefill_time_seconds_sum",
                "Seconds from a request's admission to its first token, summed over finished requests.",
            ),
            (
                "decode_sum",
                "vllm:request_decode_time_seconds_sum",
                "Seconds from a request's first token to its finish, summed over finished requests.",
            ),
            (
                "prefill_tokens_sum",
                "vllm:request_prefill_kv_computed_tokens_sum",
                "Prompt tokens computed (not taken from the prefix cache) by finished requests.",
            ),
            ("gen_tokens_sum", "vllm:request_generation_tokens_sum", "Tokens generated by finished requests."),
            ("gen_tokens_count", "vllm:request_generation_tokens_count", "Finished requests that produced a token."),
        ] {
            if let Some(value) = latency.get(key).and_then(Value::as_f64) {
                metric(name, "counter", help, &[(String::new(), value)]);
            }
        }
    }
    // the info sample scrapers read capacity and concurrency from (vLLM's cache_config_info labels)
    if let (Some(size), Some(conc)) = (
        m.get("kv").and_then(|kv| kv.get("size_tokens")).and_then(Value::as_f64),
        m.get("kv").and_then(|kv| kv.get("max_concurrency")).and_then(Value::as_f64),
    ) {
        metric(
            "vllm:cache_config_info",
            "gauge",
            "Information of the KV cache configuration.",
            &[(format!("kv_cache_size_tokens=\"{size:.0}\",kv_cache_max_concurrency=\"{conc:.0}\""), 1.0)],
        );
    }
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
                metric(
                    &format!("g6:kv_store_{key}"),
                    "gauge",
                    &format!("KV store {}.", key.replace('_', " ")),
                    &per(key),
                );
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

#[cfg(test)]
mod tests {
    use super::render;
    use serde_json::json;

    fn base() -> serde_json::Value {
        json!({"spec": {"drafts": 0, "draft_tokens": 0, "accepted": 0, "accepted_per_pos": []},
               "prefix_cache": {"queries": 0, "hits": 0},
               "requests": {"running": 0, "waiting": 0, "finished": {}},
               "tokens": {"prompt": 0, "generation": 0}})
    }

    const LAB: &str = "engine=\"0\",model_name=\"test-model\"";

    #[test]
    fn legacy_snapshot_has_no_new_metrics() {
        let t = render(&base(), "test-model");
        for name in [
            "kv_cache_size_tokens",
            "kv_active_tokens",
            "prompt_tokens_cached_total",
            "prompt_tokens_by_source_total",
            "time_to_first_token_seconds",
        ] {
            assert!(!t.contains(&format!("vllm:{name}")), "{name} must be absent");
        }
    }

    #[test]
    fn reported_values_are_exported() {
        let mut m = base();
        m["kv"] = json!({"size_tokens": 393216, "active_tokens": 77376, "reserved_tokens": 393216});
        m["latency"] = json!({"ttft_sum": 12.5, "ttft_count": 5});
        m["tokens"] = json!({"prompt": 1000, "generation": 10, "cached": 300});
        m["prefix_cache"] = json!({"queries": 900, "hits": 300});
        let t = render(&m, "test-model");
        for (name, kind) in [
            ("vllm:kv_cache_size_tokens", "gauge"),
            ("vllm:kv_active_tokens", "gauge"),
            ("g6:s2_reserved_tokens", "gauge"),
            ("vllm:prompt_tokens_cached_total", "counter"),
            ("vllm:prompt_tokens_by_source_total", "counter"),
            ("vllm:time_to_first_token_seconds_sum", "counter"),
            ("vllm:time_to_first_token_seconds_count", "counter"),
        ] {
            assert!(t.contains(&format!("# HELP {name} ")), "missing HELP for {name}");
            assert!(t.contains(&format!("# TYPE {name} {kind}\n")), "missing TYPE for {name}");
        }
        for want in [
            format!("vllm:kv_cache_size_tokens{{{LAB}}} 393216.0\n"),
            format!("vllm:kv_active_tokens{{{LAB}}} 77376.0\n"),
            format!("g6:s2_reserved_tokens{{{LAB}}} 393216.0\n"),
            format!("vllm:time_to_first_token_seconds_sum{{{LAB}}} 12.5\n"),
            format!("vllm:time_to_first_token_seconds_count{{{LAB}}} 5.0\n"),
            format!("vllm:prompt_tokens_cached_total{{{LAB}}} 300.0\n"),
            format!("vllm:prompt_tokens_by_source_total{{{LAB},source=\"local_compute\"}} 700.0\n"),
            format!("vllm:prompt_tokens_by_source_total{{{LAB},source=\"local_cache_hit\"}} 300.0\n"),
            format!("vllm:prefix_cache_queries_total{{{LAB}}} 900.0\n"),
            format!("vllm:prefix_cache_hits_total{{{LAB}}} 300.0\n"),
        ] {
            assert!(t.contains(&want), "missing {want:?} in\n{t}");
        }
    }

    #[test]
    fn partial_snapshots_do_not_fabricate_zeros() {
        let mut m = base();
        m["kv"] = json!({"size_tokens": null, "active_tokens": "unknown", "reserved_tokens": null});
        m["tokens"] = json!({"cached": 25});
        m["latency"] = json!({"ttft_count": 2});
        let t = render(&m, "test-model");
        assert!(!t.contains("vllm:kv_cache_size_tokens"));
        assert!(!t.contains("vllm:kv_active_tokens"));
        assert!(!t.contains("g6:s2_reserved_tokens"));
        assert!(t.contains("vllm:prompt_tokens_cached_total"));
        assert!(!t.contains("vllm:prompt_tokens_by_source_total"));
        assert!(!t.contains("vllm:time_to_first_token_seconds_sum"));
        assert!(t.contains("vllm:time_to_first_token_seconds_count"));

        m["tokens"] = json!({"prompt": 1000});
        m["latency"] = json!({"ttft_sum": 12.5});
        let t = render(&m, "test-model");
        assert!(!t.contains("vllm:prompt_tokens_cached_total"));
        assert!(!t.contains("vllm:prompt_tokens_by_source_total"));
        assert!(t.contains("vllm:time_to_first_token_seconds_sum"));
        assert!(!t.contains("vllm:time_to_first_token_seconds_count"));
    }

    #[test]
    fn missing_key_leaves_the_metric_out() {
        let mut m = base();
        m["kv"] = json!({"size_tokens": 100});
        let t = render(&m, "test-model");
        assert!(t.contains(&format!("vllm:kv_cache_size_tokens{{{LAB}}} 100.0\n")));
        assert!(!t.contains("vllm:kv_active_tokens"));
        assert!(!t.contains("g6:s2_reserved_tokens"));
        assert!(!t.contains("time_to_first_token"));
    }

    #[test]
    fn request_timing_and_concurrency_families() {
        let mut m = base();
        m["kv"] = json!({"size_tokens": 408832, "active_tokens": 128, "reserved_tokens": 512, "max_concurrency": 8});
        m["latency"] = json!({"ttft_sum": 3.5, "ttft_count": 2, "itl_sum": 1.25, "itl_count": 40, "prefill_sum": 3.0, "decode_sum": 4.5,
                              "prefill_tokens_sum": 9000, "gen_tokens_sum": 300, "gen_tokens_count": 2});
        let t = render(&m, "test-model");
        for want in [
            format!("vllm:inter_token_latency_seconds_sum{{{LAB}}} 1.25\n"),
            format!("g6:s2_reserved_tokens{{{LAB}}} 512.0\n"),
            format!("vllm:inter_token_latency_seconds_count{{{LAB}}} 40.0\n"),
            format!("vllm:request_prefill_time_seconds_sum{{{LAB}}} 3.0\n"),
            format!("vllm:request_decode_time_seconds_sum{{{LAB}}} 4.5\n"),
            format!("vllm:request_prefill_kv_computed_tokens_sum{{{LAB}}} 9000.0\n"),
            format!("vllm:request_generation_tokens_sum{{{LAB}}} 300.0\n"),
            format!("vllm:request_generation_tokens_count{{{LAB}}} 2.0\n"),
            format!(
                "vllm:cache_config_info{{{LAB},kv_cache_size_tokens=\"408832\",kv_cache_max_concurrency=\"8\"}} 1.0\n"
            ),
        ] {
            assert!(t.contains(&want), "missing {want:?} in\n{t}");
        }
    }

    #[test]
    fn cache_config_info_needs_both_fields() {
        let mut m = base();
        m["kv"] = json!({"size_tokens": 100});
        assert!(!render(&m, "test-model").contains("cache_config_info"));
        m["kv"] = json!({"max_concurrency": 8});
        assert!(!render(&m, "test-model").contains("cache_config_info"));
        m["latency"] = json!({"decode_sum": 1.0});
        let t = render(&m, "test-model");
        assert!(t.contains("vllm:request_decode_time_seconds_sum"));
        assert!(!t.contains("vllm:request_prefill_time_seconds_sum"));
    }

    #[test]
    fn prefix_cache_series_need_a_reported_counter() {
        let mut m = base();
        m.as_object_mut().unwrap().remove("prefix_cache");
        let t = render(&m, "test-model");
        assert!(!t.contains("vllm:prefix_cache_queries_total") && !t.contains("vllm:prefix_cache_hits_total"));
        m["prefix_cache"] = json!({"queries": 7});
        let t = render(&m, "test-model");
        assert!(t.contains(&format!("vllm:prefix_cache_queries_total{{{LAB}}} 7.0\n")));
        assert!(!t.contains("vllm:prefix_cache_hits_total"));
    }
}
