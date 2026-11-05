// The C++ request parser sessions.h had before it moved to engine/scheduler-rs, kept only for the differential test
// (tests/scheduler_rs_diff_test.cpp). Namespace g6s2ref. Delete with R2's last step.
#pragma once
#include "../../sessions.h"

namespace g6s2ref {
using g6s2::number; using g6s2::require;
inline std::vector<int> request_ids(const std::string& s) {
    std::vector<int> out; size_t a = 0;
    do {
        auto b = s.find(',', a); auto n = number(s.substr(a, b == std::string::npos ? b : b - a));
        require(n <= INT32_MAX, "request token integer overflow"); out.push_back(int(n));
        if (b == std::string::npos) break;
        a = b + 1;
    } while (true);
    return out;
}
// `ins <after>:<unless or -1>:<id,id,..>` (insert_rule.h)
inline g6rt::InsertRule parse_insert(const std::string& s) {
    const auto a = s.find(':'), b = a == std::string::npos ? a : s.find(':', a + 1);
    require(b != std::string::npos, "invalid ins header");
    g6rt::InsertRule r;
    const auto after = number(s.substr(0, a));
    require(after <= INT32_MAX, "invalid ins after");
    r.after = int(after);
    const auto unless = s.substr(a + 1, b - a - 1);
    r.unless = unless == "-1" ? -1 : int(std::min<uint64_t>(number(unless), INT32_MAX));
    r.tokens = request_ids(s.substr(b + 1));
    require(!r.tokens.empty() && r.tokens.size() <= size_t(g6rt::INSERT_MAX), "invalid ins token count");
    return r;
}
// `gram <kind>:<strict>:<arm or -1>:<hex>` (grammar.h): kind 1 = JSON, 2 = the JSON schema whose UTF-8 bytes are <hex>
inline g6rt::GrammarSpec parse_grammar(const std::string& s) {
    g6rt::GrammarSpec g;
    size_t c[3], at = 0;
    for (auto& x : c) { x = s.find(':', at); require(x != std::string::npos, "invalid gram header"); at = x + 1; }
    const auto kind = s.substr(0, c[0]), strict = s.substr(c[0] + 1, c[1] - c[0] - 1), arm = s.substr(c[1] + 1, c[2] - c[1] - 1);
    require((kind == "1" || kind == "2") && (strict == "0" || strict == "1"), "invalid gram kind/strict");
    g.kind = kind == "1" ? g6rt::GrammarSpec::JSON : g6rt::GrammarSpec::SCHEMA;
    g.strict = strict == "1";
    g.arm = arm == "-1" ? -1 : int(std::min<uint64_t>(number(arm), INT32_MAX));
    const auto hex = s.substr(c[2] + 1);
    require(hex.size() % 2 == 0 && (g.kind == g6rt::GrammarSpec::JSON) == hex.empty(), "invalid gram schema");
    auto nib = [](char h) { require(isxdigit((unsigned char)h), "invalid gram hex"); return h <= '9' ? h - '0' : (h | 0x20) - 'a' + 10; };
    for (size_t i = 0; i < hex.size(); i += 2) g.schema.push_back(char(nib(hex[i]) << 4 | nib(hex[i + 1])));
    return g;
}
inline g6rt::ServeReq parse_request(const std::string& bytes, const g6rt::steering::BootConfig& steering = {}) {
    const auto line = bytes.find('\n'); require(line != std::string::npos, "request missing prompt line");
    std::istringstream in(bytes.substr(0, line));
    g6rt::ServeReq r; std::string k, gen, seed, min, stop, extra;
    require(bool(in >> k) && k == "n_gen" && bool(in >> gen >> k) && k == "temp" &&
            bool(in >> r.temp >> k) && k == "seed" && bool(in >> seed >> k) && k == "min" &&
            bool(in >> min >> k) && k == "stop" && bool(in >> stop), "invalid request header");
    // optional, in this order: top_p <float>, lp <k>, plp <k>, sys <tokens>, ins <rule>, gram <grammar>,
    // steer <profile> <canonical-fp32-hex>
    auto next = [&] { if (!(in >> k)) k.clear(); };
    next();
    if (k == "top_p") { require(bool(in >> r.top_p), "invalid top_p header"); next(); }
    if (k == "lp") { require(bool(in >> r.lp) && r.lp >= 0 && r.lp <= g6rt::LP_MAX, "invalid lp header"); next(); }
    if (k == "plp") { require(bool(in >> r.plp) && r.plp >= 0 && r.plp <= g6rt::LP_MAX, "invalid plp header"); next(); }
    if (k == "sys") { require(bool(in >> r.sys) && r.sys >= 0 && r.sys <= INT32_MAX, "invalid sys header"); next(); }
    if (k == "ins") { std::string v; require(bool(in >> v), "invalid ins header"); r.ins = parse_insert(v); next(); }
    if (k == "gram") {
        std::string v; require(bool(in >> v), "invalid gram header"); r.gram = parse_grammar(v); next();
        require(g6rt::grammar_ready(), "gram: the engine was started without --grammar-dir");
    }
    g6rt::steering::RequestFields sf;
    if (k == "steer") {
        std::string name, bits; require(bool(in >> name >> bits), "invalid steering header"); next();
        require(g6rt::steering::valid_profile_name(name), "invalid steering profile name");
        const uint32_t scale = g6rt::steering::parse_scale_bits(bits);
        sf.has_profile = sf.has_scale = true; sf.profile = std::move(name);
        sf.scale = g6rt::steering::bits_float(scale);
    }
    require(k.empty(), "invalid request header tail (S2 accepts top_p, lp, plp, sys, ins, gram and steer)");
    require(!r.ins.on() || r.lp < 0, "ins with lp");
    r.steering = steering.resolve(sf);
    auto g = number(gen), sd = number(seed), mn = number(min);
    require(g > 0 && g <= INT32_MAX && sd <= UINT32_MAX && mn <= g &&
            std::isfinite(r.temp) && r.temp >= 0 && std::isfinite(r.top_p) && r.top_p >= 0 && r.top_p <= 1,
            "invalid request sampling/length");
    r.n_gen = int(g); r.seed = uint32_t(sd); r.min_tokens = int(mn);
    if (stop != "-") r.stop = request_ids(stop);
    std::istringstream ids(bytes.substr(line + 1)); std::string prompt;
    require(bool(ids >> prompt) && !(ids >> extra), "invalid prompt line");
    r.prompt = request_ids(prompt); return r;
}
} // namespace g6s2ref
