// route.h and the relay header (relay.h) on a synthetic 8-rank hybrid topology: the table parses, every refusal names
// its pair, the plan of each rank has the right roles, and the packed relay header is exact for every (k, src, dst).
// Fixture (synthetic, 198.18.0.0/15): ranks 0..5 on a switch (two rails, 198.18.1/24 and 198.18.2/24); cables 5<->6
// (198.18.11/24, .12/24), 0<->7 (198.18.21/24, .22/24) and 6<->7 (198.18.31/24, .32/24). Rank 6 reaches 0..4 through 5,
// rank 7 reaches 1..5 through 0 (lowest-rank common neighbour). Rank 1 also has a stray local alias in 7's cable subnet.
//   g++ -std=c++17 -O2 -Wall -Wextra -I engine/comms engine/comms/tests/route_test.cpp -o route_test && ./route_test
#include "route.h"
#include "relay.h"

#include <cstdio>
#include <functional>
#include <set>
#include <string>
#include <vector>

using namespace g6;

static int fails = 0;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); fails++; } } while (0)

static std::string ip(int net, int host) { return "198.18." + std::to_string(net) + "." + std::to_string(host); }

// the declared addresses: ports[r] = list of (subnet, host)
static std::vector<std::vector<std::pair<int, int>>> fixture_ports() {
  std::vector<std::vector<std::pair<int, int>>> p(8);
  for (int r = 0; r < 6; r++) { p[r].push_back({1, r + 1}); p[r].push_back({2, r + 1}); }
  p[5].push_back({11, 6}); p[5].push_back({12, 6});
  p[6].push_back({11, 7}); p[6].push_back({12, 7}); p[6].push_back({31, 7}); p[6].push_back({32, 7});
  p[0].push_back({21, 1}); p[0].push_back({22, 1});
  p[7].push_back({21, 8}); p[7].push_back({22, 8}); p[7].push_back({31, 8}); p[7].push_back({32, 8});
  return p;
}

// a route file in the launcher's format: rule A, the rail by (a + b) % 2 among the pair's common subnets
static std::string fixture(std::function<bool(const std::string&)> keep = nullptr, const std::string& extra = "") {
  const auto P = fixture_ports();
  std::string s = "# synthetic 8-rank hybrid table\n";
  auto common = [&](int a, int b) {
    std::vector<int> v;
    for (auto& x : P[a]) for (auto& y : P[b]) if (x.first == y.first) v.push_back(x.first);
    return v;
  };
  auto host = [&](int r, int net) { for (auto& x : P[r]) if (x.first == net) return x.second; return -1; };
  auto add = [&](const std::string& l) { if (!keep || keep(l)) s += l + "\n"; };
  for (int r = 0; r < 8; r++) {
    std::string l = "P " + std::to_string(r);
    for (auto& x : P[r]) l += " " + ip(x.first, x.second) + "/24";
    add(l);
  }
  for (int a = 0; a < 8; a++)
    for (int b = a + 1; b < 8; b++) {
      const auto c = common(a, b);
      if (!c.empty()) {
        const int n = c[(a + b) % c.size()];
        add("D " + std::to_string(a) + " " + std::to_string(b) + " " + ip(n, host(a, n)) + " " + ip(n, host(b, n)));
        continue;
      }
      int v = -1;
      for (int q = 0; q < 8 && v < 0; q++) if (q != a && q != b && !common(a, q).empty() && !common(q, b).empty()) v = q;
      for (auto [s0, d0] : {std::pair<int, int>{a, b}, {b, a}}) {
        const auto c1 = common(s0, v), c2 = common(v, d0);
        const int n1 = c1[(s0 + d0) % c1.size()], n2 = c2[(s0 + d0) % c2.size()];
        add("R " + std::to_string(s0) + " " + std::to_string(d0) + " " + std::to_string(v) + " " + ip(n1, host(s0, n1)) + " " +
            ip(n1, host(v, n1)) + " " + ip(n2, host(v, n2)) + " " + ip(n2, host(d0, n2)));
      }
    }
  return s + extra;
}

static bool refuses(const std::string& text, const std::string& needle, int nr = 8) {
  RouteTable t; std::string err;
  if (route_parse(text, nr, &t, &err)) { fprintf(stderr, "  accepted, expected a refusal with '%s'\n", needle.c_str()); return false; }
  if (err.find(needle) == std::string::npos) { fprintf(stderr, "  refusal '%s' lacks '%s'\n", err.c_str(), needle.c_str()); return false; }
  return true;
}
static std::string replace(std::string s, const std::string& a, const std::string& b) {
  const size_t k = s.find(a);
  if (k == std::string::npos) { fprintf(stderr, "  fixture has no '%s'\n", a.c_str()); fails++; return s; }
  return s.replace(k, a.size(), b);
}

// the fixture line that starts with prefix
static std::string line_of(const std::string& g, const std::string& prefix) {
  const size_t k = g.find("\n" + prefix);
  if (k == std::string::npos) { fprintf(stderr, "  fixture has no line '%s'\n", prefix.c_str()); fails++; return prefix; }
  return g.substr(k + 1, g.find('\n', k + 1) - k - 1);
}

static void good_table() {
  RouteTable t; std::string err;
  const std::string txt = fixture();
  CHECK(route_parse(txt, 8, &t, &err));
  if (!err.empty()) fprintf(stderr, "  %s\n", err.c_str());
  CHECK(t.direct.size() == 18 && t.relay.size() == 20);       // 18 direct pairs, 10 relayed pairs = 20 directed routes
  for (const RouteRelay& r : t.relay) {
    if (r.src == 6 || r.dst == 6) { if (!((r.src == 7 || r.dst == 7))) CHECK(r.via == 5); }
    if ((r.src == 7 || r.dst == 7) && r.src != 6 && r.dst != 6) CHECK(r.via == 0);
  }
  CHECK(t.relay[t.relay_of(0, 6)].via == 5);                  // two common neighbours (5, 7): the lowest
  CHECK(t.relay[t.relay_of(5, 7)].via == 0);                  // two common neighbours (0, 6): the lowest
  // the plans: the cable-only rank 7 is an origin with 5 relayed peers and ZERO relay contexts
  const RoutePlan p7 = route_plan(t, 7);
  int relayed = 0;
  for (int q = 0; q < 7; q++) if (p7.peer[q].relayed) { relayed++; CHECK(p7.peer[q].via == 0); CHECK(route_port(t, 7, p7.peer[q].ip)); }
  CHECK(relayed == 5 && p7.ctx.empty() && p7.fin.size() == 5 && p7.local_ips.size() <= 4);
  CHECK(!p7.peer[0].relayed && !p7.peer[6].relayed);
  // the two vias each relay 10 directed routes; the switch-only ranks relay none
  CHECK(route_plan(t, 0).ctx.size() == 10 && route_plan(t, 5).ctx.size() == 10);
  for (int r : {1, 2, 3, 4, 6}) CHECK(route_plan(t, r).ctx.empty());
  for (int r = 0; r < 8; r++) {
    const RoutePlan p = route_plan(t, r);
    CHECK(p.local_ips.size() <= (size_t)kRouteMaxPorts);
    for (const auto& c : p.ctx) { CHECK(route_port(t, r, c.ip_in)); CHECK(route_port(t, r, c.ip_out)); CHECK(c.src != r && c.dst != r); }
    for (const auto& f : p.fin) CHECK(route_port(t, r, f.ip) && f.via != r && f.src != r);
  }
  // my P record equals the expected local declaration (any order); a stray local alias or a missing port is refused
  CHECK(route_ports_match(t, 1, {ip(2, 2) + "/24", ip(1, 2) + "/24"}, &err));
  CHECK(!route_ports_match(t, 1, {ip(1, 2) + "/24", ip(2, 2) + "/24", ip(21, 2) + "/24"}, &err) && err.find("rank 1") != std::string::npos);
  CHECK(!route_ports_match(t, 1, {ip(1, 2) + "/24"}, &err));
  CHECK(!route_ports_match(t, 1, {ip(1, 2) + "/25", ip(2, 2) + "/24"}, &err));
  CHECK(!route_ports_match(t, 1, {"198.18.1.02/24", ip(2, 2) + "/24"}, &err));
  CHECK(!route_ports_match(t, 8, {}, &err));
  // the hash: known SHA-256 vectors, and stable for the table
  CHECK(route_sha256("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(route_sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(route_sha256(std::string(1000, 'a')) == "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
  CHECK(route_sha256(txt) == route_sha256(fixture()));
  // a TP4 table (no relay) and a forced-relay TP4 table both parse
  err.clear();
  CHECK(route_parse("P 0 198.18.1.1/24\nP 1 198.18.1.2/24\nP 2 198.18.1.3/24\nP 3 198.18.1.4/24\n"
                    "D 0 1 198.18.1.1 198.18.1.2\nD 0 2 198.18.1.1 198.18.1.3\nD 0 3 198.18.1.1 198.18.1.4\n"
                    "D 1 2 198.18.1.2 198.18.1.3\nD 1 3 198.18.1.2 198.18.1.4\nD 2 3 198.18.1.3 198.18.1.4\n", 4, &t, &err));
  CHECK(route_parse("P 0 198.18.1.1/24\nP 1 198.18.1.2/24\nP 2 198.18.1.3/24\nP 3 198.18.1.4/24\n"
                    "D 0 1 198.18.1.1 198.18.1.2\nD 0 3 198.18.1.1 198.18.1.4\nD 1 2 198.18.1.2 198.18.1.3\n"
                    "D 2 3 198.18.1.3 198.18.1.4\nD 1 3 198.18.1.2 198.18.1.4\n"
                    "R 0 2 1 198.18.1.1 198.18.1.2 198.18.1.2 198.18.1.3\nR 2 0 1 198.18.1.3 198.18.1.2 198.18.1.2 198.18.1.1\n", 4, &t, &err));
  if (!err.empty()) fprintf(stderr, "  %s\n", err.c_str());
  CHECK(route_plan(t, 1).ctx.size() == 2 && route_plan(t, 3).ctx.empty() && route_plan(t, 0).peer[2].relayed);
}

static void refusals() {
  const std::string g = fixture();
  // bad subnet: a D pair whose two addresses are on different subnets
  CHECK(refuses(replace(g, "D 1 2 " + ip(2, 2) + " " + ip(2, 3), "D 1 2 " + ip(1, 2) + " " + ip(2, 3)), "pair 1-2"));
  CHECK(refuses(replace(g, "D 1 2 " + ip(2, 2) + " " + ip(2, 3), "D 1 2 " + ip(1, 2) + " " + ip(2, 3)), "not two hosts on one subnet"));
  // a different prefix on one end of one subnet
  CHECK(refuses(replace(g, ip(1, 3) + "/24", ip(1, 3) + "/16"), "not two hosts on one subnet"));
  // missing pair
  CHECK(refuses(fixture([](const std::string& l) { return l.rfind("D 2 3 ", 0) != 0; }), "pair 2-3 has no route"));
  // the reverse of a relayed route missing
  CHECK(refuses(fixture([](const std::string& l) { return l.rfind("R 6 1 ", 0) != 0; }), "route 6-1 missing"));
  // the two directions through different vias (rank 7 as the via of 6->0 is a valid hop pair, 5 for 0->6)
  CHECK(refuses(replace(g, line_of(g, "R 6 0 5 "),
                        "R 6 0 7 " + ip(31, 7) + " " + ip(31, 8) + " " + ip(21, 8) + " " + ip(21, 1)), "different vias"));
  // 2-hop: an R record with a second via; and an R whose via is itself relayed to the destination
  CHECK(refuses(fixture(nullptr, "R 1 2 3 4 " + ip(1, 2) + " " + ip(1, 4) + " " + ip(1, 4) + " " + ip(1, 3) + " x\n"), "more than one relay hop"));
  // (7 is direct with 6 but not with 1: 7 reaches 1 through 0)
  CHECK(refuses(replace(g, line_of(g, "R 6 1 5 "), "R 6 1 7 " + ip(31, 7) + " " + ip(31, 8) + " " + ip(21, 8) + " " + ip(21, 1)),
                "R route 6->1 via 7"));
  // duplicates: a second D, R, or P record, and one address declared by two ranks
  CHECK(refuses(fixture(nullptr, "D 2 1 " + ip(2, 3) + " " + ip(2, 2) + "\n"), "second D record for pair 1-2"));
  CHECK(refuses(fixture(nullptr, line_of(g, "R 6 1 5 ") + "\n"), "second R record for route 6->1"));
  CHECK(refuses(fixture(nullptr, "P 3 " + ip(1, 4) + "/24\n"), "second P record for rank 3"));
  CHECK(refuses(replace(g, "P 4 " + ip(1, 5) + "/24", "P 4 " + ip(1, 5) + "/24 " + ip(1, 4) + "/24"), "declared by rank 3 and rank 4"));
  // a hop address that is not in the named rank's P record (rank 3's address used for rank 2, same subnet)
  CHECK(refuses(replace(g, "D 1 2 " + ip(2, 2) + " " + ip(2, 3), "D 1 2 " + ip(2, 2) + " " + ip(2, 4)), "is not in the P record of rank 2"));
  // the stray alias: rank 1 holds 198.18.21.2 locally but never declared it; a table that uses it is refused
  CHECK(refuses(replace(g, line_of(g, "R 7 1 0 "),
                        "R 7 1 0 " + ip(21, 8) + " " + ip(21, 1) + " " + ip(21, 1) + " " + ip(21, 2)), "198.18.21.2 is not in the P record of rank 1"));
  // a relay hop whose addresses are on two subnets
  CHECK(refuses(replace(g, line_of(g, "R 7 1 0 "), "R 7 1 0 " + ip(21, 8) + " " + ip(22, 1) + " " + ip(1, 1) + " " + ip(1, 2)),
                "R route 7->1 via 0 (first hop)"));
  // a pair with both a D and an R record
  CHECK(refuses(fixture(nullptr, "R 1 2 3 " + ip(1, 2) + " " + ip(1, 4) + " " + ip(1, 4) + " " + ip(1, 3) + "\nR 2 1 3 " +
                                     ip(1, 3) + " " + ip(1, 4) + " " + ip(1, 4) + " " + ip(1, 2) + "\n"), "pair 1-2 has both a D and an R record"));
  // ranks: out of range, twice in one record; the rank count
  CHECK(refuses(fixture(nullptr, "D 1 8 " + ip(1, 2) + " " + ip(1, 9) + "\n"), "bad D record"));
  CHECK(refuses(fixture(nullptr, "D 3 3 " + ip(1, 4) + " " + ip(1, 4) + "\n"), "names one rank twice"));
  CHECK(refuses(fixture(nullptr, "R 1 2 1 " + ip(1, 2) + " " + ip(1, 2) + " " + ip(1, 2) + " " + ip(1, 3) + "\n"), "names one rank twice"));
  CHECK(refuses(g, "rank count 9", 9));
  CHECK(refuses(g, "rank count 1", 1));
  CHECK(refuses(fixture([](const std::string& l) { return l.rfind("P 5 ", 0) != 0; }), "no P record for rank 5"));
  // malformed addresses and records
  for (const char* bad : {"198.18.1.256/24", "198.18.1.1", "198.18.1.1/0", "198.18.1.1/32", "198.18.01.1/24", "a.b.c.d/24", "198.18.1.1/2x"})
    CHECK(refuses(replace(g, "P 2 " + ip(1, 3) + "/24", std::string("P 2 ") + bad), "bad address"));
  CHECK(refuses(replace(g, "P 6 " + ip(11, 7) + "/24", "P 6 198.18.40.7/24 " + ip(11, 7) + "/24"), "more than 4 addresses"));
  CHECK(refuses(fixture(nullptr, "X 1 2\n"), "unknown record X"));
  CHECK(refuses(fixture(nullptr, "D 1 2 " + ip(2, 2) + "\n"), "bad D record"));
}

static void headers() {
  const uint32_t inl = 40, chunk = kRelayChunk;
  int accepted = 0, legacy_rejects = 0;
  for (uint32_t k = 0; k <= 1; k++)
    for (int s = 0; s < 8; s++)
      for (int d = 0; d < 8; d++) {
        if (s == d) continue;
        const uint32_t kind = relay_kind(k, d, s);
        CHECK((kind & 0xff) == k && (kind >> 8 & 0xff) == (uint32_t)d && (kind >> 16 & 0xff) == (uint32_t)s && !(kind >> 24));
        const RelayHdr h{4096, 8192, k ? 40u : 1u << 20, kind};
        const uint32_t bl = (uint32_t)sizeof(RelayHdr) + (k ? h.len : 0);
        uint32_t got = 99;
        CHECK(relay_header_ok(h, true, s, d, bl, inl, chunk, &got) && got == k);
        accepted++;
        // every other context refuses it: wrong src, wrong dst, or the two swapped
        for (int s2 = 0; s2 < 8; s2++)
          for (int d2 = 0; d2 < 8; d2++)
            if ((s2 != s || d2 != d) && s2 != d2) CHECK(!relay_header_ok(h, true, s2, d2, bl, inl, chunk, &got));
        // the legacy check refuses every packed kind (rank ids are nonzero in at least one field): decode is required
        uint32_t lk; if (!relay_header_ok(h, false, -1, -1, 0, inl, chunk, &lk)) legacy_rejects++;
        // unused bits, a bad k, out-of-range ranks
        CHECK(!relay_header_ok(RelayHdr{0, 0, h.len, kind | 1u << 24}, true, s, d, bl, inl, chunk, &got));
        CHECK(!relay_header_ok(RelayHdr{0, 0, h.len, kind | 0x80000000u}, true, s, d, bl, inl, chunk, &got));
        CHECK(!relay_header_ok(RelayHdr{0, 0, h.len, (kind & ~0xffu) | 2}, true, s, d, bl, inl, chunk, &got));
        // lengths: the received SEND length must match; an inline payload at most inline_max; a ring payload 1..chunk
        CHECK(!relay_header_ok(h, true, s, d, bl + 8, inl, chunk, &got));
        CHECK(!relay_header_ok(h, true, s, d, bl - 1, inl, chunk, &got));
        if (k == 1) {
          CHECK(!relay_header_ok(RelayHdr{0, 0, 41, kind}, true, s, d, 24 + 41, inl, chunk, &got));
          CHECK(relay_header_ok(RelayHdr{0, 0, 0, kind}, true, s, d, 24, inl, chunk, &got));
        } else {
          CHECK(!relay_header_ok(RelayHdr{0, 0, 0, kind}, true, s, d, 24, inl, chunk, &got));
          CHECK(!relay_header_ok(RelayHdr{0, 0, chunk + 1, kind}, true, s, d, 24, inl, chunk, &got));
          CHECK(relay_header_ok(RelayHdr{0, 0, 1, kind}, true, s, d, 24, inl, chunk, &got));
        }
      }
  CHECK(accepted == 112 && legacy_rejects == 112);
  for (int r = 0; r < 8; r++) {                                   // ranks 8+ never decode, src == dst never decodes
    uint32_t got;
    CHECK(!relay_header_ok(RelayHdr{0, 0, 1, relay_kind(0, 8, r)}, true, r, 8, 24, inl, chunk, &got));
    CHECK(!relay_header_ok(RelayHdr{0, 0, 1, relay_kind(0, r, 8)}, true, 8, r, 24, inl, chunk, &got));
    CHECK(!relay_header_ok(RelayHdr{0, 0, 1, relay_kind(0, r, r)}, true, r, r, 24, inl, chunk, &got));
  }
  // legacy ring compatibility: kinds stay exactly 0 and 1, accepted exactly as the old `kind > 1 || (kind == 1 && len >
  // inline)` test did, for any length and any received byte count (the ring check reads no byte_len)
  for (uint32_t kind = 0; kind < 600; kind++)
    for (uint32_t len : {0u, 1u, 39u, 40u, 41u, 4096u, 1u << 20, (1u << 20) + 1}) {
      uint32_t got = 99;
      const bool old_ok = !(kind > 1 || (kind == 1 && len > inl));
      CHECK(relay_header_ok(RelayHdr{0, 0, len, kind}, false, 3, 5, 12345, inl, chunk, &got) == old_ok);
      CHECK(got == kind);
    }
  CHECK(sizeof(RelayHdr) == 24 && 64 - (int)sizeof(RelayHdr) == 40);
}

int main() {
  good_table();
  refusals();
  headers();
  if (fails) { fprintf(stderr, "route_test: %d failures\n", fails); return 1; }
  printf("route_test: OK (8-rank table: 18 direct pairs, 20 relayed routes; 112 packed headers exact; refusals named)\n");
  return 0;
}
