// Hybrid fabric route table (FabricConfig::hybrid): parse, validate, and this rank's role plan. CPU only: no verbs, no
// CUDA, so engine/comms/tests/route_test.cpp drives it on any host. The launcher writes the file (one line each):
//   P <rank> <ip/prefix> [<ip/prefix> ...]                               the fabric addresses rank declared
//   D <a> <b> <ip_a> <ip_b>                                              a direct pair a<->b on one rail
//   R <src> <dst> <via> <ip_src> <ip_via_in> <ip_via_out> <ip_dst>      a directed route src->dst relayed by via
// Blank lines and lines starting with # are skipped. Every check refuses with the pair or rank named, before any QP.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace g6 {

constexpr int kRouteMaxRanks = 8;
constexpr int kRouteMaxPorts = 4;   // fabric addresses a rank may declare: two ports x two PCIe roots

struct RouteAddr { uint32_t ip = 0; int prefix = 0; std::string text; };   // ip in host order; text = "a.b.c.d"
struct RouteDirect { int a = -1, b = -1; std::string ip_a, ip_b; };          // stored with a < b
struct RouteRelay { int src = -1, dst = -1, via = -1; std::string ip_src, ip_via_in, ip_via_out, ip_dst; };

struct RouteTable {
  int nr = 0;
  std::vector<std::vector<RouteAddr>> ports;   // [rank] the P record
  std::vector<RouteDirect> direct;
  std::vector<RouteRelay> relay;
  int direct_of(int a, int b) const {
    if (a > b) std::swap(a, b);
    for (size_t i = 0; i < direct.size(); i++) if (direct[i].a == a && direct[i].b == b) return (int)i;
    return -1;
  }
  int relay_of(int src, int dst) const {
    for (size_t i = 0; i < relay.size(); i++) if (relay[i].src == src && relay[i].dst == dst) return (int)i;
    return -1;
  }
};

namespace route_detail {
inline bool parse_ip(const std::string& s, uint32_t* ip) {
  unsigned a[4]; char tail;
  if (s.empty() || s.size() > 15 || sscanf(s.c_str(), "%u.%u.%u.%u%c", &a[0], &a[1], &a[2], &a[3], &tail) != 4) return false;
  for (unsigned v : a) if (v > 255) return false;
  char back[16]; snprintf(back, sizeof back, "%u.%u.%u.%u", a[0], a[1], a[2], a[3]);
  if (s != back) return false;                      // no leading zeros or other spellings: one text per address
  *ip = a[0] << 24 | a[1] << 16 | a[2] << 8 | a[3];
  return true;
}
inline bool parse_cidr(const std::string& s, RouteAddr* out) {
  const size_t k = s.find('/');
  if (k == std::string::npos) return false;
  const std::string ip = s.substr(0, k), pre = s.substr(k + 1);
  if (pre.empty() || pre.size() > 2 || pre.find_first_not_of("0123456789") != std::string::npos) return false;
  const int p = atoi(pre.c_str());
  if (p < 1 || p > 31 || !parse_ip(ip, &out->ip)) return false;
  out->prefix = p; out->text = ip;
  return true;
}
inline bool parse_rank(const std::string& s, int nr, int* r) {
  if (s.empty() || s.size() > 2 || s.find_first_not_of("0123456789") != std::string::npos) return false;
  *r = atoi(s.c_str());
  return *r >= 0 && *r < nr;
}
inline uint32_t mask(int prefix) { return prefix <= 0 ? 0u : ~0u << (32 - prefix); }
inline std::string pair(int a, int b) { return std::to_string(a) + "-" + std::to_string(b); }
}  // namespace route_detail

// The P entry of rank r with this exact address (nullptr: r did not declare it)
inline const RouteAddr* route_port(const RouteTable& t, int r, const std::string& ip) {
  for (const RouteAddr& a : t.ports[r]) if (a.text == ip) return &a;
  return nullptr;
}

// One hop x<->y: each address must be in the P record of its named rank, and both on one subnet (same prefix)
inline bool route_hop_ok(const RouteTable& t, int x, const std::string& ipx, int y, const std::string& ipy,
                         const std::string& what, std::string* err) {
  const RouteAddr* ax = route_port(t, x, ipx);
  const RouteAddr* ay = route_port(t, y, ipy);
  if (!ax) { *err = what + ": " + ipx + " is not in the P record of rank " + std::to_string(x); return false; }
  if (!ay) { *err = what + ": " + ipy + " is not in the P record of rank " + std::to_string(y); return false; }
  const uint32_t m = route_detail::mask(ax->prefix);
  if (ax->prefix != ay->prefix || (ax->ip & m) != (ay->ip & m) || ax->ip == ay->ip) {
    *err = what + ": " + ipx + "/" + std::to_string(ax->prefix) + " and " + ipy + "/" + std::to_string(ay->prefix) +
           " are not two hosts on one subnet";
    return false;
  }
  return true;
}

// Parse and validate the whole table for nr ranks. false: *err names the line, pair, or rank.
inline bool route_parse(const std::string& text, int nr, RouteTable* out, std::string* err) {
  using namespace route_detail;
  RouteTable t; t.nr = nr; t.ports.assign(nr > 0 ? nr : 0, {});
  if (nr < 2 || nr > kRouteMaxRanks) { *err = "rank count " + std::to_string(nr) + " (want 2.." + std::to_string(kRouteMaxRanks) + ")"; return false; }
  std::vector<bool> have_p(nr, false);
  std::istringstream in(text);
  std::string line;
  int ln = 0;
  while (std::getline(in, line)) {
    ln++;
    std::istringstream ls(line);
    std::vector<std::string> w; std::string x;
    while (ls >> x) w.push_back(x);
    if (w.empty() || w[0][0] == '#') continue;
    const std::string at = "line " + std::to_string(ln);
    if (w[0] == "P") {
      int r;
      if (w.size() < 3 || !parse_rank(w[1], nr, &r)) { *err = at + ": bad P record"; return false; }
      if (have_p[r]) { *err = at + ": second P record for rank " + std::to_string(r); return false; }
      if ((int)w.size() - 2 > kRouteMaxPorts) { *err = at + ": rank " + std::to_string(r) + " declares more than " + std::to_string(kRouteMaxPorts) + " addresses"; return false; }
      have_p[r] = true;
      for (size_t i = 2; i < w.size(); i++) {
        RouteAddr a;
        if (!parse_cidr(w[i], &a)) { *err = at + ": bad address " + w[i] + " (want a.b.c.d/prefix)"; return false; }
        for (int q = 0; q < nr; q++)
          for (const RouteAddr& b : t.ports[q])
            if (b.ip == a.ip) { *err = at + ": address " + a.text + " declared by rank " + std::to_string(q) + " and rank " + std::to_string(r); return false; }
        t.ports[r].push_back(a);
      }
    } else if (w[0] == "D") {
      RouteDirect d;
      if (w.size() != 5 || !parse_rank(w[1], nr, &d.a) || !parse_rank(w[2], nr, &d.b)) { *err = at + ": bad D record"; return false; }
      if (d.a == d.b) { *err = at + ": D pair " + pair(d.a, d.b) + " names one rank twice"; return false; }
      d.ip_a = w[3]; d.ip_b = w[4];
      if (d.a > d.b) { std::swap(d.a, d.b); std::swap(d.ip_a, d.ip_b); }
      if (t.direct_of(d.a, d.b) >= 0) { *err = at + ": second D record for pair " + pair(d.a, d.b); return false; }
      t.direct.push_back(d);
    } else if (w[0] == "R") {
      RouteRelay r;
      if (w.size() > 8) { *err = at + ": R record with more than one relay hop (multi-hop routes are refused)"; return false; }
      if (w.size() != 8 || !parse_rank(w[1], nr, &r.src) || !parse_rank(w[2], nr, &r.dst) || !parse_rank(w[3], nr, &r.via)) {
        *err = at + ": bad R record"; return false;
      }
      if (r.src == r.dst || r.via == r.src || r.via == r.dst) {
        *err = at + ": R route " + pair(r.src, r.dst) + " via " + std::to_string(r.via) + " names one rank twice"; return false;
      }
      r.ip_src = w[4]; r.ip_via_in = w[5]; r.ip_via_out = w[6]; r.ip_dst = w[7];
      if (t.relay_of(r.src, r.dst) >= 0) { *err = at + ": second R record for route " + std::to_string(r.src) + "->" + std::to_string(r.dst); return false; }
      t.relay.push_back(r);
    } else {
      *err = at + ": unknown record " + w[0]; return false;
    }
  }
  for (int r = 0; r < nr; r++) if (!have_p[r]) { *err = "no P record for rank " + std::to_string(r); return false; }
  for (const RouteDirect& d : t.direct)
    if (!route_hop_ok(t, d.a, d.ip_a, d.b, d.ip_b, "D pair " + pair(d.a, d.b), err)) return false;
  for (const RouteRelay& r : t.relay) {
    const std::string what = "R route " + std::to_string(r.src) + "->" + std::to_string(r.dst) + " via " + std::to_string(r.via);
    if (!route_hop_ok(t, r.src, r.ip_src, r.via, r.ip_via_in, what + " (first hop)", err)) return false;
    if (!route_hop_ok(t, r.via, r.ip_via_out, r.dst, r.ip_dst, what + " (second hop)", err)) return false;
    // a relay hop must itself be one link: the via is a direct peer of both ends, never relayed to them
    if (t.direct_of(r.src, r.via) < 0 || t.direct_of(r.via, r.dst) < 0) {
      *err = what + ": the via is not a direct pair of both ends (a 2-hop route)"; return false;
    }
  }
  // every unordered pair: exactly one D, or an R each way through one via
  for (int a = 0; a < nr; a++)
    for (int b = a + 1; b < nr; b++) {
      const int d = t.direct_of(a, b), f = t.relay_of(a, b), g = t.relay_of(b, a);
      if (d >= 0 && (f >= 0 || g >= 0)) { *err = "pair " + pair(a, b) + " has both a D and an R record"; return false; }
      if (d >= 0) continue;
      if (f < 0 && g < 0) { *err = "pair " + pair(a, b) + " has no route"; return false; }
      if (f < 0 || g < 0) { *err = "pair " + pair(a, b) + ": route " + (f < 0 ? pair(a, b) : pair(b, a)) + " missing (an R record needs its reverse)"; return false; }
      if (t.relay[f].via != t.relay[g].via) {
        *err = "pair " + pair(a, b) + ": the two directions use different vias (" + std::to_string(t.relay[f].via) + ", " + std::to_string(t.relay[g].via) + ")";
        return false;
      }
    }
  *out = std::move(t);
  return true;
}

// The expected local declaration (--fab-ports, the box's FAB_PORTS) against my P record: the same set of ip/prefix
inline bool route_ports_match(const RouteTable& t, int me, const std::vector<std::string>& expected, std::string* err) {
  if (me < 0 || me >= t.nr) { *err = "rank " + std::to_string(me) + " is not in the table"; return false; }
  std::vector<std::string> want, have;
  for (const std::string& s : expected) {
    RouteAddr a;
    if (!route_detail::parse_cidr(s, &a)) { *err = "--fab-ports: bad address " + s; return false; }
    want.push_back(a.text + "/" + std::to_string(a.prefix));
  }
  for (const RouteAddr& a : t.ports[me]) have.push_back(a.text + "/" + std::to_string(a.prefix));
  std::sort(want.begin(), want.end()); std::sort(have.begin(), have.end());
  if (want != have) {
    auto join = [](const std::vector<std::string>& v) { std::string s; for (const auto& x : v) s += (s.empty() ? "" : ",") + x; return s; };
    *err = "the P record of rank " + std::to_string(me) + " (" + join(have) + ") is not --fab-ports (" + join(want) + ")";
    return false;
  }
  return true;
}

// This rank's roles, from a validated table. Every peer p != me has exactly one entry in peer.
struct RoutePlan {
  struct Peer { bool relayed = false; int via = -1; std::string ip; };   // ip: my address of the first hop
  struct Ctx { int src = -1, dst = -1; std::string ip_in, ip_out; };     // I relay src->dst: in from src, out to dst
  struct Fin { int src = -1, via = -1; std::string ip; };                // src->me arrives from via on my ip
  std::vector<Peer> peer;    // [nr]; peer[me] unused
  std::vector<Ctx> ctx;      // the routes I relay
  std::vector<Fin> fin;      // the relayed routes that end at me
  std::vector<std::string> local_ips;   // every distinct local address the plan uses, first-use order
};
inline RoutePlan route_plan(const RouteTable& t, int me) {
  RoutePlan p; p.peer.assign(t.nr, {});
  auto use = [&](const std::string& ip) {
    if (std::find(p.local_ips.begin(), p.local_ips.end(), ip) == p.local_ips.end()) p.local_ips.push_back(ip);
  };
  for (int q = 0; q < t.nr; q++) {
    if (q == me) continue;
    const int d = t.direct_of(me, q);
    if (d >= 0) {
      const RouteDirect& x = t.direct[d];
      p.peer[q] = RoutePlan::Peer{false, -1, x.a == me ? x.ip_a : x.ip_b};
    } else {
      const RouteRelay& r = t.relay[t.relay_of(me, q)];
      p.peer[q] = RoutePlan::Peer{true, r.via, r.ip_src};
    }
    use(p.peer[q].ip);
  }
  for (const RouteRelay& r : t.relay) {
    if (r.via == me) { p.ctx.push_back(RoutePlan::Ctx{r.src, r.dst, r.ip_via_in, r.ip_via_out}); use(r.ip_via_in); use(r.ip_via_out); }
    if (r.dst == me) { p.fin.push_back(RoutePlan::Fin{r.src, r.via, r.ip_dst}); use(r.ip_dst); }
  }
  return p;
}

// SHA-256 of the route file bytes: every engine logs it ("routes sha"), so a run shows the same table on every rank
inline std::string route_sha256(const std::string& m) {
  static const uint32_t K[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
      0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
      0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
      0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::string b = m; b += (char)0x80;
  while (b.size() % 64 != 56) b += (char)0;
  const uint64_t bits = (uint64_t)m.size() * 8;
  for (int i = 7; i >= 0; i--) b += (char)(bits >> (i * 8) & 0xff);
  auto rr = [](uint32_t x, int n) { return x >> n | x << (32 - n); };
  for (size_t o = 0; o < b.size(); o += 64) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
      w[i] = (uint32_t)(uint8_t)b[o + 4 * i] << 24 | (uint32_t)(uint8_t)b[o + 4 * i + 1] << 16 | (uint32_t)(uint8_t)b[o + 4 * i + 2] << 8 | (uint8_t)b[o + 4 * i + 3];
    for (int i = 16; i < 64; i++) {
      const uint32_t s0 = rr(w[i - 15], 7) ^ rr(w[i - 15], 18) ^ (w[i - 15] >> 3), s1 = rr(w[i - 2], 17) ^ rr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], bb = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; i++) {
      const uint32_t t1 = hh + (rr(e, 6) ^ rr(e, 11) ^ rr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
      const uint32_t t2 = (rr(a, 2) ^ rr(a, 13) ^ rr(a, 22)) + ((a & bb) ^ (a & c) ^ (bb & c));
      hh = g; g = f; f = e; e = d + t1; d = c; c = bb; bb = a; a = t1 + t2;
    }
    h[0] += a; h[1] += bb; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  char out[65];
  for (int i = 0; i < 8; i++) snprintf(out + 8 * i, 9, "%08x", h[i]);
  return std::string(out, 64);
}

}  // namespace g6
