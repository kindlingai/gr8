"""The mentat side of `--mentat localhost`: a native agent link to the local daemon, and the group's membership.

Modelled on mentat's python/ray/register.py (mentat's own PROTOCOL.md: "Agent link", "Client link", "Snapshot"). Each box registers
one agent in the group (`gr8-glm53` by default) with one UMA GPU and its gr8 state in `meta`. The members are the
group's live agents that report a GPU, one per node_ip; rank r is the r-th address in numeric order, so every box
computes the same ranks from the same snapshot: each box publishes in `gr8.offers` the ranks its model set can serve
(`any`, or a list), the boxes that offer particular ranks get them first, and the rest take the free ranks in
address order. The head's API is announced under a second agent id, so the member row never changes.

With --shape ring there is no switch, so the cables decide the order. A daemon's snapshot holds only its own probes,
so each box publishes in `gr8.ring` its rdma addresses (with their PCIe root) and the peer addresses its daemon's
probes reached directly, and every box derives the same ring from the members' meta (ring_plan): the rank order along
the cables, and each rank's addresses toward the next and previous ranks.
"""
import ipaddress
import itertools
import json
import os
import select
import socket
import struct
import threading
import time
import urllib.request

PROTO = '0.99'
PING_S = 2.0
# the service a KV node (./gr8 kv up) announces
KV_SERVICE = 'gr8-kv'
# --shape ring: the agent meta key with this box's probed rdma links; how long every member's links must hold before
# a plan (longer than mentat's default 15 s probe round); and how long a full group whose links form no ring waits
# for more probes before the boot is refused
LINKS_META = 'gr8.ring'
META_MAX = 1024
RING_SETTLE_S = 45.0
RING_STABLE_S = 20.0


def frame(header):
    hb = json.dumps(header).encode()
    return struct.pack('<II', len(hb), 0) + hb


def read_frame(sock):
    def exact(n):
        buf = b''
        while len(buf) < n:
            c = sock.recv(n - len(buf))
            if not c:
                raise ConnectionError('mentat daemon closed the connection')
            buf += c
        return buf
    hlen, plen = struct.unpack('<II', exact(8))
    header = json.loads(exact(hlen))
    exact(plen)
    return header


def major_ok(peer):
    """Majors 0 and 1 read each other while 0.99 becomes 1.0 (PROTOCOL.md "Version")."""
    head, dot, tail = str(peer).partition('.')
    if not dot or not head.isdigit() or not tail.isdigit():
        return False
    a, b = '0', head.lstrip('0') or '0'
    return a == b or {a, b} <= {'0', '1'}


class Daemon:
    """Where the local daemon listens: control (agent and client links) and HTTP."""

    def __init__(self, spec):
        host, _, port = (spec or 'localhost').partition(':')
        self.host = '127.0.0.1' if host in ('', 'localhost') else host
        self.port = int(port or 6379)
        self.http = f'http://{self.host}:{self.port + 1}'

    def local_status(self):
        """This daemon's own snapshot: its node_ip, addresses, tags and probe table."""
        with urllib.request.urlopen(self.http + '/status', timeout=5) as r:
            return json.load(r)

    def group_status(self, group):
        """The head's snapshot of the group, over a client link the local daemon relays to the head."""
        with socket.create_connection((self.host, self.port), timeout=10) as s:
            s.sendall(frame({'t': 'hello', 'req': 1, 'proto': PROTO, 'client_id': f'gr8-status-{os.getpid()}',
                             'group': group, 'session': False, 'kind': 'cli'}))
            h = read_frame(s)
            if h.get('t') != 'hello_ok':
                raise RuntimeError(f'mentat refused the client link: {h.get("error", h.get("t"))}')
            s.sendall(frame({'t': 'status', 'req': 2, 'group': group}))
            h = read_frame(s)
            if h.get('t') != 'status_ok':
                raise RuntimeError(f'mentat status failed: {h.get("error", h.get("t"))}')
            return h['snapshot']


def node_view(snap):
    """(node_ip, lan address, sorted rdma IPv4 addresses) of the daemon that produced this snapshot."""
    tags = snap.get('addr_tags') or {}
    node = snap.get('node_ip') or ''
    lan = [a for a in snap.get('addrs') or [] if 'lan' in tags.get(a, [])]
    return node, (lan[0] if lan else node), rdma_addrs(tags)


def rdma_addrs(tags):
    out = [a for a, t in tags.items() if 'rdma' in t and ':' not in a]
    return sorted(out, key=ipaddress.ip_address)


def member_rows(snap, group):
    """{node_ip: (node_id, meta)} of the group's live GPU agents, the first agent of each node."""
    agents = ((snap.get('groups') or {}).get(group) or {}).get('agents') or {}
    seen = {}
    for a in agents.values():
        if a.get('alive') and (a.get('machine') or {}).get('gpus') and a.get('node_ip'):
            seen.setdefault(a['node_ip'], (a.get('node_id', ''), a.get('meta') or {}))
    return seen


def members(snap, group):
    """[(node_ip, node_id, offers)] of the group's live GPU agents, one per node, sorted by address. offers is the
    agent's gr8.offers meta, or None before it has published one."""
    seen = member_rows(snap, group)
    return sorted(((ip, n, m.get('gr8.offers')) for ip, (n, m) in seen.items()), key=lambda x: ipaddress.ip_address(x[0]))


def member_links(snap, group):
    """{node_ip: gr8.ring meta text, or None before the box has published it} of the group's members."""
    return {ip: m.get(LINKS_META) for ip, (_, m) in member_rows(snap, group).items()}


def kv_nodes(snap, group):
    """[(address, port, meta)] of the group's live KV nodes (agents announcing KV_SERVICE), by address. meta has
    gr8.kv.bytes and gr8.kv.rdma, the node's addresses with a RoCE v2 GID."""
    agents = ((snap.get('groups') or {}).get(group) or {}).get('agents') or {}
    out = []
    for a in agents.values():
        svc = (a.get('services') or {}).get(KV_SERVICE)
        if a.get('alive') and svc and a.get('node_ip'):
            out.append((svc.get('host') or a['node_ip'], int(svc['port']), a.get('meta') or {}))
    return sorted(out, key=lambda x: ipaddress.ip_address(x[0]))


def kv_target(nodes, rails, rank=0):
    """host:port of the KV node a box with these fabric addresses should use: the first node with an RDMA address on
    one of the rails' /24 subnets, else the first node's LAN address (TCP). Rank r tries rail r % 2 first, so the ranks
    split their traffic over both of the node's PCIe roots."""
    nets = [a.rsplit('.', 1)[0] for a in rails if a]
    nets = nets[rank % len(nets):] + nets[:rank % len(nets)] if nets else nets
    for net in nets:
        for _, port, meta in nodes:
            for a in filter(None, (meta.get('gr8.kv.rdma') or '').split(',')):
                if a.rsplit('.', 1)[0] == net:
                    return f'{a}:{port}'
    return f'{nodes[0][0]}:{nodes[0][1]}' if nodes else None


def format_offers(ranks):
    """None (any rank) -> 'any', else '2' or '1,3' ('' for none)."""
    return 'any' if ranks is None else ','.join(map(str, ranks))


def parse_offers(text):
    return None if text == 'any' else [int(x) for x in text.split(',') if x]


class NoAssignment(RuntimeError):
    pass


def assign_ranks(ips, offers, tp, head=None):
    """The members in rank order. ips: sorted by address; offers: {ip: None (any) or [ranks]}. The pinned head takes
    rank 0. Boxes that offer particular ranks take them next, fewest choices first. The rest take the free ranks in
    address order. Raises NoAssignment, naming every box's offer, when no assignment fits."""
    def fail(why):
        raise NoAssignment(f'{why}. The model set on each box offers: '
                           + ', '.join(f'{ip} {format_offers(offers[ip]) or "no rank"}' for ip in ips))
    taken = {}
    if head:
        if head not in ips:
            fail(f'--head {head} is not a member')
        if offers[head] is not None and 0 not in offers[head]:
            fail(f'--head {head} must be rank 0')
        taken[0] = head
    picky = [ip for ip in ips if offers[ip] is not None and ip not in taken.values()]
    for ip in sorted(picky, key=lambda ip: len(offers[ip])):
        free = [r for r in offers[ip] if r not in taken and 0 <= r < tp]
        if not free:
            fail(f'no rank left for {ip}')
        taken[free[0]] = ip
    rest = [ip for ip in ips if ip not in taken.values()]
    for ip, r in zip(rest, [r for r in range(tp) if r not in taken]):
        taken[r] = ip
    return [taken[r] for r in range(tp)]


def ip_key(a):
    return ipaddress.ip_address(a)


def own_links(local, usable):
    """This box's side of a ring, as the gr8.ring meta text: 'addr=root,...;addr>peer_addr,...'. usable: {rdma address
    with a RoCE v2 device: PCIe root index}. A link is a pair the local daemon's probe reached, and not through another
    box (direct is not false)."""
    tags = local.get('addr_tags') or {}
    mine = {a: r for a, r in usable.items() if 'rdma' in tags.get(a, [])}
    links = set()
    for peer in (local.get('peers') or {}).values():
        for x, row in ((peer or {}).get('probes') or {}).items():
            if x not in mine:
                continue
            for y, p in (row or {}).items():
                if (p or {}).get('ok') is True and p.get('direct') is not False and ':' not in y:
                    links.add((x, y))
    text = (','.join(f'{a}={mine[a]}' for a in sorted(mine, key=ip_key)) + ';'
            + ','.join(f'{x}>{y}' for x, y in sorted(links, key=lambda p: (ip_key(p[0]), ip_key(p[1])))))
    if len(text) > META_MAX:
        raise RuntimeError(f'this box\'s ring links take {len(text)} bytes, more than mentat\'s {META_MAX}: '
                           'tag only the ConnectX ports rdma')
    return text


def parse_links(text):
    """gr8.ring meta text -> ({address: root}, {(address, peer address)})."""
    head, _, tail = (text or '').partition(';')
    roots = {}
    for item in filter(None, head.split(',')):
        a, _, r = item.partition('=')
        roots[a] = int(r)
    links = {tuple(item.split('>', 1)) for item in filter(None, tail.split(',')) if '>' in item}
    return roots, links


def ring_text(order, ports):
    """One line naming the ring plan, so the handshake can check every box derived the same one."""
    return ' '.join(f'{ip}:{",".join(p)}' for ip, p in zip(order, ports))


def ring_plan(boxes, tp, offers, head=None):
    """The ring a set of boxes forms through their cables. boxes: {lan: gr8.ring meta text}; offers: {lan: None (any)
    or [ranks]}. Returns (order, ports): the LAN addresses in rank order, and for each rank its addresses as
    (relay-out, direct-next, relay-in, direct-prev), the order of --ring-devs. Every box computes the same answer
    from the same meta. Rank 0 is --head, or else the lowest address the offers allow, and rank 1 its lower
    neighbour. On each cable the pair from the higher PCIe roots carries the direct writes and a second pair, on other
    ports at both ends, the relay (TP=4). Raises NoAssignment, naming the boxes, when the links form no ring."""
    n = int(tp)
    ips = sorted(boxes, key=ip_key)
    parsed = {ip: parse_links(boxes[ip]) for ip in ips}
    owner = {a: ip for ip in ips for a in parsed[ip][0]}
    root = {a: r for ip in ips for a, r in parsed[ip][0].items()}
    pairs = {}
    for ip in ips:
        roots, links = parsed[ip]
        for x, y in links:
            b = owner.get(y)
            if x not in roots or b is None or b == ip:
                continue
            pairs.setdefault((ip, b), set()).add((x, y))
            pairs.setdefault((b, ip), set()).add((y, x))
    nbrs = {ip: sorted({b for a, b in pairs if a == ip}, key=ip_key) for ip in ips}
    if head and head not in ips:
        raise NoAssignment(f'--head {head} is not a member ({", ".join(ips)})')
    cycles = [] if len(ips) != n else [
        (ips[0], *p) for p in itertools.permutations(ips[1:])
        if all(((ips[0], *p)[i], (ips[0], *p)[(i + 1) % n]) in pairs for i in range(n))]
    if not cycles:
        short = [ip for ip in ips if len(nbrs[ip]) < 2]
        idle = [f'{a} on {ip}' for ip in ips for a in sorted(parsed[ip][0], key=ip_key)
                if not any(a in p for k, v in pairs.items() if k[0] == ip for p in v)]
        why = f'the probed links of {len(ips)} boxes form no ring of {n}'
        if len(short) == 2 and short[1] not in nbrs[short[0]]:
            why += f': no cable works between {short[0]} and {short[1]}'
        elif short:
            why += ': ' + ', '.join(f'{ip} reaches {" and ".join(nbrs[ip]) or "no box"}' for ip in short)
        if idle:
            why += f'. Ports that reach no box: {", ".join(idle)}'
        links = sorted({tuple(sorted(k, key=ip_key)) for k in pairs}, key=lambda k: (ip_key(k[0]), ip_key(k[1])))
        raise NoAssignment(why + '. Links: ' + (', '.join(f'{a}-{b}' for a, b in links) or 'none'))
    orders = {d[k:] + d[:k] for c in cycles for d in (c, c[::-1]) for k in range(n)}
    fits = [o for o in orders if (not head or o[0] == head)
            and all(offers.get(o[r]) is None or r in offers[o[r]] for r in range(n))]
    if not fits:
        rings = sorted({min((c, (c[0], *c[:0:-1])), key=lambda o: [ip_key(a) for a in o]) for c in cycles})
        raise NoAssignment(f'no rank order along the ring ({"; ".join(" - ".join(c) for c in rings)}) fits '
                           + (f'--head {head} and ' if head else '') + 'the model set on each box, which offers: '
                           + ', '.join(f'{ip} {format_offers(offers.get(ip)) or "no rank"}' for ip in ips))
    order = min(fits, key=lambda o: [ip_key(a) for a in o])
    edges = []
    for r in range(n):
        a, b = order[r], order[(r + 1) % n]
        ps = sorted(pairs[(a, b)], key=lambda p: (root[p[0]], root[p[1]], ip_key(p[0]), ip_key(p[1])))
        direct = ps[-1]
        if n == 3:      # TP=3 opens only the direct devices (fabric.cpp): the relay fields repeat them
            edges.append((direct, direct))
            continue
        relay = next((p for p in ps if p[0] != direct[0] and p[1] != direct[1]), None)
        if relay is None:
            ends = lambda ip: ', '.join(f'{x} (root {root[x]})' for x in sorted(parsed[ip][0], key=ip_key))  # noqa: E731
            raise NoAssignment(f'{a} and {b} link on one port pair only ({direct[0]} - {direct[1]}): TP=4 relays '
                               'over a second pair on the same cable, from the other PCIe root at both ends. '
                               f'{a} has {ends(a)}; {b} has {ends(b)}. Is that pair addressed, tagged rdma, and '
                               'answering mentat\'s probes?')
        edges.append((direct, relay))
    ports = []
    for r in range(n):
        (dn, ro), (dp, ri) = edges[r], edges[r - 1]
        p = (ro[0], dn[0], ri[1], dp[1])
        both = {p[0], p[1]} & {p[2], p[3]}
        if both:
            raise NoAssignment(f'{order[r]} reaches both its ring neighbours from {", ".join(sorted(both))}: '
                               '--shape ring wants one cable to each neighbour')
        ports.append(p)
    return list(order), ports


def rails_of(snap, node_id):
    """The rdma addresses of a member, from the head's snapshot (its own row is the top level)."""
    if node_id == snap.get('node_id'):
        return rdma_addrs(snap.get('addr_tags') or {})
    peer = (snap.get('peers') or {}).get(node_id) or {}
    return rdma_addrs(peer.get('addr_tags') or {})


def probe_problems(local, ranks, me, my_rails):
    """Fabric pairs the local daemon's probe table says fail, as text. ranks: [(lan, node_id, rails)]."""
    out = []
    peers = local.get('peers') or {}
    for lan, node_id, rails in ranks:
        if lan == me:
            continue
        probes = (peers.get(node_id) or {}).get('probes') or {}
        for k in range(2):
            ok = ((probes.get(my_rails[k]) or {}).get(rails[k]) or {}).get('ok')
            cross = ((probes.get(my_rails[k]) or {}).get(rails[1 - k]) or {}).get('ok')
            if ok is False and cross:
                out.append(f'rail {k}: {my_rails[k]} reaches {lan} on its rail {1 - k} ({rails[1 - k]}), not on '
                           f'{rails[k]}: the boxes number their PCIe roots differently')
            elif ok is False:
                out.append(f'rail {k}: mentat cannot reach {rails[k]} ({lan}) from {my_rails[k]}')
    return out


class Agent:
    """One agent registration held open, redialled when the daemon closes it. set_meta() re-registers on a fresh
    link first and then closes the old one, which the daemon allows (it marks an agent lost only on the EOF of the
    link that owns the current registration)."""

    def __init__(self, daemon, group, agent_id, node_ip, gpus=1, services=None, log=print):
        self.daemon, self.group, self.agent_id, self.node_ip = daemon, group, agent_id, node_ip
        self.gpus, self.services, self.meta, self.log = gpus, services or {}, {}, log
        self.sock, self.lock, self.dirty = None, threading.Lock(), threading.Event()
        self.stopped = threading.Event()
        self.error = None

    def frame(self):
        gpu = [{'index': 0, 'vendor': 'nvidia', 'name': 'NVIDIA GB10', 'memory': 0, 'uma': True}]
        return {'t': 'agent_register', 'req': 1, 'proto': PROTO, 'agent_id': self.agent_id, 'group': self.group,
                'node_ip': self.node_ip, 'container': 'gr8', 'pid': os.getpid(),
                'machine': {'memory': 0, 'cpus': os.cpu_count() or 1, 'gpus': gpu[:self.gpus]},
                'services': self.services, 'resume': [], 'unacked_refs': [],
                'meta': {k: str(v)[:1024] for k, v in self.meta.items()}}

    def register(self):
        s = socket.create_connection((self.daemon.host, self.daemon.port), timeout=10)
        try:
            s.sendall(frame(self.frame()))
            h = read_frame(s)
            if h.get('t') == 'err':
                raise RuntimeError(f'mentat refused the registration: {h.get("error")} ({h.get("code", "")})')
            if not major_ok(h.get('proto', '')):
                raise RuntimeError(f'mentat daemon speaks {h.get("proto")!r}, gr8 speaks {PROTO}')
            if h.get('t') != 'agent_register_ok':
                raise RuntimeError(f'expected agent_register_ok, got {h.get("t")!r}')
            s.settimeout(None)
            return s
        except BaseException:
            s.close()
            raise

    def start(self):
        threading.Thread(target=self.run, daemon=True, name=f'mentat-{self.agent_id}').start()

    def set_meta(self, **kv):
        with self.lock:
            self.meta.update({k: v for k, v in kv.items() if v is not None})
        self.dirty.set()

    def stop(self):
        self.stopped.set()
        self.dirty.set()

    def run(self):
        delay = 1.0
        while not self.stopped.is_set():
            try:
                with self.lock:
                    self.dirty.clear()
                    self.sock = self.register()
                delay = 1.0
                self.hold()
            except (OSError, RuntimeError, ValueError) as e:
                if 'refused the registration' in str(e) or 'speaks' in str(e):
                    self.error = str(e)
                self.log(f'mentat: {self.agent_id}: {e}; retry in {delay:.0f}s')
                self.stopped.wait(delay)
                delay = min(delay * 2, 30.0)
        if self.sock:
            self.sock.close()

    def hold(self):
        """Ping every 2 s, drain the daemon's frames, and re-register when the meta changes."""
        while not self.stopped.is_set():
            if self.dirty.wait(PING_S):
                if self.stopped.is_set():
                    return
                with self.lock:
                    self.dirty.clear()
                    fresh = self.register()
                    old, self.sock = self.sock, fresh
                old.close()
                continue
            self.sock.sendall(frame({'t': 'ping', 'req': 0}))
            while select.select([self.sock], [], [], 0)[0]:
                h = read_frame(self.sock)
                if h.get('t') == 'ping':
                    self.sock.sendall(frame({'t': 'pong', 'req': h.get('req', 0)}))


def elect(daemon, group, tp, head=None, timeout=600, hold_s=5.0, log=print, sleep=time.sleep, clock=time.monotonic,
          ring=None, ring_settle=RING_SETTLE_S, ring_stable=RING_STABLE_S):
    """Wait until the group has tp live members, with their offers, the same on two reads hold_s apart; returns
    (ranks, snapshot), where ranks is [(lan, node_id, (rail0, rail1))] in rank order. Raises NoAssignment when the
    offers fit no assignment, and RuntimeError on a timeout or a bad membership.

    ring: for --shape ring, a callable run before each read that republishes this box's gr8.ring meta. The ranks
    then follow the cables (ring_plan), every member's gr8.ring counts toward the two equal reads, and the third
    field of each rank is its (relay-out, direct-next, relay-in, direct-prev) addresses. The members' gr8.ring must
    hold for ring_stable seconds before a plan, and links that form no ring get ring_settle seconds of new probes
    before the NoAssignment."""
    t0, last, said, unsettled, since = clock(), None, -60, None, None
    while True:
        if ring:
            ring()
        try:
            snap = daemon.group_status(group)
            ms = members(snap, group)
            links = member_links(snap, group) if ring else {}
        except (OSError, RuntimeError, ValueError) as e:
            snap, ms, links = None, [], {}
            err = str(e)
        else:
            err = ''
        if len(ms) > tp:
            raise RuntimeError(f'group {group} has {len(ms)} members for --tp {tp} '
                               f'({", ".join(m[0] for m in ms)}); stop one, or use --group')
        waited = clock() - t0
        if timeout and waited >= timeout:
            raise RuntimeError(f'waited {waited:.0f}s for {tp} members in group {group}, have {len(ms)} '
                               f'({", ".join(m[0] for m in ms) or "none"})'
                               + (', but their ring links kept changing' if len(ms) == tp and ring else '')
                               + (f': {err}' if err else ''))
        if len(ms) == tp and all(o is not None for _, _, o in ms) and all(v is not None for v in links.values()):
            key = (ms, sorted(links.items()))
            if key != last:
                since = clock()
            # a ring's links count once no box's gr8.ring changed for a probe round, so every box plans from the same
            # tables (a plan from other tables is refused by the handshake: ring= in the cluster options)
            if key == last and ring and clock() - since < ring_stable:
                if waited - said >= 60:
                    log(f'waiting for the ring links of group {group} to settle')
                    said = waited
                sleep(hold_s)
                continue
            if key == last and ring:
                offers = {ip: parse_offers(o) for ip, _, o in ms}
                try:
                    order, ports = ring_plan(links, tp, offers, head)
                except NoAssignment as e:
                    if unsettled is None:
                        unsettled = clock()
                        log(f'{e}; waiting up to {ring_settle:.0f}s for more probes')
                    elif clock() - unsettled >= ring_settle:
                        raise
                    sleep(hold_s)
                    continue
                ids = {ip: n for ip, n, _ in ms}
                return [(ip, ids[ip], p) for ip, p in zip(order, ports)], snap
            if key == last:
                ips = assign_ranks([ip for ip, _, _ in ms], {ip: parse_offers(o) for ip, _, o in ms}, tp, head)
                ids = {ip: n for ip, n, _ in ms}
                ranks = []
                for ip in ips:
                    rails = rails_of(snap, ids[ip])
                    if len(rails) != 2:
                        raise RuntimeError(f'{ip} has {len(rails)} rdma-tagged addresses ({", ".join(rails) or "none"}), '
                                           'want two. Tag both ConnectX roots in mentat: MENTAT_ANNOUNCE_IFACES='
                                           'en*f*np*=connectx+rdma,en*=lan (or pass --ips and --ips2)')
                    ranks.append((ip, ids[ip], tuple(rails)))
                return ranks, snap
            last = key
            sleep(hold_s)
            continue
        last = None
        if waited - said >= 60:
            log(f'waiting for {tp} members in group {group}, have {len(ms)} ({", ".join(m[0] for m in ms) or "none"})'
                + (f': {err}' if err else ''))
            said = waited
        sleep(hold_s)
