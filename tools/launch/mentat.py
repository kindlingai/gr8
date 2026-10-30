"""The mentat side of `--mentat localhost`: a native agent link to the local daemon, and the group's membership.

Modelled on mentat's python/ray/register.py (mentat's own PROTOCOL.md: "Agent link", "Client link", "Snapshot"). Each box registers
one agent in the group (`gr8-glm53` by default) with one UMA GPU and its gr8 state in `meta`. The members are the
group's live agents that report a GPU, one per node_ip; rank r is the r-th address in numeric order, so every box
computes the same ranks from the same snapshot: each box publishes in `gr8.offers` the ranks its model set can serve
(`any`, or a list), the boxes that offer particular ranks get them first, and the rest take the free ranks in
address order. The head's API is announced under a second agent id, so the member row never changes.

With --shape ring there is no switch, so the cables decide the order. Every box sends the same ring claim (mentat
0.24.0+: PROTOCOL.md "Claims"). The head solves it once over the group's nodes, and every holder gets that one view:
the cycle and every probed link between members. Each box publishes the PCIe root of its RoCE addresses in
`gr8.roots`, and ring_plan turns the view into the rank order and each rank's addresses toward the next and previous
ranks. A box holds the claim on an open client link (ClaimLink) until the handshake, so every box sees one view.
"""
import ipaddress
import itertools
import json
import os
import select
import socket
import struct
import subprocess
import threading
import time
import urllib.request

PROTO = '0.99'
PING_S = 2.0
# the service a KV node (./gr8 kv up) announces
KV_SERVICE = 'gr8-kv'
# --shape ring: the agent meta key with this box's {RoCE address: PCIe root}; how long a full group whose links form
# no ring waits for more probes before the boot is refused (three default 15 s probe rounds); and how many claims
# (one probe round apart) a TP=4 ring gets to find its relay pairs
ROOTS_META = 'gr8.roots'
RING_SETTLE_S = 45.0
RING_RETRY_S = 15.0
RING_ATTEMPTS = 4


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


def member_meta(snap, group, key):
    """{node_ip: the meta value, or None before the box has published it} of the group's members."""
    return {ip: m.get(key) for ip, (_, m) in member_rows(snap, group).items()}


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


def local_nets():
    """{address: its IPv4 network} for every address on this box, from `ip -j -4 addr`. Empty where ip is missing."""
    try:
        out = subprocess.run(['ip', '-j', '-4', 'addr', 'show'], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {i['local']: ipaddress.ip_interface(f"{i['local']}/{i['prefixlen']}").network
            for link in json.loads(out) for i in link.get('addr_info', [])}


def kv_target(nodes, rails, rank=0, nets=None):
    """host:port of the KV node a box with these fabric addresses should use: the first node with an RDMA address on
    one of the rails' subnets, else the first node's LAN address (TCP). nets: {rail: its network}, as local_nets()
    reads them; a rail without one is taken as a /24. Rank r tries rail r % 2 first, so the ranks split their
    traffic over both of the node's PCIe roots."""
    nets = nets or {}
    subnets = [nets.get(a) or ipaddress.ip_interface(f'{a}/24').network for a in rails if a]
    subnets = subnets[rank % len(subnets):] + subnets[:rank % len(subnets)] if subnets else subnets
    for net in subnets:
        for _, port, meta in nodes:
            for a in filter(None, (meta.get('gr8.kv.rdma') or '').split(',')):
                if ipaddress.ip_address(a) in net:
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


def head_address(head):
    """--head as a LAN address. A name, such as a LeaderWorkerSet's LWS_LEADER_ADDRESS (the leader pod's DNS name, its
    box's address on the host network), resolves to its IPv4 address. Raises OSError, naming it, while it does not."""
    try:
        ipaddress.ip_address(head)
        return head
    except ValueError:
        pass
    try:
        return socket.gethostbyname(head)
    except OSError as e:
        raise OSError(f'--head {head} does not resolve ({e})') from None


def format_roots(roots):
    """{address: PCIe root} -> the gr8.roots meta text 'addr=root,...'."""
    return ','.join(f'{a}={r}' for a, r in sorted(roots.items(), key=lambda x: ip_key(x[0])))


def parse_roots(text):
    out = {}
    for item in filter(None, (text or '').split(',')):
        a, _, r = item.partition('=')
        out[a] = int(r)
    return out


def ring_shape(tp):
    """The claim every box of a --shape ring group sends. driver: ignore (mentat 0.23.0+) makes the answer and its
    order independent of which box claims first."""
    return {'sets': [{'name': 'ring', 'bundles': int(tp), 'link': 'rdma', 'layout': 'ring'}], 'driver': 'ignore'}


def ring_name(group, tp, attempt):
    return f'{group}-ring{tp}' + (f'-{attempt}' if attempt else '')


class Incomplete(NoAssignment):
    """The claimed ring lacks a pair the plan needs, perhaps because its probes were not all in yet."""


class Stale(NoAssignment):
    """The claim's view is not of this group's members: a box of an earlier boot still holds the name."""


def view_pairs(view, lan_of, key='ring'):
    """{(lan, lan): {(address, address)}} of the probed links in a claim view (links[set], mentat 0.23.0+), both ways."""
    pairs = {}
    for link in (view.get('links') or {}).get(key) or []:
        a, b = link.get('a') or {}, link.get('b') or {}
        x, y = lan_of.get(a.get('node')), lan_of.get(b.get('node'))
        if x is None or y is None or x == y or not a.get('addr') or not b.get('addr'):
            continue
        pairs.setdefault((x, y), set()).add((a['addr'], b['addr']))
        pairs.setdefault((y, x), set()).add((b['addr'], a['addr']))
    return pairs


def ring_why(pairs, roots, tp, why):
    """Why the probed links of a group form no ring, naming the pair: pairs as view_pairs gives them (from any view of
    the members), roots {lan: {address: root}}, why mentat's own reason."""
    ips = sorted(roots, key=ip_key)
    nbrs = {ip: sorted({b for a, b in pairs if a == ip}, key=ip_key) for ip in ips}
    short = [ip for ip in ips if len(nbrs[ip]) < 2]
    idle = [f'{a} on {ip}' for ip in ips for a in sorted(roots[ip], key=ip_key)
            if not any(a == p[0] for k, v in pairs.items() if k[0] == ip for p in v)]
    out = f'the probed links of {len(ips)} boxes form no ring of {tp}'
    if len(short) == 2 and short[1] not in nbrs[short[0]]:
        out += f': no cable works between {short[0]} and {short[1]}'
    elif short:
        out += ': ' + ', '.join(f'{ip} reaches {" and ".join(nbrs[ip]) or "no box"}' for ip in short)
    if idle:
        out += f'. Ports that reach no box: {", ".join(idle)}'
    links = sorted({tuple(sorted(k, key=ip_key)) for k in pairs}, key=lambda k: (ip_key(k[0]), ip_key(k[1])))
    return out + '. Links: ' + (', '.join(f'{a}-{b}' for a, b in links) or 'none') + f'. mentat: {why}'


def ring_plan(view, ids, roots, tp, offers, head=None):
    """The ring a mentat ring claim found, as gr8 runs it. view: the claim_ok view; ids: {lan: node_id} of the members;
    roots: {lan: {address: PCIe root}} (each box's gr8.roots); offers: {lan: None (any) or [ranks]}. Returns (order,
    ports): the LAN addresses in rank order, and for each rank its addresses as (relay-out, direct-next, relay-in,
    direct-prev), the order of --ring-devs. Every box gets the same view, so every box computes the same answer.
    mentat's cycle is turned and reflected so rank 0 is --head, or else the lowest address the offers allow, and rank
    1 its lower neighbour. On each cable the pair from the higher PCIe roots carries the direct writes and a second
    pair, other addresses at both ends, the relay (TP=4). Raises NoAssignment when the ring does not fit gr8, and
    Incomplete when the view lacks the TP=4 relay pair."""
    n = int(tp)
    ips = sorted(ids, key=ip_key)
    lan_of = {v: k for k, v in ids.items()}
    nodes = [m.get('node') for m in (view.get('sets') or {}).get('ring') or []]
    if sorted(nodes) != sorted(ids.values()) or len(nodes) != n:
        hosts = ', '.join(lan_of.get(x) or str(x) for x in nodes)
        raise Stale(f"mentat's ring ({hosts or 'empty'}) is not the group's {n} members ({', '.join(ips)})")
    if head and head not in ids:
        raise NoAssignment(f'--head {head} is not a member ({", ".join(ips)})')
    cycle = tuple(lan_of[x] for x in nodes)
    root = {a: r for ip in ips for a, r in (roots.get(ip) or {}).items()}
    # only addresses behind a RoCE v2 device on their box (gr8.roots) can carry the engine's ring
    pairs = {k: {p for p in v if p[0] in root and p[1] in root} for k, v in view_pairs(view, lan_of).items()}
    orders = {d[k:] + d[:k] for d in (cycle, cycle[::-1]) for k in range(n)}
    fits = [o for o in orders if (not head or o[0] == head)
            and all(offers.get(o[r]) is None or r in offers[o[r]] for r in range(n))]
    if not fits:
        ring = min((cycle, (cycle[0], *cycle[:0:-1])), key=lambda o: [ip_key(a) for a in o])
        raise NoAssignment(f'no rank order along the ring ({" - ".join(ring)}) fits '
                           + (f'--head {head} and ' if head else '') + 'the model set on each box, which offers: '
                           + ', '.join(f'{ip} {format_offers(offers.get(ip)) or "no rank"}' for ip in ips))
    order = min(fits, key=lambda o: [ip_key(a) for a in o])
    edges = []
    for r in range(n):
        a, b = order[r], order[(r + 1) % n]
        ps = sorted(pairs.get((a, b)) or (), key=lambda p: (root[p[0]], root[p[1]], ip_key(p[0]), ip_key(p[1])))
        if not ps:
            raise NoAssignment(f"mentat's ring puts {a} next to {b}, but no link between them joins two RoCE v2 "
                               'addresses (gr8.roots): is each ring port addressed and its RoCE device ACTIVE?')
        direct = ps[-1]
        if n == 3:      # TP=3 opens only the direct devices (fabric.cpp): the relay fields repeat them
            edges.append((direct, direct))
            continue
        relay = next((p for p in ps if p[0] != direct[0] and p[1] != direct[1]), None)
        if relay is None:
            ends = lambda ip: ', '.join(f'{x} (root {root[x]})' for x in sorted(roots.get(ip) or {}, key=ip_key))  # noqa: E731,E501
            raise Incomplete(f'{a} and {b} link on one port pair only ({direct[0]} - {direct[1]}): TP=4 relays '
                             'over a second pair on the same cable, from the other PCIe root at both ends. '
                             f'{a} has {ends(a) or "no address"}; {b} has {ends(b) or "no address"}. Is that pair '
                             'addressed, tagged rdma, and answering mentat\'s probes?')
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


def ring_text(order, ports, claim=''):
    """One line naming the ring plan and the claim it came from, so the handshake can check every box has the same."""
    return ' '.join(f'{ip}:{",".join(p)}' for ip, p in zip(order, ports)) + (f' claim={claim}' if claim else '')


def rails_of(snap, node_id):
    """The rdma addresses of a member, from the head's snapshot (its own row is the top level)."""
    if node_id == snap.get('node_id'):
        return rdma_addrs(snap.get('addr_tags') or {})
    peer = (snap.get('peers') or {}).get(node_id) or {}
    return rdma_addrs(peer.get('addr_tags') or {})


def member_rails(snap, node_ids, names=None):
    """{node_id: (rail 0, rail 1)}: one rdma address per PCIe root for each member. A member with more than two (a
    direct cable to another box beside its switch port) uses the pair on the fabric every member shares: the local
    box's two addresses whose mentat probes reach exactly one address of every other member, and the addresses they
    reach. None until the probes settle that. Raises RuntimeError for a member with fewer than two, named by names
    ({node_id: LAN address}) where given."""
    rails = {n: rails_of(snap, n) for n in node_ids}
    for n, r in rails.items():
        if len(r) < 2:
            raise RuntimeError(f'{(names or {}).get(n, n)} has {len(r)} rdma-tagged addresses '
                               f'({", ".join(r) or "none"}), want two. Tag both ConnectX roots in mentat: '
                               'MENTAT_ANNOUNCE_IFACES=en*f*np*=connectx+rdma,en*=lan (or pass --ips and --ips2)')
    if all(len(r) == 2 for r in rails.values()):
        return {n: tuple(r) for n, r in rails.items()}
    me = snap.get('node_id')
    others = [n for n in node_ids if n != me]
    if me not in rails or not others:
        return {n: tuple(r[:2]) for n, r in rails.items()}      # nobody to probe: the lowest two
    peers = snap.get('peers') or {}
    reach = {}
    for a in rails[me]:
        hit = {}
        for n in others:
            probes = ((peers.get(n) or {}).get('probes') or {}).get(a) or {}
            ok = [p for p in rails[n] if (probes.get(p) or {}).get('ok')]
            if len(ok) != 1:
                break
            hit[n] = ok[0]
        else:
            reach[a] = hit
    if len(reach) != 2:
        return None
    mine = sorted(reach, key=ipaddress.ip_address)
    return {me: tuple(mine), **{n: tuple(reach[a][n] for a in mine) for n in others}}


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


class ClaimLink:
    """A client link (session: false) that sends this box's claims and holds them while it is open (mentat 0.24.0+:
    a client with no session lets go of its claims when its last link closes). The local daemon relays it to the
    head, the one daemon that solves claims. Kept open from the claim through the handshake, so a box that claims
    later gets the same view; closed after it, so the next boot solves afresh."""

    def __init__(self, daemon, group, client_id=None):
        self.daemon, self.group = daemon, group
        self.client_id = client_id or f'gr8-ring-{socket.gethostname()}-{os.getpid()}'
        self.sock, self.req, self.tag = None, 0, ''

    def claim(self, name, shape):
        """The daemon's reply to a claim: claim_ok, or err with code and error. OSError when the link fails (its
        claims go with it)."""
        try:
            if self.sock is None:
                s = socket.create_connection((self.daemon.host, self.daemon.port), timeout=30)
                try:
                    s.sendall(frame({'t': 'hello', 'req': 1, 'proto': PROTO, 'client_id': self.client_id,
                                     'group': self.group, 'session': False, 'kind': 'cli'}))
                    h = read_frame(s)
                except BaseException:
                    s.close()
                    raise
                if h.get('t') != 'hello_ok':
                    s.close()
                    return h
                self.sock, self.req = s, 1
            self.req += 1
            self.sock.sendall(frame({'t': 'claim', 'req': self.req, 'name': name, 'shape': shape}))
            return read_frame(self.sock)
        except (OSError, ValueError):
            self.close()
            raise

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None


# the codes a claim gets while the cables or the head are still settling
CLAIM_WAIT = ('unsolvable_claim', 'not_head', 'no_head')


def why_no_ring(daemon, group, tp, ids, roots, why):
    """The probed links between the members, from a claim that asks for any layout (it lists every link, ring or
    not), worded as ring_why. The claim's link closes at once, so it holds nothing."""
    link = ClaimLink(daemon, group, f'gr8-ring-why-{socket.gethostname()}-{os.getpid()}')
    try:
        h = link.claim(f'{group}-ring-why', {'sets': [{'name': 'ring', 'bundles': int(tp), 'link': 'rdma',
                                                        'layout': 'any'}], 'driver': 'ignore'})
    except OSError:
        h = {}
    finally:
        link.close()
    lan_of = {v: k for k, v in ids.items()}
    pairs = view_pairs(h.get('view') or {}, lan_of) if h.get('t') == 'claim_ok' else {}
    pairs = {k: v for k, v in pairs.items() if all(p[0] in roots.get(k[0], {}) for p in v)}
    return ring_why(pairs, roots, tp, why)


def claim_ring(daemon, link, group, tp, ms, snap, head, deadline, hold_s, log, sleep, clock, settle, retry,
               attempts):
    """--shape ring: claim the ring on link, and plan it (ring_plan). ms: the members. Every box tries the same names
    in the same order: a TP=4 view without a relay pair is the same on every box, so each moves to the next name,
    whose claim solves on probes a round newer. Returns [(lan, node_id, ports)] in rank order, and sets link.tag to
    the claim it came from."""
    ids = {ip: n for ip, n, _ in ms}
    offers = {ip: parse_offers(o) for ip, _, o in ms}
    roots = {ip: parse_roots(t) for ip, t in member_meta(snap, group, ROOTS_META).items() if ip in ids}
    shape, attempt, unsolved, said = ring_shape(tp), 0, None, None
    while True:
        name = ring_name(group, tp, attempt)
        try:
            h = link.claim(name, shape)
        except OSError as e:
            h = {'t': 'err', 'code': 'link', 'error': f'the claim link failed: {e}'}
        if h.get('t') == 'claim_ok':
            view = h.get('view') or {}
            if 'links' not in view:
                raise NoAssignment("mentat's claim view has no links: --shape ring needs mentatd 0.24.0 or later on "
                                   'every box')
            try:
                order, ports = ring_plan(view, ids, roots, tp, offers, head)
            except Incomplete as e:
                attempt += 1
                if attempt >= attempts:
                    raise NoAssignment(f'{e} ({attempts} claims, {retry:.0f}s apart)')
                if clock() >= deadline:
                    raise RuntimeError(f'the ring claim of group {group} timed out: {e}')
                log(f'{e}; claiming again as {ring_name(group, tp, attempt)} in {retry:.0f}s')
                sleep(retry)
                continue
            except Stale as e:
                # let go of the old view, and claim again once its last holder has gone
                link.close()
                if clock() >= deadline:
                    raise RuntimeError(f'the ring claim of group {group} timed out: {e}')
                if said is None or clock() - said >= 60:
                    said = clock()
                    log(f'{e}: a box of an earlier boot still holds the claim {name}; waiting for it to let go')
                sleep(hold_s)
                continue
            link.tag = f'{name}@{h.get("head_node_id", "")}:{h.get("generation", "")}'
            return [(ip, ids[ip], p) for ip, p in zip(order, ports)]
        code, err = h.get('code') or h.get('t'), h.get('error') or ''
        if code not in CLAIM_WAIT and code != 'link':
            raise NoAssignment(f'mentat refused the ring claim {name}: {err} ({code})')
        now = clock()
        if code == 'unsolvable_claim':
            if unsolved is None:
                unsolved = now
                log(f'mentat: no ring yet ({err}); waiting up to {settle:.0f}s for more probes')
            elif now - unsolved >= settle:
                raise NoAssignment(why_no_ring(daemon, group, tp, ids, roots, err))
        elif said is None or now - said >= 60:
            said = now
            log(f'mentat: the ring claim waits: {err} ({code})')
        if now >= deadline:
            raise RuntimeError(f'the ring claim of group {group} timed out: {err} ({code})')
        sleep(hold_s)


def elect(daemon, group, tp, head=None, timeout=600, hold_s=5.0, log=print, sleep=time.sleep, clock=time.monotonic,
          ring=None, publish=None, ring_settle=RING_SETTLE_S, ring_retry=RING_RETRY_S, ring_attempts=RING_ATTEMPTS):
    """Wait until the group has tp live members, with their offers, the same on two reads hold_s apart; returns
    (ranks, snapshot), where ranks is [(lan, node_id, (rail0, rail1))] in rank order. Raises NoAssignment when the
    offers fit no assignment, and RuntimeError on a timeout or a bad membership.

    ring: for --shape ring, a ClaimLink. Every member's gr8.roots then counts toward the two equal reads, the ranks
    follow the ring mentat's claim solves (claim_ring), and the third field of each rank is its (relay-out,
    direct-next, relay-in, direct-prev) addresses. publish: a callable run before each read, which republishes this
    box's gr8.roots. A ring the cables do not form gets ring_settle seconds of new probes before the NoAssignment."""
    t0, last, said = clock(), None, -60
    while True:
        if publish:
            publish()
        try:
            snap = daemon.group_status(group)
            ms = members(snap, group)
            roots = member_meta(snap, group, ROOTS_META) if ring else {}
        except (OSError, RuntimeError, ValueError) as e:
            snap, ms, roots = None, [], {}
            err = str(e)
        else:
            err = ''
        lan_head = None
        if head:
            try:
                lan_head = head_address(head)
            except OSError as e:     # a pod's name resolves once the pod has an address: wait for it like a member
                err = str(e)
        if len(ms) > tp:
            raise RuntimeError(f'group {group} has {len(ms)} members for --tp {tp} '
                               f'({", ".join(m[0] for m in ms)}); stop one, or use --group')
        waited = clock() - t0
        if timeout and waited >= timeout:
            raise RuntimeError(f'waited {waited:.0f}s for {tp} members in group {group}, have {len(ms)} '
                               f'({", ".join(m[0] for m in ms) or "none"})' + (f': {err}' if err else ''))
        if len(ms) == tp and (lan_head or not head) and all(o is not None for _, _, o in ms) \
                and all(v is not None for v in roots.values()):
            key = (ms, sorted(roots.items()))
            if key == last and ring:
                deadline = t0 + timeout if timeout else float('inf')
                return claim_ring(daemon, ring, group, tp, ms, snap, lan_head, deadline, hold_s, log, sleep, clock,
                                  ring_settle, ring_retry, ring_attempts), snap
            if key == last:
                ips = assign_ranks([ip for ip, _, _ in ms], {ip: parse_offers(o) for ip, _, o in ms}, tp, lan_head)
                ids = {ip: n for ip, n, _ in ms}
                rails = member_rails(snap, [ids[ip] for ip in ips], {n: ip for ip, n in ids.items()})
                if rails is not None:
                    return [(ip, ids[ip], rails[ids[ip]]) for ip in ips], snap
                if waited - said >= 60:
                    log('waiting for mentat to probe which rdma addresses every member shares')
                    said = waited
            last = key
            sleep(hold_s)
            continue
        last = None
        if waited - said >= 60:
            log(f'waiting for {tp} members in group {group}, have {len(ms)} ({", ".join(m[0] for m in ms) or "none"})'
                + (f': {err}' if err else ''))
            said = waited
        sleep(hold_s)
