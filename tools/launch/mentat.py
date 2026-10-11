"""The mentat side of `--mentat localhost`: a native agent link to the local daemon, and the group's membership.

Modelled on mentat's python/ray/register.py (mentat's own PROTOCOL.md: "Agent link", "Client link", "Snapshot"). Each box registers
one agent in the group (`gr8-glm53` by default) with one UMA GPU and its gr8 state in `meta`. The members are the
group's live agents that report a GPU, one per node_ip; rank r is the r-th address in numeric order, so every box
computes the same ranks from the same snapshot: each box publishes in `gr8.offers` the ranks its model set can serve
(`any`, or a list), the boxes that offer particular ranks get them first, and the rest take the free ranks in
address order. The head's API is announced under a second agent id, so the member row never changes.
"""
import ipaddress
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


def members(snap, group):
    """[(node_ip, node_id, offers)] of the group's live GPU agents, one per node, sorted by address. offers is the
    agent's gr8.offers meta, or None before it has published one."""
    agents = ((snap.get('groups') or {}).get(group) or {}).get('agents') or {}
    seen = {}
    for a in agents.values():
        if a.get('alive') and (a.get('machine') or {}).get('gpus') and a.get('node_ip'):
            seen.setdefault(a['node_ip'], (a.get('node_id', ''), (a.get('meta') or {}).get('gr8.offers')))
    return sorted(((ip, n, o) for ip, (n, o) in seen.items()), key=lambda x: ipaddress.ip_address(x[0]))


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


def elect(daemon, group, tp, head=None, timeout=600, hold_s=5.0, log=print, sleep=time.sleep, clock=time.monotonic):
    """Wait until the group has tp live members, with their offers, the same on two reads hold_s apart; returns
    (ranks, snapshot), where ranks is [(lan, node_id, (rail0, rail1))] in rank order. Raises NoAssignment when the
    offers fit no assignment, and RuntimeError on a timeout or a bad membership."""
    t0, last, said = clock(), None, -60
    while True:
        try:
            snap = daemon.group_status(group)
            ms = members(snap, group)
        except (OSError, RuntimeError, ValueError) as e:
            snap, ms = None, []
            err = str(e)
        else:
            err = ''
        if len(ms) > tp:
            raise RuntimeError(f'group {group} has {len(ms)} members for --tp {tp} '
                               f'({", ".join(m[0] for m in ms)}); stop one, or use --group')
        if len(ms) == tp and all(o is not None for _, _, o in ms):
            if ms == last:
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
            last = ms
            sleep(hold_s)
            continue
        last = None
        waited = clock() - t0
        if timeout and waited >= timeout:
            raise RuntimeError(f'waited {waited:.0f}s for {tp} members in group {group}, have {len(ms)} '
                               f'({", ".join(m[0] for m in ms) or "none"}){": " + err if err else ""}')
        if waited - said >= 60:
            log(f'waiting for {tp} members in group {group}, have {len(ms)} ({", ".join(m[0] for m in ms) or "none"})'
                + (f': {err}' if err else ''))
            said = waited
        sleep(hold_s)
