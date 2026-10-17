"""--shape hybrid: the per-pair route table, computed on the head from every rank's declared fabric ports.

Each rank declares its fabric addresses with prefix (FAB_PORTS, a box option). A subnet is a rail. Two ranks are
adjacent on a rail when both declare an address in it, unless FAB_DENY names the pair. Nothing is discovered: an
address a box holds but does not declare (a stray alias on a port) never makes a pair adjacent.

A subnet that more than two ranks declare is a switch rail; one that exactly two declare is a cable rail. A pair is
adjacent over switch rails or over cable rails, never both: a pair that shares both kinds is refused, so a switch pair
never goes over a cable and a cable pair never goes over the switch.

Rule A, for every unordered pair a < b:
  - direct, when the pair is adjacent: one rail.
  - else relay via v, the lowest rank adjacent to both (neither hop denied): two directed routes a->b and b->a, both
    through v. At most one relay hop.
  - else the boot is refused, naming the pair.

Rails:
  - Canonical switch-rail order: the order in which the ranks declare their switch subnets in FAB_PORTS. Every rank
    that declares two or more switch subnets must list them in the same relative order, or the table is refused. (So
    an operator who wants rail 0 to be the subnet a mesh env lists first in IPS lists that subnet first in FAB_PORTS.)
  - A direct switch pair a, b takes rail (a + b) % n of its common switch rails in canonical order (n = 2 on a
    two-rail switch): exactly the mesh layout.
  - A direct cable pair and every relay hop take the least-loaded common rail of its kind, by directed flow count:
    the flows the sending end already sends on the rail plus the flows the far end already receives on it, then the
    larger of the two (a direct pair counts once each way); ties go to the lower rail in canonical order (switch) or address order (cable). Pairs
    are placed in a fixed order: switch pairs, cable pairs, then relay routes by (src, dst).

The table is text, one line each (render), and its sha256 goes in the handshake verdict and the option digest:
  P <rank> <ip/prefix> [<ip/prefix> ...]                          declared ports, in declared order
  D <a> <b> <ip_a> <ip_b>                                         direct pair a<b on this rail
  R <src> <dst> <via> <ip_src> <ip_via_in> <ip_via_out> <ip_dst>  directed relayed route src->dst
"""
import hashlib
import ipaddress
import re
import sys


class RouteError(Exception):
    pass


def parse_port(text):
    """'10.0.0.1/24' -> IPv4Interface. Raises RouteError for anything else (no prefix, IPv6, a network address)."""
    if not re.fullmatch(r'\d{1,3}(\.\d{1,3}){3}/\d{1,2}', text or ''):
        raise RouteError(f'{text!r}: want an IPv4 address with its prefix, such as 192.0.2.1/24')
    try:
        i = ipaddress.IPv4Interface(text)
    except ValueError as e:
        raise RouteError(f'{text!r}: {e}')
    if not 8 <= i.network.prefixlen <= 30:
        raise RouteError(f'{text!r}: want a prefix from /8 to /30')
    if i.ip in (i.network.network_address, i.network.broadcast_address):
        raise RouteError(f'{text!r}: the network or broadcast address is not a port address')
    return i


def parse_ports(text):
    """A FAB_PORTS value: 1-4 comma-separated ip/prefix entries, no address or subnet twice."""
    items = [x for x in re.split(r'[,\s]+', text or '') if x]
    if not 1 <= len(items) <= 4:
        raise RouteError(f'FAB_PORTS {text!r}: want 1 to 4 ip/prefix entries')
    out = [parse_port(x) for x in items]
    nets = [i.network for i in out]
    for k, a in enumerate(nets):
        for b in nets[k + 1:]:
            if a.overlaps(b):
                raise RouteError(f'FAB_PORTS {text!r}: {a} and {b} overlap: one address per rail')
    return out


def parse_deny(text, nr):
    """FAB_DENY 'a-b,c-d' -> {(min, max)}. Ranks in 0..nr-1, distinct."""
    out = set()
    for item in [x for x in re.split(r'[,\s]+', text or '') if x]:
        m = re.fullmatch(r'(\d+)-(\d+)', item)
        if not m:
            raise RouteError(f'FAB_DENY {item!r}: want a-b (two ranks)')
        a, b = int(m.group(1)), int(m.group(2))
        if a == b or not (0 <= a < nr and 0 <= b < nr):
            raise RouteError(f'FAB_DENY {item!r}: want two different ranks in 0..{nr - 1}')
        out.add((min(a, b), max(a, b)))
    return out


class Table:
    """The routes for one cluster. direct: {(a, b): (ip_a, ip_b)} with a < b. relay: {(src, dst): (via, ip_src, ip_via_in,
    ip_via_out, ip_dst)}. text and sha: the rendered file."""

    def __init__(self, ports, direct, relay, switch_order):
        self.ports, self.direct, self.relay, self.switch_order = ports, direct, relay, switch_order
        self.text = render(ports, direct, relay)
        self.sha = hashlib.sha256(self.text.encode()).hexdigest()

    def route(self, a, b):
        """('direct', ip_a, ip_b) or ('relay', via, ip_a, ip_via_in, ip_via_out, ip_b) for directed a->b."""
        if (min(a, b), max(a, b)) in self.direct:
            x, y = self.direct[(min(a, b), max(a, b))]
            return ('direct', x, y) if a < b else ('direct', y, x)
        return ('relay', *self.relay[(a, b)])


def render(ports, direct, relay):
    lines = [f'P {r} ' + ' '.join(str(i) for i in p) for r, p in enumerate(ports)]
    lines += [f'D {a} {b} {x} {y}' for (a, b), (x, y) in sorted(direct.items())]
    lines += [f'R {s} {d} ' + ' '.join(str(v) for v in r) for (s, d), r in sorted(relay.items())]
    return '\n'.join(lines) + '\n'


def compute(ports, deny=(), names=None):
    """ports: per rank, a FAB_PORTS string or a list of ip/prefix strings. deny: {(a, b)} unordered. names: per rank, a
    label for refusals (the LAN address). Returns Table; raises RouteError naming the pair or the rank."""
    nr = len(ports)
    name = (lambda r: f'rank {r} ({names[r]})') if names else (lambda r: f'rank {r}')
    decl = []
    for r, p in enumerate(ports):
        try:
            decl.append(parse_ports(p if isinstance(p, str) else ','.join(p)))
        except RouteError as e:
            raise RouteError(f'{name(r)}: {e}')
    deny = {(min(a, b), max(a, b)) for a, b in deny}
    seen = {}
    for r, p in enumerate(decl):
        for i in p:
            if i.ip in seen:
                raise RouteError(f'{name(seen[i.ip])} and {name(r)} both declare {i.ip}')
            seen[i.ip] = r
    # rail -> {rank: interface}; overlapping but unequal subnets on two ranks are a declaration mistake
    rails = {}
    for r, p in enumerate(decl):
        for i in p:
            for net in rails:
                if net != i.network and net.overlaps(i.network):
                    raise RouteError(f'{name(r)} declares {i} in {i.network}, which overlaps {net} declared by '
                                     f'{name(next(iter(rails[net])))}: one prefix per rail')
            rails.setdefault(i.network, {})[r] = i
    switch = {n for n, m in rails.items() if len(m) > 2}
    order = switch_order(decl, switch, name)
    rank_of = {n: k for k, n in enumerate(order)}

    def common(a, b):
        return [n for n in rails if a in rails[n] and b in rails[n]]

    adj = {}
    for a in range(nr):
        for b in range(a + 1, nr):
            c = common(a, b)
            if not c or (a, b) in deny:
                continue
            kinds = {n in switch for n in c}
            if len(kinds) > 1:
                raise RouteError(f'{name(a)} and {name(b)} share switch rails and cable rails '
                                 f'({", ".join(map(str, sorted(c)))}): a pair goes over one or the other')
            if True in kinds:
                adj[(a, b)] = ('switch', sorted(c, key=rank_of.get))
            else:
                adj[(a, b)] = ('cable', sorted(c))

    def adjacent(a, b):
        return (min(a, b), max(a, b)) in adj

    out, inn = {}, {}   # directed flows per (rank, rail): sent at the source end, received at the far end

    def count(x, y, n):
        out[(x, n)] = out.get((x, n), 0) + 1
        inn[(y, n)] = inn.get((y, n), 0) + 1

    def place(x, y, fixed=False):
        """The rail for directed hop x->y (adjacent). fixed: a direct pair, which carries both directions."""
        kind, rs = adj[(min(x, y), max(x, y))]
        if kind == 'switch' and fixed:
            n = rs[(x + y) % len(rs)]
        else:
            def cost(n):
                a, b = out.get((x, n), 0), inn.get((y, n), 0)
                if fixed:
                    a, b = max(a, out.get((y, n), 0)), max(b, inn.get((x, n), 0))
                return a + b, max(a, b), rs.index(n)
            n = min(rs, key=cost)
        count(x, y, n)
        if fixed:
            count(y, x, n)
        return str(rails[n][x].ip), str(rails[n][y].ip)

    direct, relay = {}, {}
    for want in ('switch', 'cable'):
        for (a, b), (kind, _) in sorted(adj.items()):
            if kind == want:
                direct[(a, b)] = place(a, b, fixed=True)
    for a in range(nr):
        for b in range(a + 1, nr):
            if adjacent(a, b):
                continue
            vias = [v for v in range(nr) if v not in (a, b) and adjacent(a, v) and adjacent(v, b)]
            if not vias:
                why = 'denied by FAB_DENY' if (a, b) in deny else 'no common rail'
                raise RouteError(f'no route between {name(a)} and {name(b)}: {why}, and no rank is adjacent to both '
                                 '(routes take at most one relay hop)')
            v = vias[0]
            for s, d in ((a, b), (b, a)):
                ip_s, ip_vi = place(s, v)
                ip_vo, ip_d = place(v, d)
                relay[(s, d)] = (v, ip_s, ip_vi, ip_vo, ip_d)
    return Table(decl, direct, relay, order)


def switch_order(decl, switch, name):
    """The canonical order of the switch rails: their declaration order, which every rank must share."""
    order = []
    for r, p in enumerate(decl):
        mine = [i.network for i in p if i.network in switch]
        prev = [n for n in order if n in mine]
        if prev != [n for n in mine if n in order]:
            raise RouteError(f'{name(r)} lists the switch rails {", ".join(map(str, mine))} in another order than the '
                             f'lower ranks ({", ".join(map(str, order))}): list them in one order on every box (it is '
                             'the rail order)')
        order += [n for n in mine if n not in order]
    return order


def main(argv):
    """Dry run: routes.py PORTS0 PORTS1 ... [--deny a-b,...]. Prints the table and its sha256; exit 2 on refusal."""
    deny, ports = '', []
    i = 0
    while i < len(argv):
        if argv[i] == '--deny':
            deny = argv[i + 1]
            i += 2
            continue
        ports.append(argv[i])
        i += 1
    try:
        t = compute(ports, parse_deny(deny, len(ports)))
    except (RouteError, IndexError) as e:
        print(f'routes: {e}', file=sys.stderr)
        return 2
    sys.stdout.write(t.text)
    print(f'routes sha {t.sha}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
