"""This box's addresses: which peer it is, and its two fabric rails.

A rail address is an IPv4 address with a RoCE v2 GID on an ACTIVE RDMA port, read from /sys/class/infiniband the way
engine/comms/fabric.cpp finds the device for an address. Rail 0 is the lowest address, rail 1 the next, so every box
lists its PCIe roots in the same order when each root has its own subnet.
"""
import ipaddress
import os
import socket


def is_local(addr):
    """Whether addr is one of this box's addresses: binding to it works only then."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((addr, 0))
        return True
    except OSError:
        return False


def own_index(members, node_ip=None):
    """The index of this box in the member list. node_ip, when given, says which address is this box's."""
    if node_ip:
        if node_ip not in members:
            raise RuntimeError(f'--node-ip {node_ip} is not in the peer list {",".join(members)}')
        return members.index(node_ip)
    hits = [i for i, a in enumerate(members) if is_local(a)]
    if not hits:
        raise RuntimeError(f'none of {",".join(members)} is an address of this box')
    if len(hits) > 1:
        raise RuntimeError(f'{",".join(members[i] for i in hits)} are all addresses of this box; '
                           'pass --node-ip to say which one it is')
    return hits[0]


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def gid_ipv4(gid):
    """'0000:0000:0000:0000:0000:ffff:0a01:0001' -> '10.1.0.1', or None for a GID that maps no IPv4 address."""
    parts = gid.lower().split(':')
    if len(parts) != 8 or parts[:6] != ['0000'] * 5 + ['ffff']:
        return None
    try:
        b = bytes.fromhex(parts[6] + parts[7])
    except ValueError:
        return None
    return socket.inet_ntoa(b)


def roce_addresses(root='/sys/class/infiniband'):
    """[(address, device, port, state)] for every RoCE v2 GID that maps an IPv4 address."""
    out = []
    for dev in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        ports = os.path.join(root, dev, 'ports')
        for port in sorted(os.listdir(ports)) if os.path.isdir(ports) else []:
            base = os.path.join(ports, port)
            state = _read(os.path.join(base, 'state')) or ''
            gids = os.path.join(base, 'gids')
            for gi in sorted(os.listdir(gids), key=lambda x: int(x) if x.isdigit() else 0) if os.path.isdir(gids) else []:
                addr = gid_ipv4(_read(os.path.join(gids, gi)) or '')
                typ = _read(os.path.join(base, 'gid_attrs', 'types', gi)) or ''
                if addr and addr != '0.0.0.0' and 'v2' in typ.lower():
                    out.append((addr, dev, port, state))
    return out


def rails(root='/sys/class/infiniband'):
    """(rail0, rail1) of this box. Raises RuntimeError naming what is missing."""
    found = roce_addresses(root)
    active = sorted({a for a, _, _, st in found if 'ACTIVE' in st}, key=ipaddress.ip_address)
    if len(active) >= 2:
        if len(active) > 2:
            raise RuntimeError(f'more than two RoCE v2 addresses ({", ".join(active)}): pass --ips and --ips2')
        return active[0], active[1]
    down = sorted({f'{a} ({d} port {p}: {st or "no state"})' for a, d, p, st in found if 'ACTIVE' not in st})
    if not found:
        raise RuntimeError(f'no RoCE v2 address under {root}: is the ConnectX cabled and addressed? '
                           'Or pass --ips and --ips2')
    raise RuntimeError(f'{len(active)} active rail address{"" if len(active) == 1 else "es"} '
                       f'({", ".join(active) or "none"}), want two' + (f'; not active: {", ".join(down)}' if down else '')
                       + '. An uncabled ConnectX port leaves the PCI bus, so a missing one is often a cable')


def same_subnet(a, b, prefix=24):
    return ipaddress.ip_network(f'{a}/{prefix}', strict=False) == ipaddress.ip_network(f'{b}/{prefix}', strict=False)
