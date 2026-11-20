"""GPU clocks and heat on every box of the cluster.

Stages run at the stage clocks: no lock, or a box's own cap (--cap HOST=MHZ). Between stages every box is locked at
the cool clock, and the next stage waits until the server is idle and every box's GPU temperature has stopped
falling. Each box is sampled once a second the whole time (temperature, SM clock, power, utilization, throttle
reasons), so every stage reports the clocks it actually ran at.

A host is an ssh destination (the head's own address works too), or "local" for the box this runs on. Clock control
needs passwordless sudo for nvidia-smi on every box. Without it the clocks are left alone and the report says so.

A timed stage runs with no limit, so a guard stands in for the stopped limiter: a box that reaches guard_temp (90 C)
is locked at the cool clock for the rest of the stage, and the stage's numbers are marked as affected.

A service that sets clocks itself (sparkos-thermal moves a cap every 15 s) would undo these locks, so it is stopped
for the run and started again at the end. A systemd timer on each box starts it again after `restore_after` seconds
too, in case this process dies first. A box without the service, or without sudo, is left as it is."""
import collections
import os
import subprocess
import threading
import time

QUERY = 'temperature.gpu,clocks.sm,power.draw,utilization.gpu,clocks_throttle_reasons.active'
# clocks_throttle_reasons bits worth reporting (idle and the application clock setting are not throttles)
THROTTLE = {0x4: 'sw power cap', 0x8: 'hw slowdown', 0x20: 'sw thermal', 0x40: 'hw thermal', 0x80: 'power brake'}


def _num(s):
    try:
        return float(s)
    except ValueError:
        return None


class Host:
    def __init__(self, name):
        self.name = name
        self.samples = collections.deque(maxlen=4 * 3600)   # (time, temp, sm, power, util, throttle)
        self.proc = None

    def argv(self, cmd):
        if self.name == 'local':
            return ['bash', '-c', cmd]
        return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', self.name, cmd]

    def run(self, cmd, timeout=60):
        return subprocess.run(self.argv(cmd), capture_output=True, text=True, timeout=timeout)

    def between(self, t0, t1):
        return [s for s in list(self.samples) if t0 <= s[0] <= t1]


class Thermal:
    def __init__(self, hosts, caps=None, cool_mhz=1800, cool_temp=55, guard_temp=90, rate=1.0, window=60, min_wait=30,
                 max_wait=300, idle_util=10, log=print):
        self.hosts = [Host(h) for h in hosts]
        self.caps = dict(caps or {})
        self.cool_mhz, self.cool_temp, self.guard_temp, self.rate, self.window = cool_mhz, cool_temp, guard_temp, rate, window
        self.min_wait, self.max_wait, self.idle_util = min_wait, max_wait, idle_util
        self.log = log
        self.clock_control = True
        self.clock_error = None
        self.paused = []   # (host, service, restore timer unit)
        self.guarding = False
        self.trips = []    # guard trips: {host, temp, time}
        self._tripped = set()
        self._lock = threading.Lock()

    def start(self):
        for h in self.hosts:
            cmd = f'exec nvidia-smi --query-gpu={QUERY} --format=csv,noheader,nounits -lms 1000'
            h.proc = subprocess.Popen(h.argv(cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=self._read, args=(h,), daemon=True).start()

    def _read(self, h):
        for line in h.proc.stdout:
            f = [x.strip() for x in line.split(',')]
            if len(f) != 5:
                continue
            thr = int(f[4], 16) if f[4].startswith('0x') else 0
            h.samples.append((time.time(), _num(f[0]), _num(f[1]), _num(f[2]), _num(f[3]), thr))
            temp = h.samples[-1][1]
            if self.guarding and temp is not None and temp >= self.guard_temp:
                with self._lock:
                    if h.name in self._tripped:
                        continue
                    self._tripped.add(h.name)
                self.trips.append({'host': h.name, 'temp': temp, 'time': time.time()})
                self.log(f'{h.name} at {temp:.0f} C: held at {self.cool_mhz} MHz for the rest of the stage')
                threading.Thread(target=self._clock, args=(h, self.cool_mhz), daemon=True).start()

    def stop(self):
        self.guarding = False
        for h in self.hosts:
            self._clock(h, self.caps.get(h.name))
        self.resume_services()
        for h in self.hosts:
            if h.proc:
                h.proc.terminate()

    def pause_services(self, names, restore_after):
        """Stop each named service where it runs. -> {host: [stopped services]}; missing ones are skipped."""
        out = {}
        for h in self.hosts:
            for name in names:
                try:
                    if h.run(f'systemctl is-active --quiet {name}', timeout=20).returncode != 0:
                        continue
                    unit = f'gr8bench-restore-{name}-{os.getpid()}'
                    p = h.run(f'sudo -n systemd-run --quiet --on-active={restore_after} --unit={unit} '
                              f'systemctl start {name} && sudo -n systemctl stop {name}', timeout=60)
                except subprocess.TimeoutExpired:
                    p = None
                if p is None or p.returncode:
                    self.log(f'{h.name}: could not stop {name} ({p.stderr.strip() if p else "timed out"}): left running')
                    continue
                self.paused.append((h, name, unit))
                out.setdefault(h.name, []).append(name)
        return out

    def resume_services(self):
        for h, name, unit in self.paused:
            try:
                p = h.run(f'sudo -n systemctl start {name}; sudo -n systemctl stop {unit}.timer 2>/dev/null; true',
                          timeout=60)
                if h.run(f'systemctl is-active --quiet {name}', timeout=20).returncode:
                    self.log(f'{h.name}: {name} did not start again ({p.stderr.strip()}); its timer will retry')
            except subprocess.TimeoutExpired:
                self.log(f'{h.name}: starting {name} timed out; its timer will retry')
        self.paused = []

    def _clock(self, h, mhz):
        if not self.clock_control:
            return
        cmd = f'sudo -n nvidia-smi --lock-gpu-clocks=0,{mhz}' if mhz else 'sudo -n nvidia-smi --reset-gpu-clocks'
        try:
            p = h.run(cmd + ' >/dev/null', timeout=30)
            err = p.returncode and (p.stderr.strip() or f'exit {p.returncode}')
        except subprocess.TimeoutExpired:
            err = 'timed out'
        if err:
            self.clock_control = False
            self.clock_error = f'{h.name}: {err}'
            self.log(f'clock control off ({self.clock_error}): stages and cooldowns run at whatever clocks the boxes have')

    def stage_clocks(self):
        """Uncapped, except the boxes with a cap of their own, with the guard on."""
        with self._lock:
            self._tripped = set()
        for h in self.hosts:
            self._clock(h, self.caps.get(h.name))
        self.guarding = True

    def cool_clocks(self):
        self.guarding = False
        for h in self.hosts:
            cap = self.caps.get(h.name)
            self._clock(h, min(self.cool_mhz, cap) if cap else self.cool_mhz)

    def _slope(self, h, now):
        """GPU temperature trend over the last window (degrees C per minute), or None until the window is covered."""
        s = [x for x in h.between(now - self.window, now) if x[1] is not None]
        if len(s) < 10 or s[-1][0] - s[0][0] < 0.8 * self.window:
            return None
        tm = sum(x[0] for x in s) / len(s)
        ym = sum(x[1] for x in s) / len(s)
        var = sum((x[0] - tm) ** 2 for x in s)
        return 60 * sum((x[0] - tm) * (x[1] - ym) for x in s) / var if var else 0.0

    def settle(self, busy):
        """Lock the cool clock, then wait until busy() is false, every box's GPU is idle, and every box is either at or
        under cool_temp or has stopped cooling (trend above -rate C/min over the window). -> what the wait found."""
        self.cool_clocks()
        t0 = time.time()
        idle_since = None
        timed_out = False
        slopes = {}
        while True:
            time.sleep(2)
            now = time.time()
            if now - t0 > self.max_wait:
                timed_out = True
                self.log(f'not settled after {self.max_wait} s (server busy or temperatures still falling); going on')
                break
            if busy():
                idle_since = None
                continue
            idle_since = idle_since or now
            utils = {h.name: max((x[4] or 0) for x in h.between(now - 10, now)) for h in self.hosts if h.samples}
            if not all(u < self.idle_util for u in utils.values()):
                continue
            temps = {h.name: h.samples[-1][1] for h in self.hosts if h.samples}
            if all(t is not None and t <= self.cool_temp for t in temps.values()):
                break
            # a trend counts only over a window that starts after the last request left
            slopes = {h.name: self._slope(h, now) for h in self.hosts if h.samples}
            flat = all(v is not None and v > -self.rate for v in slopes.values())
            if flat and now - t0 >= self.min_wait and now - idle_since >= self.window:
                break
        temps = {h.name: h.samples[-1][1] for h in self.hosts if h.samples}
        trace = {h.name: [(round(x[0] - t0), x[1], x[4]) for x in h.between(t0, time.time())][::5] for h in self.hosts}
        return {'seconds': round(time.time() - t0, 1), 'timed_out': timed_out, 'temps': temps,
                'slopes': {k: round(v, 2) for k, v in slopes.items() if v is not None},
                'trace': trace}   # every 5 s: seconds into the wait, temperature, utilization

    def window_stats(self, t0, t1, under_mhz=2000):
        """Per box between t0 and t1: temperatures, SM clock, power and the throttle reasons seen."""
        out = {}
        for h in self.hosts:
            s = [x for x in h.between(t0, t1) if x[1] is not None and x[2] is not None]
            if not s:
                continue
            sm = [x[2] for x in s]
            busy = [x for x in s if (x[4] or 0) >= self.idle_util] or s
            bits = 0
            for x in s:
                bits |= x[5]
            out[h.name] = {
                'samples': len(s), 'temp_start': s[0][1], 'temp_max': max(x[1] for x in s),
                'sm_mean': round(sum(x[2] for x in busy) / len(busy)), 'sm_min': min(x[2] for x in busy),
                'sm_max': max(sm), 'under_frac': round(sum(x[2] < under_mhz for x in busy) / len(busy), 3),
                'power_mean': round(sum(x[3] or 0 for x in busy) / len(busy), 1),
                'throttle': [n for b, n in THROTTLE.items() if bits & b], 'cap_mhz': self.caps.get(h.name)}
        return out

    def now(self):
        return {h.name: {'temp': h.samples[-1][1], 'sm': h.samples[-1][2]} for h in self.hosts if h.samples}
