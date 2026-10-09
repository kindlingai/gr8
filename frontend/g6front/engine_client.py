"""One multiplexed Unix-socket connection to the engine (PROTOCOL.md). Reconnects on loss."""
import asyncio
import itertools
import logging

from . import wire

log = logging.getLogger('g6front.engine')


class EngineError(Exception):
    pass


class EngineClient:
    def __init__(self, path: str):
        self.path = path
        self.hello: dict | None = None
        self._reader = None
        self._writer = None
        self._queues: dict[str, asyncio.Queue] = {}
        self._metrics_waiters: dict[int, asyncio.Future] = {}
        self._seq = itertools.count(1)
        self._connected = asyncio.Event()
        self._task = None

    # -- lifecycle ------------------------------------------------------------------------------------------------
    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task:
            self._task.cancel()
        if self._writer:
            self._writer.close()

    @property
    def ready(self) -> bool:
        return self._connected.is_set()

    async def wait_ready(self, timeout: float | None = None):
        await asyncio.wait_for(self._connected.wait(), timeout)

    async def _run(self):
        backoff = 0.2
        while True:
            try:
                reader, writer = await asyncio.open_unix_connection(self.path, limit=wire.MAX_FRAME)
                hello = await wire.read_frame(reader)
                if not hello or hello.get('type') != 'hello' or hello.get('version') != 1:
                    raise EngineError(f'bad hello: {hello!r}')
                self.hello, self._reader, self._writer = hello, reader, writer
                self._connected.set()
                log.info('engine connected: %s', hello)
                backoff = 0.2
                await self._read_loop(reader)
                log.warning('engine closed the connection')
            except asyncio.CancelledError:
                raise
            except (OSError, EngineError, wire.FrameError) as e:
                if self._connected.is_set() or backoff >= 5:
                    log.warning('engine connection: %s', e)
            finally:
                self._connected.clear()
                if self._writer:
                    self._writer.close()
                self._writer = None
                self._fail_all('engine connection lost')
            await asyncio.sleep(backoff)
            # until the first connection, the shim is starting (the launcher starts the front end early): retry often
            backoff = min(backoff * 2, 5.0) if self.hello else 0.25

    async def _read_loop(self, reader):
        while True:
            f = await wire.read_frame(reader)
            if f is None:
                return
            t = f['type']
            if t == 'metrics':
                fut = self._metrics_waiters.pop(f.get('seq'), None)
                if fut and not fut.done():
                    fut.set_result(f)
                continue
            q = self._queues.get(f.get('id'))
            if q is not None:  # frames for cancelled/unknown ids are dropped
                q.put_nowait(f)
                if t == 'finish':
                    self._queues.pop(f['id'], None)

    def _fail_all(self, msg):
        for rid, q in list(self._queues.items()):
            q.put_nowait({'type': 'finish', 'id': rid, 'reason': 'error', 'message': msg,
                          'prompt_tokens': 0, 'cached_tokens': 0, 'completion_tokens': 0})
        self._queues.clear()
        for fut in self._metrics_waiters.values():
            if not fut.done():
                fut.set_exception(EngineError(msg))
        self._metrics_waiters.clear()

    def _send(self, obj):
        if self._writer is None:
            raise EngineError('engine not connected')
        wire.write_frame(self._writer, obj)

    # -- API ------------------------------------------------------------------------------------------------------
    async def generate(self, req: dict):
        """Async iterator over this request's engine frames (tokens / prompt_logprobs / finish).
        Closing the iterator early (client gone, stop string) sends `cancel`."""
        rid = req['id']
        if rid in self._queues:
            raise EngineError(f'duplicate request id {rid}')
        q: asyncio.Queue = asyncio.Queue()
        self._queues[rid] = q
        finished = False
        try:
            self._send(dict(req, type='generate'))
            while True:
                f = await q.get()
                if f['type'] == 'finish':
                    finished = True
                yield f
                if finished:
                    return
        finally:
            if not finished:
                self._queues.pop(rid, None)
                try:
                    self._send({'type': 'cancel', 'id': rid})
                except EngineError:
                    pass

    async def metrics(self, timeout: float = 5.0) -> dict:
        seq = next(self._seq)
        fut = asyncio.get_running_loop().create_future()
        self._metrics_waiters[seq] = fut
        try:
            self._send({'type': 'metrics', 'seq': seq})
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._metrics_waiters.pop(seq, None)
