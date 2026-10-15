"""The checkpoint's own tokenizer.json via HF `tokenizers`, plus incremental detokenization."""
import asyncio
import functools
import json
import os

from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

# encode() on long text runs in a thread so a 200K-token prompt does not stall the event loop
_THREAD_CHARS = 20000


class GlmTokenizer:
    def __init__(self, model_dir: str):
        self.tok = Tokenizer.from_file(os.path.join(model_dir, 'tokenizer.json'))
        with open(os.path.join(model_dir, 'tokenizer_config.json')) as f:
            cfg = json.load(f)
        gen_path = os.path.join(model_dir, 'generation_config.json')
        gen = json.load(open(gen_path)) if os.path.exists(gen_path) else {}
        self.eos_token = cfg.get('eos_token', '<|endoftext|>')
        self.eos_token_id = self.tok.token_to_id(self.eos_token)
        eos = gen.get('eos_token_id', self.eos_token_id)
        self.eos_ids = list(eos) if isinstance(eos, list) else [eos]
        self.pad_token = cfg.get('pad_token')
        # sampling defaults the old vLLM stacks took from generation_config.json (--generation-config auto)
        self.gen_defaults = {k: gen[k] for k in ('temperature', 'top_p', 'top_k', 'min_p') if k in gen}
        self.special_ids = {i for i, t in self.tok.get_added_tokens_decoder().items() if t.special}
        self.vocab_size = self.tok.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self.tok.encode(text, add_special_tokens=add_special_tokens).ids

    async def aencode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if len(text) < _THREAD_CHARS:
            return self.encode(text, add_special_tokens)
        return await asyncio.to_thread(self.encode, text, add_special_tokens)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    @functools.lru_cache(maxsize=1 << 16)
    def token_str(self, tid: int) -> str:
        """Text of one token as vLLM shows it in logprob entries (decoded, special tokens kept)."""
        return self.tok.decode([tid], skip_special_tokens=False)

    def id_to_token(self, tid: int) -> str | None:
        return self.tok.id_to_token(tid)

    def detokenizer(self, skip_special_tokens: bool) -> 'IncrementalDetok':
        return IncrementalDetok(self, skip_special_tokens)


class IncrementalDetok:
    """Text deltas for a growing id stream. Holds back incomplete UTF-8 byte sequences (DecodeStream)."""

    def __init__(self, t: GlmTokenizer, skip_special_tokens: bool):
        self.t = t
        self.skip = skip_special_tokens
        self.ds = DecodeStream(skip_special_tokens=skip_special_tokens)

    def step(self, tid: int) -> str:
        return self.ds.step(self.t.tok, tid) or ''
