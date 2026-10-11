"""OpenAI/vLLM request fields -> PROTOCOL.md `generate` fields, with vLLM's defaults and validation."""
from .errors import ApiError

_INT64 = 1 << 63


def _num(body, key, lo=None, hi=None, kind=float):
    v = body.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (kind is int and not float(v).is_integer()):
        raise ApiError(f'`{key}` must be a {"integer" if kind is int else "number"}', param=key)
    v = kind(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ApiError(f'`{key}` out of range: {v}', param=key)
    return v


def build(body: dict, prompt_len: int, max_model_len: int, eos_ids: list[int], gen_defaults: dict,
          default_max_tokens: int | None, max_logprobs: int, chat: bool) -> dict:
    """Returns the sampling part of a `generate` frame plus front-end-only fields (stop strings etc.)."""
    if (body.get('n') or 1) != 1:
        raise ApiError('only n=1 is supported by the gr8 front end', param='n')
    if body.get('best_of') not in (None, 1):
        raise ApiError('best_of is not supported', param='best_of')
    max_tokens = _num(body, 'max_completion_tokens' if chat and body.get('max_completion_tokens') is not None
                      else 'max_tokens', lo=0, kind=int)
    room = max_model_len - prompt_len
    if room <= 0 or (max_tokens is not None and max_tokens > room):
        want = prompt_len + (max_tokens or 0)
        raise ApiError(f"This model's maximum context length is {max_model_len} tokens. However, you requested "
                       f'{want} tokens ({prompt_len} in the messages, {max_tokens or 0} in the completion). '
                       'Please reduce the length of the messages or completion.', param='max_tokens')
    if max_tokens is None:
        max_tokens = room if default_max_tokens is None else min(default_max_tokens, room)
    temperature = _num(body, 'temperature', lo=0, hi=2)
    top_p = _num(body, 'top_p', lo=0, hi=1)
    top_k = _num(body, 'top_k', lo=-1, kind=int)
    min_p = _num(body, 'min_p', lo=0, hi=1)
    # vLLM: request value, else generation_config.json value, else neutral
    temperature = temperature if temperature is not None else gen_defaults.get('temperature', 1.0)
    top_p = top_p if top_p is not None else gen_defaults.get('top_p', 1.0)
    top_k = top_k if top_k is not None else gen_defaults.get('top_k', 0)
    min_p = min_p if min_p is not None else gen_defaults.get('min_p', 0.0)
    if top_p == 0:
        raise ApiError('top_p must be in (0, 1]', param='top_p')
    seed = _num(body, 'seed', lo=-_INT64, hi=_INT64 - 1, kind=int)
    min_tokens = _num(body, 'min_tokens', lo=0, kind=int) or 0
    if min_tokens > max_tokens:
        raise ApiError('min_tokens must be <= max_tokens', param='min_tokens')
    if max_tokens == 0:
        raise ApiError('max_tokens must be at least 1', param='max_tokens')
    stop = body.get('stop')
    stop = [stop] if isinstance(stop, str) else list(stop or [])
    if not all(isinstance(s, str) for s in stop):
        raise ApiError('`stop` must be a string or a list of strings', param='stop')
    stop = [s for s in stop if s]
    stop_token_ids = body.get('stop_token_ids') or []
    if not isinstance(stop_token_ids, list) or not all(isinstance(i, int) for i in stop_token_ids):
        raise ApiError('`stop_token_ids` must be a list of integers', param='stop_token_ids')
    ignore_eos = bool(body.get('ignore_eos', False))
    stop_ids = list(dict.fromkeys(([] if ignore_eos else list(eos_ids)) + stop_token_ids))
    for k in ('frequency_penalty', 'presence_penalty'):
        if body.get(k) not in (None, 0, 0.0):
            raise ApiError(f'{k} is not supported by the gr8 engine yet', param=k)
    if body.get('repetition_penalty') not in (None, 1, 1.0):
        raise ApiError('repetition_penalty is not supported by the gr8 engine yet', param='repetition_penalty')

    # logprobs
    if chat:
        lp = None
        if body.get('logprobs'):
            tl = _num(body, 'top_logprobs', lo=0, kind=int)
            lp = tl or 0
        elif body.get('top_logprobs') is not None:
            raise ApiError('when using `top_logprobs`, `logprobs` must be set to true.', param='top_logprobs')
    else:
        lp = _num(body, 'logprobs', lo=0, kind=int)
    plp = _num(body, 'prompt_logprobs', lo=-1, kind=int)
    if plp == -1:
        plp = max_logprobs
    for name, v in (('logprobs', lp), ('prompt_logprobs', plp)):
        if v is not None and v > max_logprobs:
            raise ApiError(f'Requested {name} of {v}, which is greater than max allowed: {max_logprobs}', param=name)
    return {
        'max_tokens': max_tokens, 'min_tokens': min_tokens, 'temperature': float(temperature), 'top_p': float(top_p),
        'top_k': 0 if top_k in (-1, None) else int(top_k), 'min_p': float(min_p), 'seed': seed,
        'stop_ids': stop_ids, 'logprobs': lp, 'prompt_logprobs': plp,
        # front-end only
        '_stop': stop, '_include_stop': bool(body.get('include_stop_str_in_output', False)),
    }
