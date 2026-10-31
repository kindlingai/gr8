"""The models ./gr8 serves, and what the launcher does differently for each. GR8_MODEL picks one; ./gr8 <name> sets it.
Nothing picks one for the runner. `./gr8 models` (this file run) lists each model and the values its options take."""
import json
import os
from dataclasses import dataclass


# The files the front end and the boxes' set id read from the tokenizer directory. GLM's release also carries the chat template
# and generation config; DeepSeek's does not (its prompt comes from encoding.py, ported in frontend/g6front-rs/src/dschat.rs).
TOKENIZER_GLM = ('chat_template.jinja', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json')
TOKENIZER_DS = ('tokenizer.json', 'tokenizer_config.json')


class ModelMismatch(Exception):
    pass


@dataclass(frozen=True)
class Model:
    name: str
    title: str                  # the model's own name, for messages and ./gr8 models
    release: str                # its published exlr8 release, for help text only: --model always names the release
    engine: str                 # bin/<engine>, built by `./gr8 <name> build` (scripts/build.sh MODEL=<build>)
    build: str
    tp: tuple                   # the TP sizes the engine is built for
    variants: dict              # exlr8 v2 width manifests, exlr8-manifest-<name>.json: mean routed-expert bits
    default_variant: dict       # TP: the names tried in order, first found wins; a TP not listed takes `home`
    group: str                  # the default mentat group, so two models' boxes never join one group
    architecture: str           # config.json `architectures[0]` of the release
    drafter: bool               # DFlash2 drafter (--drafter, DFLASH=1)
    int4_kv: bool               # KV_FORMAT=int4 (bin/<engine>-int4)
    thinking_off: bool          # the chat template can turn thinking off (reasoning_effort none)
    fabric: bool = True         # the ranks talk over RDMA: rails and /dev/infiniband
    dialect: str = 'glm'        # the front end's --dialect: glm (GLM chat template and <tool_call>) or ds41 (DeepSeek DSML)
    served: str = ''            # the front end's --served-model-name (empty: the dialect's own default)
    defaults: tuple = ()        # (KEY, value) option defaults for this model, over the table's: origin '<name> default'
    vocab: int = 0              # the shim's --vocab-size: the ids the engine's embedding table takes (0: the shim's own, GLM-5.3's)
    eos_ids: tuple = ()         # the shim's --eos-ids (empty: the shim's own, GLM-5.3's)

    @property
    def tp_sizes(self):
        return ', '.join(self.tp[:-1]) + ' or ' + self.tp[-1] if len(self.tp) > 1 else self.tp[0]


# GLM-5.3 variants: home, down1 and up1 follow the TR3 checkpoint's per-expert widths. The bitNNN profiles spend their
# bits by tensor type, then by layer, in the order that lowered full-vocabulary KL against bf16 the most per byte.
# The TP=3 default: bit225 is down1's size with 5-10% lower KL against bf16, at par on code and ~2% slower on prose. A
# release from before it boots down1. home's experts take 99.8 GB of a rank at TP=3, which leaves no room for the KV
# pool on a 128 GB box. At TP=4 home stays: bit325 lowers KL by 1.5-7% at home's size but decoded ~2.5% slower in
# three boots each.
GLM53 = Model('glm53', 'GLM-5.3', 'https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4', 'g6m4', 'glm',
              ('4', '6', '3', '5'),
              {'home': 3.25, 'down1': 2.25, 'up1': 4.0,
               'bit225': 2.25, 'bit243': 2.43, 'bit260': 2.60, 'bit325': 3.25, 'bit338': 3.38},
              {3: ('bit225', 'down1')}, 'gr8-glm53', 'GlmMoeDsaForCausalLM', True, True, False)
# GLM-5.3-Flash: uniform-width releases, K4 by default.
GLM53F = Model('glm53f', 'GLM-5.3-Flash', 'https://huggingface.co/kindlingai/glm-5.3-flash-exlr8-k2-k3-k4',
               'g6m4_flash', 'flash', ('4', '3', '2'), {'k2': 2.0, 'k3': 3.0, 'k4': 4.0},
               {2: ('k4',), 3: ('k4',), 4: ('k4',)}, 'gr8-glm53f', 'Glm5NextForConditionalGeneration', False, False, True)
# DeepSeek-V4.1-Flash-1017 (docs/DEEPSEEK.md): the release as published (fp8 trunk, MXFP4 routed experts), no exlr8 variants,
# TP=4 first. The DSpark drafter is in the checkpoint (MTP layers), not a DFlash2 drafter. thinking_off: the chat
# template turns thinking off. Owner of the entry: B8 (front end + launcher); B1 made the scaffold.
# Front end: the ds41 dialect (frontend/g6front-rs/src/dialect.rs), served as deepseek-v4.1-flash. max_tokens for a request
# that names none is the lane's 32768; the reasoning effort for one that names none is low (50: 18/18 at 27% of the
# think tokens of high, reports/l3-ds-think-less.md). The binary's own fallback stays 75, the release's default.
DS41 = Model('ds41', 'g6m4_ds41', 'ds41', ('4',), {}, {}, 'gr8-ds41', 'DeepseekV41ForCausalLM', False, False, True,
             dialect='ds41', served='deepseek-v4.1-flash', vocab=129280, eos_ids=(1,),   # engine/model_geom_ds41.h VOCAB, TOK_EOS
             defaults=(('DEFAULT_MAX_TOKENS', '32768'), ('DEFAULT_REASONING_EFFORT', 'low')))
# The pseudo engine (frontend/pseudo-engine): GLM-5.3's tokens and chat template, answers from a TOML file of rules, on
# a CPU. Its release is a directory with responses.toml and the tokenizer files.
<<<<<<< ours
PSEUDO = Model('pseudo', 'the pseudo engine', '', 'g6pseudo', 'pseudo', ('1', '2', '3', '4'), {}, {}, 'gr8-pseudo',
               'GlmMoeDsaForCausalLM', False, False, False, fabric=False)
MODELS = {m.name: m for m in (GLM53, GLM53F, PSEUDO)}
=======
PSEUDO = Model('pseudo', 'g6pseudo', 'pseudo', ('1', '2', '3', '4'), {}, {}, 'gr8-pseudo', 'GlmMoeDsaForCausalLM',
               False, False, False, fabric=False)
MODELS = {m.name: m for m in (GLM53, GLM53F, DS41, PSEUDO)}
>>>>>>> theirs


# The models a runner names. pseudo, the test engine, is served when named but left out of the lists.
SERVED = (GLM53, GLM53F)
CHOICES = ' or '.join(f'{m.name} ({m.title})' for m in SERVED)


def of(o):
    """The Model the options select: GR8_MODEL, which has no default (options.validate refuses a boot without it)."""
    return MODELS[o.get('GR8_MODEL')]


def check_config(model, directory):
    """Raises ModelMismatch when directory/config.json names another architecture than model's. A directory without
    config.json, or a config.json without `architectures`, passes."""
    path = os.path.join(directory or '', 'config.json')
    if not directory or not os.path.isfile(path):
        return
    with open(path) as f:
        arch = (json.load(f).get('architectures') or [None])[0]
    if arch and arch != model.architecture:
        want = next((m.name for m in MODELS.values() if m.architecture == arch), None)
        hint = f': ./gr8 {want} serves it' if want else ''
        raise ModelMismatch(f'{path} is {arch}, and ./gr8 {model.name} serves {model.architecture}{hint}')


def describe(m):
    """What `./gr8 models` prints for m: the name to give and the values the model's options take."""
    defaults = ', '.join(f'TP={tp} ' + ' else '.join(m.default_variant.get(int(tp), ('home',))) for tp in m.tp)
    extra = [o for o, on in (('--drafter URL (DFlash2)', m.drafter), ('--kv-format int4', m.int4_kv),
                             ('--default-reasoning-effort none', m.thinking_off)) if on]
    return '\n'.join([f'{m.name:7} {m.title}: ./gr8 {m.name} <command>, or GR8_MODEL={m.name}',
                      f'        --model    an exlr8 release of {m.architecture} (config.json `architectures`), such as',
                      f'                   {m.release}',
                      f'        --tp       {m.tp_sizes}',
                      f'        --variant  {", ".join(m.variants)}',
                      f'                   by default {defaults}',
                      *([f'        also       {", ".join(extra)}'] if extra else [])])


if __name__ == '__main__':
    print('\n\n'.join(describe(m) for m in SERVED))
    print('\n./gr8 knobs lists every option. The pseudo engine (./gr8 pseudo, no GPU) is for tests: '
          'frontend/pseudo-engine.')
