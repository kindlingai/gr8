"""The models ./gr8 serves, and what the launcher does differently for each. GR8_MODEL picks one; ./gr8 <name> sets it."""
import json
import os
from dataclasses import dataclass


class ModelMismatch(Exception):
    pass


@dataclass(frozen=True)
class Model:
    name: str
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

    @property
    def tp_sizes(self):
        return ', '.join(self.tp[:-1]) + ' or ' + self.tp[-1]


# GLM-5.3 variants: home, down1 and up1 follow the TR3 checkpoint's per-expert widths. The bitNNN profiles spend their
# bits by tensor type, then by layer, in the order that lowered full-vocabulary KL against bf16 the most per byte.
# The TP=3 default: bit225 is down1's size with 5-10% lower KL against bf16, at par on code and ~2% slower on prose. A
# release from before it boots down1. home's experts take 99.8 GB of a rank at TP=3, which leaves no room for the KV
# pool on a 128 GB box. At TP=4 home stays: bit325 lowers KL by 1.5-7% at home's size but decoded ~2.5% slower in
# three boots each.
GLM53 = Model('glm53', 'g6m4', 'glm', ('4', '6', '3'),
              {'home': 3.25, 'down1': 2.25, 'up1': 4.0,
               'bit225': 2.25, 'bit243': 2.43, 'bit260': 2.60, 'bit325': 3.25, 'bit338': 3.38},
              {3: ('bit225', 'down1')}, 'gr8-glm53', 'GlmMoeDsaForCausalLM', True, True, False)
# GLM-5.3-Flash: uniform-width releases, K4 by default.
GLM53F = Model('glm53f', 'g6m4_flash', 'flash', ('4', '2'), {'k2': 2.0, 'k3': 3.0, 'k4': 4.0},
               {2: ('k4',), 4: ('k4',)}, 'gr8-glm53f', 'Glm5NextForConditionalGeneration', False, False, True)
MODELS = {m.name: m for m in (GLM53, GLM53F)}


def of(o):
    """The Model the options select (GR8_MODEL, default glm53)."""
    name = o.get('GR8_MODEL') or GLM53.name
    if name not in MODELS:
        raise KeyError(name)
    return MODELS[name]


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
