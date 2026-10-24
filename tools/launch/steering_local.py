"""Validate a local steering directory against the resolved model before rank preparation."""
import json
import os
import sys

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
from steering import release
import modelset
import options


def prepare(o, ms):
    directory = o.get('STEERING_PROFILE')
    if not directory:
        return
    try:
        if o.get('BASE_PACKS') or (ms.set.get('source', {}).get('kind') != 'exlr8-release' and
                                   (o.get('V2') or o.get('V2_DENSE'))):
            raise ValueError('local profiles require a self-contained release or complete rank-pack set; mixed external weights are unsupported')
        template = os.path.join(ms.tokenizer, 'chat_template.jinja')
        actual_template = options.file_sha256(template)
        if actual_template != o.get('STEERING_CHAT_TEMPLATE_SHA256'):
            raise ValueError('profile identity does not match the served chat template')
        weights = ms.set.get('source', {}).get('weights') or ms.id
        if o.get('STEERING_WEIGHT_SET') and o.get('STEERING_WEIGHT_SET') != weights:
            raise ValueError('profile identity does not match the resolved weight set')
        # Packs can lack config.json. Where present, compare actual dimensions and revision metadata too.
        candidates = [os.path.join(ms.tokenizer, 'config.json')]
        for config_root in (ms.src_dir, o.get('V2_DENSE')):
            if config_root:
                candidates.append(os.path.join(config_root, 'config.json'))
        for path in candidates:
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                config = json.load(f)
            if not isinstance(config, dict):
                raise ValueError('resolved model config must be an object')
            for field, want in (('hidden_size', 6144), ('num_hidden_layers', 78)):
                if config.get(field) != want:
                    raise ValueError('resolved model has incompatible ' + field)
            if config.get('_commit_hash') and config['_commit_hash'] != o.get('STEERING_MODEL_REVISION'):
                raise ValueError('resolved model revision differs from profile identity')
        with open(os.path.join(ms.tokenizer, 'tokenizer_config.json')) as f:
            token_config = json.load(f)
        if not isinstance(token_config, dict):
            raise ValueError('resolved tokenizer config must be an object')
        if token_config.get('_commit_hash') and token_config['_commit_hash'] != o.get('STEERING_TOKENIZER_REVISION'):
            raise ValueError('resolved tokenizer revision differs from profile identity')
        expected = {'name': 'glm-5.3', 'revision': o.get('STEERING_MODEL_REVISION'),
                    'weight_set': weights, 'tokenizer_revision': o.get('STEERING_TOKENIZER_REVISION'),
                    'chat_template_sha256': actual_template, 'hidden_size': 6144, 'layer_count': 78}
        manifest = release.validate_directory(directory, expected)
        o.values['STEERING_WEIGHT_SET'] = weights
        o.values['STEERING_CHAT_TEMPLATE'] = template
        return manifest
    except (OSError, ValueError, release.ProfileError) as e:
        raise modelset.ModelError(f'--steering-profile {directory}: {e}') from None
