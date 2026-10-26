#!/usr/bin/env python3
"""docker/model-loader.yaml against the launcher and docker/compose.yaml: every key it sets is a launcher option, it
sets FETCH=model-loader and points MODEL_CACHE and MODEL_LOADER into the one read-only mount of the store, and the
service it waits on mounts the same store. Stdlib only (no YAML parser): the file keeps to a plain subset, which this
reads line by line.
    python3 -m unittest discover -s tools/launch"""
import os
import re
import shutil
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import options as opt  # noqa: E402

OVERRIDE = os.path.join(REPO, 'docker', 'model-loader.yaml')
COMPOSE = os.path.join(REPO, 'docker', 'compose.yaml')


def services(path):
    """{service: {'environment': [...], 'volumes': [...], 'depends_on': [...], 'command': [...]}} of a compose file
    written as compose.yaml is: two-space indents, block lists."""
    out, svc, key, in_services = {}, None, None, False
    with open(path) as f:
        for raw in f:
            line = raw.split(' #')[0].rstrip() if not raw.lstrip().startswith('#') else ''
            if not line.strip():
                continue
            ind = len(line) - len(line.lstrip())
            if ind == 0:
                in_services, svc, key = line.startswith('services:'), None, None
            elif in_services and ind == 2 and line.strip().endswith(':'):
                svc, key = line.strip()[:-1], None
                out[svc] = {}
            elif svc and ind == 4:
                k, _, v = line.strip().partition(':')
                key = k
                out[svc][key] = [] if not v.strip() else v.strip()
            elif svc and key and ind >= 6 and line.strip().startswith('- '):
                if isinstance(out[svc][key], list):
                    out[svc][key].append(line.strip()[2:])
            elif svc and key and ind == 6 and line.strip().endswith(':') and isinstance(out[svc][key], list):
                out[svc][key].append(line.strip()[:-1])                # depends_on: <name>:
    return out


def mount(spec):
    """(host, container, ro) of a volume line."""
    parts = re.sub(r'\$\{[^}]*\}', lambda m: m.group(0).replace(':', '\0'), spec).split(':')
    parts = [p.replace('\0', ':') for p in parts]
    return parts[0], parts[1], len(parts) > 2 and parts[2] == 'ro'


class ComposeTests(unittest.TestCase):
    def setUp(self):
        self.o, self.base = services(OVERRIDE), services(COMPOSE)

    def test_the_override_sets_launcher_options_only(self):
        env = dict(e.split('=', 1) for e in self.o['gr8']['environment'])
        for k in env:
            self.assertIn(k, opt.KEYS, f'{k}: not a launcher option')
        self.assertEqual(env['FETCH'], 'model-loader')
        # no option default lives in compose files (compose.yaml's rule): only the switch and the container's paths
        self.assertEqual(set(env), {'FETCH', 'MODEL_CACHE', 'MODEL_LOADER'})

    def test_the_store_is_one_read_only_mount(self):
        env = dict(e.split('=', 1) for e in self.o['gr8']['environment'])
        mounts = [mount(v) for v in self.o['gr8']['volumes']]
        self.assertEqual(len(mounts), 1)
        host, where, ro = mounts[0]
        self.assertTrue(ro, 'the store must be read-only in gr8')
        self.assertEqual(env['MODEL_CACHE'], where)
        self.assertEqual(env['MODEL_LOADER'], f'unix:{where}/model-loader.sock')
        self.assertNotIn(where, [mount(v)[1] for v in self.base['gr8']['volumes']])   # compose.yaml's stay as they are
        wait = self.o['model-loader-ready']
        self.assertIn(f'model-loader -check unix:{where}/model-loader.sock', ' '.join(wait['command']))
        self.assertEqual([mount(v) for v in wait['volumes']], [(host, where, True)])

    def test_gr8_waits_for_model_loader(self):
        self.assertIn('model-loader-ready', self.o['gr8']['depends_on'])
        with open(OVERRIDE) as f:
            self.assertIn('condition: service_completed_successfully', f.read())

    def test_compose_still_sets_the_legacy_cache(self):
        env = dict(e.split('=', 1) for e in self.base['gr8']['environment'])
        self.assertEqual(env['MODEL_CACHE'], '/var/lib/gr8/cache')
        self.assertNotIn('FETCH', env)                            # the default, legacy, comes from the launcher

    def test_the_override_parses_as_options(self):
        env = dict(e.split('=', 1) for e in self.o['gr8']['environment'])
        o = opt.parse(['--no-config', '--peers', '192.0.2.1,192.0.2.2,192.0.2.3,192.0.2.4',
                       '--model', 'https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4'], {**env, 'GR8_MODEL': 'glm53'})
        opt.validate(o)
        self.assertEqual((o.get('FETCH'), o.origin['FETCH']), ('model-loader', 'environment'))


class ModelLoaderTreeTests(unittest.TestCase):
    """model-loader's own deploy trees (its deploy/compose/gr8.yml, deploy/spark-os/gr8.override.yml and
    deploy/k8s/examples/gr8) set MODEL_CACHE and MODEL_LOADER, not FETCH: those two alone select model-loader."""

    GLM = {'GR8_MODEL': 'glm53', 'MODEL': 'https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4'}
    TREE = {'MODEL_CACHE': '/var/lib/model-loader', 'MODEL_LOADER': 'unix:/var/lib/model-loader/model-loader.sock'}

    def test_model_loader_alone_selects_it(self):
        o = opt.parse(['--no-config'], {**self.GLM, **self.TREE})
        self.assertEqual(o.get('FETCH'), 'model-loader')
        self.assertIn('MODEL_LOADER', o.origin['FETCH'])

    def test_unset_stays_legacy(self):
        o = opt.parse(['--no-config'], {**self.GLM, 'MODEL_CACHE': '/opt/gr8/cache'})
        self.assertEqual((o.get('FETCH'), o.origin['FETCH']), ('legacy', 'default'))

    def test_legacy_with_model_loader_is_refused(self):
        o = opt.parse(['--no-config', '--peers', '10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4', '--fetch', 'legacy'],
                      {**self.GLM, **self.TREE})
        with self.assertRaisesRegex(opt.OptionError, 'fetch legacy with --model-loader'):
            opt.validate(o)


CHART = os.path.join(REPO, 'deploy', 'k8s')


@unittest.skipUnless(shutil.which('helm'), 'needs helm')
class ChartTests(unittest.TestCase):
    """deploy/k8s with modelLoader.enabled: the same contract as docker/model-loader.yaml."""

    def render(self, *sets):
        # GR8_MODEL and MODEL are required values (#245)
        argv = ['helm', 'template', 'gr8', CHART, '--set', 'options.GR8_MODEL=glm53',
                '--set', 'options.MODEL=https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4']
        for s in sets:
            argv += ['--set', s]
        return subprocess.run(argv, capture_output=True, text=True, check=True).stdout

    @staticmethod
    def env(text):
        return dict(re.findall(r'- name: ([A-Z0-9_]+)\n\s+value: (\S+)', text))

    def test_off_by_default(self):
        out = self.render()
        self.assertNotIn('model-loader', out)
        self.assertEqual(self.env(out)['MODEL_CACHE'], '/var/lib/gr8/cache')
        self.assertIn('name: cache', out)

    def test_on(self):
        out = self.render('modelLoader.enabled=true', 'modelLoader.store=/srv/ml')
        env = self.env(out)
        self.assertEqual((env['FETCH'], env['MODEL_CACHE'], env['MODEL_LOADER']),
                         ('model-loader', '/var/lib/model-loader', 'unix:/var/lib/model-loader/model-loader.sock'))
        self.assertNotIn('name: cache', out)                     # no cache claim or mount
        self.assertRegex(out, r'- name: model-loader-store\n\s+mountPath: /var/lib/model-loader\n\s+readOnly: true')
        self.assertRegex(out, r'- name: model-loader-store\n\s+hostPath:\n\s+path: /srv/ml')
        self.assertIn("model-loader -check unix:/var/lib/model-loader/model-loader.sock", out)
        self.assertEqual(out.count('readOnly: true'), 2)             # gr8 and the init container both read-only


if __name__ == '__main__':
    unittest.main()
