#!/usr/bin/env python3
"""Declarative Model attributes and release_metadata auto-discovery for adding models to the launcher."""
from dataclasses import replace
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models  # noqa: E402
import options as opt  # noqa: E402
import supervisor as S  # noqa: E402


def hf_model(**kw):
    base = dict(
        name='qwen3_next',
        title='Qwen3-Next-80B',
        release='https://huggingface.co/example/qwen3-next-exlr8',
        engine='g6m4_qwen',
        build='qwen',
        tp=('4', '2'),
        variants={},
        default_variant={},
        group='gr8-qwen3',
        architecture='Qwen3NextForCausalLM',
        drafter=False,
        int4_kv=False,
        thinking_off=True,
        dialect='hf',
        served=('qwen3-next',),
        vocab=0,
        eos_ids=(),
        max_positions=0,
    )
    base.update(kw)
    return models.Model(**base)


class ReleaseMetadataTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.t, True)

    def write_json(self, name, payload):
        with open(os.path.join(self.t, name), 'w') as f:
            json.dump(payload, f)

    def test_explicit_model_values_win_over_release_files(self):
        self.write_json('config.json', {
            'vocab_size': 999999,
            'eos_token_id': [7, 8],
            'max_position_embeddings': 32768,
        })
        self.write_json('generation_config.json', {'eos_token_id': [9, 10]})
        self.assertEqual(
            models.release_metadata(models.DS41, self.t),
            {'vocab': 129280, 'eos_ids': (1,), 'max_positions': 1048576},
        )
        # GLM53 keeps its explicit max_positions=1048576 even when config.json reports a smaller window,
        # and keeps the legacy vocabulary/EOS even when no directory is passed.
        self.assertEqual(models.release_metadata(models.GLM53, self.t)['max_positions'], 1048576)
        self.assertEqual(
            models.release_metadata(models.GLM53, ''),
            {'vocab': 154880, 'eos_ids': (154820, 154827, 154829), 'max_positions': 1048576},
        )
        partial = hf_model(vocab=152064)
        self.assertEqual(
            models.release_metadata(partial, self.t),
            {'vocab': 152064, 'eos_ids': (9, 10), 'max_positions': 32768},
        )

    def test_hf_model_discovers_vocab_scalar_eos_and_max_positions(self):
        self.write_json('config.json', {
            'architectures': ['Qwen3NextForCausalLM'],
            'vocab_size': 151936,
            'eos_token_id': 151643,
            'max_position_embeddings': 262144,
        })
        self.assertEqual(
            models.release_metadata(hf_model(), self.t),
            {'vocab': 151936, 'eos_ids': (151643,), 'max_positions': 262144},
        )

    def test_generation_config_eos_overrides_config(self):
        self.write_json('generation_config.json', {'eos_token_id': [151645, 151643]})
        self.write_json('config.json', {'vocab_size': 151936, 'eos_token_id': [151643, 151650]})
        self.assertEqual(
            models.release_metadata(hf_model(), self.t)['eos_ids'],
            (151645, 151643),
        )
        self.write_json('generation_config.json', {'eos_token_id': 2})
        self.write_json('config.json', {'vocab_size': 32000, 'eos_token_id': [2, 32000, 32007]})
        self.assertEqual(
            models.release_metadata(hf_model(), self.t)['eos_ids'],
            (2,),
        )

    def test_vlm_text_config_nesting(self):
        self.write_json('config.json', {
            'architectures': ['ExampleVLForConditionalGeneration'],
            'vision_config': {'hidden_size': 1152},
            'text_config': {
                'vocab_size': 262144,
                'max_position_embeddings': 131072,
                'eos_token_id': [1, 106],
            },
        })
        self.write_json('generation_config.json', {'eos_token_id': 106})
        self.assertEqual(
            models.release_metadata(hf_model(), self.t),
            {'vocab': 262144, 'eos_ids': (106,), 'max_positions': 131072},
        )

    def test_missing_or_malformed_files_fall_back_cleanly(self):
        m = hf_model()
        for bad_dir in (None, '', os.path.join(self.t, 'missing'), self.t):
            self.assertEqual(models.release_metadata(m, bad_dir), {'vocab': 0, 'eos_ids': (), 'max_positions': 0})
        with open(os.path.join(self.t, 'config.json'), 'w') as f:
            f.write('{not valid json')
        with open(os.path.join(self.t, 'generation_config.json'), 'w') as f:
            json.dump([1, 2, 3], f)
        self.assertEqual(models.release_metadata(m, self.t), {'vocab': 0, 'eos_ids': (), 'max_positions': 0})
        # Booleans are ints in Python; they must never be read as vocab_size / eos_token_id / max_position_embeddings.
        self.write_json('config.json', {'vocab_size': True, 'eos_token_id': [True, False, 42],
                                        'max_position_embeddings': True})
        self.assertEqual(models.release_metadata(m, self.t), {'vocab': 0, 'eos_ids': (42,), 'max_positions': 0})


class DeclarativeSupervisorTests(unittest.TestCase):
    MEMBERS = ['10.0.0.1', '10.0.0.2', '10.0.0.3', '10.0.0.4']

    class Spawned(Exception):
        pass

    def setUp(self):
        self.t = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.t, True)

    def make_sup(self, model_obj, *argv):
        reg = {**models.MODELS, model_obj.name: model_obj}
        p = mock.patch.dict(models.MODELS, reg, clear=True)
        p.start()
        self.addCleanup(p.stop)
        home = tempfile.mkdtemp(dir=self.t)
        o = opt.parse(['--no-config', '--work-dir', home, '--peers', ','.join(self.MEMBERS), *argv],
                      {'GR8_MODEL': model_obj.name})
        opt.validate(o)
        s = S.Supervisor(o)
        s.lan = self.MEMBERS[0]
        s.rank = 0
        return s

    def test_builtin_models_declare_expected_flags(self):
        self.assertEqual(
            (models.GLM53.native_release, models.GLM53.second_rail, models.GLM53.per_slot_pool, models.GLM53.stage_relay),
            (False, 'all', False, False),
        )
        self.assertEqual(
            (models.GLM53F.native_release, models.GLM53F.second_rail, models.GLM53F.per_slot_pool, models.GLM53F.stage_relay),
            (False, 'prefill', False, False),
        )
        self.assertEqual(
            (models.DS41.native_release, models.DS41.second_rail, models.DS41.per_slot_pool, models.DS41.stage_relay),
            (True, 'all', True, True),
        )
        self.assertEqual(
            (models.PSEUDO.native_release, models.PSEUDO.second_rail, models.PSEUDO.per_slot_pool, models.PSEUDO.stage_relay),
            (False, 'prefill', False, False),
        )

    def test_native_release_flag_resolves_in_place_and_names_model_in_errors(self):
        m = hf_model(native_release=True)
        rel = os.path.join(self.t, 'native')
        os.makedirs(os.path.join(rel, 'dense'))
        os.makedirs(os.path.join(rel, 'manifests'))
        for n in ('dense/exlr8-vocab.safetensors', *models.TOKENIZER_DS):
            with open(os.path.join(rel, n), 'w') as f:
                f.write('{}')
        with open(os.path.join(rel, 'config.json'), 'w') as f:
            json.dump({'architectures': [m.architecture]}, f)
        with open(os.path.join(rel, 'manifests', 'exlr8-manifest-native.json'), 'w') as f:
            json.dump({'set_id': 'c' * 64}, f)

        s = self.make_sup(m, '--model', rel)
        ms, variant, err = S.resolve_model(s.o)
        self.assertEqual((err, variant.name, ms.tokenizer, s.o.get('V2_DENSE')), ('', 'home', rel, rel))
        self.assertFalse(s.o.get('V2'))

        empty = os.path.join(self.t, 'empty')
        os.makedirs(empty)
        s_bad = self.make_sup(m, '--model', empty)
        self.assertIn('want a local copy of the Qwen3-Next-80B exlr8 native release', S.resolve_model(s_bad.o)[2])
        s_none = self.make_sup(m)
        self.assertIn('no model: pass --model <Qwen3-Next-80B release directory>', S.resolve_model(s_none.o)[2])

    def test_boot_uses_second_rail_and_stage_relay_flags(self):
        for rail, relay_on, want_flag in (('all', False, '--ips2'), ('prefill', True, '--pf-ips2')):
            m = hf_model(second_rail=rail, stage_relay=relay_on)
            s = self.make_sup(m)
            os.makedirs(s.out, exist_ok=True)
            s.ms = mock.Mock(dir='/m', tokenizer=self.t)
            s.model_id = 'id1'
            s.hs = {'epoch': 7, 'ips': self.MEMBERS, 'ips2': ['198.51.100.1', '198.51.100.2', '198.51.100.3', '198.51.100.4']}
            plan = {'env': [], 'args': ['--serve', os.path.join(s.home, 's2')], 'knobs': {'READY_TMO': '5'}}

            def fake_popen(argv, **_kw):
                os.makedirs(os.path.join(s.home, 's2'), exist_ok=True)
                with open(os.path.join(s.home, 's2', 'ready.s2'), 'w') as f:
                    f.write('ready\n')
                proc = mock.Mock(pid=4321)
                proc.poll.return_value = None
                return proc

            with mock.patch.object(S.opt, 'engine_plan', return_value=plan), \
                    mock.patch.object(s, 'kv_store', return_value=''), \
                    mock.patch.object(s, 'kv_rows', return_value=''), \
                    mock.patch.object(S.subprocess, 'Popen', side_effect=fake_popen) as popen, \
                    mock.patch.object(s, 'start_front'), \
                    mock.patch.object(s, 'start_shim'), \
                    mock.patch.object(s, 'donate'), \
                    mock.patch.object(s, 'run_forever'), \
                    mock.patch.object(s, 'set_state'), \
                    mock.patch.object(s, 'watch'), \
                    mock.patch.object(S, 'log'), \
                    mock.patch.object(S, 'StageRelay', wraps=S.StageRelay) as relay_cls:
                s.boot()
            argv = popen.call_args[0][0]
            self.assertIn(want_flag, argv)
            self.assertEqual(relay_cls.call_count, 1 if relay_on else 0)

    def test_front_and_shim_follow_declarative_model_and_release_metadata(self):
        tok_dir = os.path.join(self.t, 'tok')
        os.makedirs(tok_dir)
        with open(os.path.join(tok_dir, 'config.json'), 'w') as f:
            json.dump({
                'vocab_size': 151936,
                'max_position_embeddings': 131072,
                'eos_token_id': 151643,
            }, f)
        with open(os.path.join(tok_dir, 'generation_config.json'), 'w') as f:
            json.dump({'eos_token_id': [151645, 151643]}, f)

        m = hf_model(per_slot_pool=True)
        s = self.make_sup(m, '--allow-thinking-off', '1')
        s.ms = mock.Mock(tokenizer=tok_dir)
        with mock.patch.object(s, 'spawn') as spawn, mock.patch.object(S.os, 'access', return_value=True):
            s.start_front({'knobs': {'SHARE_LIMIT': '65536', 'REUSE': '0'}})
            front_no_reuse = spawn.call_args[0][1]
            s.start_front({'knobs': {'SHARE_LIMIT': '65536', 'REUSE': '1'}})
            front_reuse = spawn.call_args[0][1]
        self.assertEqual(front_no_reuse[front_no_reuse.index('--dialect') + 1], 'hf')
        self.assertEqual(front_no_reuse[front_no_reuse.index('--served-model-name') + 1], 'qwen3-next')
        self.assertNotIn('--allow-thinking-off', front_no_reuse)
        self.assertNotIn('--enable-prompt-tokens-details', front_no_reuse)
        self.assertIn('--enable-prompt-tokens-details', front_reuse)

        with mock.patch.object(S.opt, 'engine_plan', return_value={'knobs': {'SESS': '6'}}), \
                mock.patch.object(s, 'spawn', side_effect=self.Spawned) as spawn:
            with self.assertRaises(self.Spawned):
                s.start_shim('/work')
        shim_argv = spawn.call_args[0][1]
        self.assertEqual(shim_argv[shim_argv.index('--sessions') + 1], '6')
        self.assertEqual(shim_argv[shim_argv.index('--capacity-slots') + 1], '6')
        self.assertEqual(shim_argv[shim_argv.index('--model-name') + 1], 'qwen3-next')
        self.assertEqual(shim_argv[shim_argv.index('--vocab-size') + 1], '151936')
        self.assertEqual(shim_argv[shim_argv.index('--eos-ids') + 1], '151645,151643')
        self.assertEqual(shim_argv[shim_argv.index('--max-model-len') + 1], '131072')

        s_shared = self.make_sup(replace(m, per_slot_pool=False))
        s_shared.ms = mock.Mock(tokenizer=tok_dir)
        with mock.patch.object(S.opt, 'engine_plan', return_value={'knobs': {'SESS': '6'}}), \
                mock.patch.object(s_shared, 'spawn', side_effect=self.Spawned) as spawn:
            with self.assertRaises(self.Spawned):
                s_shared.start_shim('/work')
        self.assertNotIn('--capacity-slots', spawn.call_args[0][1])


if __name__ == '__main__':
    unittest.main()
