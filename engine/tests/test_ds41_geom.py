"""engine/model_geom_ds41.h against the release's config.json (engine/ref/ds41/config.json, sha256 8be45ce0...).
DS41_GEOM_DUMP=<binary built from ds41_geom_dump.cpp> python3 -m unittest -q test_ds41_geom   (scripts/host_tests.sh)"""
import hashlib
import json
import os
import subprocess
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, '..', 'ref', 'ds41', 'config.json')
CONFIG_SHA256 = '8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879'


def dump():
    out = subprocess.run([os.environ['DS41_GEOM_DUMP']], check=True, capture_output=True, text=True).stdout
    scalars, layers = {}, {}
    for line in out.splitlines():
        f = line.split()
        if f[0] == 'layer':
            layers[int(f[1])] = dict(zip(f[2::2], map(int, f[3::2])))
        else:
            scalars[f[0]] = float(f[1])
    return scalars, layers


class Ds41Geom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONFIG, 'rb') as fh:
            raw = fh.read()
        cls.sha = hashlib.sha256(raw).hexdigest()
        cls.cfg = json.loads(raw)
        cls.t = cls.cfg['text_config']
        cls.g, cls.layers = dump()

    def test_config_is_the_1017_release(self):
        self.assertEqual(self.sha, CONFIG_SHA256)
        self.assertEqual(self.cfg['architectures'], ['DeepseekV41ForCausalLM'])

    def test_scalars_equal_config(self):
        t, g = self.t, self.g
        pairs = {
            'HID': 'hidden_size', 'VOCAB': 'vocab_size', 'L_MAIN': 'num_hidden_layers', 'N_NEXTN': 'num_nextn_predict_layers',
            'RMS_EPS': 'rms_norm_eps', 'HEADS': 'num_attention_heads', 'HEAD_DIM': 'head_dim', 'QK_ROPE': 'qk_rope_head_dim',
            'QA': 'q_lora_rank', 'O_LORA': 'o_lora_rank', 'O_GROUPS': 'o_groups', 'WINDOW': 'sliding_window',
            'ROPE_THETA': 'rope_theta', 'COMPRESS_ROPE_THETA': 'compress_rope_theta', 'MAX_POS': 'max_position_embeddings',
            'IDX_HEADS': 'index_n_heads', 'IDX_DIM': 'index_head_dim', 'IDX_TOPK': 'index_topk',
            'CAND_SOURCE': 'candidate_source_layer_id', 'CAND_TOPK_BLOCKS': 'candidate_topk_blocks',
            'CAND_BLOCK': 'candidate_block_size', 'NEXP': 'n_routed_experts', 'TOPK': 'num_experts_per_tok',
            'EXP_I': 'moe_intermediate_size', 'ROUTE_SCALE': 'routed_scaling_factor', 'SWIGLU_LIMIT': 'swiglu_limit',
            'HC': 'hc_mult', 'HC_ITERS': 'hc_sinkhorn_iters', 'HC_EPS': 'hc_eps', 'NGRAM_MAX': 'engram_max_ngram_size',
            'ENGRAM_HEADS': 'engram_n_heads', 'ENGRAM_HEAD_DIM': 'engram_head_dim', 'ENGRAM_VOCAB': 'engram_vocab_size',
            'ENGRAM_CVOCAB': 'engram_compressed_vocab_size', 'ENGRAM_PAD': 'engram_pad_token_id',
            'DS_NEXP': 'dspark_n_routed_experts', 'DS_TOPK': 'dspark_num_experts_per_tok', 'DS_BLOCK': 'dspark_block_size',
            'DS_MARKOV': 'dspark_markov_rank', 'DS_NOISE_TOKEN': 'dspark_noise_token_id',
        }
        for name, key in pairs.items():
            self.assertAlmostEqual(g[name], float(t[key]), delta=abs(float(t[key])) * 1e-6, msg=f'{name} vs {key}')
        r = t['rope_scaling']
        self.assertEqual((g['YARN_ORIG_POS'], g['YARN_FACTOR'], g['YARN_BETA_FAST'], g['YARN_BETA_SLOW']),
                         (r['original_max_position_embeddings'], r['factor'], r['beta_fast'], r['beta_slow']))
        self.assertEqual((g['TOK_BOS'], g['TOK_EOS'], g['TOK_PAD'], g['TOK_IMAGE']),
                         (self.cfg['bos_token_id'], self.cfg['eos_token_id'], self.cfg['pad_token_id'], self.cfg['image_token_id']))
        self.assertEqual((g['FP8_BLOCK'], g['FP8_BLOCK']), tuple(self.cfg['quantization_config']['weight_block_size']))
        self.assertEqual((g['ENGRAM_ROWS_L0'], g['ENGRAM_ROWS_L1']), tuple(t['engram_num_embeddings']))
        self.assertEqual(g['DS_TARGET0'], t['dspark_target_layer_ids'][0])
        self.assertEqual(t['dspark_target_layer_ids'], [37, 38, 39])

    def test_derived(self):
        g, t = self.g, self.t
        self.assertEqual(g['QK_NOPE'], t['head_dim'] - t['qk_rope_head_dim'])
        self.assertEqual(g['KV_LORA'], g['QK_NOPE'])
        self.assertEqual(g['HC_MIX'], (2 + t['hc_mult']) * t['hc_mult'])
        self.assertEqual(g['NLAYERS'], len(t['compress_ratios']))
        self.assertEqual(g['MTP_LAYER'], t['num_hidden_layers'])
        self.assertEqual(g['DENSE_I'], 0)

    def test_layer_tables(self):
        t = self.t
        for l in range(t['num_hidden_layers'] + t['num_nextn_predict_layers']):
            row = self.layers[l]
            self.assertEqual(row['ratio'], t['compress_ratios'][l], f'layer {l} ratio')
            self.assertEqual(row['engram'], int(l in t['engram_layer_ids']), f'layer {l} engram')
            self.assertEqual(row['full_index'], int(l in t['index_source_layer_ids']), f'layer {l} full_index')
            for key, srcs in (('kv_source', t['kv_source_layer_ids']), ('index_source', t['index_source_layer_ids'])):
                want = max([s for s in srcs if s <= l], default=-1) if l < t['num_hidden_layers'] else -1
                self.assertEqual(row[key], want, f'layer {l} {key}')


if __name__ == '__main__':
    unittest.main()
