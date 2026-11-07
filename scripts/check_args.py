#!/usr/bin/env python3
"""The engine's option checks over the launcher's argument lists, through `--check-args`.

    scripts/check_args.py cases > engine/runtime/s2/tests/options_cases.txt
    scripts/check_args.py run --model glm53|glm53f|ds41 --bin BIN [--nr 4] [--lib DIR] [--cases FILE] --out FILE

`cases` writes one line per case: MODEL, NAME, ENV and ARGS, tab separated (ENV and ARGS space separated); MODEL@NR
limits a case to builds of that width. The lists come from tools/launch/options.py engine_plan, run as the supervisor
runs it for rank 0, with fixture paths for every file (the checks open none of them), for each model and launcher
variant. Then one mutation of a model's default list
per refusal in engine/runtime/engine_options.h, and a few the engine makes after the parse (steering, grammar).
In ARGS, @IPS and @IPS2 stand for the build's NR rail addresses and @EMPTY for an empty argument.
engine/runtime/s2/tests/options_vectors_test.cpp parses the same cases without CUDA.

`run` starts BIN once per case of MODEL, with --check-args first, the case's ENV over PATH alone, and writes each
case's exit code, stdout and stderr to FILE. The lib folder beside BIN, then each --lib (bin/lib holds a build's cuBLAS
for every model), is the library path. --check-args stops before any CUDA work, so it runs anywhere the binary runs.
Two transcripts (an old and a new build) compare with diff.

R6a's comparison: before R6a only DS41 took --check-args. check_args_base.patch adds it to GLM and Flash at R6a's exit
point (DS41 keeps its own), for the comparison only. Apply it to a checkout of the base commit (74d62d56, `git apply
scripts/check_args_base.patch` from this tree), build both trees' engines, run `run` once per model with each build's
binary and the same --cases file, and diff each pair of transcripts.
"""
import argparse
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ('glm53', 'glm53f', 'ds41')
IPS2_FLAG = {'glm53': '--ips2', 'glm53f': '--pf-ips2', 'ds41': '--ips2'}
FX = '/fx'
SOURCE = {'glm53': ['--v2', FX + '/v2', '--v2-slices', FX + '/slices', '--v2-widths', FX + '/widths.json',
                    '--v2-dense', FX + '/release'],
          'glm53f': ['--v2', FX + '/v2', '--v2-slices', FX + '/slices', '--v2-widths', FX + '/widths.json',
                     '--v2-dense', FX + '/release'],
          'ds41': ['--v2-dense', FX + '/release']}
KV_SSD = 'ssd:' + FX + '/kv,size=8G'
# launcher variants per model: (name, flags). A variant the launcher refuses is left out.
VARIANTS = {
    'glm53': [('tp4', ['--tp', '4']), ('tp6', ['--tp', '6']), ('tp3', ['--tp', '3']), ('tp5', ['--tp', '5']),
              ('dflash0', ['--dflash', '0']), ('adapt0', ['--spec-adapt', '0']),
              ('dflash0-adapt0', ['--dflash', '0', '--spec-adapt', '0']),
              ('kvstore', ['--kv-store', KV_SSD]), ('kvstore-sessions', ['--kv-store', KV_SSD, '--kv-keep', 'sessions']),
              ('reuse0', ['--reuse', '0']), ('int4', ['--kv-format', 'int4']), ('mtp1', ['--mtp', '1']),
              ('mtp3', ['--mtp', '3']), ('sess1', ['--sess', '1']), ('sess4', ['--sess', '4']), ('nohash', ['--nohash', '1']),
              ('rejection', ['--spec-accept', 'rejection']), ('tp6-dflash0', ['--tp', '6', '--dflash', '0']),
              ('tp3-mtp3', ['--tp', '3', '--mtp', '3'])],
    'glm53f': [('tp4', ['--tp', '4']), ('tp3', ['--tp', '3']), ('tp2', ['--tp', '2']), ('tp1', ['--tp', '1']),
               ('dflash0', ['--dflash', '0']), ('adapt0', ['--spec-adapt', '0']),
               ('dflash0-adapt0', ['--dflash', '0', '--spec-adapt', '0']), ('cache', ['--flash-cache', 'aligned']),
               ('cache-kvstore', ['--flash-cache', 'aligned', '--kv-store', KV_SSD]),
               ('cache-terminal', ['--flash-cache', 'terminal', '--dflash', '0']), ('int4', ['--kv-format', 'int4']),
               ('mtp2', ['--mtp', '2']), ('mtp3', ['--mtp', '3']), ('sess1', ['--sess', '1']), ('sess4', ['--sess', '4']),
               ('nohash', ['--nohash', '1'])],
    'ds41': [('tp4', ['--tp', '4']), ('chunk1', ['--chunk', '1']), ('chunk16', ['--chunk', '16']), ('eager', ['--eager', '1']),
             ('mtp1', ['--mtp', '1']), ('sess1', ['--sess', '1']), ('sess8', ['--sess', '8']), ('sess32', ['--sess', '32']),
             ('reuse0', ['--reuse', '0']), ('int4', ['--kv-format', 'int4']), ('nohash', ['--nohash', '1']),
             ('eager-sess1', ['--eager', '1', '--sess', '1'])],
}


def plan(model, flags):
    """(env, args) of rank 0's engine for these launcher flags, or None when the launcher refuses them."""
    sys.path.insert(0, os.path.join(REPO, 'tools', 'launch'))
    import options as opt
    tp = flags[flags.index('--tp') + 1] if '--tp' in flags else '4'
    peers = ','.join(f'10.0.0.{i + 1}' for i in range(int(tp)))
    try:
        o = opt.parse(['--no-config', '--peers', peers] + SOURCE[model] + flags, {'GR8_MODEL': model})
        opt.validate(o)
        p = opt.engine_plan(o, rank=0, rank_dir=FX + '/rank', aux_dir=FX + '/aux', home=FX + '/home', epoch=1,
                            set_id='ab12')
    except opt.OptionError as e:
        print(f'# {model} {" ".join(flags)}: the launcher refuses: {e}', file=sys.stderr)
        return None
    args = ['--rank', '0', '--ips', '@IPS', IPS2_FLAG[model], '@IPS2', '--proxy-cpu', '3'] + p['args']
    return list(p['env']), args


# ---- mutations: edits of a model's default (env, args)
def env_set(*kv):
    def f(env, args):
        keys = {x.split('=', 1)[0] for x in kv}
        return [e for e in env if e.split('=', 1)[0] not in keys] + list(kv), args
    return f


def env_del(*keys):
    return lambda env, args: ([e for e in env if e.split('=', 1)[0] not in keys], args)


def arg_set(flag, value):
    """--flag's value replaced, or --flag value appended."""
    def f(env, args):
        a = list(args)
        if flag in a:
            a[a.index(flag) + 1] = value
        else:
            a += [flag, value]
        return env, a
    return f


def arg_add(*extra):
    return lambda env, args: (env, list(args) + list(extra))


def arg_del(flag, values=1):
    def f(env, args):
        a = list(args)
        while flag in a:
            i = a.index(flag)
            del a[i:i + 1 + values]
        return env, a
    return f


def arg_del_prefix(prefix):
    """every --prefix* flag and its value"""
    def f(env, args):
        a, out, i = list(args), [], 0
        while i < len(a):
            if a[i].startswith(prefix):
                i += 2
                continue
            out.append(a[i])
            i += 1
        return env, out
    return f


def chain(*fs):
    def f(env, args):
        for g in fs:
            env, args = g(env, args)
        return env, args
    return f


SERVE = chain(arg_del('--serve'), arg_del('--s2-port'), arg_del('--s2-sessions'), arg_del('--s2-epoch'))
OFFLINE = chain(SERVE, arg_del_prefix('--dflash-'), arg_del('--spec-adaptive', 0), arg_del('--reuse-prefix', 0),
                arg_del('--kv-store'))
PACKS = chain(arg_del('--v2-dense'), arg_add('--manifest', FX + '/rank0.manifest.json', '--pack', FX + '/rank0.pack'))
DFLASH = arg_add('--dflash-manifest', FX + '/dflash.manifest.json', '--dflash-pack', FX + '/dflash.pack')
G, F, D = ('glm53',), ('glm53f',), ('ds41',)
GF, ALL = ('glm53', 'glm53f'), MODELS
MUTATIONS = [
    # every model's checks, in engine_options.h order
    (G + D, 'moe-topk', env_set('GR8_MOE_TOPK=4')),
    (F, 'moe-topk-screen', env_set('GR8_MOE_TOPK=4')),
    (G + D, 'router-split', env_set('GR8_PF_ROUTER_SPLIT=1')),
    (F, 'router-split-on', env_set('GR8_PF_ROUTER_SPLIT=1')),
    (ALL, 'missing-value', arg_add('--cap')),
    (ALL, 'dense-form', arg_add('--v2-dense-form', 'exact')),
    (ALL, 'spec-accept-bad', arg_add('--spec-accept', 'maybe')),
    (ALL, 'unknown', arg_add('--bogus')),
    (G + D, 'spec-fixed-unknown', arg_add('--spec-fixed')),
    (ALL, 'set-id-bad', arg_set('--set-id', 'XYZ')),
    (ALL, 'set-id-long', arg_set('--set-id', 'a' * 65)),
    (ALL, 'kvb-packs', chain(env_set('GR8_KVB=fp8'), PACKS)),
    (ALL, 'rank-neg', arg_set('--rank', '-1')),
    (ALL, 'ips-short', arg_set('--ips', '10.0.0.1')),
    (ALL, 'no-pack', arg_del('--v2-dense')),
    (ALL, 'release-manifest', arg_add('--manifest', FX + '/rank0.manifest.json')),
    (GF, 'release-no-v2', arg_del('--v2')),
    (D, 'release-no-image', arg_del('--image', 0)),
    (ALL, 'layout-packs', chain(PACKS, arg_add('--layout', FX + '/layout.json'))),
    (G, 'shared-scratch-no-pfv2', arg_del('--pfv2', 0)),
    (F + D, 'shared-scratch', env_set('GR8_DEC_SHARED_SCRATCH=1')),
    (ALL, 'layers-file', arg_set('--layers', '@' + FX + '/missing-layers.txt')),
    (ALL, 'chunk0', arg_set('--chunk', '0')),
    (ALL, 'chunk-big', arg_set('--chunk', '999')),
    (ALL, 'mtp9', arg_set('--mtp', '9')),
    (ALL, 'cands-big', arg_set('--top-p-cands', '9999')),
    (ALL, 'cands-neg', arg_set('--top-p-cands', '-1')),
    (F, 'lanes-explicit', arg_set('--pf-lanes', '1')),
    (ALL, 'resid-dump', env_set('GR8_RESID_DUMP=' + FX + '/resid')),
    (ALL, 'logit-dump', env_set('GR8_LOGIT_DUMP=' + FX + '/logit')),
    (GF, 'dflash-no-pack', arg_del('--dflash-pack')),
    (GF, 'dflash-sel', arg_add('--dflash-sel', '2')),
    (GF, 'dflash-k0', arg_add('--dflash-k', '0')),
    (GF, 'dflash-k8', arg_add('--dflash-k', '8')),
    (GF, 'dflash-k3', arg_add('--dflash-k', '3')),
    (GF, 'dflash-offline-mtp', chain(OFFLINE, DFLASH)),
    (D, 'dflash-no-pack', arg_add('--dflash-manifest', FX + '/dflash.manifest.json')),
    (ALL, 'rejection-no-draft', chain(arg_set('--mtp', '0'), arg_del_prefix('--dflash-'), arg_add('--spec-accept', 'rejection'))),
    (F, 'spec-reset-serve', arg_add('--spec-reset-per-request')),
    (F, 'spec-fixed-adapt', arg_add('--spec-fixed')),
    (F, 'spec-fixed-ok', chain(arg_del('--spec-adaptive', 0), arg_add('--spec-fixed'))),
    (F, 'offline-adapt', chain(OFFLINE, arg_add('--prompt', '1,2,3'))),
    (F, 'offline-reset', chain(OFFLINE, arg_add('--spec-reset-per-request'))),
    (G, 'adapt-no-dflash', arg_del_prefix('--dflash-')),
    (D, 'adapt-no-dflash', arg_add('--spec-adaptive')),
    (GF, 'levels-no-adapt', chain(arg_del('--spec-adaptive', 0), arg_add('--spec-levels', '1,2'))),
    (D, 'levels', arg_add('--spec-levels', '1')),
    (GF, 'levels-range', arg_add('--spec-levels', '9')),
    (GF, 'levels-zero', arg_add('--spec-levels', ',')),
    (GF, 'levels-empty-file', arg_add('--spec-levels', '@/dev/null')),
    (GF, 'levels-ok', arg_add('--spec-levels', '3,5,7')),
    (ALL, 's2-port0', arg_set('--s2-port', '0')),
    (ALL, 's2-port-big', arg_set('--s2-port', '70000')),
    (GF, 's2-sessions9', arg_set('--s2-sessions', '9')),
    (D, 's2-sessions33', arg_set('--s2-sessions', '33')),
    (ALL, 's2-sessions0', arg_set('--s2-sessions', '0')),
    (ALL, 's2-epoch0', arg_set('--s2-epoch', '0')),
    (GF, 'serve-no-graph', arg_del('--graph', 0)),
    (D, 'serve-eager', arg_del('--graph', 0)),
    (GF, 'serve-mtp0', arg_set('--mtp', '0')),
    (GF, 'serve-no-prefill', arg_del('--prefill')),
    (ALL, 'test-knob', env_set('GR8_TEST_S2_RUNS=0:4')),
    (ALL, 'rt-store-offline', chain(OFFLINE, env_set('GR8_RT_STORE=mem:'))),
    (ALL, 'rt-poison', env_set('GR8_TEST_RT_POISON=1')),
    (ALL, 'rt-poison-offline', chain(OFFLINE, env_del('GR8_DS_MTP'), env_set('GR8_TEST_RT_POISON=1'))),
    (ALL, 'rt-local-no-store', env_set('GR8_RT_LOCAL=4')),
    (ALL, 'rt-place-no-local', env_set('GR8_RT_PLACE=8')),
    (ALL, 'rt-local-l', env_set('GR8_RT_STORE=mem:', 'GR8_RT_LOCAL=4', 'GR8_RT_L=1024')),
    (ALL, 'rt-local-ok', env_set('GR8_RT_STORE=mem:', 'GR8_RT_LOCAL=4', 'GR8_RT_PLACE=8')),
    (ALL, 'steering-no-hash', arg_add('--steering-profile', FX + '/steering', '--no-hash')),
    (ALL, 'steering-no-weight-set', arg_add('--steering-profile', FX + '/steering')),
    (ALL, 'kstop-tau4', env_set('GR8_MTP_KSTOP_TAU=0.1,0.2,0.3,0.4')),
    (ALL, 'kstop-offline', chain(OFFLINE, env_set('GR8_MTP_KSTOP=1'))),
    (G, 'kstop-seed3', env_set('GR8_MTP_KSTOP_SEED3=0')),
    (G, 'kstop1', env_set('GR8_MTP_KSTOP=1', 'GR8_MTP_KSTOP_KMAX=2', 'GR8_MTP_KSTOP_TAU=0,0.3,0.6')),
    (G, 'kstop-no-dflash-mode2', chain(arg_del_prefix('--dflash-'), arg_del('--spec-adaptive', 0), env_set('GR8_MTP_KSTOP=2'))),
    (F, 'kstop-mtp1', env_set('GR8_MTP_KSTOP=2')),
    (D, 'kstop', env_set('GR8_MTP_KSTOP=1')),
    (ALL, 'kstop-lplog', env_set('GR8_MTP_LPLOG=1')),
    (ALL, 'kstopb-tau8', env_set('GR8_MTP_KSTOPB_TAU=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8')),
    (ALL, 'kstopb-tau-s8', env_set('GR8_MTP_KSTOPB_TAU_S1=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8')),
    (ALL, 'kstopb-widths65', env_set('GR8_MTP_KSTOPB_WIDTHS=' + ','.join(['2'] * 65))),
    (ALL, 'kstopb-widths-big', env_set('GR8_MTP_KSTOPB_WIDTHS=999')),
    (ALL, 'kstopb-offline', chain(OFFLINE, env_set('GR8_MTP_KSTOPB=1'))),
    (ALL, 'kstopb-explore-neg', env_set('GR8_MTP_KSTOPB=1', 'GR8_MTP_KSTOPB_EXPLORE=-1')),
    (ALL, 'kstopb-ok', env_set('GR8_MTP_KSTOPB=1', 'GR8_MTP_KSTOPB_TAU=0.6,0.88', 'GR8_MTP_KSTOPB_TAU_S2=0.7,0.8',
                               'GR8_MTP_KSTOPB_WIDTHS=2,3,5')),
    (ALL, 'cache-bad', env_set('GR8_GLM53F_CACHE=bogus')),
    (G + D, 'cache-not-flash', env_set('GR8_GLM53F_CACHE=aligned')),
    (F, 'serve-reuse', arg_add('--reuse-prefix')),
    (F, 'serve-rejection', arg_add('--spec-accept', 'rejection')),
    (F, 'cache-terminal-dflash', env_set('GR8_GLM53F_CACHE=terminal')),
    (F, 'sessions-rows', chain(arg_set('--s2-sessions', '8'), arg_set('--mtp', '7'), env_set('GR8_MTP_KSTOP=0'))),
    (F, 'cache-offline', chain(OFFLINE, env_set('GR8_GLM53F_CACHE=aligned'))),
    (F, 'cache-two-stores', chain(env_set('GR8_GLM53F_CACHE=aligned'), arg_add('--kv-store', KV_SSD, '--kv-store', 'mem:,size=1G'))),
    (F, 'cache-budget0', env_set('GR8_GLM53F_CACHE=aligned', 'GR8_GLM53F_STATE_BUDGET_MIB=0')),
    (F, 'kvstore-no-cache', arg_add('--kv-store', KV_SSD)),
    (F, 'kv-chunks', env_set('GR8_GLM53F_KV_CHUNKS=1')),
    (G, 'kvstore-no-reuse', chain(arg_del('--reuse-prefix', 0), arg_add('--kv-store', KV_SSD))),
    (G, 'kvtier', arg_add('--kv-tier', KV_SSD)),
    (G, 'kvtier-system', chain(arg_add('--kv-tier', KV_SSD + ',keep=shared'), arg_add('--kv-shared-after', '3'))),
    (ALL, 'kv-keep-bad', arg_add('--kv-keep', 'bogus')),
    (ALL, 'kv-system-after0', arg_add('--kv-system-after', '0')),
    (ALL, 'kv-write-limit', arg_add('--kv-write-limit', '2000')),
    (ALL, 'kv-write-limit-nan', arg_add('--kv-write-limit', 'nan')),
    (ALL, 'kv-format-int4', arg_add('--kv-format', 'int4')),
    (ALL, 'kv-format-bad', arg_add('--kv-format', 'fp16')),
    (D, 'topk-decode', env_set('GR8_DS_MOE_TOPK=4', 'GR8_DS_MOE_TOPK_DECODE=5')),
    (D, 'topk-decode-ok', env_set('GR8_DS_MOE_TOPK=4', 'GR8_DS_MOE_TOPK_DECODE=3')),
    (D, 'reuse-offline', chain(OFFLINE, env_del('GR8_DS_MTP'), arg_add('--reuse-prefix'))),
    (D, 'kvstore', arg_add('--kv-store', KV_SSD)),
    (D, 'prefill', arg_add('--prefill', '4096')),
    (D, 'mtp', arg_set('--mtp', '1')),
    (D, 'dsp-mtp-bad', env_set('GR8_DS_MTP=9')),
    (D, 'dsp-block', env_set('GR8_DS_BLOCK=4')),
    (D, 'dsp-relaxed-bad', env_set('GR8_DS_RELAXED_ACCEPT=bogus')),
    (D, 'dsp-verify-cap', env_set('GR8_DS_VERIFY_CAP=2')),
    (D, 'dsp-conf-temp', env_set('GR8_DS_CONF_TEMP=-1')),
    (D, 'dsp-conf-temp3', env_set('GR8_DS_CONF_TEMP=1,1,1')),
    (D, 'dsp-prefill-keep', env_set('GR8_DS_PREFILL_KEEP=5')),
    (D, 'dsp-offline', chain(OFFLINE, env_set('GR8_DS_MTP=3'))),
    (D, 'sess-drafts-no-mtp', env_set('GR8_DS_SESS_DRAFTS=1', 'GR8_DS_MTP=0')),
    (D, 'sess-drafts', chain(env_set('GR8_DS_SESS_DRAFTS=1'), arg_set('--s2-sessions', '16'))),
    (D, 'relaxed-no-mtp', env_set('GR8_DS_RELAXED_ACCEPT=ratio:0.5', 'GR8_DS_MTP=0')),
    (D, 'relaxed-topp-no-cands', chain(env_set('GR8_DS_RELAXED_ACCEPT=topp:0.9'), arg_set('--top-p-cands', '0'))),
    (D, 'relaxed-ok', env_set('GR8_DS_RELAXED_ACCEPT=ratio:0.5,topp:0.9')),
    (D, 'dflash', DFLASH),
    (D, 'grammar', arg_add('--grammar-dir', FX + '/grammar')),
    (D, 'dec-split', env_set('GR8_DEC_SPLIT=3')),
    (D, 'engram-mtp', env_set('GR8_DS_ENGRAM=1', 'GR8_DS_ENGRAM_PREFIX=' + FX + '/engram', 'GR8_DS_ENGRAM_MTP=0')),
    (D, 'engram-no-prefix', env_set('GR8_DS_ENGRAM=1')),
    (D, 'engram-layers', chain(env_set('GR8_DS_ENGRAM=1', 'GR8_DS_ENGRAM_PREFIX=' + FX + '/engram'), arg_set('--layers', '0,3'))),
    (D, 'engram-ok', env_set('GR8_DS_ENGRAM=1', 'GR8_DS_ENGRAM_PREFIX=' + FX + '/engram')),
    (D, 'c8pack-one', chain(env_set('GR8_DS_C8PACK=1'), arg_set('--s2-sessions', '1'))),
    (D, 'c8pack-ok', env_set('GR8_DS_C8PACK=1')),
    (D, 'prefill-min', env_set('GR8_DS_PREFILL_MIN=1')),
    (D, 'park-no-reuse', chain(env_set('GR8_DS_PARK=1'), arg_del('--reuse-prefix', 0))),
    (D, 'park-ok', env_set('GR8_DS_PARK=1')),
    (ALL, 'reuse-dflash-offline', chain(OFFLINE, arg_set('--mtp', '0'), DFLASH, arg_add('--reuse-prefix'))),
    (ALL, 'empty-prompt', arg_add('--prompt', '@EMPTY')),
    (ALL, 'prompt-file', arg_add('--prompt', '@' + FX + '/missing-prompt.txt')),
    (ALL, 'two-prompts', arg_add('--prompt', '1,2,3', '--prompt', '4,5,6,7,8,9,10,11,12')),
    (ALL, 'cap-small', arg_set('--cap', '16')),
    (ALL, 'gen-big', arg_add('--gen', '100000000')),
    (ALL, 'offline', chain(OFFLINE, arg_set('--mtp', '0'))),
    (ALL, 'offline-pf-rows', chain(OFFLINE, arg_add('--prompt', '1,2,3,4,5,6,7,8,9,10', '--pf-rows', '-3'))),
    (GF, 'grammar-offline', chain(OFFLINE, arg_add('--grammar-dir', FX + '/grammar'))),
    (G, 'qp-wire-no-roll', env_set('GR8_PF_QP_ROLL=0', 'GR8_PF_QP_WIRE_E=2')),
    (G, 'qp-roll-no-own4', env_set('GR8_PF_OWN=2')),
    (G, 'qp-roll-pack', arg_add('--qp-pack', FX + '/qp.pack')),
    (G, 'qp-roll-no-ctx', env_del('GR8_PF_QP_CTX')),
    (G, 'qp-roll-wire-e', env_set('GR8_PF_QP_WIRE_E=99')),
    (G, 'own4-no-roll', env_set('GR8_PF_QP_ROLL=0')),
    (G, 'own4-packs', chain(env_set('GR8_PF_QP_ROLL=0'), arg_add('--qp-manifest', FX + '/qp.json', '--qp-pack', FX + '/qp.pack'))),
    (G, 'own4-tp6', env_set('GR8_PF_OWN=4', 'GR8_PF_QP_ROLL=2', 'GR8_PF_QP_CTX=65536')),
    (ALL, 'qp-no-own4', chain(env_set('GR8_PF_OWN=2', 'GR8_PF_QP_ROLL=0'), arg_add('--qp-manifest', FX + '/qp.json'))),
    (F + D, 'qp-roll', env_set('GR8_PF_QP_ROLL=2', 'GR8_PF_QP_CTX=65536')),
    # after the parse, in main: the steering profile and the grammar are files
    (ALL, 'steering-missing', arg_add('--steering-profile', FX + '/steering', '--steering-weight-set', 'ab12')),
    (GF, 'grammar-missing', arg_add('--grammar-dir', FX + '/grammar')),
    # a refusal of the parse together with one of main's: the parse's comes first
    (G, 'steering-and-kv', chain(arg_add('--steering-profile', FX + '/steering', '--steering-weight-set', 'ab12'),
                                 arg_add('--kv-keep', 'bogus'))),
    (G, 'grammar-and-qp', chain(arg_add('--grammar-dir', FX + '/grammar'), env_set('GR8_PF_QP_ROLL=0'))),
]


def cases():
    out = []
    base = {}
    for model in MODELS:
        for name, flags in VARIANTS[model]:
            p = plan(model, flags)
            if p:
                out.append((model, name, *p))
        base[model] = plan(model, [])
    # GLM's default list carries GR8_PF_OWN=4, which a TP6 build refuses before most checks: the TP6 shape's mutations
    # start from it with GR8_PF_OWN=3 (TP6's own) and no pair-owner knobs
    env, args = base['glm53']
    base['glm53@4'] = base['glm53']
    base['glm53@6'] = ([e for e in env if not e.startswith(('GR8_PF_OWN=', 'GR8_PF_QP_'))] + ['GR8_PF_OWN=3'], args)
    for models, name, edit in MUTATIONS:
        for model in models:
            for tag in ('glm53@4', 'glm53@6') if model == 'glm53' else (model,):
                env, args = edit(*base[tag])
                out.append((tag, 'mut-' + name, env, args))
    return out


def expand(args, nr):
    sub = {'@IPS': ','.join(f'10.0.0.{i + 1}' for i in range(nr)), '@IPS2': ','.join(f'10.1.0.{i + 1}' for i in range(nr)),
           '@EMPTY': ''}
    return [sub.get(a, a) for a in args]


def read_cases(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            model, name, env, args = line.rstrip('\n').split('\t')
            out.append((model, name, env.split(), args.split()))
    return out


def run(a):
    bin_ = os.path.abspath(a.bin)
    lib = os.path.join(os.path.dirname(bin_), 'lib')
    with open(a.out, 'w') as out, tempfile.TemporaryDirectory() as cwd:
        for model, name, env, args in read_cases(a.cases):
            if model not in (a.model, f'{a.model}@{a.nr}'):
                continue
            e = {'PATH': '/usr/bin:/bin'}
            libs = [d for d in [lib] + a.lib if os.path.isdir(d)]
            if libs:
                e['LD_LIBRARY_PATH'] = ':'.join(os.path.abspath(d) for d in libs)
            e.update(x.split('=', 1) for x in env)
            r = subprocess.run([bin_, '--check-args'] + expand(args, a.nr), env=e, cwd=cwd, capture_output=True,
                               text=True, timeout=60)
            out.write(f'== {name}\nrc {r.returncode}\n')
            out.writelines(f'! {x}\n' for x in r.stderr.splitlines())
            out.writelines(f'> {x}\n' for x in r.stdout.splitlines())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('cases')
    r = sub.add_parser('run')
    r.add_argument('--model', choices=MODELS, required=True)
    r.add_argument('--bin', required=True)
    r.add_argument('--nr', type=int, default=4)
    r.add_argument('--lib', action='append', default=[])
    r.add_argument('--cases', default=os.path.join(REPO, 'engine', 'runtime', 's2', 'tests', 'options_cases.txt'))
    r.add_argument('--out', required=True)
    a = p.parse_args()
    if a.cmd == 'cases':
        print('# scripts/check_args.py cases: MODEL, NAME, ENV, ARGS (tab separated); @IPS, @IPS2, @EMPTY are filled in')
        for model, name, env, args in cases():
            print(f'{model}\t{name}\t{" ".join(env)}\t{" ".join(args)}')
    else:
        run(a)


if __name__ == '__main__':
    main()
