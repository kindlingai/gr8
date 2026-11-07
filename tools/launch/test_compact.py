"""The prepare-only compact step uses the same local knob assignment as the engine."""
import os
import types
import unittest
from unittest import mock

import fetch_ranks
import launch


class CompactPrepareTests(unittest.TestCase):
    def options(self, extra=''):
        values = {'TP': '4', 'V2': 'experts', 'V2_SLICES': 'slices', 'EXTRA_ENV': extra, 'LAYOUT': 'layout.json', 'V2_DENSE': 'release'}
        return types.SimpleNamespace(get=values.get)

    def test_unset_and_empty_do_not_prepare(self):
        for env, extra in [({}, ''), ({'GR8_LOAD_COMPACT': ''}, ''),
                           ({'GR8_LOAD_COMPACT': 'copies'}, 'GR8_LOAD_COMPACT=')]:
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(fetch_ranks, 'compact_rank') as run:
                launch.prepare_compact(self.options(extra), 1, types.SimpleNamespace(path='widths.json'))
                run.assert_not_called()

    def test_extra_env_wins_and_layout_is_kept(self):
        with mock.patch.dict(os.environ, {'GR8_LOAD_COMPACT': 'old'}, clear=True), \
                mock.patch.object(fetch_ranks, 'compact_rank', return_value={'checked': 3, 'bad': []}) as run, \
                mock.patch.object(launch.opt, 'engine_plan', return_value={'env': ['GR8_PF_OWN=3', 'GR8_KVB=fp8']}):
            launch.prepare_compact(self.options('GR8_LOAD_COMPACT=first GR8_LOAD_COMPACT=copies'), 1,
                                   types.SimpleNamespace(path='widths.json'))
            run.assert_called_once_with('experts', 'copies', 'slices', 4, 1, 'widths.json', 'layout.json',
                                        dense='release', dense_form='trellis', qo=True, kvb='fp8')

    def test_source_without_slices_is_refused(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(launch.modelset.ModelError, 'index slices'):
                launch.prepare_compact(self.options('GR8_LOAD_COMPACT=copies'), 1, types.SimpleNamespace(path=None))
