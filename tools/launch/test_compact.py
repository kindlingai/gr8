"""The prepare-only compact step uses the same local knob assignment as the engine; a scale names a directory per width."""
import hashlib
import os
import shutil
import tempfile
import types
import unittest
from unittest import mock

import fetch_ranks
import supervisor as sv


class CompactPrepareTests(unittest.TestCase):
    def options(self, extra=''):
        values = {'TP': '4', 'V2': 'experts', 'V2_SLICES': 'slices', 'EXTRA_ENV': extra, 'LAYOUT': 'layout.json'}
        return types.SimpleNamespace(get=values.get)

    def test_unset_and_empty_do_not_prepare(self):
        for env, extra in [({}, ''), ({'GR8_LOAD_COMPACT': ''}, ''),
                           ({'GR8_LOAD_COMPACT': 'copies'}, 'GR8_LOAD_COMPACT=')]:
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(fetch_ranks, 'compact_rank') as run:
                sv.prepare_compact(self.options(extra), 1, types.SimpleNamespace(path='widths.json'))
                run.assert_not_called()

    def test_extra_env_wins_and_layout_is_kept(self):
        with mock.patch.dict(os.environ, {'GR8_LOAD_COMPACT': 'old'}, clear=True), \
                mock.patch.object(fetch_ranks, 'compact_rank', return_value={'checked': 3, 'bad': []}) as run:
            done = sv.prepare_compact(self.options('GR8_LOAD_COMPACT=first GR8_LOAD_COMPACT=copies'), 1,
                                      types.SimpleNamespace(path='widths.json'))
            run.assert_called_once_with('experts', 'copies', 'slices', 4, 1, 'widths.json', 'layout.json')
            self.assertEqual(done, ('copies', {'checked': 3, 'bad': []}))

    def test_source_without_slices_is_refused(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(sv.modelset.ModelError, 'index slices'):
                sv.prepare_compact(self.options('GR8_LOAD_COMPACT=copies'), 1, types.SimpleNamespace(path=None))


class ScaleCompactTests(unittest.TestCase):
    """scale_compact names GR8_LOAD_COMPACT per width; layout_file names the planned layout by its bytes."""
    M = ['10.0.0.1', '10.0.0.2', '10.0.0.3']

    def name(self, extra, layout='{"a": 1}', members=M):
        out = sv.scale_compact(extra, 3, 'bit260', members, layout)
        return sv.opt.extra_env_value(types.SimpleNamespace(get={'EXTRA_ENV': out[1]}.get), 'GR8_LOAD_COMPACT') \
            if out else out

    def test_no_knob_no_words(self):
        for extra in (None, '', 'GR8_LOAD_MERGE_KB=1', 'GR8_LOAD_COMPACT=/c GR8_LOAD_COMPACT='):
            self.assertEqual(sv.scale_compact(extra, 3, 'bit260', self.M, None), [], extra)

    def test_one_name_per_width(self):
        a = self.name('GR8_LOAD_MERGE_KB=1 GR8_LOAD_COMPACT=/c')
        self.assertRegex(a, r'^/c\.tp3-[0-9a-f]{8}$')
        self.assertEqual(self.name('GR8_LOAD_COMPACT=/c'), a)                    # same inputs, same name
        self.assertEqual(self.name(f'GR8_LOAD_COMPACT={a}'), a)                  # re-suffixed, not stacked
        self.assertEqual(self.name('GR8_LOAD_COMPACT=/c.tp4-0123abcd'), a)
        self.assertNotEqual(self.name('GR8_LOAD_COMPACT=/c', layout='{"a": 2}'), a)
        self.assertNotEqual(self.name('GR8_LOAD_COMPACT=/c', members=self.M[::-1]), a)
        out = sv.scale_compact('A=1 G6_LOAD_COMPACT=/c B=2', 3, 'bit260', self.M, '{"a": 1}')
        self.assertEqual(out, ['--extra-env', f'A=1 G6_LOAD_COMPACT={a} B=2'])   # other words kept, in order

    def test_layout_file_is_written_once(self):
        d = tempfile.mkdtemp(prefix='gr8layout-')
        self.addCleanup(shutil.rmtree, d, True)
        text = '{"plan": 1}'
        p = sv.layout_file(d, text)
        self.assertEqual(os.path.basename(p), f'layout-{hashlib.sha256(text.encode()).hexdigest()[:16]}.json')
        st = os.stat(p)
        self.assertEqual(sv.layout_file(d, text), p)
        self.assertEqual((os.stat(p).st_mtime_ns, os.stat(p).st_ino), (st.st_mtime_ns, st.st_ino))
        self.assertNotEqual(sv.layout_file(d, '{"plan": 2}'), p)

    def test_a_done_scale_removes_earlier_widths_copies_only(self):
        t = tempfile.mkdtemp(prefix='gr8compact-')
        self.addCleanup(shutil.rmtree, t, True)
        names = ('c', 'c.tp4-0123abcd', 'c.tp3-89abcdef', 'c.tp2-deadbeef', 'c.tp3-notmine', 'other.tp4-0123abcd')
        for n in names:
            os.makedirs(os.path.join(t, n))
        sup = sv.Supervisor.__new__(sv.Supervisor)
        sup.home, sup.pending, sup.stop_peer = t, {'id': 'x'}, lambda: None
        sup.scale_go = sup.scale_abort = sup.scale_done = mock.Mock()
        sup.o = types.SimpleNamespace(get={'EXTRA_ENV': f'GR8_LOAD_COMPACT={t}/c.tp3-89abcdef'}.get)
        with mock.patch.object(sv, 'log'):
            sup.finish_scale()
        self.assertEqual(sorted(os.listdir(t)), ['c', 'c.tp3-89abcdef', 'c.tp3-notmine', 'other.tp4-0123abcd'])
        sup.pending = {'id': 'y'}   # the operator's own directory: nothing named per width to clear
        sup.o = types.SimpleNamespace(get={'EXTRA_ENV': f'GR8_LOAD_COMPACT={t}/c'}.get)
        sup.finish_scale()
        self.assertEqual(len(os.listdir(t)), 4)
