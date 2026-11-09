import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import query


class WindowTests(unittest.TestCase):
    def rows(self, text, cols=20):
        with contextlib.redirect_stdout(io.StringIO()):
            w = query.Window(cols, 10, '')
        w.spans = [('', text)]
        return [''.join(c for _, c in r) for r in w.wrap()]

    def test_latin_breaks_at_a_space_and_cjk_anywhere(self):
        self.assertEqual(self.rows('Rotate the plant occasionally so it grows evenly. 如果叶子变黄或徒长'),
                         ['Rotate the plant ', 'occasionally so it ', 'grows evenly. 如果叶', '子变黄或徒长'])

    def test_a_word_longer_than_the_row_is_cut(self):
        self.assertEqual(self.rows('a ' + 'x' * 25), ['a ', 'x' * 20, 'x' * 5])


class ReplayTests(unittest.TestCase):
    def test_replay_prints_thinking_dim_then_the_answer(self):
        with tempfile.TemporaryDirectory() as d:
            rec = os.path.join(d, 'r.json')
            with open(rec, 'w') as f:
                json.dump({'events': [[0, 'think', 'hmm'], [0, 'answer', '天空'], [0, 'answer', ' blue']]}, f)
            out = subprocess.run([sys.executable, os.path.join(HERE, 'query.py'), '--replay', rec,
                                  '--record', os.path.join(d, 'again.json')],
                                 capture_output=True, text=True, check=True).stdout
            self.assertEqual(out, query.OFF + query.DIM + 'hmm' + query.OFF + '\n\n天空 blue' + query.OFF + '\n')
            self.assertEqual(json.load(open(os.path.join(d, 'again.json')))['events'],
                             [[0, 'think', 'hmm'], [0, 'answer', '天空'], [0, 'answer', ' blue']])


if __name__ == '__main__':
    unittest.main()
