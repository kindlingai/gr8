import unittest

from tools.trace import nicprof


def samples(rows):
    """[(ts seconds, xmit words, rcv words, retrans)] -> sampler lines of one port, the other trouble counters 0."""
    zeros = [0] * (len(nicprof.TROUBLE) - 1)
    return [{'ts': int(t * 1e6), 'ports': {'d0': [x, r, re] + zeros}} for t, x, r, re in rows]


class NicprofTests(unittest.TestCase):
    def test_rates_are_gbps_of_4_byte_words(self):
        ev = nicprof.tracks(2, samples([(0, 0, 0, 0), (1, 250_000_000, 125_000_000, 0)]))
        self.assertEqual(ev, [{'name': 'd0 Gb/s', 'ph': 'C', 'pid': 2, 'ts': 1_000_000, 'args': {'tx': 8.0, 'rx': 4.0}}])

    def test_trouble_is_the_increase_per_interval(self):
        ev = nicprof.tracks(0, samples([(0, 0, 0, 5), (1, 0, 0, 8)]))
        self.assertEqual(ev[1]['args'], {'roce_adp_retrans': 3})

    def test_summary_counts_busy_intervals_only(self):
        lines = nicprof.summary(1, samples([(0, 0, 0, 0), (1, 0, 0, 0), (2, 250_000_000, 0, 2)]))
        self.assertEqual(lines, ['rank 1 d0: tx mean 8.0 peak 8.0 Gb/s, rx mean 0.0 peak 0.0 Gb/s over 1 busy of 2 intervals; '
                                 'roce_adp_retrans 2'])


if __name__ == '__main__':
    unittest.main()
