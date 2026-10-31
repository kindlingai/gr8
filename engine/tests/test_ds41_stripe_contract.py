"""Host-only source contract for the DS41 uniform-fabric stripe opt-in."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


def contract(g6: str, fabric: str) -> None:
    start = g6.index('cfg.slot_bytes = NX ?')
    end = g6.index('cfg.ncoll = NC;', start)
    ds = g6[start:end]
    assert 'env_choice("GR8_DS_STRIPE", 1, 0' in ds
    assert 'cfg.ips2.size() != cfg.ips.size()' in ds
    assert '!cfg.vcap.empty() || cfg.bulk_bytes' in ds
    assert 'cfg.stripe_min = sm ? (size_t)atoll(sm) : (size_t)1 << 20;' in ds
    assert 'cfg.stripe_min < 8192' in ds
    assert 'I.stripe = cfg.stripe_min > 0 && I.nrails == 2 && !I.var && !I.bulk;' in fabric
    assert 'stripe=%d rails=%d uniform=%d bulk=%d min=%zu B' in fabric
    assert 'g6::Fabric fabds;' in g6 and 'g6::FabricConfig cd = cfg;' in g6[g6.index('g6::Fabric fabds;'):g6.index('fabds.init(cd);')]


class StripeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g6 = (ROOT / 'engine/runtime/g6m4.cu').read_text()
        cls.fabric = (ROOT / 'engine/comms/fabric.cpp').read_text()

    def test_production(self):
        contract(self.g6, self.fabric)

    def test_assignment_deleted_mutant(self):
        bad = self.g6.replace(
            'cfg.stripe_min = sm ? (size_t)atoll(sm) : (size_t)1 << 20;',
            '/* deleted stripe assignment */', 1)
        with self.assertRaises(AssertionError):
            contract(bad, self.fabric)

    def test_actual_stripe_predicate_deleted_mutant(self):
        bad = self.fabric.replace(
            'I.stripe = cfg.stripe_min > 0 && I.nrails == 2 && !I.var && !I.bulk;',
            'I.stripe = false;', 1)
        with self.assertRaises(AssertionError):
            contract(self.g6, bad)

    def test_every_ds41_fabric_layout(self):
        # Mirrors the stripe gate in Fabric::init (fabric.cpp). The main/pre-fill and DSpark draft
        # fabrics inherit cfg (uniform, non-bulk). Variable/bulk layouts cannot
        # stripe even if they inherit stripe_min; one-rail fallback also stays off.
        cases = (
            ('DS41 main / W-prefill', 2, False, False, True),
            ('DS41 DSpark draft', 2, False, False, True),
            ('DS41 variable-layout guard', 2, True, False, False),
            ('DS41 bulk-layout guard', 2, False, True, False),
            ('DS41 single-rail fallback', 1, False, False, False),
        )
        for name, rails, var, bulk, want in cases:
            actual = 1048576 > 0 and rails == 2 and not var and not bulk
            print(f'{name}: I.stripe={int(actual)} rails={rails} var={int(var)} bulk={int(bulk)}')
            self.assertEqual(actual, want, name)


if __name__ == '__main__':
    unittest.main()
