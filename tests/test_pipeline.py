import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import parse, write_result
from kg_evidence import evidence

ROOT = Path(__file__).resolve().parents[1]

class PipelineTests(unittest.TestCase):
    def test_counts_and_labels(self):
        train = parse(ROOT / 'FoKG26-train.nt.txt')
        test = parse(ROOT / 'FoKG26-test.nt.txt')
        self.assertEqual(len(train), 765)
        self.assertEqual(len(test), 327)
        self.assertTrue(all('y' in r for r in train))
        self.assertTrue(all('y' not in r for r in test))

    def test_cache_distinguishes_unknown_and_empty(self):
        r = parse(ROOT / 'FoKG26-test.nt.txt')[0]
        from kg_evidence import key
        unknown = evidence(r, {})
        observed_empty = evidence(r, {key(r): []})
        exact = evidence(r, {key(r): [r['object'][1:-1]]})
        self.assertEqual(unknown['kg_known'], 0)
        self.assertEqual(observed_empty['kg_known'], 1)
        self.assertEqual(exact['kg_exact'], 1)

if __name__ == '__main__':
    unittest.main()
