"""Check the GERBIL prediction RDF file against all test statement identifiers."""
import argparse
import math
import re
from collections import Counter
from main import parse, VALUE_NS, OUTPUT_TYPE

LINE = re.compile(r'^<([^>]*)>\s+<([^>]*)>\s+"([^"]+)"\^\^<([^>]*)>\s+\.$')

def verify(test_file: str, result_file: str) -> None:
    expected = {r['id'] for r in parse(test_file)}
    with open(result_file, encoding='utf-8') as f:
        lines = [s.strip() for s in f if s.strip()]
    ids = []
    for number, line in enumerate(lines, 1):
        m = LINE.fullmatch(line)
        if not m:
            raise ValueError(f'Line {number}: invalid RDF output syntax')
        ident, predicate, raw, dtype = m.groups()
        if predicate != VALUE_NS or dtype != OUTPUT_TYPE:
            raise ValueError(f'Line {number}: wrong predicate/datatype')
        score = float(raw)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f'Line {number}: truth score outside [0,1]')
        ids.append(ident)
    counts = Counter(ids)
    if len(lines) != len(expected) or set(ids) != expected or any(n != 1 for n in counts.values()):
        raise ValueError(f'Mismatch: {len(lines)} triples, {len(expected)} expected; missing={len(expected-set(ids))}, unexpected={len(set(ids)-expected)}, duplicates={sum(n-1 for n in counts.values())}')
    print(f'PASS: {len(lines)} predictions; identifiers, predicate, datatype and scores valid')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default='FoKG26-test.nt.txt')
    parser.add_argument('--result', default='result.ttl')
    a = parser.parse_args()
    verify(a.test, a.result)
