"""Optional, cached DBpedia evidence retrieval for FoKG26.

No gold labels or test scores are used for queries. Missing evidence is distinct
from an observed graph without an exact match (open-world assumption).
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

ENDPOINT = 'https://dbpedia.org/sparql'

def iri(token):
    if not token.startswith('<') or not token.endswith('>'):
        raise ValueError(f'Expected RDF IRI: {token}')
    v = token[1:-1]
    if any(c in v for c in ('>', '<', '"', ' ', '\n', '\r')):
        raise ValueError('Invalid IRI')
    return v

def key(record):
    return iri(record['subject']) + '\t' + iri(record['predicate'])

def load_cache(path):
    file = Path(path)
    if not file.exists():
        return {}
    data = json.loads(file.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('KG cache must be a JSON mapping')
    return data

def save_cache(path, cache):
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_suffix(file.suffix + '.tmp')
    temp.write_text(json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True), encoding='utf-8')
    temp.replace(file)

def fetch_one(s, p, endpoint=ENDPOINT, timeout=20):
    """Query explicitly asserted DBpedia predicate values for an (s,p) pair.

    Returns only URI objects, so literals do not masquerade as resource matches.
    """
    query = f'SELECT DISTINCT ?o WHERE {{ <{s}> <{p}> ?o . FILTER(isIRI(?o)) }} LIMIT 500'
    data = urlencode({'query': query, 'format': 'application/sparql-results+json'}).encode('utf-8')
    req = Request(endpoint, data=data, headers={
        'Accept': 'application/sparql-results+json',
        'User-Agent': 'FoKG26-research-fact-checker/1.0 (academic project)',
    })
    with urlopen(req, timeout=timeout) as response:
        payload = json.load(response)
    return sorted({row['o']['value'] for row in payload['results']['bindings']
                   if row.get('o', {}).get('type') == 'uri'})

def populate(records, cache, endpoint=ENDPOINT, delay=0.1, max_queries=0, timeout=20):
    """Fetch uncached subject/predicate pairs, persist after each 25 requests.

    max_queries=0 means no cap. Failed queries are NOT stored as empty results.
    """
    pairs = {key(r) for r in records}
    missing = sorted(pairs - cache.keys())
    if max_queries:
        missing = missing[:max_queries]
    done, failures = 0, 0
    for pair in missing:
        s, p = pair.split('\t', 1)
        try:
            cache[pair] = fetch_one(s, p, endpoint=endpoint, timeout=timeout)
            done += 1
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            failures += 1
            if failures <= 5:
                print(f'  DBpedia query failed: {type(exc).__name__}: {exc}')
        if delay:
            time.sleep(delay)
    return done, failures, len(pairs - cache.keys())

def evidence(record, cache):
    pair = key(record)
    if pair not in cache:
        return {'kg_known': 0., 'kg_exact': 0., 'kg_has_values': 0.,
                'kg_num_values': 0., 'kg_best_similarity': 0.}
    values = cache[pair]
    obj = iri(record['object'])
    exact = float(obj in values)
    from difflib import SequenceMatcher
    from urllib.parse import unquote
    def name(v):
        return unquote(v.rsplit('/', 1)[-1]).replace('_', ' ').lower()
    sim = max((SequenceMatcher(None, name(obj), name(v)).ratio() for v in values), default=0.)
    return {'kg_known': 1., 'kg_exact': exact, 'kg_has_values': float(bool(values)),
            'kg_num_values': float(min(len(values), 20)), 'kg_best_similarity': sim}
