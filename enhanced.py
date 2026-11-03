"""DBpedia-assisted supervised fact checking; outputs GERBIL-compatible .ttl."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from main import parse, label, write_result
from kg_evidence import load_cache, save_cache, populate, evidence, ENDPOINT

NUMERIC = ('kg_known', 'kg_exact', 'kg_has_values', 'kg_num_values', 'kg_best_similarity')

class FactChecker:
    def __init__(self, cache):
        self.cache = cache
        self.vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=2, sublinear_tf=True)
        self.pred_enc = OneHotEncoder(handle_unknown='ignore')
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=2500, C=1.0)

    @staticmethod
    def strings(rows):
        return [label(r['subject'])+' '+label(r['predicate'])+' '+label(r['object']) for r in rows]

    def matrix(self, rows, fit=False):
        txt = self.vec.fit_transform(self.strings(rows)) if fit else self.vec.transform(self.strings(rows))
        predicates = np.array([label(r['predicate']) for r in rows]).reshape(-1,1)
        pred = self.pred_enc.fit_transform(predicates) if fit else self.pred_enc.transform(predicates)
        numeric = np.array([[evidence(r, self.cache)[k] for k in NUMERIC] for r in rows], dtype=float)
        numeric = self.scaler.fit_transform(numeric) if fit else self.scaler.transform(numeric)
        return hstack([txt, pred, csr_matrix(numeric)], format='csr')

    def fit(self, rows):
        self.model.fit(self.matrix(rows, fit=True), np.array([r['y'] for r in rows]))
        return self

    def predict(self, rows):
        return self.model.predict_proba(self.matrix(rows))[:, list(self.model.classes_).index(1.)]

def run(args):
    train, test = parse(args.train), parse(args.test)
    if not all('y' in r for r in train):
        raise ValueError('Training file has unlabeled statements')
    if any('y' in r for r in test):
        raise ValueError('Test file must be unlabeled')
    cache = load_cache(args.cache)
    if args.fetch_kg:
        print('Querying DBpedia for subject/predicate evidence...')
        fetched, failed, remaining = populate(train + test, cache, endpoint=args.endpoint,
                                               delay=args.delay, max_queries=args.max_queries,
                                               timeout=args.timeout)
        save_cache(args.cache, cache)
        print(f'KG cache: {fetched} fetched, {failed} failed, {remaining} pairs still uncached')
    elif not cache:
        print('WARNING: KG cache empty. This run is lexical-only; use --fetch-kg to obtain graph evidence.')
    covered = sum(evidence(r,cache)['kg_known'] for r in train+test)
    print(f'Knowledge graph coverage: {int(covered)}/{len(train)+len(test)} examples')

    yy = np.array([r['y'] for r in train])
    tr_idx, va_idx = train_test_split(np.arange(len(train)), test_size=0.25,
                                      stratify=yy, random_state=42)
    tr = [train[i] for i in tr_idx]
    va = [train[i] for i in va_idx]
    model = FactChecker(cache).fit(tr)
    scores = model.predict(va)
    print(f'Validation ROC AUC: {roc_auc_score(yy[va_idx], scores):.4f}')
    print(f'Validation accuracy: {accuracy_score(yy[va_idx], scores>=0.5):.4f}')
    # Test set has no labels; this is a local validation number, NOT GERBIL AUC.
    model = FactChecker(cache).fit(train)
    result = model.predict(test)
    write_result(test, result, args.output)
    print(f'Wrote {len(test)} predictions to {args.output}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', default='FoKG26-train.nt.txt')
    parser.add_argument('--test', default='FoKG26-test.nt.txt')
    parser.add_argument('--output', default='result.ttl')
    parser.add_argument('--cache', default='dbpedia_cache.json')
    parser.add_argument('--fetch-kg', action='store_true', help='Fetch actual DBpedia evidence (needs internet)')
    parser.add_argument('--endpoint', default=ENDPOINT)
    parser.add_argument('--delay', type=float, default=0.1)
    parser.add_argument('--timeout', type=float, default=20)
    parser.add_argument('--max-queries', type=int, default=0, help='Limit new KG requests for incremental runs; 0=all')
    run(parser.parse_args())
