"""FoKG26 baseline: parse reified RDF facts, validate and export GERBIL predictions.

This is a lexical supervised baseline, not yet an external knowledge-graph verifier.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

LINE = re.compile(r'^<([^>]*)>\s+<([^>]*)>\s+(.*)\s+\.$')
RDF_NS = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
VALUE_NS = 'http://swc2017.aksw.org/hasTruthValue'
OUTPUT_TYPE = 'http://www.w3.org/2001/XMLSchema#double'

def parse(path: str):
    statements = {}
    for lineno, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        m = LINE.fullmatch(line.strip())
        if not m:
            raise ValueError(f'{path}:{lineno}: invalid N-Triples statement')
        statement, predicate, obj = m.groups()
        if predicate.startswith(RDF_NS):
            property_name = predicate[len(RDF_NS):]
        elif predicate == VALUE_NS:
            property_name = 'truth'
        else:
            raise ValueError(f'{path}:{lineno}: unexpected property {predicate}')
        data = statements.setdefault(statement, {'id': statement})
        if property_name in data:
            raise ValueError(f'{path}:{lineno}: duplicate property {property_name}')
        data[property_name] = obj
    for uri, x in statements.items():
        if not all(k in x for k in ('subject','predicate','object','type')):
            raise ValueError(f'{uri}: missing RDF fact fields')
        if x['type'] != f'<{RDF_NS}Statement>':
            raise ValueError(f'{uri}: unexpected rdf:type')
        for field in ('subject','predicate','object'):
            if not (x[field].startswith('<') and x[field].endswith('>')):
                raise ValueError(f'{uri}: expected URI for {field}')
        if 'truth' in x:
            raw = x['truth']
            m = re.fullmatch(r'"([01](?:\.0+)?)"\^\^<http://www.w3.org/2001/XMLSchema#double>',raw)
            if not m:
                raise ValueError(f'{uri}: invalid truth value {raw}')
            x['y'] = float(m.group(1))
    return list(statements.values())

def label(uri: str) -> str:
    from urllib.parse import unquote
    return unquote(uri.strip('<>').rsplit('/',1)[-1]).replace('_',' ').lower()

def features(records):
    # Different text perspectives help with sparse labeled examples.
    return [
        {'sub':label(r['subject']), 'obj':label(r['object']),
         'pair':label(r['subject'])+' [SEP] '+label(r['object']),
         'pred':label(r['predicate'])} for r in records
    ]

def fit_model(records):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.pipeline import FeatureUnion
    from sklearn.preprocessing import FunctionTransformer
    # Text features retain names, predicate acts as a separate categorical feature.
    from scipy.sparse import hstack
    texts = [label(r['subject'])+' '+label(r['predicate'])+' '+label(r['object']) for r in records]
    predicates = [label(r['predicate']) for r in records]
    vectorizer = TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,sublinear_tf=True)
    enc = OneHotEncoder(handle_unknown='ignore')
    X = hstack([vectorizer.fit_transform(texts), enc.fit_transform(np.array(predicates).reshape(-1,1))]).tocsr()
    y = np.array([r['y'] for r in records])
    model = LogisticRegression(max_iter=2000,C=1.0)
    model.fit(X,y)
    return model,vectorizer,enc

def predict(model_bundle, records):
    from scipy.sparse import hstack
    model,vec,enc = model_bundle
    texts = [label(r['subject'])+' '+label(r['predicate'])+' '+label(r['object']) for r in records]
    preds = [label(r['predicate']) for r in records]
    X = hstack([vec.transform(texts),enc.transform(np.array(preds).reshape(-1,1))]).tocsr()
    return model.predict_proba(X)[:,list(model.classes_).index(1.0)]

def write_result(records, probabilities, output):
    if len(records)!=len(probabilities):
        raise ValueError('number of results does not match test facts')
    with open(output,'w',encoding='utf-8') as file:
        for record,score in zip(records,probabilities):
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f'invalid score for {record["id"]}')
            file.write(f'<{record["id"]}> <{VALUE_NS}> "{float(score):.12f}"^^<{OUTPUT_TYPE}> .\n')

def run(train_path,test_path,out_path):
    train = parse(train_path)
    test = parse(test_path)
    if any('y' not in r for r in train):
        raise ValueError('training facts must all have labels')
    if any('y' in r for r in test):
        raise ValueError('test file must not contain truth labels')
    # Statement identifiers are reused across splits for different fact triples;
    # their numeric suffixes must never be used as predictive features.
    y = np.array([r['y'] for r in train])
    indices = np.arange(len(train))
    i_train,i_valid = train_test_split(indices,test_size=0.25,random_state=42,stratify=y)
    partial = [train[i] for i in i_train]
    valid = [train[i] for i in i_valid]
    scores = predict(fit_model(partial),valid)
    print(f'Validation ROC AUC: {roc_auc_score(y[i_valid],scores):.4f}')
    print(f'Validation accuracy at 0.5: {accuracy_score(y[i_valid],scores>=0.5):.4f}')
    test_scores = predict(fit_model(train),test)
    write_result(test,test_scores,out_path)
    print(f'Wrote {len(test)} predictions to {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',default='FoKG26-train.nt.txt')
    parser.add_argument('--test',default='FoKG26-test.nt.txt')
    parser.add_argument('--output',default='result.ttl')
    args = parser.parse_args()
    run(args.train,args.test,args.output)
