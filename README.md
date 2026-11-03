# FoKG 2026 — Knowledge Graph Fact Checking


## Team Members

**University:** Paderborn University  
**Course:** Foundations of Knowledge Graphs (FoKG)  
**Project:** Knowledge Graph Fact Checking using DBpedia

| Name | Student ID |
|------|------------|
| Mihir Sane | 4086665 |
| Aryan Solanki | 4106036 |
| Jenish Jai | 4090405 |

---


A reproducible RDF fact-checking mini-project for the Foundations of Knowledge Graphs (FoKG) course. The system assigns each candidate triple a truth-confidence score between 0 and 1, using **DBpedia SPARQL evidence** and a **supervised logistic-regression classifier**.

## Results

| Evaluation | Dataset | ROC AUC |
| --- | --- | ---: |
| Local stratified validation (seed 42; 25% of training) | FoKG26 training | **0.9748** |
| GERBIL Fact Checking (2026-11-03) | FoKG 2026 Test | **0.9657** |
| Course minimum | — | **> 0.60** |

GERBIL experiment: https://gerbil-kbc.aksw.org/gerbil/experiment?id=202611030000

The GERBIL result is from the supplied `result.ttl`, evaluated against **FoKG 2026 Test**. It is separate from the local validation result.

## Input and output

- `FoKG26-train.nt.txt`: **765** reified RDF statements with binary `hasTruthValue` labels.
- `FoKG26-test.nt.txt`: **327** reified RDF statements without labels.
- `result.ttl`: **327** generated RDF triples, one numeric truth-confidence score per original test statement ID.
- `dbpedia_cache.json`: **549** previously retrieved subject/predicate evidence entries. The cache is included so offline runs can be reproduced without dependence on the live DBpedia endpoint.

These statement IDs identify the examples and are **not** used as features; identifiers may repeat across training and test files for different facts.

## Requirements

- Python **3.10+** (developed and executed with Python 3.13 on Windows).
- Python dependencies listed in `requirements.txt` (`numpy`, `scipy`, `scikit-learn`, `requests`).
- Internet is **not required** for the cached evaluation/prediction path. It is required only for fetching new DBpedia evidence.

## Installation and execution

Open a terminal **in the folder containing `enhanced.py`**. In Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe enhanced.py --output result_reproduced.ttl
.\.venv\Scripts\python.exe verify_result.py --result result_reproduced.ttl
```

For macOS/Linux, activate the venv with `source .venv/bin/activate` and use `python` for the same commands. Alternatively on Windows, use the system `python` if a virtual environment is not desired.

Expected program output with included cache (small numerical differences may occur with different library versions):

```text
Knowledge graph coverage: 1090/1092 examples
Validation ROC AUC: 0.9748
Validation accuracy: 0.9062
Wrote 327 predictions to result_reproduced.ttl
```

### Requery DBpedia (optional)

```powershell
python enhanced.py --fetch-kg --output result_new.ttl
```

This queries only missing subject/predicate entries, then updates `dbpedia_cache.json`. The DBpedia public endpoint may return HTTP errors or time out, and newly fetched knowledge can change predictions. **To reproduce the submitted experiment, use the included cache without `--fetch-kg`.** To limit fresh requests, use `--max-queries 20`.

### Lexical-only baseline

```powershell
python main.py --output result_baseline.ttl
```

This baseline does **not** use external graph evidence and is not the system associated with the reported GERBIL score.

## Method

1. Parse RDF reification (statement identifier, subject, predicate, object, type, and for training facts, the truth label).
2. Retrieve the set of URI objects asserted by DBpedia for each subject/predicate pair using SPARQL (`kg_evidence.py`), or load the locally cached retrieved values.
3. Construct features: subject/predicate/object text as character n-gram TF-IDF; one-hot predicate; KG evidence availability, exact object match, whether any object values exist, number of returned object values, and maximum object-label similarity.
4. Fit logistic regression on labeled training statements. For local validation, use a **stratified 75/25 split with random state 42**, train on 75%, and calculate ROC AUC and accuracy on the withheld 25%.
5. Retrain on all 765 training statements and predict confidence scores for the 327 unlabeled test statements.
6. Export one N-Triples-compatible Turtle line per test statement, e.g.:

```ttl
<http://swc2017.aksw.org/task2/dataset/stmt_60> <http://swc2017.aksw.org/hasTruthValue> "0.500000000000"^^<http://www.w3.org/2001/XMLSchema#double> .
```

**Open-world caveat:** absence of an assertion in DBpedia is not proof of falsehood. The model represents missing evidence separately from a query that returned no matching objects. The cache contains data from DBpedia, not ground-truth training labels. This is an evidence-assisted binary classifier, not a general-purpose logical proof engine.

## Verification

```powershell
python -m unittest discover -s tests -v
python verify_result.py --result result.ttl
```

The verifier checks RDF output count, all test IDs exactly once, predicate, datatype and that every numeric score is finite and within [0, 1]. Use it for regenerated output too.

## File overview

| File | Purpose |
|---|---|
| `enhanced.py` | DBpedia-assisted classification, validation and prediction |
| `kg_evidence.py` | SPARQL lookup, caching and graph evidence features |
| `main.py` | RDF parser, output writer and lexical baseline |
| `dbpedia_cache.json` | Cached DBpedia evidence for offline execution |
| `FoKG26-{train,test}.nt.txt` | Course-supplied inputs |
| `result.ttl` | Submitted GERBIL predictions |
| `verify_result.py` | Independent checks of RDF predictions |
| `tests/test_pipeline.py` | Unit tests |
| `requirements.txt` | Python dependencies |

## Submission information

- GERBIL task: **Fact Checking**
- Reference: **FoKG 2026 Test**
- System: `FoKG26-DBpedia-FactChecker-v2`
- Experiment: https://gerbil-kbc.aksw.org/gerbil/experiment?id=202611030000
- University final submission deadline in lecture slides: **January 15, 2027**.

The GERBIL evaluation is a benchmark attempt. Confirm with the course's PANDA instructions whether a repository link, ZIP, and/or any additional report is required for the **final university submission**.

## Team

Add team members' names and university IDs here before the final university submission, if the course requires them.
