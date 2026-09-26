# Prior Authorization Decision Support

An API and demo UI that evaluates health-insurance prior-authorization requests against
**published CMS coverage determinations**, and returns one of three outcomes:

| Outcome | When |
|---|---|
| `approve` | Every coverage criterion in the governing policy is explicitly satisfied by the clinical note. |
| `deny` | At least one criterion is explicitly not satisfied, with the failing criterion named. |
| `route_to_human` | The system cannot responsibly decide — and says exactly why. |

Every response cites the verbatim policy passage it was decided from.

---

## The problem

Before an insurer pays for an expensive service — bariatric surgery, a CPAP machine, an
MRI — the provider must get it pre-approved. Today a clinical reviewer reads the note,
reads the plan's coverage policy, and decides. It is slow, and two reviewers can reach
different conclusions on the same file.

The obvious thing to build is a model that reads both and answers. That is also the wrong
thing to build, because **a model asked "should we approve this?" will always answer** —
and in prior authorization a wrong approval costs money while a wrong denial delays
someone's care.

## The design

```
  clinical note + procedure/diagnosis codes
                    │
                    ▼
      retrieval over policy corpus ──── weak match? ───► route_to_human
                    │                                    (LLM never called)
                    ▼
      LLM: per-criterion judgement            ← the model's only job
        met │ not_met │ unknown
                    │
                    ▼
      deterministic rule engine               ← the decision's only author
                    │
                    ▼
      approve │ deny │ route_to_human
```

**The model never decides.** It is asked only to list the criteria the retrieved policy
imposes and judge each against the note, with `unknown` as a first-class answer for
anything the note is silent on. A plain Python rule engine maps those judgements to an
outcome.

That split buys three things:

1. **Auditability** — every outcome traces to a named rule and a quoted policy passage.
2. **Testability** — the rules are pure functions. All 31 tests run with no API key and no
   model. See `tests/test_decision.py`.
3. **Safety** — approval requires *every* criterion to be explicitly met, so no prompt,
   accidental or adversarial, can produce an approval the policy does not support.

---

## What it looks like

### Same patient, one sentence apart

**Full note** — BMI 41.2, type 2 diabetes, 14 months of failed supervised diet:

```
Approve · confidence 58%
  [met] body-mass index >= 35 kg/m2            "52F, BMI 41.2"
  [met] at least one co-morbidity              "Type 2 diabetes mellitus diagnosed 2019"
  [met] previously unsuccessful with           "Completed 14 months of physician-supervised
        medical treatment for obesity           diet and exercise with maximum 6 kg loss"
  cited: ncd-100.1-bariatric-surgery.txt (0.577)
```

**The same note with the diet sentence deleted:**

```
Route to human reviewer · confidence 37% · incomplete_documentation
  reason: Clinical documentation does not address: The beneficiary has been
          previously unsuccessful with medical treatment for obesity
  [met]     body-mass index >= 35 kg/m2        "BMI 41.2"
  [met]     at least one co-morbidity          "Type 2 diabetes mellitus diagnosed 2019"
  [unknown] previously unsuccessful with medical treatment for obesity
  cited: ncd-100.1-bariatric-surgery.txt (0.559)
```

The escalation is not a failure mode. It is a work-routing output with a reason: the
reviewer is told *which* criterion is undocumented, so the plan can request one specific
item rather than the provider waiting days to learn the submission was incomplete.

### Confidence is arithmetic, not a vibe

```
confidence = retrieval_similarity × (criteria_resolved / criteria_total)
```

58% is `0.577 × 3/3`. 37% is `0.559 × 2/3`. A strong policy match whose criteria are
mostly `unknown` is not a confident decision, and neither is a fully-resolved evaluation
against a passage that barely matched.

---

## Escalation paths

| `escalation_reason` | Meaning |
|---|---|
| `insufficient_policy_match` | No passage scored above the retrieval floor. The LLM is never called. |
| `policy_not_applicable` | Passages cleared the floor, but do not govern this procedure. |
| `no_criteria_extracted` | The matched policy yielded no evaluable criteria. |
| `incomplete_documentation` | Nothing failed, but the note does not address a required criterion. |
| `evaluation_unavailable` | Upstream failure. The service escalates rather than defaulting to an answer. |

That last one is not theoretical. During development Gemini returned `503` for four
consecutive models; the service returned `route_to_human` with a cited policy rather than
guessing or crashing.

The first two are independent safety nets, and the second catches what the first lets
through. A cataract request scores 0.380 against the sleep-apnea policy — above the 0.35
floor, so the LLM is called — and the model then reports `policy_applies=false`, producing
`policy_not_applicable` at 38% confidence with no criteria evaluated.

---

## Upstream resilience

`app/llm.py` distinguishes three failure classes:

- **Transient** (429, 5xx) — retry the same model with exponential backoff.
- **Model unavailable** (404) — skip immediately. Providers retire models on their own
  schedule and keep advertising them in list-models, so retrying cannot help.
- **Everything else** (400, 401, 403) — raise at once. A malformed request or a bad key is
  a bug, and failing over would hide it behind pointless retries.

Tested against a fake client and a fake clock, so the suite asserts the backoff schedule
is `[1.0, 2.0]` without sleeping (`tests/test_llm_resilience.py`).

---

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

copy .env.example .env          # then add your Gemini API key
python scripts/check_llm.py     # verifies key, model access and schema, layer by layer
python scripts/ingest_policies.py
uvicorn app.main:app --reload
```

- Demo UI: <http://localhost:8000/>
- API docs: <http://localhost:8000/docs>

```bash
pytest -v
ruff check app tests
```

CI runs lint, tests and a Docker build on every push.

---

## Design notes

**Storage is behind an interface.** `app/store.py` defines `VectorStore`; `NumpyStore` is
the file-backed implementation. At this corpus size exact cosine search is faster and
simpler than an ANN index, and the interface means retrieval and decision code does not
change if that stops being true.

**Chunking overlaps by design.** Coverage criteria are numbered lists, and those lists
cross page boundaries constantly. A hard cut drops the tail of the list, which silently
turns a `not_met` into an `unknown`. Oversized paragraphs are split on sentence
boundaries — PDF extraction routinely returns a whole page as one blob.

**The embedding model loads once, at startup.** Loading it per request is the standard way
to turn a 200 ms endpoint into a 30 s one.

**Retrieval queries combine codes and prose.** Codes alone match the procedure table but
miss the criteria text; the note alone matches symptom language but drifts to the wrong
procedure.

**Ingestion reports empty files.** A scanned PDF has no text layer and yields nothing. Left
unreported, that looks identical to a broken retriever at query time.

---

## Known limitations

**Over-escalation on conditional policy text.** Evaluating a CPAP request, the system
correctly finds the harder rule — an AHI of 9 qualifies under the "5–14 events with
documented symptoms" branch, which a naive threshold check would deny. But it also
extracts a footnote requiring a minimum event count *when the AHI was computed from under
two hours of recorded sleep*. The note never says the study was short, so the rule should
not apply; instead it becomes `unknown` and escalates a request that should be approved.

Two prompt revisions have narrowed this — alternative qualifying paths now collapse into a
single criterion, dropping the extracted count from five to four — but distinguishing a
genuine criterion from one whose precondition is absent is not fully solved. A more robust
fix than prompting would be a second pass that asks, per criterion, whether its triggering
circumstance is present at all.

**No labelled evaluation set.** Correctness is currently argued from unit tests and
inspected cases. Measuring the escalation rate and the false-approval rate needs a corpus
of labelled requests.

**In-process retry only.** Fine for one request, wrong for a queue: a dead upstream would
be hammered by every concurrent caller. That wants a circuit breaker.

**Three policies.** Enough to demonstrate the mechanism; a production corpus is thousands
of documents, where exact cosine search would need replacing.

---

## Data

Coverage policies are US government public documents — CMS National Coverage
Determinations 100.1 (bariatric surgery), 240.4 (CPAP therapy) and 240.4.1 (sleep
testing), retrieved from the Medicare Coverage Database. No patient data is used anywhere;
all requests in tests, examples and the demo UI are synthetic.

## Stack

Python · FastAPI · Pydantic · sentence-transformers · Google Gemini · NumPy · pytest ·
Docker · GitHub Actions
