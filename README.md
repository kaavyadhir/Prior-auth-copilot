# Prior Authorization Decision Support

An API and demo UI that evaluates health-insurance prior-authorization requests against
**published CMS coverage determinations**, and returns one of three outcomes:

| Outcome | When |
|---|---|
| `approve` | Every coverage criterion in the governing policy is explicitly satisfied by the clinical note. |
| `deny` | At least one criterion is explicitly not satisfied, with the failing criterion named. |
| `route_to_human` | The system cannot responsibly decide — and says exactly why. |

Every response cites the verbatim policy passage it was decided from.

**[Try it live &rarr;](https://prior-auth-copilot.onrender.com)**  — free tier, so the
first request after a quiet spell takes a few seconds to wake the service.

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
Approve · confidence 73%
  [met] body-mass index >= 35 kg/m2            "BMI 41.2"
  [met] at least one co-morbidity              "Type 2 diabetes mellitus diagnosed 2019"
  [met] previously unsuccessful with           "Completed 14 months of physician-supervised
        medical treatment for obesity           diet and exercise with maximum 6 kg loss"
  cited: ncd-100.1-bariatric-surgery.txt (0.727)
```

**The same note with the diet sentence deleted:**

```
Route to human reviewer · confidence 49% · incomplete_documentation
  reason: Clinical documentation does not address: The patient must have been
          previously unsuccessful with medical treatment for obesity
  [met]     body-mass index >= 35 kg/m2        "BMI 41.2"
  [met]     at least one co-morbidity          "Type 2 diabetes mellitus diagnosed 2019"
  [unknown] previously unsuccessful with medical treatment for obesity
  cited: ncd-100.1-bariatric-surgery.txt (0.729)
```

The escalation is not a failure mode. It is a work-routing output with a reason: the
reviewer is told *which* criterion is undocumented, so the plan can request one specific
item rather than the provider waiting days to learn the submission was incomplete.

### The case a threshold check gets wrong

A CPAP request with an AHI of 9 — below the headline threshold of 15 in NCD 240.4:

```
Approve · confidence 77%
  [met] qualifying sleep test (PSG, or Type II/III/IV HST with >= 3 channels)
  [met] ordered by treating physician and furnished under supervision
  [met] AHI >= 15, OR AHI 5-14 with documented daytime sleepiness, impaired
        cognition, mood disorders, insomnia, hypertension, ischemic heart
        disease or stroke                      "AHI 9 ... severe daytime
                                                sleepiness and documented hypertension"
  cited: ncd-240.4-cpap-therapy-osa.txt (0.772)
```

A keyword or threshold check reads "9 < 15" and denies. The policy has a second
qualifying branch, and the note satisfies it.

### Confidence is arithmetic, not a vibe

```
confidence = retrieval_similarity × (criteria_resolved / criteria_total)
```

73% is `0.727 × 3/3`. 49% is `0.729 × 2/3`. A strong policy match whose criteria are
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

`app/upstream.py` distinguishes three failure classes, shared by generation and embedding:

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

**Embeddings moved from a local model to the provider's API.** The first version ran
sentence-transformers locally, which needed PyTorch and roughly 500 MB resident - more
than any free container host allows. Calling the embedding API instead took the image
from ~1.5 GB to ~300 MB and runtime memory to about 150 MB.

The trade-off is real: retrieval now needs a network round trip and depends on the same
upstream as generation. It is mitigated by sharing the retry/fallback policy in
`app/upstream.py`, and by embedding the corpus once at startup rather than per request.

**Indexed passages and queries use different embedding task types.** `RETRIEVAL_DOCUMENT`
for the corpus, `RETRIEVAL_QUERY` for the incoming request. Asymmetric retrieval
embeddings measurably beat using one type for both.

**The embedding chain has no fallback, on purpose.** Generation falls back across seven models; embedding falls back to nothing. The available alternative does not accept retrieval task types and aggregates multi-input requests into a single vector, so falling back to it would build an index whose semantics differ from the queries run against it. That degrades retrieval silently. Failing loudly and escalating every request to a human is the safer failure.

**Batch embedding verifies it got one vector per input.** Some embedding models return a
single *aggregated* vector for a multi-input request. Accepting that silently would
misalign every passage with its vector and corrupt the index in a way that looks like
poor retrieval rather than a bug, so the code checks and falls back to one request per
text. See `tests/test_embeddings.py`.

**Retrieval queries combine codes and prose.** Codes alone match the procedure table but
miss the criteria text; the note alone matches symptom language but drifts to the wrong
procedure.

**Ingestion reports empty files.** A scanned PDF has no text layer and yields nothing. Left
unreported, that looks identical to a broken retriever at query time.

---

## Known limitations

**Criterion extraction needed two revisions to get right.** The first version treated
every policy sentence as a potential criterion. On the CPAP case it correctly found the
hard rule — AHI 9 qualifies under the "5–14 with documented symptoms" branch — but also
extracted a footnote requiring a minimum event count *when the AHI was computed from
under two hours of recorded sleep*. The note never said the study was short, so the rule
did not apply, yet it landed as `unknown` and escalated a request that should have been
approved.

Two prompt revisions fixed it: alternative qualifying paths now collapse into a single
criterion rather than several, and conditional rules whose triggering circumstance is
absent from the note are not extracted at all. That case now approves at 77%. The more
robust fix, if the corpus grew, would be a second pass asking per criterion whether its
precondition is present, rather than relying on prompt instructions.

**No labelled evaluation set.** Correctness is currently argued from unit tests and
inspected cases. Measuring the escalation rate and the false-approval rate needs a corpus
of labelled requests.

**In-process retry only.** Fine for one request, wrong for a queue: a dead upstream would
be hammered by every concurrent caller. That wants a circuit breaker.

**Three policies.** Enough to demonstrate the mechanism; a production corpus is thousands
of documents, where exact cosine search would need replacing.

**Every request costs two API round trips** — one to embed the query, one to evaluate
criteria. A local embedding model would remove the first, at the memory cost that made
this change necessary. A query-embedding cache would remove most of it for repeated
procedures.

---

## Data

Coverage policies are US government public documents — CMS National Coverage
Determinations 100.1 (bariatric surgery), 240.4 (CPAP therapy) and 240.4.1 (sleep
testing), retrieved from the Medicare Coverage Database. No patient data is used anywhere;
all requests in tests, examples and the demo UI are synthetic.

## Stack

Python · FastAPI · Pydantic · Google Gemini (generation + embeddings) · NumPy · pytest ·
Docker · GitHub Actions
