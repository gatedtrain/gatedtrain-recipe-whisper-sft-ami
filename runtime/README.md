# Area "audio": Whisper base.en SFT on AMI meeting speech with the HF Seq2SeqTrainer (prototype, unpromotable)

Multimodal prototype, scope row 5 ("Audio | Hugging Face Trainer | Whisper-tiny or base SFT, per-utterance
WER"; fourth in the run order, hence the original directory name `area4-whisper`), gatedtrain's `MULTIMODAL-SPEC.md` §6
row 5. Runtime kind
`venv_in_container` (MM-32), as areas 1 and 3: a pinned CUDA base image plus this directory's host venv, built from
the committed `uv.lock` and mounted read-only. Nothing here starts a container or uses the GPU; an operator runs
the commands below through `ext-launch` and `gpu-step.sh` after approval. Numbers marked **estimate** are unmeasured
on the GPU; the smoke run measures them.

Status (2026-11-13): runtime prepared, CPU sanity checks and a CPU power pilot passed. Proposals are marked
"proposal" and collected in "Open questions".

## Pins

| Item | Value |
|---|---|
| Base image | the area-1 image, `docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266` (as area 3) |
| Python | CPython 3.12.16, `<data-root>/python/cpython-3.12.16-linux-x86_64-gnu` (shared with areas 1-3, unchanged) |
| Lock | `uv.lock` (60 packages, `exclude-newer = 2026-10-29`, Linux x86_64), sha256 `bf53ae6b…bd690` (the published run's lock, `520d9773…430176`, differs only in the project name; `../RESULT.md`) |
| Venv | `.venv` (`uv sync --frozen --compile-bytecode`, copy link mode, 5.7 GB, 34,303 files) |
| Framework | **transformers 5.21.0** (`Seq2SeqTrainer`), accelerate 1.15.0, torch 2.15.0 (PyPI build, CUDA 13.2 runtime wheels, as area 3), safetensors 0.8.0, tokenizers 0.23.3, huggingface-hub 2.3.0, numpy 2.5.4, pyarrow 26.0.0 (host data step only), soundfile 0.14.0 (libsndfile 1.2.2), regex 2026.9.29 (the Whisper normalizer) |

Rebuild (host, no GPU): `UV_PROJECT_ENVIRONMENT=$PWD/.venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
uv sync --frozen --compile-bytecode --python <data-root>/python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12`
(the build used a temporary `UV_CACHE_DIR` inside this directory, deleted afterwards; no global cache was written).
CUDA: same torch build and fallback as area 3 README "Pins".

## Model choice (measured on CPU, 500 registered items, greedy)

| Base | Mean per-utterance WER | sd | at 0 | max | corpus WER | CPU s/item | Verdict |
|---|---|---|---|---|---|---|---|
| `openai/whisper-tiny.en` @ `87c71024…9063` | 0.207 | 0.673 | 25 % | **14.67** (a "be here, be here, …" loop on a 9-word utterance) | 0.190 | 0.09 | not chosen: hallucination loops make the per-utterance WER heavy-tailed (sd 0.18 without that one item), which ruins power |
| **`openai/whisper-base.en` @ `911407f4214e0e1d82085af863093ec0b66f9cd6`** | **0.153** | 0.168 | 30 % | 1.0 | 0.150 | 0.18 | **chosen** (Apache-2.0, ungated, 74 M parameters, 290 MB) |

Both are far below the GPU budget (Profiles), so cost did not decide; the tail did. tiny.en was removed after the
pilot (its revision is noted in `fetch_models.py`).

## Why fine-tuning measurably lowers WER (and what it measures)

Data: **AMI Meeting Corpus, individual headset microphones** (`edinburghcstr/ami`, config `ihm`, Kaldi s5
segmentation and text; CC-BY-4.0, ungated). Spontaneous multi-party meetings, mostly **non-native English speakers**,
transcribed **verbatim**: repetitions, false starts, partial words ("IF YOU IF YOU", "THE THE", "COMPU"). Whisper was
trained to produce clean, readable transcripts, so on AMI it mainly **deletes** disfluent words: baseline errors on
the 500 items are 711 deletions, 423 substitutions, 74 insertions. SFT on AMI training meetings teaches the verbatim
convention and the meeting vocabulary and accents. Measured (CPU pilot, 60 steps x 16 = 960 examples, lr 1e-5):
mean per-utterance WER 0.153 → **0.120**, corpus WER 0.150 → 0.112, deletions 711 → 247 (substitutions 423 → 526,
insertions 74 → 124), 191 items better, 75 worse, 234 tied; paired z ≈ 6.3.

**Disclosure (proposal for the registration text):** a large part of the gain is transcription-convention learning
(keeping disfluencies the reference keeps), as Tier B's answer-format learning; the Whisper English normalizer removes
fillers (uh, um, mm, hmm) and casing/punctuation but not repetitions. A pass shows "the model transcribes AMI-style
meeting speech of unseen speakers closer to the AMI references", not a general accuracy gain. Whisper's training data
is undisclosed; AMI may be in it. That affects both arms equally (the gate compares base and cut on the same items)
and is not checkable.

## Data (`fetch_models.py`, `make_data.py`)

Downloads (2026-11-12): fresh `HF_HOME=hf-home/` (still empty), `token=False`, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`,
Xet off, plain HTTPS; a token variable in the environment is removed from the process before huggingface_hub is
imported, its value never read. Plain files only; huggingface_hub's download metadata moved to `hf-meta/`.
`models/MODELS.json` (sha256 `9bc54cf0…bdb63`) and `data-source/DATA.json` (sha256 `c909aa8d…863b1`) hold every
file's SHA-256 and size; the safetensors and parquet hashes equal the Hub's LFS hashes. `fetch_models.py --verify`
re-hashes.

| Source | Revision | Files kept | Size |
|---|---|---|---|
| `models/whisper-base.en` | `openai/whisper-base.en` @ `911407f4…cd6` (Apache-2.0) | config, generation config, `model.safetensors` (`d4dd5542…71db5`), preprocessor config, tokenizer files, `normalizer.json` (`bf1c507d…26dd`); no `.bin`, `.h5`, `.msgpack` | 294 MB, 11 files |
| `data-source/ami-ihm/` | `edinburghcstr/ami` @ `46f28f2503e2ec48f8867a84eef356c70476beab` (CC-BY-4.0) | `README.md`; train shards 00000 (EN2001), 00010 (ES2006-ES2007), 00026 (IS1002-IS1003), 00034 (TS3006-TS3007) — one per recording series, chosen from the shards' parquet statistics; all 4 test shards (EN2002, ES2004, IS1009, TS3003) | 2.49 GB, 9 files |

`make_data.py` (seed 20261112, `numpy default_rng(SeedSequence([0xA4D10, seed]))`, pools sorted by utterance id):

| Set | Rule | Result | SHA-256 |
|---|---|---|---|
| Training shard `data/train.jsonl` + `data/audio/*.wav` | AMI **train** split, >= 4 words (drops "YEAH", "OKAY", "MM-HMM" backchannels), <= 30 s; 3,000 of the 5,894 drawn | 3,000 utterances, 3.42 h, 29 speakers; EN2001 841, ES2006 333, ES2007 418, IS1002 213, IS1003 513, TS3006 59, TS3007 623 | `2278c534…7b372` |
| Items `items/items.jsonl` + `items/audio/*.wav` | AMI **test** split, >= 8 words after the normalizer, <= 30 s; 500 of 4,393 drawn | 500 utterances, 0.70 h, 16 speakers; EN2002 172, ES2004 107, IS1009 105, TS3003 116; 8-46 normalized words | `94d0bea4…d29fc` |

Disjoint by meeting (AMI's own split) and **by speaker** (0 shared of 29 + 16, checked against the whole training
pool); exact raw-text and normalized-text overlap of items with the training pool: 0 and 0 (any would have been
dropped, `dropped_text_overlap` 0). Audio files are the exact WAV bytes of the parquet cells (16 kHz mono float32);
each row carries the file's SHA-256, re-checked by the trainer and the evaluator at load. Meta: `data/data_meta.json`,
`items/item_set_meta.json`. Item row: `{item, text, reference_normalized, audio, audio_sha256, speaker_id,
meeting_id, duration, n_words}`; training row: `{utt_id, text, speaker_id, meeting_id, audio, audio_sha256, duration,
n_words}`. Both use the field **`text`** for the raw AMI transcript (the MM-11 overlap field).

## Trainer (`train_whisper.py`)

transformers `Seq2SeqTrainer`, full fine-tune of `WhisperForConditionalGeneration`, token cross-entropy on the
transcript (AMI text lower-cased with one leading space; the tokenizer's `<|startoftranscript|>` prefix is dropped by
the collator as in the transformers Whisper fine-tuning recipe). AdamW lr 1e-5, linear decay, warm-up 30, batch 32,
no gradient accumulation, grad-norm clip 1.0, bf16 autocast on the GPU (fp32 master weights; no loss scaler, so no
skipped optimizer step), 4 dataloader workers. Whisper's config has dropout 0 and SpecAugment off, so the data order
is the only randomness. Launcher-side additions (no transformers source edited, MM-27):

- **Observed consumption, `row`.** The collator carries each example's `utt_id`; `compute_loss` takes them, and a
  callback writes them in `on_pre_optimizer_step`, i.e. before `optimizer.step()` (transformers 5.21.0
  `trainer.py:1914`; the micro-batches of a step are prefetched by `get_batch_samples`, `trainer.py:1856`, and all
  pass through `compute_loss` before that call). `samples.jsonl`: one line per (step, example) `{step, epoch,
  micro_batch, position, utt_id, row = sha256(utt_id), attempt, utc}`, fsync; `train_steps.jsonl`: `{step, epoch,
  n_examples, example_ids}`. A run that ends with examples loaded after the last step fails.
- **Atomic cuts.** Trainer checkpointing is off (`save_strategy="no"`). In `on_step_end` (after the optimizer step and
  `global_step += 1`, `trainer.py:1927-1929`) at every `save_steps` and at the last step, the cut is written into
  `.checkpoint-<S>.partial/`, files fsynced, and renamed to `checkpoint-<S>/`; `cut.json` is written last.
- **Seeds (MM-29).** `Seq2SeqTrainingArguments(seed=S, data_seed=S)`: `set_seed(S)` at Trainer construction
  (`trainer.py:419`); the order comes from accelerate's seedable sampler with
  `data_seed` (`trainer.py:836`). Run seed **4000**. The order used is the observed log, never a reconstruction.
- **Bindings and pins.** `env-freeze.txt` and `env-runtime.json` (lock, script and decoder-code SHA-256) before torch
  is imported; `run_args.json` with the base binding (repo, revision, file hashes from `MODELS.json`, re-hashed),
  `model_name_or_path` = `base_model` = the base directory, `train_file` and `train_file_sha256`, `seed`,
  `profile`, `dry_run: false`, `max_steps`, `TrainingArguments`. Preemption: as areas 2-3 (earlier files move to
  `preempted-attempt-<k>/`, the run restarts, no resume).
- `full_determinism` (transformers `enable_full_determinism`) is on for the CPU profiles only (needed for identical
  CPU cut hashes; untested on the GPU, so off there; GPU cuts are not claimed bit-reproducible).

**Cut layout** `checkpoint-<S>/` (S = optimizer step): `model.safetensors` (fp32, 290 MB, header canonicalized as in
area 3), `config.json` (`use_cache: true` restored), `generation_config.json`, `processor_config.json` (feature
extractor), `tokenizer.json`, `tokenizer_config.json`, `decoding.json` (the deployment decoder), `base_ref.json`
(repo, revision, per-file SHA-256 of the base), `trainer_state.json` (TrainerState, resume-only), then **`cut.json`
last** (`global_step`, `examples_consumed`, SHA-256 of every other file, `deployable` list, `script_sha256`,
`decoder_code_sha256`, `seed`, `profile`; no wall-clock field, so the cut hash is reproducible).

**Deployment decoder** (`decoding.json`, the "SHARED DECODER" block, byte-identical in both scripts, SHA-256
`478dd30e…88cc5f`): greedy, 1 beam, no sampling, no temperature fallback, short-form (every utterance <= 30 s, one
window zero-padded to 30 s), no timestamps, batch bound 200 new tokens, and **per utterance at most 16 + ceil(12 x
seconds) text tokens** (the first ones of the greedy output, which equals generating with that bound; the fastest
reference is 9.7 tokens/s, every reference is under half its bound). The bound cuts a repetition loop at a length
proportional to the audio (tiny.en's 14.67 item) instead of 200 tokens; `truncated` is a sub-metric (0 of 500 in
both pilot arms).

## Metric and tier (proposal)

`wer` (evaluator `asr.whisper_items@0.1`, `eval_asr.py`): per-utterance word error rate (S + D + I) / N, word-level
Levenshtein after the **Whisper English normalizer** (transformers 5.21.0 `EnglishTextNormalizer` with the base
repo's `normalizer.json`, the scorer's own copy, never the candidate's) on reference and hypothesis; **lower is
better, range [0, null]** (an insertion-heavy transcript exceeds 1). Empty transcript = score (WER 1, MM-4);
`decode_error` (generate raised) and `scorer_error` (normalizer or alignment raised, or the normalized reference
differs from the registered one) are errors. Sub-metrics: substitutions, deletions, insertions, ref_words,
hyp_words, empty, truncated. Corpus WER (ratio of sums, 05 §3.1 since revision 11) is reported in `summary.json`
only, descriptive (spec MM-GAP-13 choice: per-item mean decides). The alignment was cross-checked against an
independent Levenshtein on 3,000 random cases.

**Tier: T3 (proposed).** 05 §6.1 grades an evaluator against each *training reward*; SFT has none. The training
signal is token cross-entropy of the Whisper tokenizer, a different quantity from normalized word edit distance, and
the scorer is a deterministic program (as "geometry checks" or OCR edit distance in the T3 row) with no learned
weights. Two honest caveats for the registration: (1) the item references come from the **same AMI transcription
convention and project** as the training labels; read literally, "shares training data or labelers" is a T0 trigger,
but the T0 row is about the *scorer* (a reward model or judge trained on those labels), whereas held-out references
are the item set itself, as GSM8K answers are in area 1 (T3); (2) the gain is largely convention-aligned (see
above), analogous to R15-1's "construct-aligned". Scorer validity (G-21): exact and deterministic (re-scoring gives the
same value; two CPU evaluations gave byte-identical outputs); the pilot baseline mean 0.153 lies far from the floor 0.

**MM-38 degeneracy (deciding metric only), baseline on CPU:** all 500 items: mode 0 with share 0.30 (< 0.5), sd 0.168
(> 0), mean 0.153; the smoke's first 20 items: mean 0.186, sd 0.160, mode share 0.15. The [0, null] range has no
upper bound, so the "within 5 % of a bound" rule only concerns 0: mean 0.153 is far above it. Not degenerate.

## Items and power (proposal)

**N = 500, K = 1, `eval_seeds [11]`** (greedy decoding is deterministic; the seed is recorded, nothing is drawn).
CPU pilot (cut 60 of `--profile cpu-pilot` vs base, all 500 items): mean oriented difference 0.0334, **σ_d = 0.119**.
Single deciding clause, superiority, δ 0, one-sided α 0.025, power 0.8: n = ((1.960 + 0.842) σ_d / Δ)² → 1,107 /
492 / 277 / 123 items for Δ = 0.01 / 0.015 / 0.02 / 0.03; at N = 500 the MDE is **0.0149** (0.0155 with the ucb80
factor ≈ 1.04 for df 499). The pilot's effect 0.033 is about 2.2x that MDE after only 960 examples; the full run's
cut moves further, and area 3 showed σ_d grows with the distance from the base, so N = 500 keeps margin. Proposal:
`sigma_plan: fixed` 0.12 from this pilot (or `internal_pilot` from the smoke as areas 1-3), planning effect 0.02, `min_paired` 10,
`max_unpaired` 0. N cannot grow far without new test shards: the pool after filters is 4,393.

## Outputs of `train_whisper.py` (`/out`)

| File | Written | Purpose |
|---|---|---|
| `env-freeze.txt`, `env-runtime.json` | first, before torch | MM-33 pins; `script_sha256` (EXT-83), `decoder_code_sha256`, lock SHA-256 |
| `run_args.json` | before training | settings, `TrainingArguments`, seed and derivation, base and data bindings, decoding |
| `samples.jsonl`, `train_steps.jsonl` | before each optimizer step, fsync | observed consumption |
| `train_log.jsonl` | per logging step | loss, grad norm, learning rate (Trainer logs) |
| `memory.jsonl` | per cut | seconds per step, examples consumed, peak GPU memory, peak RSS |
| `attempts.jsonl` | start and end | attempts |
| `checkpoint-<S>/` | at `save_steps` and the last step | the cuts |

## Evaluator outputs (`eval_asr.py`, `/out`)

`outputs/<item>.json` (raw and normalized hypothesis, normalized reference), `scores.jsonl`, `summary.json` (per-item
and corpus WER, S/D/I totals, MM-38 checks, peak memory), `env-runtime.json`, and **one arm `arm-wer/`** with
`items.jsonl` (MM-1: item, repeat 0, seed, score, error, sub_metrics, output_ref), `eval_meta.json` (MM-2: evaluator
id, adapter `hf_seq2seq`, metric `wer`, `lower_is_better`, `[0, null]`, checkpoint hashes as passed, `item_set_hash`
as passed, `eval_seeds`, sampler = decoding + batch size 16 + dtype fp32 + order, `sampler_hash` = SHA-256 of compact
sorted JSON (not RFC 8785, as areas 2-3), scorer `{per_utterance_wer, version incl. transformers version and
normalizer.json hash, weights null, code_sha256 = eval_asr.py}`, framework transformers) and hard-linked `outputs/`.
A cut is checked before use: every `cut.json` file hash, `decoder_code_sha256` equal to the evaluator's, `decoding.json`
equal to the evaluator's decoder, `base_ref.json` equal to `MODELS.json`. Examples: `sanity-evidence/`.

## Profiles and cost

| Profile | Steps x batch | Examples (epochs) | Cuts | Time |
|---|---|---|---|---|
| smoke | 60 x 32 | 1,920 (0.64) | 30, 60 | estimate ~1 min training + ~1 min load; baseline 20 items + cut 60 eval < 1 min each |
| full | 300 x 32 (set from the smoke if wanted, `--max-steps`) | 9,600 (3.2) | 100, 200, 300 | estimate ~2-3 min training |
| cpu-sanity (host) | 4 x 4 | 16 | 2, 4 | measured 7.4 s training, 12.5 s wall, peak RSS 3.9 GB |
| cpu-pilot (host) | 60 x 16 | 960 | 60 | measured 6.7 s/step, 6.7 min, peak RSS 8.4 GB |

GPU time per step (estimate, 4090, bf16): ~5.8 TFLOP per 32-example step → ~0.1 s compute; log-mel extraction ~15-20
ms per utterance on 4 workers ≈ 0.15 s; **≈ 0.2-0.4 s/step**. Peak GPU memory (estimate): weights + AdamW fp32 ≈ 1.2
GB, activations for 32 x 1500 encoder frames bf16 ≈ 5-8 GB → **≈ 7-10 GB**, far under 22.5 GB. Evaluation: fp32,
batch 16, 500 items ≈ 1 min (CPU measured 0.18 s/item), < 2 GB. **Area total (estimate, with container starts and
model loads): ≈ 0.1-0.15 GPU-h** (smoke 0.03, full 0.05, two arms 0.03), well under the 1 GPU-h target.
Alternative: the whole area runs on CPU (`--device cpu --profile full` with `bf16` off would need a CPU profile; full
≈ 300 x ~13 s ≈ 65 min estimate, 0 GPU-h) — open question 3.

## CPU sanity check (2026-11-13, host, `CUDA_VISIBLE_DEVICES=` empty, no container, selected outputs in `results/2026-11-13/sanity-evidence/`)

| Check | Result |
|---|---|
| `uv lock` / `uv sync --frozen` | 60 packages; imports OK |
| Training micro-run (`--profile cpu-sanity`) | rc 0; cuts 2 and 4 with `cut.json` last; `samples.jsonl` 16 lines, `train_steps.jsonl` 4 steps x 4 ids (`sanity-evidence/cpu-sanity-train_steps.jsonl`) |
| Reproducibility | two runs: **every file of both cuts identical** (cut 4 `model.safetensors` `f37b8ce0…a715`, `cut.json` `f1008eec…59bf`; `sanity-evidence/cpu-sanity-run-{a,b}-sha256.txt`), same order and logs. Found and fixed: losses differed at step 3 by 1e-7 without `full_determinism`; `cut.json` carried a timestamp |
| Cut → evaluation | `eval_asr.py --checkpoint checkpoint-4` reloads the cut (processor + model), checks hashes, decoder and base; 16 items: mean WER base 0.207 → cut 0.201, 0 errors, equal `sampler_hash`; arms in `sanity-evidence/cpu-sanity-{base,cut4}-arm-wer/` |
| Evaluator determinism | two base evaluations of all 500 items: byte-identical `outputs/` |
| Power pilot | `sanity-evidence/cpu-pilot/` (summaries, paired.json, training log) |

## What gatedtrain needs (adapter `hf_seq2seq`, trainer id `sft.hf_seq2seq@0.1` as spec §6 row 5)

1. **Cuts and completeness (MM-20).** `checkpoint-(\d+)/` with `cut.json` whose `files` SHA-256 all match, whose
   `global_step` equals the directory number, and whose `files` keys are exactly the 9 files above;
   `.checkpoint-*.partial/` or a directory without `cut.json` is incomplete. The last cut's step equals `run_args.json`
   `max_steps`. `last_step_included` = `global_step`. A top-level `hf-trainer/` directory is removed when empty; it is
   never a cut. (Unlike area 1 the Trainer's own `checkpoint-*` writer is off; `trainer_state.json` is in the cut but
   not the marker.)
2. **Identity (MM-17).** Full weights, no LoRA: `model_hash` = T0-09 manifest hash over `cut.json` `deployable`
   (`model.safetensors`, `config.json`, `generation_config.json`, `processor_config.json`, `tokenizer.json`,
   `tokenizer_config.json`, `decoding.json`); `cut_hash` adds `base_ref.json`, `trainer_state.json`, `cut.json`. No
   composition with the base. Readable without torch: all JSON and the safetensors header.
3. **Base binding (MM-21).** `base_ref.json` `{repo, revision}` = `openai/whisper-base.en` @ `911407f4…cd6`
   (40-hex) and its `files` equal `models/MODELS.json`; `run_args.json` `model_name_or_path` (= `base_model`) is the
   container path of the base directory inside the read-only identity mount
   (`<runtime>/models/whisper-base.en`). The spec's base artifact is that
   directory; **baseline arm: `checkpoint.model_hash` = its manifest hash** (11 files; note the base's processor files
   differ from a cut's: `preprocessor_config.json` + vocab/merges vs `processor_config.json` + `tokenizer.json`).
4. **Data binding (EXT-65 analogue).** Spec data = `data/` (one shard `train.jsonl`, plus `data/audio/` and
   `data_meta.json` inside the snapshot manifest); `run_args.json` `train_file` =
   `<runtime>/data/train.jsonl` (identity mount), `train_file_sha256` =
   `sha256:2278c534…7b372`.
5. **Seed and pins.** `run_args.json` `seed` = launch `training_seed` (4000); `env-freeze.txt` has
   `transformers==5.21.0` (framework pin; also accelerate, torch, tokenizers); `env-runtime.json` `script_sha256` =
   the record's; `cut.json` `decoder_code_sha256` = the evaluator's.
6. **Consumption (MM-23 with sample log): precision `row`, derivation `observed`.** One ledger record per
   `samples.jsonl` line: `data_ref.snapshot` = manifest hash of `data/`, shard `train.jsonl`, `row` = the line's `row`
   (sha256 of `utt_id`), `step` = its step, `sample_id` = H(run, attempt, epoch, step, utt_id) (suggest NUL-joined as
   area 3), `rewards` null, `seed_inputs` null (no per-sample randomness), outcome `consumed`. Refuse when: steps are
   not consecutive from 1; a step's `samples.jsonl` ids differ from its `train_steps.jsonl` `example_ids`; a step's
   count is not `per_device_train_batch_size` except the last step of an epoch (3,000 = 93 x 32 + 24: steps 94, 188,
   282 carry 24); an id is not a `utt_id` of `train.jsonl`; a cut's `examples_consumed` ≠ the sum of counts of steps ≤
   its `global_step`. Records after the last cut are orphaned (none in a finished run). No example is filtered.
7. **Exact overlap (MM-11).** `payload_text_field: "text"` on both sides (items' raw AMI `text` vs training rows'
   `text`); 0 by construction (normalized overlap 0 too).
8. **Arm.** `<out>/arm-wer/` (one metric); `ext-arm-fill` (or the request) supplies `model_hash`, `base_model_hash`
   (null or the base hash; no LoRA), `item_set_hash`, and the RFC 8785 `sampler_hash`. `normalizer_sha256` stays
   null (no embodied normalizer; the text normalizer is in the scorer version).
9. **Registry rows (proposal).** Trainer: `framework: "transformers"`, `pins ("transformers", "accelerate", "torch")`,
   components as item 2, `consumption {target: row, fallback: stage}`, `gpu_h_estimate 0.15`, `input_roots`: the
   Python directory, `<runtime>`, `<output-root>/whisper-sft-ami`;
   `scan_skip`: `<runtime>/.venv` (34,303 files; only huggingface_hub's `_git_credential.py` and its `.pyc`
   match a credential-like name) and the interpreter's site-packages; `interpreter_trees`: the CPython 3.12.16 tree;
   `interpreter`: `<runtime>/.venv/bin/python`. Outside the venv the mount holds 3,577 files plus
   `sanity-evidence/`; none matches a credential-like name (checked). Evaluator `asr.whisper_items@0.1`: metric
   `wer`, `lower_is_better`, `score_range (0, null)`, `samples "500, 1"`, `error_classes ("decode_error",
   "scorer_error")`, `score_outcomes ("empty_transcript", "hallucination")`, `payload_text_field "text"`,
   `item_payload ("text", "reference_normalized", "audio", "audio_sha256")`, `binary false`, tier T3 (proposed),
   scorer `code_sha256` = SHA-256 of `eval_asr.py` (at the gate, the gate-time version, which is not published; now
   `rows.EVAL_ASR_SHA256`, `../RESULT.md`).

## Launch request facts

| Host | Container | Mode |
|---|---|---|
| `<data-root>/python` | same | ro |
| `<runtime>` (this directory: scripts, lock, venv, `models/`, `data/`, `items/`) | same | ro |
| evaluation of a cut only: `<output-root>/whisper-sft-ami/<run>/checkpoint-<S>` | `/inputs/cut` | ro |
| `<output-root>/whisper-sft-ami/<run>` (new, empty, owned by the launching user) | `/out` | rw |
| tmpfs | `/tmp` | `rw,exec,nosuid,nodev,size=8g` |

Environment names (none a credential): `HF_HUB_OFFLINE=1`, `HOME=/tmp/home`, `XDG_CACHE_HOME=/tmp/cache`,
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`, `TOKENIZERS_PARALLELISM=false`; `env_paths`: `HOME`,
`XDG_CACHE_HOME`. Network `none` (`local_files_only=True` everywhere), `gpus: true`, user `1000:1000`, `--cap-drop
ALL`, `--security-opt no-new-privileges`, `--shm-size=8g` (dataloader workers). argv[0]
`<runtime>/.venv/bin/python`.

```
smoke:     argv [<python>, "<runtime>/train_whisper.py",
                 "--profile", "smoke", "--out", "/out", "--seed", "4000"]
           training_seed 4000, label mm-a4-whisper-smoke, gpu_est_h 0.03
full:      argv [<python>, "<runtime>/train_whisper.py", "--profile", "full", "--out", "/out", "--seed", "4000"]
           training_seed 4000, label mm-a4-whisper-full, gpu_est_h 0.05
baseline:  argv [<python>, "<runtime>/eval_asr.py", "--base-only", "--items", "<runtime>/items",
                 "--out", "/out", "--eval-seeds", "11", "--model-hash", "<H base>", "--item-set-hash", "<H>"]
                                                         # smoke: add "--limit", "20"
candidate: argv [<python>, "<runtime>/eval_asr.py", "--checkpoint", "/inputs/cut", "--items", "<runtime>/items",
                 "--out", "/out", "--eval-seeds", "11", "--model-hash", "<H>", "--item-set-hash", "<H>"]
```

As area 3, `eval_asr.py` does not hash the base it loads with `--base-only`; a request should name no `--base` other
than the default inside the read-only mount. `framework`: `{name: "transformers", version: "5.21.0", commit: null}`;
`runtime`: `{kind: "venv_in_container", image: <area-1 digest>, venv_lock: "<runtime>/uv.lock"}`.

## Open questions

Marked **[design]** where a design or option choice is involved, **[routine]** where an operator can decide.

1. **[design] Tier of `wer`.** (a) T3, blocking, with the disclosure that the gain is largely AMI-convention learning
   and that the references share the training labels' transcription project (recommended: the T0 row targets
   scorers trained on those labels, not held-out references; consistent with area 1); (b) T3 and additionally a
   "construct-aligned"-style statement in the registration (as R15-1 for area 3); (c) T0, non-blocking. If the
   maintainers want the "shares labelers" reading settled, it is a design clarification of 05 §6.1 (DA3-4 candidate text:
   "Held-out item references produced by the same annotation process as supervised training labels do not make a
   programmatic scorer T0; the registration discloses the shared convention.").
2. **[routine] Deployment decoder bound.** Keep the duration-proportional token bound (recommended; it changes no
   pilot transcript and caps loops) or the plain 200-token bound of spec §6 row 5's "greedy, no fallback".
3. **[design] GPU or CPU.** (a) GPU as above, ≈ 0.1-0.15 GPU-h (recommended: fast, matches the plan);
   (b) CPU only (`gpus: false`, ≈ 1 h wall for the full run, 0 GPU-h; needs a CPU `full` profile and loses the bf16
   path's realism).
4. **[routine] Power plan.** `sigma_plan: fixed` 0.12 from the CPU pilot (recommended, as area 2 used a pilot-derived
   σ) or `internal_pilot` from the smoke; planning effect 0.02, N 500, K 1.
5. **[routine] Steps.** Full 300 steps (3.2 epochs) at lr 1e-5; the pilot moved WER by 0.033 after 960 examples, so
   no tuning is expected; adjust only if the smoke shows divergence.
6. **[routine] Mount size.** `data-source/` (2.49 GB raw parquet) sits in the read-only mount; it is never read at
   run time. Keep for provenance (recommended) or move out like area 3's reference clone.
7. **[routine] Corpus WER.** Report only (recommended, per spec MM-GAP-13 choice) or register the ratio-of-sums form
   now supported by 05 §3.1/§3.4 as a second, non-deciding clause.
