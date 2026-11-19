# Area "audio" (Whisper base.en SFT on AMI) gate: evaluation plan (DRAFT)

prototype, unpromotable. Draft written 2026-11-13 (UTC) on the ptc-skel side, before the area's smoke run. The
operator writes the real files under `<data-root>/ptc-mm/area4-whisper/gate/` (this draft never touches
the data directory). Every GPU step runs through `ext-launch` and `gpu-step.sh` after approval (MM-31). Adapter
`hf_seq2seq` (tests/README.md section 11.21, EXT-138 to EXT-153). Below, `PTC` is
the ptc command line (`python -m ptc_skel` at the time), `A4` is
`<data-root>/ptc-mm/area4-whisper`, `OUT` is `<data-root>/ptc-mm/out/area4-whisper`, `REC` is
`<data-root>/ptc-mm/records/area4-whisper`, `REQ` is a working directory of launch requests, `DRAFT`
this directory. No `PTC_SKEL_EXT_*` override may be set for any real command.

## 1. Files

| File | Purpose | Hash |
|---|---|---|
| `spec.yaml` | workflow `mm-a4-whisper-sft`: stage `sft.hf_seq2seq@0.1`, gate `gate.superiority@1.0` on `asr.whisper_items@0.1`, `mde` 0.02 | pins base, data, items, trainer config; registration hash after `created_at` |
| `registration.yaml` | v1; `wer`, lower, superiority, margin 0, alpha 0.025, power 0.8, `sigma_plan: fixed` 0.12 (CPU pilot), planning effect 0.02, N 500, K 1, seed 11, `baseline_pilot_mean` 0.153, `shared_annotation` (R16-1); `created_at` to be added | pin in spec.yaml after `created_at` |
| `trainer.yaml` | `{backend: hf_seq2seq, profile: full}` | `sha256:0288b678...c1f42f` |
| `smoke_scores.py` | helper: the smoke record's baseline scores, error count and degeneracy checks from the filled 20-item arm (stdlib only) | none |

Fixed hashes (computed 2026-11-13 with ptc-skel on the real directories, read only):
- base `A4/models/whisper-base.en/` manifest `sha256:583e09fedd5421398d92af8b40bb97631823c277782b1c904c67b1cbd8d4b805`
  (11 files, equal to `models/MODELS.json`; the baseline arm's `checkpoint.model_hash` and the smoke record's
  `base_model_hash`);
- training data `A4/data/` manifest `sha256:d5080d22d85c45c9628dc7dc2d21ec273b442643f1614fb6fbf9f82c670927f6` (the
  ledger's `data_ref.snapshot`; shard `train.jsonl` `sha256:2278c534...7b372` = `run_args.json` `train_file_sha256`,
  3,000 rows);
- item set `A4/items/` manifest `sha256:38b26785063a54688cb650cc97bbe1bedbb02caa9f0975e82fddd36ec25a5fc7` (500 items;
  `items.jsonl`, `item_set_meta.json`, `audio/`; the README's `94d0bea4...` is the file hash of `items.jsonl` only);
- evaluator identity `asr.whisper_items@0.1` `sha256:d18c705e5677c6883b48c5654035d269d54927d6d0d1f56913d6e28a570a91b0`
  (scorer code = SHA-256 of `eval_asr.py`, `sha256:6703cd48...`; version string as its 2026-11-13 CPU evaluation wrote
  it);
- evaluation sampler RFC 8785 hash `sha256:2649163a23ba8d35c2454a07547ef3d62cfe36bc8a1c58d6363f79bcf2b6be19`, equal to
  the hash `eval_asr.py` writes (compact JSON; the sampler holds integers and strings only, so `ext-arm-fill` verifies
  it and converts nothing);
- deployment decoder code (`SHARED DECODER` block of both scripts) `sha256:478dd30e...88cc5f`, pinned in the adapter;
  every cut's `cut.json` `decoder_code_sha256` must equal it.
- plan: n_required 283 of 500 (sigma 0.12, planning effect 0.02), MDE at N 500 = 0.0150 (checked with ptc-skel).

## 2. Gate files (before the smoke)

The registration takes no value from the smoke (sigma fixed from the CPU pilot), so it is written first and the smoke
is launched with `--spec`:
```
mkdir A4/gate
cp DRAFT/spec.yaml DRAFT/registration.yaml DRAFT/trainer.yaml A4/gate/
# append to A4/gate/registration.yaml:  created_at: "<now, RFC 3339 UTC>"
sha256sum A4/gate/registration.yaml      # write the hex into spec.yaml artifacts.registration (replace the <...>)
```
(`A4` is read-only for the ptc-skel side; the operator writes `gate/`, as for areas 2 and 3. `gate/` lies outside
the hashed base, data and items.)

## 3. Smoke (MM-38, MM-39)

1. **Training smoke** (`REQ/smoke.request.yaml`, label `mm-a4-whisper-smoke`, output `OUT/smoke-1`, 60 steps x 32,
   cuts 30 and 60, gpu_est_h 0.03):
   ```
   mkdir -p OUT/smoke-1 REC        # new, empty, owned by the launching user
   PTC ext-launch --adapter hf_seq2seq --profile smoke --request REQ/smoke.request.yaml --gpu-log <data-root>/ptc-tierb/gpu-log.md --spec A4/gate/spec.yaml --out-record REC/smoke-1.launch.json
   PTC ext-launch --adapter hf_seq2seq --recheck REC/smoke-1.launch.json --spec A4/gate/spec.yaml
   <the printed gpu-step.sh command>
   ```
   Read `OUT/smoke-1/memory.jsonl`: `seconds_per_step`, `peak_reserved_gb`, `max_rss_mb`; `nvidia-smi` sampling: peak
   GPU memory. Expect `checkpoint-30/`, `checkpoint-60/` with `cut.json`, `samples.jsonl` 1,920 lines,
   `train_steps.jsonl` 60 lines of 32 ids (epoch 0), `run_args.json` `base_ref.verified true`.
2. **Smoke baseline** (`REQ/eval-smoke-base.request.yaml`, output `OUT/eval-smoke-base-1`, the base on the first 20
   items, `--limit 20`):
   ```
   mkdir OUT/eval-smoke-base-1
   PTC ext-launch --adapter hf_seq2seq --profile smoke --request REQ/eval-smoke-base.request.yaml --gpu-log <data-root>/ptc-tierb/gpu-log.md --out-record REC/eval-smoke-base-1.launch.json
   PTC ext-launch --adapter hf_seq2seq --recheck REC/eval-smoke-base-1.launch.json
   <the printed gpu-step.sh command>
   PTC ext-arm-fill --adapter hf_seq2seq --checkpoint A4/models/whisper-base.en --items A4/items OUT/eval-smoke-base-1/arm-wer
   ```
   Expect `filled checkpoint.model_hash` (`sha256:583e09fe...`) and `filled item_set_hash` (`sha256:38b26785...`); the
   sampler hash is verified, not converted. No `--base` is needed (full weights); `base_model_hash` stays null.
3. **Smoke cut** (`REQ/eval-smoke-cut.request.yaml`, output `OUT/eval-smoke-cut-1`, `OUT/smoke-1/checkpoint-60`
   read-only at `/inputs/cut`, the same 20 items): exercises the cut-to-evaluation path (`eval_asr.py` checks the cut's
   file hashes, decoder code, `decoding.json` and `base_ref.json`) before the full run.
   ```
   mkdir OUT/eval-smoke-cut-1
   PTC ext-launch --adapter hf_seq2seq --profile smoke --request REQ/eval-smoke-cut.request.yaml --gpu-log <data-root>/ptc-tierb/gpu-log.md --out-record REC/eval-smoke-cut-1.launch.json
   PTC ext-launch --adapter hf_seq2seq --recheck REC/eval-smoke-cut-1.launch.json
   <the printed gpu-step.sh command>
   PTC ext-arm-fill --adapter hf_seq2seq --checkpoint OUT/smoke-1/checkpoint-60 --items A4/items OUT/eval-smoke-cut-1/arm-wer
   ```
   Expect 20 scored items, 0 errors (count them into the smoke record's `error_count` with the baseline's), and a
   mean WER at or below the baseline's (the CPU pilot moved it 0.153 to 0.120 after 960 examples).
4. **Smoke record.** `python3 -I DRAFT/smoke_scores.py OUT/eval-smoke-base-1/arm-wer` prints `baseline_scores` (20
   per-item WER), `error_count` and the degeneracy checks. Write `REC/smoke-1.smoke.json`: `schema
   ptc_skel.smoke_record`, `schema_version 1.0`, `tags`, `adapter hf_seq2seq`, `area 5`, `base_model_hash
   sha256:583e09fe...`, `item_set_hash sha256:38b26785...`, `peak_gpu_mem_gb`, `host_ram_peak_gb`, `swap_peak_gb`,
   `seconds_per_step`, `projected_gpu_h` (300 x seconds per step + model load and cut saves, plus two 500-item
   evaluations; README estimate about 0.1), `gpu_log_total_h` (gpu-log.md running total), `baseline_scores`,
   `error_count` (both smoke arms), `verdict PASS`, `reasons []`. ptc-skel recomputes the verdict on `wer`: [0, null] has no mean rule;
   FAIL when 10 or more of the 20 scores share one value or their SD is 0 (CPU: mode share 0.15, sd 0.16).

## 4. Full run

`REQ/full.request.yaml` (label `mm-a4-whisper-full`, output `OUT/full-1`, 300 steps; keep `--max-steps` quoted, `-
'300'`, if it is changed):
```
mkdir OUT/full-1
PTC ext-launch --adapter hf_seq2seq --profile full --request REQ/full.request.yaml --gpu-log <data-root>/ptc-tierb/gpu-log.md --smoke-record REC/smoke-1.smoke.json --spec A4/gate/spec.yaml --out-record REC/full-1.launch.json
PTC ext-launch --adapter hf_seq2seq --recheck REC/full-1.launch.json --spec A4/gate/spec.yaml
<the printed gpu-step.sh command>
```
Expect `checkpoint-100/`, `-200/`, `-300/` with `cut.json` `examples_consumed` 3,192, 6,384, 9,576 (an epoch is 94
steps: 93 x 32 + 24, so steps 94, 188 and 282 carry 24); `samples.jsonl` 9,576 lines; `train_steps.jsonl` 300 lines;
`run_args.json` `seed 4000`, `profile full`, `max_steps 300`. Any step count works if changed: the last step always
saves a cut, and collect requires the last cut's step to equal `max_steps`.

## 5. Arms

`REQ/eval-base.request.yaml` (the base on all 500 items) and `REQ/eval-cand.request.yaml` (`OUT/full-1/checkpoint-300`
read-only at `/inputs/cut`), each:
```
mkdir OUT/eval-base-1        # resp. OUT/eval-cand-1
PTC ext-launch --adapter hf_seq2seq --profile full --request REQ/eval-base.request.yaml --gpu-log <data-root>/ptc-tierb/gpu-log.md --smoke-record REC/smoke-1.smoke.json --out-record REC/eval-base-1.launch.json
PTC ext-launch --adapter hf_seq2seq --recheck REC/eval-base-1.launch.json
<the printed gpu-step.sh command>
```
(the GPU estimate becomes the smoke's projection: conservative; these records are never collected). Then:
```
PTC ext-arm-fill --adapter hf_seq2seq --checkpoint A4/models/whisper-base.en --items A4/items OUT/eval-base-1/arm-wer
PTC ext-arm-fill --adapter hf_seq2seq --checkpoint OUT/full-1/checkpoint-300 --items A4/items OUT/eval-cand-1/arm-wer
```
Corpus WER (ratio of sums) is read from `OUT/eval-base-1/summary.json` and `OUT/eval-cand-1/summary.json`
(`corpus_wer`) and reported beside the gate, descriptive only.

## 6. Collect

```
PTC ext-collect A4/gate/spec.yaml --adapter hf_seq2seq --stage-out OUT/full-1 --launch-record REC/full-1.launch.json --smoke-record REC/smoke-1.smoke.json --eval-baseline OUT/eval-base-1/arm-wer --eval-candidate OUT/eval-cand-1/arm-wer --runs-root <data-root>/ptc-mm/runs/area4-whisper
```
**Expect:** `metric: wer`, `direction: lower_is_better`, `independence_tier: T3`, a `shared_annotation:` line (the
registration's statement) and a `shared_annotation_manifest:` line (the registry row's, since the texts differ),
`n_items 500`, `samples_per_item 1`, unpaired 0/0, `baseline_model_hash sha256:583e09fe...`, `candidate_model_hash` =
the hash `ext-arm-fill` printed, `contamination: exact_overlap=0` (item `text` against the 3,000 training rows' `text`),
`consumption_precision: row` (observed; 9,576 ledger records, one per (step, example)), `gpu_seconds` = the wrapper time.
`PTC impact A4/data/train.jsonl --run-dir <run>` returns cuts 100, 200, 300 with answer `consumed` at `row`.

## 7. Assumption: the evaluated base is the default one

`eval_asr.py --base-only` loads `--base` (default `A4/models/whisper-base.en`, inside the read-only identity mount)
without hashing it; its `env-runtime.json` records the resolved `model_dir` but no hash. Every evaluation request
therefore uses the default base: none passes `--base`. `ext-launch` and `--recheck` enforce it for `hf_seq2seq`
(EXT-150), as for area 3 (EXT-133); they also refuse `train_whisper.py --no-verify-base` and a `--profile` other than
smoke or full. Since review round 9 (EXT-156) they also refuse an `--items` other than `A4/items`, `--limit` under
`--profile full` and an argv[1] that is not `A4/train_whisper.py` or `A4/eval_asr.py`; collect checks every output's
`reference_normalized` against the registered items (EXT-155). A cut's evaluation checks its own `base_ref.json` against
`MODELS.json`.
