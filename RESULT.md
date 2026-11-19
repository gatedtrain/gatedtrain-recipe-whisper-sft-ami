Prototype result of a single registered run; not a benchmark claim; not affiliated with or endorsed by the owners of the models, datasets, or frameworks used.

# Result (2026-11-13): gate PASS; carried forward to ptc-hub 0.2.0 and gatedtrain 0.3.0 as post-hoc amendments

The gate of `results/2026-11-13/gate/` (Whisper base.en full-parameter SFT on AMI IHM, 300 steps, against the base) was
collected on 2026-11-13 with ptc-skel 0.1.0 (ptc-skel, ptc-hub and ptc are earlier names of gatedtrain; see "Rename to
gatedtrain 0.3.0"): **PASS**, deciding per-utterance `wer` (lower is better, T3 with the
shared-annotation disclosure) 0.153 (base) to 0.108 (cut 300), delta 0.0455 [0.0348, 0.0557] (95% cluster bootstrap),
p <= 9.6e-05, achieved power 0.961, N 500, K 1, exact overlap 0, consumption precision `row` (9,576 records). The run's
outputs are not part of this repository.

That verdict remains reproducible with ptc `78eb913` and the original `registration.yaml` and `spec.yaml` (the
gate-time files, not published; the copies here are redacted, README.md "Results"). `78eb913` and the other
framework commits cited here (`e20736e` among them) are unpublished pre-release commits of the framework, earlier
named ptc-skel and ptc-hub; the published framework is `gatedtrain` `v0.3.0`. Likewise `6d2681b` is a pre-release
commit of this repository, not in its published history. Checked 2026-11-17: a re-collect with ptc `78eb913` and
this recipe at `6d2681b` (before the rename) on the
original records, arms and gate files gives PASS with every report field equal to the published report except
`run_id`, `created_at` and `lockfile_hash`.
Under ptc-hub 0.2.0 the amended registration below reports it **INCONCLUSIVE (post_hoc)**: every statistic is
unchanged, but the amendment changes the evaluator after the result was observed. Under gatedtrain 0.3.0 its
further amendment (last section) reports the same.

## Rename to ptc-hub 0.2.0

ptc-hub 0.2.0 renamed the persisted identifiers (schema ids, ledger writer id, analysis code version). The files under
results/2026-11-13/ keep the names ptc-skel 0.1.0 wrote (including the byte-copied arms under
`sanity-evidence/cpu-sanity-*-arm-wer/`); they are read by ptc `78eb913`. The gate definition is carried forward by
results/2026-11-13/gate/registration-ptc-hub.yaml (amends sha256:fc0f47b2b59509eae5230a031e10ad3baa8ae791cd7bf25b490726f762041b81,
post_hoc: true) and spec-ptc-hub.yaml. Re-pinned for the rename: runtime/eval_asr.py, from the gate-time version (not
published) to 126480747ca5970a5b95f7c09f17f8cacff484bb5a8c61f64d6db2b0189ac81b (only the new pin is accepted).

| File | Reason | SHA-256 before | SHA-256 now |
|---|---|---|---|
| `runtime/eval_asr.py` (the scorer; `rows.EVAL_ASR_SHA256`, part of the evaluator identity) | schema id `ptc_hub.item_eval`, `ptc-hub` in the docstring, internal decision-document name removed from the docstring | not published | `126480747ca5970a5b95f7c09f17f8cacff484bb5a8c61f64d6db2b0189ac81b` |
| `runtime/train_whisper.py` (its hash is recorded as `script_sha256` by every run) | internal decision-document name removed from the docstring | not published | `ca0955dd498ff3f9c25ee0101031a49a42df0bc3a813e3b93a3c9f33d0528507` |
| `runtime/pyproject.toml` | internal decision-document name removed from the header comment | not published | `b2326cfa6bbb5f259ab85885aabd20f49f67a5256a6cb24b8be55c6b924a5259` |

The historical runs (the 2026-11-13 gate and the CPU sanity evidence) used the old versions. Only comments and
docstrings changed besides the schema id: the shared decoder block (`decoder_code_sha256` `478dd30e...`), the scoring,
the sampler and the item set (`item_set_hash` `38b26785...`) are unchanged; `runtime/uv.lock` is unchanged.

## The amendment

`registration-ptc-hub.yaml` (sha256 `a8c52aeeabb095da53b16240a4ba2e9a615296ac52b6b31d2228be6c0bc628a3`) is a copy of
`registration.yaml` (sha256 `fc0f47b2...`, the published copy of the gate-time file, which it equals except
for redacted comments and the reworded free text of `sigma_pilot.source`) that differs in
`analysis_code_version: ptc-hub 0.2.0`, `amends`, `created_at` (2026-11-17), `post_hoc: true`, the name map in
comments, a header comment, and `evaluators[0].manifest_hash`
`sha256:d18c705e5677c6883b48c5654035d269d54927d6d0d1f56913d6e28a570a91b0` ->
`sha256:015cfae5001d6961dcd498d50bca6a2c03b4981ba8409c785132bfd3d041f329` (the evaluator identity with the re-pinned
`eval_asr.py`: an evaluator change, not only a rename, so it is not a version-only amendment). `spec-ptc-hub.yaml`
(sha256 `ed8629ae1cdbae430cdaae3eb2312ab1f746b939a664225fa5237d2801dc249c`) differs from `spec.yaml` only in the
registration URI and the name map in comments. With `<data-root>` replaced, `ptc-hub validate spec-ptc-hub.yaml`
exits 0; the gate-time spec exits 3 (REG-FIELD: the analysis code version names the framework before the rename).

## Re-judgement on a scratch replay (2026-11-17, ptc `e20736e`)

The run's records carry the old ids, which ptc-hub 0.2.0 refuses, so the gate was re-collected from scratch copies
(nothing under the real run, record or output directories was written):

1. Copies of `full-1.launch.json`, `smoke-1.smoke.json` and the two evaluation directories `eval-base-1/`,
   `eval-cand-1/` (arm `arm-wer/` with its outputs, `env-runtime.json`, `summary.json`). In the copies only these bytes
   changed: the schema ids (`ptc_hub.launch_record`, `ptc_hub.smoke_record`, `ptc_hub.item_eval`) and, in both arms'
   `eval_meta.json`, `scorer.code_sha256` from the gate-time pin to `12648074...` (the re-pinned scorer; approved for the
   scratch copy only, since this scorer changed beyond the rename and no proof carries the gate-time hash over).
2. The launch record's `label` is a checksum of the whole record and is logged in the host's guard log, so the
   renamed copy was moved onto a scratch host-work directory: a copy of `gpu-step.sh` with only its `WORK=` line
   changed, the run's two guard-log lines copied under the new label, and `host_work`, `gpu_step_sha256`,
   `host_config_sha256` and `label` recomputed with ptc's own functions (`host_config_sha256`, `launch_label`;
   label `mm-a4-whisper-full-d1d08b9c0dea` -> `mm-a4-whisper-full-812bcacd6465`). The real guard log was not written.
3. `ptc-hub ext-collect spec-ptc-hub.yaml --adapter hf_seq2seq` with the original stage output, the scratch copies,
   the original input roots and the runtime override at the original runtime directory.

Compared with the published report of run `run-20261113T200428Z-226e3bf9`: `delta_hat` 0.0454833, interval
[0.0347963, 0.0557288], `p_value` 9.59e-05, `achieved_power` 0.960861, `n_items` 500, `baseline_score` 0.153036,
`candidate_score` 0.107553, `sd_diff`, the plan, the model, checkpoint, item-set (`38b26785...`) and sampler
(`2649163a...`) hashes and 62 report fields in all are equal. The differences are exactly the expected ones:
`status` PASS -> INCONCLUSIVE with `status_reasons` [post_hoc], `post_hoc` false -> true, `evaluator_hash` and
`evaluator_hash_registered` `d18c705e...` -> `015cfae5...` (the re-pinned scorer), `registration_hash`,
`registration_amends`, `analysis_code_version`, the report `schema` id, `lockfile_hash`, `run_id` and `created_at`.
The ledger (9,576 rows) is equal except `writer_id` (`ptc-hub.ledger@0.2.0`) and the run-id-derived `run_id` and
`sample_id`; the lineage tables are equal except `run_id`, and the lineage edges except the registration hash and
the schema id.

## Rename to gatedtrain 0.3.0 (2026-11-18)

The framework was renamed again, from ptc-hub to gatedtrain, before its first publication (distribution, import
package and CLI `gatedtrain`, entry-point group `gatedtrain.adapters`, schema ids `gatedtrain.<x>`, ledger writer id
`gatedtrain.ledger@0.3.0`, analysis code version `gatedtrain 0.3.0`, environment variables `GATEDTRAIN_<X>`); this
repository became `gatedtrain-recipe-whisper-sft-ami` (import package `gatedtrain_recipe_whisper_sft_ami`, runtime
directory variable `GATEDTRAIN_RECIPE_WHISPER_SFT_AMI_RUNTIME`). The sections above keep the names in use when they
were written (ptc, ptc-skel, ptc-hub). The files under results/2026-11-13/ keep their bytes, including
`registration-ptc-hub.yaml` and `spec-ptc-hub.yaml`, and so do the original host's directory names the specs cite
(`<data-root>/ptc-mm/...`). The committed data and model manifests (`runtime/data-source/DATA.json`,
`runtime/data/data_meta.json`, `runtime/items/item_set_meta.json`, `runtime/models/MODELS.json`) keep their bytes
and their schema ids `ptc_mm.whisper_*` (the item set's hash is registered; the others record what the run used), and
`make_data.py` and `fetch_models.py`, which write them, keep those ids so that they reproduce the committed files.

- Amendment (05 section 3.7, R17-1/R17-3): `registration-gatedtrain.yaml` (SHA-256
  `058d2144da78878c8111724ca2b9466006643106fe965c0008fe569df22630bd`) differs from `registration-ptc-hub.yaml` only in
  `analysis_code_version: gatedtrain 0.3.0`, `amends` (`sha256:a8c52aee...`, the `-ptc-hub` registration),
  `created_at` (2026-11-18T09:58:29Z), the evaluator's `manifest_hash` (`sha256:015cfae5...` ->
  `sha256:bd5e979233e0511ee7d4f15ec6dc2fb80b30b5c1f943f8f89d82805c4dc764eb`) and its comment, a header comment, and
  a `rename_proof`: the new `runtime/eval_asr.py` is exactly gatedtrain's `renamed.apply_name_map` of the ptc-hub-era
  file, kept byte for byte as `results/2026-11-13/gate/scorer-ptc-hub/eval_asr.py` (SHA-256 `12648074...`, from git).
  So this step is version-only (gatedtrain reports `registration_amendment: version_only`), but `post_hoc` stays
  `true`: the prior amendment changed the evaluator after the result was observed, and a later version-only amendment
  does not undo that. `spec-gatedtrain.yaml` (SHA-256
  `121870a62d0b5829a17e7feebf31e5b3e9f97f2fb436c15c9e0d76112324faa1`) differs from `spec-ptc-hub.yaml` only in the
  registration uri and the name map in comments. With `<data-root>` replaced, `gatedtrain validate
  spec-gatedtrain.yaml` exits 0; `spec-ptc-hub.yaml` exits 3 (REG-FIELD: the analysis code version names the
  framework before the rename).
- Re-judged on a scratch replay with gatedtrain 0.3.0, starting from the scratch copies of the ptc-hub replay (above).
  In the copies only these bytes changed: the schema ids (`gatedtrain.launch_record`, `gatedtrain.smoke_record`,
  `gatedtrain.item_eval`) and, in both arms' `eval_meta.json`, `scorer.code_sha256` `12648074...` -> `b5335561...`
  (the pair the rename proof proves). The launch record moved onto a new scratch host-work directory as before
  (`host_work`, `gpu_step_sha256`, `host_config_sha256` and `label` recomputed with gatedtrain's own functions, the
  run's two guard-log lines under the new label); the stage output is the original, read in place. Against the
  published report of run `run-20261113T200428Z-226e3bf9`: `delta_hat` 0.0454833, interval [0.0347963, 0.0557288],
  `p_value` 9.59e-05, `achieved_power` 0.960861, `n_items` 500, `baseline_score` 0.153036, `candidate_score`
  0.107553, the plan, the model, checkpoint, item-set (`38b26785...`) and sampler (`2649163a...`) hashes and 61 report
  fields in all are equal. The differences are exactly the expected ones: `status` PASS -> INCONCLUSIVE with
  `status_reasons` [post_hoc], `post_hoc` false -> true, `evaluator_hash` and `evaluator_hash_registered`
  `d18c705e...` -> `bd5e9792...`, `registration_hash`, `registration_amends`, `registration_amendment`
  (`version_only`), `analysis_code_version`, the report `schema` id, `lockfile_hash`, `run_id` and `created_at`. The
  ledger (9,576 rows) is equal except `writer_id` (`gatedtrain.ledger@0.3.0`) and the run-id-derived `run_id` and
  `sample_id`; the lineage tables are equal except `run_id`, and the lineage edges except the registration hash and
  the schema id. Against the ptc-hub replay the verdict and every statistic are equal.
- Re-pinned for the rename: `runtime/eval_asr.py` `126480747ca5970a5b95f7c09f17f8cacff484bb5a8c61f64d6db2b0189ac81b`
  -> `b53355619b480361ddbcde8d8ad5eee072275797aa2b908ae576275f08b21f84` (`rows.EVAL_ASR_SHA256`; only the new pin is
  accepted). Not pinned by the adapter: `runtime/train_whisper.py`
  `ca0955dd498ff3f9c25ee0101031a49a42df0bc3a813e3b93a3c9f33d0528507` ->
  `b175df0b29d281ae1ccd99ff9426e5cf7735c7aacb3bc6eb700f9aedfb36e22e` (comments, the `cut.json` and `run_args.json`
  schema ids `gatedtrain_mm.whisper_sft_cut` and `gatedtrain_mm.whisper_sft_run_args`; the shared decoder block,
  `decoder_code_sha256` `478dd30e...`, is unchanged in both scripts); `runtime/uv.lock`
  `520d9773fb6602c1d206729d5dcf2703c8acd913ff0a96a494f4e74121430176` ->
  `bf53ae6b6a7f53550e32eda8ec5765fe848cd75afea5eb71a8ebb778124bd690` (regenerated with `uv lock` after the runtime
  project was renamed to `gatedtrain-recipe-whisper-sft-ami-runtime`; the 60 locked packages and their versions,
  sources and hashes are unchanged, only the root project's entry moved).
