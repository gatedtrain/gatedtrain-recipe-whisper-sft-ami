"""Registry rows of adapter `hf_seq2seq`: Whisper SFT on AMI (area 4).

The training row, the evaluator rows and the adapter class, loaded by `gatedtrain.ext.registry` through the
`gatedtrain.adapters` entry point (MULTIMODAL-SPEC section 6)."""

from __future__ import annotations

from typing import Any

from gatedtrain.ext.rowkit import CONTAINER_PATHS, runtime_dir, venv_runtime_rows

NAME = "hf_seq2seq"
# The runtime directory: this repository's runtime/ (scripts, its .venv, models, data, items; mounted read-only at its
# own host path), or the override; earlier runs lie under the host's output root as `{output_root}/<OUT_NAME>`.
RUNTIME_ENV = "GATEDTRAIN_RECIPE_WHISPER_SFT_AMI_RUNTIME"
OUT_NAME = "whisper-sft-ami"
RUNTIME = runtime_dir(RUNTIME_ENV, __file__)
HOST_ROWS = venv_runtime_rows(RUNTIME, OUT_NAME)

TRAINER_ROWS: dict[str, dict[str, Any]] = {}
EVALUATOR_ROWS: dict[str, dict[str, Any]] = {}

# Area "audio" (MULTIMODAL-SPEC section 6 row 5 and its deviations; the maintainers' area 4 decisions, not
# published; runtime/README.md "What gatedtrain needs" items 1-9, read 2026-11-13). Full-parameter
# SFT of openai/whisper-base.en with transformers' Seq2SeqTrainer on AMI IHM; cuts by the launcher's own writer
# (train_whisper.py save_cut, cut.json last), observed per-example consumption.
TRAINER_ROWS["hf_seq2seq"] = {
    "uses": ("sft.hf_seq2seq@0.1",),
    "pins": ("transformers", "accelerate", "torch"),                      # from env-freeze.txt (MM-33)
    "outputs": ("checkpoint-<step>/ full-weight cuts with cut.json written last", "run_args.json", "samples.jsonl",
                "train_steps.jsonl"),
    # EXT-140: train_whisper.py DEPLOYABLE (cut.json `deployable`): weights, model and generation config, processor
    # (feature extractor and tokenizer), the deployment decoder (MM-17); base_ref.json and trainer_state.json are not.
    "components": ("model.safetensors", "config.json", "generation_config.json", "processor_config.json",
                   "tokenizer.json", "tokenizer_config.json", "decoding.json"),
    # EXT-143: observed per-example log (MM-23 with sample log); a log that does not match the run refuses (no fallback,
    # as ddpo_sd15; the README proposed `stage`).
    "consumption": {"target": "row", "fallback": None},
    "gpu_h_estimate": 0.15,                          # estimate (README "Profiles and cost")
    "evaluators": ("asr.whisper_items@0.1",),
    "framework": "transformers",
    # EXT-149: the interpreter, the runtime directory (scripts, venv, models, data, items) and the training runs' outputs
    # (an evaluation mounts a cut read-only at /inputs/cut).
    "input_roots": HOST_ROWS["input_roots"],
    # The base, the data and the items lie inside the runtime's identity mount; --spec optional, checked when given.
    "spec_binding": False,
    # README "Launch request facts": every name, none a credential.
    "env_names": ("HF_HUB_OFFLINE", "HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "TOKENIZERS_PARALLELISM",
                  "XDG_CACHE_HOME"),
    "scan_skip": HOST_ROWS["scan_skip"],
    "interpreter_trees": HOST_ROWS["interpreter_trees"],
    "interpreter": HOST_ROWS["interpreter"],
    "container_paths": CONTAINER_PATHS,
    "env_paths": ("HOME", "XDG_CACHE_HOME"),
}

# Area "audio" evaluator (MULTIMODAL-SPEC section 6 row 5; decisions "Area 4 decisions", not published: deciding, T3,
# blocking). Scorer:
# eval_asr.py (per-utterance word error rate after the Whisper English normalizer with the base repo's normalizer.json,
# word-level Levenshtein), code_sha256 its SHA-256; the version is the string it writes with transformers 5.21.0 (its
# 2026-11-13 CPU sanity evaluation). An item is one registered AMI IHM test utterance; its raw AMI `text` is the
# exact-overlap key against the training rows' `text` (EXT-144). Corpus WER (a ratio of sums) is descriptive only: the
# estimator is the per-item mean (items.EVALUATORS, MM-GAP-13).
EVAL_ASR_SHA256 = "sha256:b53355619b480361ddbcde8d8ad5eee072275797aa2b908ae576275f08b21f84"
EVALUATOR_ROWS["asr.whisper_items@0.1"] = {
    "metric": "wer",
    "direction": "lower_is_better",
    "score_range": (0, None),                        # an insertion-heavy transcript exceeds 1
    "binary": False,
    # 05 section 6.1: a deterministic program, not the training loss (token cross-entropy); R16-1: the references come
    # from the training labels' annotation process but no training stage consumed them, so it stays T3, and the
    # registration and the report state the shared convention (`shared_annotation`, below).
    "tier": "T3",
    "shared_annotation": {
        "convention": ("AMI Meeting Corpus manual verbatim transcription (IHM, Kaldi s5 segmentation; repetitions, "
                       "false starts and partial words kept)"),
        "statement": ("The references of wer are held-out AMI IHM test transcripts made by the same AMI manual verbatim "
                      "transcription process as the training labels; no training stage consumed them, so the "
                      "programmatic scorer stays T3 (05 section 6.1, R16-1). Part of a gain may be learning that "
                      "convention (keeping the disfluencies the references keep): a pass shows that the model "
                      "transcribes AMI-style meeting speech of unseen speakers closer to the AMI references, not a "
                      "general accuracy gain.")},
    "error_classes": ("decode_error", "scorer_error"),   # MM-4: generate raised; the normalizer or alignment raised
    "score_outcomes": ("empty_transcript", "truncated_transcript"),   # MM-4: scored (WER 1; the per-utterance bound)
    "payload_text_field": "text",
    "item_payload": ("text", "reference_normalized", "audio", "audio_sha256"),
    "normalizer": False,                             # no embodied normalizer; the text normalizer is in the scorer
    "env_version": False,
    # EXT-145: eval_asr.py `sampler`: the deployment decoder (every cut's decoding.json) plus the evaluation's batch
    # size, dtype and item order, exactly these keys.
    "sampler_keys": ("batch_size", "do_sample", "dtype", "input", "language", "long_form", "max_new_tokens",
                     "max_text_tokens_rule", "num_beams", "order", "return_timestamps", "sample_rate", "strategy",
                     "temperature_fallback"),
    "framework": "transformers",
    "samples": "500, 1",
    "scorer": {"name": "per_utterance_wer",
               "version": ("eval_asr@0.1 transformers 5.21.0 EnglishTextNormalizer normalizer.json "
                           "sha256:bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd; word-level "
                           "Levenshtein"),
               "weights_sha256": None, "code_sha256": EVAL_ASR_SHA256},
}

PLUGIN = {"adapters": {"hf_seq2seq": "gatedtrain_recipe_whisper_sft_ami.adapter:HfSeq2SeqAdapter"},
          "trainer_rows": TRAINER_ROWS, "evaluator_rows": EVALUATOR_ROWS}
