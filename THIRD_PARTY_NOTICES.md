# Third-party notices

This repository redistributes the following third-party material. Everything else in it is covered by `LICENSE`
(Apache License 2.0).

## AMI Meeting Corpus (transcripts)

- **What:** utterance transcripts with their AMI meeting ids and speaker ids, in `runtime/items/items.jsonl` (500 rows
  of the AMI test split: meetings EN2002, ES2004, IS1009, TS3003) and `tests/fixtures/whisper/items.sample.jsonl`,
  `tests/fixtures/whisper/train.sample.jsonl` (3 rows each). No audio is redistributed.
- **Source:** the AMI Meeting Corpus, https://groups.inf.ed.ac.uk/ami/corpus/, individual headset microphone (IHM)
  segmentation and text of the Kaldi `egs/ami/s5` recipe, as published in the Hugging Face dataset
  `edinburghcstr/ami` (config `ihm`) at revision `46f28f2503e2ec48f8867a84eef356c70476beab`. The shard files and their
  SHA-256 are listed in `runtime/data-source/DATA.json`.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/legalcode. The corpus's licence page
  (https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) states: "The AMI corpus and its annotations are released
  under the Creative Commons Attribution 4.0 license agreement (also called CC BY 4.0)."
- **Changes:** utterances selected by `runtime/make_data.py` (seeded draw, filters on length and word count, see
  `runtime/README.md` "Data"); each row reshaped to `{item, text, speaker_id, meeting_id, duration, n_words, ...}`; a
  `reference_normalized` field added (the Whisper English text normalizer applied to `text`); `audio` and
  `audio_sha256` refer to WAV files rebuilt locally from the source shards. The transcript text itself is unchanged.
- **No endorsement:** the AMI corpus authors do not endorse this repository or its results.
