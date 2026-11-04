# Smoke test sample

`hello.wav` is the audio file the platform's smoke test uploads to this profile's
`POST /v1/audio/transcriptions` endpoint (the `file` part of the smoke test's
multipart form in `ai-deploy.yaml`). The control plane reads it from the profile as
it stores it and sends it through the API gateway; it is not shipped to the
inference node with the model.

## Format

- 16 kHz, mono, 16-bit PCM WAV, 2.47 seconds, 79084 bytes.
- SHA-256: `6b1a98f08e1cb6fb787f0bfccb95acfa592ac73ab9905ab2ea289a50a383a5c5`.

## Source

- Dataset: LibriSpeech ASR corpus, `dev-clean` subset
  (<https://www.openslr.org/12>).
- Speaker 1272, chapter 128104, utterance `1272-128104-0008`. Its transcript:
  "AS FOR ETCHINGS THEY ARE OF TWO KINDS BRITISH AND FOREIGN".
- Downloaded as one FLAC file (16 kHz, mono, 5.12 seconds) through the Hugging Face
  dataset viewer of the `hf-internal-testing/librispeech_asr_dummy` dataset, a
  73-utterance excerpt of LibriSpeech `dev-clean`
  (<https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy>), row 8
  of its `clean` configuration's `validation` split:
  <https://datasets-server.huggingface.co/rows?dataset=hf-internal-testing/librispeech_asr_dummy&config=clean&split=validation&offset=8&length=1>.
- Cut from 0.45 s to 2.92 s of the utterance and converted with:

  ```bash
  ffmpeg -i audio.flac -ss 0.45 -to 2.92 -ac 1 -ar 16000 -c:a pcm_s16le \
    -map_metadata -1 -fflags +bitexact -flags:a +bitexact hello.wav
  ```

  The cut points are pauses that ffmpeg's `silencedetect` (-35 dB) finds in the
  utterance: the leading silence ends at 0.55 s, and a pause runs from 2.86 s to
  2.95 s. The clip holds the beginning of the transcript up to that pause; the exact
  last word was not checked by ear.

## License

LibriSpeech is licensed under the Creative Commons Attribution 4.0 International
license (CC BY 4.0, <https://creativecommons.org/licenses/by/4.0/>).

Attribution: "LibriSpeech ASR corpus" by Vassil Panayotov, Guoguo Chen, Daniel Povey
and Sanjeev Khudanpur, <https://www.openslr.org/12>, licensed under CC BY 4.0. The
corpus is derived from LibriVox audiobooks. Changes: this file is a 2.47-second excerpt
of utterance `1272-128104-0008`, converted from FLAC to 16-bit PCM WAV.
