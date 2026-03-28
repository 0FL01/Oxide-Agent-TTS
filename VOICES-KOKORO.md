# Voices

## American English

- `lang_code='a'` in [`misaki[en]`](https://github.com/hexgrad/misaki)
- espeak-ng `en-us` fallback

| Name | Traits | Target Quality | Training Duration | Overall Grade | SHA256 |
| ---- | ------ | -------------- | ----------------- | ------------- | ------ |
| **af\_heart** | 🚺❤️ | | | **A** | `0ab5709b` |
| af_alloy | 🚺 | B | MM minutes | C | `6d877149` |
| af_aoede | 🚺 | B | H hours | C+ | `c03bd1a4` |
| af_bella | 🚺🔥 | **A** | **HH hours** | **A-** | `8cb64e02` |
| af_jessica | 🚺 | C | MM minutes | D | `cdfdccb8` |
| af_kore | 🚺 | B | H hours | C+ | `8bfbc512` |
| af_nicole | 🚺🎧 | B | **HH hours** | B- | `c5561808` |
| af_nova | 🚺 | B | MM minutes | C | `e0233676` |
| af_river | 🚺 | C | MM minutes | D | `e149459b` |
| af_sarah | 🚺 | B | H hours | C+ | `49bd364e` |
| af_sky | 🚺 | B | _M minutes_ 🤏 | C- | `c799548a` |
| am_adam | 🚹 | D | H hours | F+ | `ced7e284` |
| am_echo | 🚹 | C | MM minutes | D | `8bcfdc85` |
| am_eric | 🚹 | C | MM minutes | D | `ada66f0e` |
| am_fenrir | 🚹 | B | H hours | C+ | `98e507ec` |
| am_liam | 🚹 | C | MM minutes | D | `c8255075` |
| am_michael | 🚹 | B | H hours | C+ | `9a443b79` |
| am_onyx | 🚹 | C | MM minutes | D | `e8452be1` |
| am_puck | 🚹 | B | H hours | C+ | `dd1d8973` |
| am_santa | 🚹 | C | _M minutes_ 🤏 | D- | `7f2f7582` |

## Notes

For each voice, the given grades are intended to be estimates of the **quality and quantity** of its associated training data, both of which impact overall inference quality.

Subjectively, voices will sound better or worse to different people.

Most voices perform best on a "goldilocks range" of 100-200 tokens out of ~500 possible. Voices may perform worse at the extremes:
- **Weakness** on short utterances, especially less than 10-20 tokens. Root cause could be lack of short-utterance training data and/or model architecture. One possible inference mitigation is to bundle shorter utterances together.
- **Rushing** on long utterances, especially over 400 tokens. You can chunk down to shorter utterances or adjust the `speed` parameter to mitigate this.

**Target Quality**
- How high quality is the reference voice? This grade may be impacted by audio quality, artifacts, compression, & sample rate.
- How well do the text labels match the audio? Text/audio misalignment (e.g. from hallucinations) will lower this grade.

**Training Duration**
- How much audio was seen during training? Smaller durations result in a lower overall grade.
- 10 hours <= **HH hours** < 100 hours
- 1 hour <= H hours < 10 hours
- 10 minutes <= MM minutes < 100 minutes
- 1 minute <= _M minutes_ 🤏 < 10 minutes
