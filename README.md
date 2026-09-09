# speakctl

Local text-to-speech on macOS (Apple Silicon) using [MLX-Audio](https://github.com/Blaizzy/mlx-audio)
with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache 2.0, by
[hexgrad](https://huggingface.co/hexgrad)). No cloud calls after the model is cached.

## Install

Requires an Apple Silicon Mac (MLX wheels are arm64-macOS only).

```bash
git clone <repo-url>
cd TextToSpeak
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
speak "hello world"

# two voices in one run
speak -j af_heart:"Hello there" -j am_michael:"And here's the reply" -d out/
```

First run downloads Kokoro-82M plus `misaki[en]` (pulls in torch/spacy) —
a few GB, cached under `~/.cache/huggingface`.

## Use

Pair each snippet with the voice it should be spoken in. All snippets run in one
process, so the model loads once no matter how many you pass:

```bash
speak -j af_heart:"Hello there" -j am_michael:"And here's the reply" -d out/
# out/001_af_heart.wav
# out/002_am_michael.wav
```

Same thing from stdin, one `voice: text` line per snippet:

```bash
printf 'af_heart: first line\nam_michael: second line\n' | speak -d out/
```

Single snippet to a named file:

```bash
speak "hello world" -o hello.wav --voice bf_emma
```

Flags: `-j/--job VOICE:TEXT` (repeatable), `-o/--output`, `-d/--outdir`,
`--voice` (default for bare text), `--model`, `--speed`.

Voices are Kokoro's: the first letter is the language pipeline, the second is
gender (`f`/`m`) — `a`=American, `b`=British, `e`=Spanish, `f`=French,
`h`=Hindi, `i`=Italian, `j`=Japanese, `p`=Brazilian Portuguese, `z`=Mandarin.
Any MLX-Audio TTS model id works via `--model`.

All 54 voices (fetched from the [prince-canuma/Kokoro-82M](https://huggingface.co/prince-canuma/Kokoro-82M)
mirror, which mlx-audio's model conversion pulls voice files from):

```
af_alloy    af_aoede    af_bella    af_heart    af_jessica  af_kore
af_nicole   af_nova     af_river    af_sarah    af_sky
am_adam     am_echo     am_eric     am_fenrir   am_liam     am_michael
am_onyx     am_puck     am_santa
bf_alice    bf_emma     bf_isabella bf_lily
bm_daniel   bm_fable    bm_george   bm_lewis
ef_dora     em_alex     em_santa
ff_siwis
hf_alpha    hf_beta     hm_omega    hm_psi
if_sara     im_nicola
jf_alpha    jf_gongitsune jf_nezumi jf_tebukuro jm_kumo
pf_dora     pm_alex     pm_santa
zf_xiaobei  zf_xiaoni   zf_xiaoxiao zf_xiaoyi
zm_yunjian  zm_yunxi    zm_yunxia   zm_yunyang
```

## Layout

`src/speakctl/engine.py` holds all model logic (`TTSEngine`); `cli.py` is a thin
wrapper around it.
