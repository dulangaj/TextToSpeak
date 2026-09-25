# speakctl

Local, offline text-to-speech on the Mac with [MLX-Audio](https://github.com/Blaizzy/mlx-audio) and
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M): a `speak` command that writes WAV files, a
`texttospeak` command that turns a multi-voice conversation into one WAV, and a `speak-serve` server that
streams audio to your apps.

Requirements: Apple Silicon Mac, Python 3.10+ (3.12 tested).

## Quick start (60 seconds)

```bash
git clone https://github.com/dulangaj/TextToSpeak.git && cd TextToSpeak
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
speak "hello world"
```

The first run downloads Kokoro-82M plus the misaki/spaCy English model (a few GB, cached under
`~/.cache/huggingface`), so it takes a few minutes once. Everything after that runs offline.

```bash
speak -j af_heart:"Hello there" -j am_michael:"And here's the reply" -d out/   # two voices to files
speak-serve                                                                   # stream to apps on 127.0.0.1:7333
```

### Conversations

```bash
texttospeak "voice1: Hello there" "voice2: Hi, how are you?" -o chat.wav
```

`voice1` to `voice4` are `af_heart`, `am_michael`, `bf_emma` and `bm_george`; bare text is `voice1`, and any
voice name works as a speaker too (`am_adam: Hi`). Swap voices with `--cast voice1=bf_alice,voice2=am_adam`,
add `--split` to also get one file per line (`001_voice1.wav`, ...), and see every name with `--list-voices`.
Lines can come from stdin (`texttospeak -o chat.wav < script.txt`); `--pause` sets the gap (default 0.35 s).
A Claude Code skill for this command lives in `.claude/skills/texttospeak/`; symlink that folder into
`~/.claude/skills/` to use it from any project.

## Using the CLI

`speak` prints the path of each WAV it writes. All snippets in one run share one model load.

```bash
speak "hello world" -o hello.wav --voice bf_emma --speed 1.1   # one snippet, named file
speak -j af_heart:"Hi" -j am_michael:"Hey" -d out/             # out/001_af_heart.wav, out/002_am_michael.wav
printf 'af_heart: first line\nam_michael: second line\n' | speak -d out/
speak -d out/ < requests.txt                                    # big batch: one "voice: text" per line
```

| Flag | Meaning |
|------|---------|
| `-j, --job VOICE:TEXT` | a snippet and its voice (repeatable) |
| `-o, --output FILE` | output file (single snippet only) |
| `-d, --outdir DIR` | directory for generated files (default `.`) |
| `--voice NAME` | voice for bare text and stdin lines without `voice:` (default `af_heart`) |
| `--speed X` | speech rate multiplier (default 1.0) |
| `--model ID` | MLX-Audio model id (default `mlx-community/Kokoro-82M-bf16`) |
| `--watch` | stay running and render each stdin line as it arrives |

`--watch` is for a steady trickle of requests from another process without reloading the model each time:

```bash
mkfifo /tmp/speak.fifo
speak --watch -d out/ < /tmp/speak.fifo &
echo 'af_heart: first' > /tmp/speak.fifo
```

A misspelled voice name doesn't error up front, so check a big batch first: `cut -d: -f1 requests.txt | sort -u`.

## Using the server (for apps)

`speak-serve` loads the model once and streams audio back sentence by sentence, so playback starts before
the whole text is rendered: first audio arrives in well under a second after warm-up. One connection can
run several requests at once and cancel them.

```bash
speak-serve                           # TCP on 127.0.0.1:7333
speak-serve --socket /tmp/speak.sock  # Unix socket instead
```

It logs `ready, listening on ...` once it's serving; connections made earlier wait until then.

| Flag | Meaning |
|------|---------|
| `--host HOST` | TCP host (default `127.0.0.1`) |
| `--port PORT` | TCP port (default 7333) |
| `--socket PATH` | listen on a Unix socket instead of TCP |
| `--model ID` | MLX-Audio model id |
| `--speed X` | default speech rate for requests that don't set one (default 1.0) |
| `--no-warmup` | skip the warm-up synthesis at startup |
| `--log-level LEVEL` | `DEBUG`, `INFO` (default), `WARNING` or `ERROR` |
| `--max-text-chars N` | longest accepted request text (default 10000) |
| `--split-pattern REGEX` | where text is split for streaming (default: sentence ends and newlines) |

Because of that splitting, a line from the server can sound slightly different from the same line rendered
by `speak`.

### Python client

`SpeakClient` uses only the standard library, so the client app doesn't need numpy or MLX.

```python
import wave
from speakctl import SpeakClient

with SpeakClient() as client, wave.open("out.wav", "wb") as wav:  # or SpeakClient("/tmp/speak.sock")
    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(client.sample_rate)
    for pcm in client.stream("Welcome back. Today we talk about bees.", voice="af_heart"):
        wav.writeframes(pcm)  # or hand each chunk to an audio player as it arrives

    pcm, rate = client.synthesize("One-shot, whole clip.", voice="am_michael", speed=1.1)
    response = client.submit("A long intro...")  # returns at once; read response.chunks()
    response.cancel()
```

Failures raise `SpeakError`; `.code` is a server code below, or `connection_closed` / `timeout` from the
client. `stream()`, `synthesize()` and `chunks()` take `timeout=` (max wait per frame; on expiry the request
is cancelled).

### Wire protocol

For clients in other languages. Every frame, both directions, is a 9-byte header (`struct` format `!BII`)
followed by the payload:

| Field | Type | Notes |
|-------|------|-------|
| `type` | u8 | frame type, below |
| `request_id` | u32 big-endian | chosen by the client; 0 is reserved |
| `payload_len` | u32 big-endian | payload bytes that follow |

| Type | Value | Direction | Payload |
|------|-------|-----------|---------|
| `REQUEST` | `0x01` | client to server | JSON `{"text": str, "voice": str, "speed": number}` (speed optional) |
| `CANCEL` | `0x02` | client to server | empty |
| `HELLO` | `0x10` | server to client | JSON, once on connect, id 0: `protocol` (1), `sample_rate`, `channels`, `format`, `model`, `max_text_chars`, `max_inflight` |
| `START` | `0x11` | server to client | JSON `{"sample_rate": int, "channels": 1, "format": "s16le"}` |
| `AUDIO` | `0x12` | server to client | raw signed 16-bit little-endian mono PCM |
| `END` | `0x13` | server to client | empty |
| `ERROR` | `0x14` | server to client | JSON `{"code": str, "message": str}` |

JSON is UTF-8. Client payloads are capped at 1 MiB.

Per request: `START`, zero or more `AUDIO`, then `END`; or `ERROR` at any point instead of `END`. `END` or
`ERROR` is always the last frame for that id, after which the id may be reused.

| Code | Meaning |
|------|---------|
| `bad_request` | id 0 or already in flight, bad JSON, empty or too-long text, voice not matching `[A-Za-z0-9_.-]{1,64}`, speed outside 0.5 to 2.0 |
| `too_many_requests` | the connection already has `max_inflight` requests queued or rendering |
| `cancelled` | the request was cancelled |
| `engine_error` | the model failed on this request; the connection carries on |
| `protocol_error` | unreadable framing; sent with id 0, then the connection closes |

Rules a client must follow:

- Use a unique, nonzero `request_id` for each request in flight. Requests may be pipelined without waiting.
- Route every frame by `request_id`. Requests render one at a time in arrival order, but a validation
  `ERROR` is sent immediately and can land between another request's `AUDIO` frames. An `ERROR` for a
  duplicate id refers to the rejected frame; the original keeps streaming.
- An `ERROR` with id 0 is about the connection: the server closes it next.
- Keep the write side open until the last `END`. Closing or half-closing it counts as a disconnect and
  cancels everything in flight on that connection.
- Read audio promptly. A connection that falls more than 64 MiB behind (about 23 minutes of audio) has its
  requests cancelled and is dropped.
- `CANCEL` stops a queued request, or a running one at its next chunk; it then ends with `ERROR cancelled`.
  A `CANCEL` for a finished or unknown id is ignored.

With `--socket PATH` the protocol is identical over a Unix socket. A stale socket file from a crashed server
is replaced; if another server is listening there, startup fails. The file is removed on shutdown.

Minimal client loop:

```
connect
read 9 bytes -> unpack "!BII" -> (type, id, len); read len bytes    # HELLO
send pack("!BII", 0x01, 1, len(json)) + json                      # REQUEST id 1
loop:
    read 9 bytes; unpack "!BII"; read len bytes
    0x11 START -> open audio output at sample_rate
    0x12 AUDIO -> write payload to it
    0x13 END   -> done with this id
    0x14 ERROR -> fail this id with code/message (id 0: the connection is closing)
```

## Voices

The first letter is the language (`a` American, `b` British, `e` Spanish, `f` French, `h` Hindi, `i` Italian,
`j` Japanese, `p` Brazilian Portuguese, `z` Mandarin), the second is gender (`f`/`m`).

```
af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky
am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa
bf_alice bf_emma bf_isabella bf_lily bm_daniel bm_fable bm_george bm_lewis
ef_dora em_alex em_santa ff_siwis hf_alpha hf_beta hm_omega hm_psi if_sara im_nicola
jf_alpha jf_gongitsune jf_nezumi jf_tebukuro jm_kumo pf_dora pm_alex pm_santa
zf_xiaobei zf_xiaoni zf_xiaoxiao zf_xiaoyi zm_yunjian zm_yunxi zm_yunxia zm_yunyang
```

`--model` accepts any MLX-Audio TTS model id.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```
