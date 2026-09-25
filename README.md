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
`--voice` (default for bare text), `--model`, `--speed`, `--watch`.

To fire many requests without paying the ~4s model-load cost each time, use
`--watch`: it keeps the process alive and synthesizes one `voice:text` line
per stdin line as it arrives. Pipe requests into it from a FIFO or another
process:

```bash
mkfifo /tmp/speak.fifo
speak --watch -d out/ < /tmp/speak.fifo &
echo 'af_heart: first' > /tmp/speak.fifo
echo 'am_michael: second' > /tmp/speak.fifo
```

### Large batches

For a big batch (e.g. 500 lines), skip `--watch` — plain stdin mode already
loads the model once and renders every line in one process:

```bash
speak -d out/ < requests.txt
```

`requests.txt` is just `voice: text` per line:

```
af_heart: hi how are you today
am_michael: i'm fine thanks
```

Before a large run, sanity-check the voice names you used against the list
below — a typo (e.g. `am_micheal`) won't error, it'll just fail or mispronounce:

```bash
cut -d: -f1 requests.txt | sort -u
```

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

## Server mode

`speak-serve` keeps the model loaded and streams speech to other local
processes over a socket. Audio comes back sentence by sentence as it's
produced, so playback can start before the whole text is rendered.

```bash
speak-serve                          # TCP on 127.0.0.1:7333
speak-serve --socket /tmp/speak.sock # Unix socket instead
```

It binds first, then loads the model and runs a short warm-up (skip it with
`--no-warmup`), and logs `ready, listening on ...` once requests will be
served. Connections made while it's loading wait until it's ready. Other
flags: `--host`, `--port`, `--model`, `--speed` (default rate),
`--split-pattern`, `--max-text-chars`, `--log-level`.

To keep latency low, the server has the model split text at sentence
boundaries and streams each sentence as soon as it's rendered. `--split-pattern`
changes that regex. The `speak` file commands don't split this way; they keep
the model's own chunking, so a line read through the server can sound
slightly different from the same line rendered by `speak`.

Use `--watch` when you just want WAV files on disk from a shell pipeline. Use
the server when another program wants the audio itself: it gets raw PCM back
as it's produced, can run several requests at once over one connection, and
can cancel them.

### Python client

```python
from speakctl import SpeakClient

with SpeakClient() as client:  # or SpeakClient("/tmp/speak.sock")
    for pcm in client.stream("Welcome back. Today we talk about bees.", voice="af_heart"):
        player.write(pcm)  # s16le mono at client.sample_rate

    pcm, rate = client.synthesize("One-shot, whole clip.", voice="am_michael", speed=1.1)

    response = client.submit("A long intro...")  # returns immediately
    response.cancel()
```

Failures raise `SpeakError`, whose `.code` is one of the server codes below
or a client-side one: `connection_closed` (the socket dropped) or `timeout`.
`chunks()`, `stream()` and `synthesize()` take `timeout=`, the longest wait
for the next frame; when it expires the request is cancelled. The package
imports only the standard library for `SpeakClient`, so a client app doesn't
need numpy or MLX installed.

### Wire protocol

Every frame, in both directions, is a 9-byte header followed by a payload:

| Field         | Type                    | Notes                                   |
|---------------|-------------------------|-----------------------------------------|
| `type`        | u8                      | frame type, below                       |
| `request_id`  | u32, big-endian         | chosen by the client; 0 is reserved     |
| `payload_len` | u32, big-endian         | bytes of payload that follow            |

| Type      | Value  | Direction        | Payload                                                         |
|-----------|--------|------------------|-----------------------------------------------------------------|
| `REQUEST` | `0x01` | client to server | JSON `{"text": str, "voice": str, "speed": number}`, speed optional |
| `CANCEL`  | `0x02` | client to server | empty                                                           |
| `HELLO`   | `0x10` | server to client | JSON, sent once on connect with `request_id` 0                  |
| `START`   | `0x11` | server to client | JSON `{"sample_rate": int, "channels": 1, "format": "s16le"}`   |
| `AUDIO`   | `0x12` | server to client | raw signed 16-bit little-endian mono PCM                        |
| `END`     | `0x13` | server to client | empty                                                           |
| `ERROR`   | `0x14` | server to client | JSON `{"code": str, "message": str}`                            |

`HELLO` carries `protocol` (1), `sample_rate`, `channels`, `format`, `model`,
`max_text_chars` and `max_inflight`. JSON is UTF-8. Client frames may carry
at most 1 MiB of payload.

A request's frames arrive as `START`, zero or more `AUDIO`, then `END`; or
`ERROR` at any point instead of `END`. `END` or `ERROR` is always the last
frame for that id, after which the id may be reused.

| Code                | Meaning                                                                 |
|---------------------|-------------------------------------------------------------------------|
| `bad_request`       | id 0 or already in flight, bad JSON, empty or too-long text, voice not matching `[A-Za-z0-9_.-]{1,64}`, speed outside 0.5 to 2.0 |
| `too_many_requests` | the connection already has `max_inflight` requests queued or rendering |
| `cancelled`         | the request was cancelled                                               |
| `engine_error`      | the model failed on this request; the server and connection carry on   |
| `protocol_error`    | unreadable framing; sent with `request_id` 0, then the server closes the connection |

**Pipelining.** A client may send many `REQUEST`s without waiting. One
worker renders requests in arrival order across all connections, so frames
for different ids rarely interleave. Clients must still route frames by
`request_id`: a validation `ERROR` for one request is sent straight away and
can land between another request's `AUDIO` frames. An `ERROR` for a
duplicate id refers to the rejected frame; the original request keeps
streaming. An `ERROR` with `request_id` 0 is about the connection as a whole.

**Cancellation.** `CANCEL` stops a queued request before it starts, or a
running one at the next chunk. Either way the request ends with
`ERROR cancelled` (unless it had already finished; a late `CANCEL` is
ignored). The server treats end-of-stream from the client as a disconnect and
cancels everything that connection had in flight, so keep the write side open
until the last `END` arrives: don't half-close it to signal "no more requests".

**Unix socket.** With `--socket PATH` the server listens on a Unix domain
socket; the protocol is identical. If the path exists, the server connects
to it first: if another server answers, startup fails with "address in use";
if nothing does (a leftover from a crashed server), the file is replaced. The
server removes its socket file on shutdown.

**Slow readers.** Audio waiting to be sent is buffered per connection, so
a client that plays audio in real time as it arrives never holds up other
clients or the model. If a client falls more than 64 MiB behind (about 23
minutes of audio), the server cancels its requests and drops the connection.

A client in any language only needs this loop:

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

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pytest
.venv/bin/pytest -q
```

Tests use a fake backend, so they need neither MLX nor the model.

## Layout

`src/speakctl/engine.py` holds `TTSEngine`, which turns text into 16-bit PCM
through a backend; `mlx_backend.py` is the MLX-Audio backend and the only
module that imports MLX. `protocol.py` defines the server's wire format,
`server.py` the server and `client.py` the Python client. `cli.py` provides
the `speak` and `speak-serve` commands.
