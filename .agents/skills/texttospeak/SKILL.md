---
name: texttospeak
description: Turn text into spoken audio with local Kokoro TTS on this Mac — a podcast-style conversation between voices, or a single-narrator explanation — and hand back an M4A/WAV. Use when the user asks for a podcast, narration, voiceover, audiobook, audio explanation, "read this aloud", a spoken version, TTS, or a dialogue between voices. Needs Apple Silicon and the Bash sandbox disabled for the render step.
allowed-tools: Bash(bash ${CLAUDE_SKILL_DIR}/texttospeak.sh *)
argument-hint: [text or topic] [podcast|narrate]
---

# texttospeak

**Rule zero:** do not read this repo's source or README; everything you need is here. This runs only on Apple Silicon Macs.

## 1. Write the script

Write the script to `script.txt` in the scratchpad (`$SCRATCH` below).

- One utterance per line. EVERY line starts with a speaker label and a colon: `voice1: ...`
- `voice1` = host or narrator (warm American female), `voice2` = guest (American male), `voice3` = British female, `voice4` = British male. Any other label errors.
- Blank lines are ignored.
- Keep each line to 1 to 4 sentences. Avoid one-word lines (voices sound weak on very short lines) and 400+ word lines (they rush).
- Plain prose only: no markdown, headings, bullets, emoji, stage directions or parentheses. Parentheses are silently dropped.
- Control pauses with commas, semicolons, em dashes and ellipses inside a line, and with `--pause` between lines.

Numbers and symbols:

- Digits, ordinals (1st), years, $3.50 and 50% are read correctly as-is.
- A bare 4-digit number is read as a year ("1500 people" becomes "fifteen hundred people"). Write "one thousand five hundred" when it is a count.
- Write out times ("ten thirty", not 10:30), units ("five kilometres", not 5 km; unknown abbreviations are dropped silently), ranges ("three to five", not 3-5), fractions and phone numbers.
- Common acronyms like NASA are spoken as words. For letter-by-letter, space them: `A P I`.
- ALL CAPS on a normal word adds emphasis.
- Hard names: give phonemes as `[Name](/fəˈnimz/)`. This is the only markup allowed.

Length: about 150 words per minute of audio, so a 5-minute piece is about 750 words.

Podcast pattern: voice1 opens with a hook and frames the topic, voice2 explains, voice1 asks the listener's questions, 5 to 8 exchanges per point, voice1 closes with the takeaway.

Explanation pattern: voice1 only, short paragraphs, one idea per line.

## 2. Render

Run with the sandbox DISABLED (`dangerouslyDisableSandbox: true`). The GPU is unreachable inside the sandbox and the error will mention Metal.

```bash
bash "${CLAUDE_SKILL_DIR}/texttospeak.sh" -o "$SCRATCH/episode.wav" < "$SCRATCH/script.txt"
```

Options:

- `--pause 0.5` seconds between lines (default 0.35)
- `--speed 0.95` speech rate, 0.5 to 2.0
- `--split` also writes one WAV per line
- `--cast voice1=bf_emma,voice2=am_fenrir` swaps voices. Good alternatives: af_heart, af_bella, am_michael, am_fenrir, am_puck, bf_emma, bm_george, bm_fable
- `--list-voices` prints every voice

Expect a model-load pause of a few seconds, then rendering many times faster than real time (two lines take under 5 s total). The command prints the output path.

First run pause: the very first run on a machine downloads the model and may take a few minutes (needs network). Wait for it; do not cancel and retry.

## 3. Deliver

Convert for sharing (WAV is about 2.9 MB per minute):

```bash
afconvert -f m4af -d aac -b 64000 "$SCRATCH/episode.wav" "$SCRATCH/episode.m4a"
```

Send the m4a with SendUserFile, and through the discord skill too if that skill is available and the user follows there. Keep the WAV if the user asked for WAV or wants to edit it.

## Troubleshooting

- Any error mentioning Metal or MLX: the sandbox was on. Rerun with it disabled.
- `unknown speaker 'X'`: a line did not start with voice1 to voice4 (or a valid voice name).
- Exit 127: run the printed install command, then retry.
- Odd pronunciation: fix the text per section 1 (time, unit, range and year rules) or use a phoneme override.
