# clipfarm

Turn long videos (podcasts, interviews, talks) into captioned vertical shorts. An AI reads the
transcript and what the audience already liked, picks the best moments, and each one comes out
as a ready-to-post 1080x1920 video. Nothing gets posted on its own: you approve or pass on every
clip on a local review page.

It's one Python file. The AI and the transcription are both swappable, so you can run it on
Claude, OpenAI, a local model, or anything else that takes a prompt.

## What it does

1. **Ingest.** Downloads the audio with [yt-dlp](https://github.com/yt-dlp/yt-dlp) (YouTube and
   [most other video sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)), or takes
   a video file you already have. From YouTube it also grabs the top 500 comments and the "most
   replayed" graph. Then it transcribes the audio with word-level timestamps.
2. **Pick.** Sends the AI the numbered transcript plus three audience signals: timestamps people
   typed in the comments, the most-liked comments, and the most-replayed peaks. The AI proposes six
   moments, each with a score, a title, a short on-screen headline, a description, the evidence
   behind it, and an honest rating of how easily it could be misread out of context. It answers with
   sentence numbers rather than times, so cuts always land on sentence boundaries.
3. **Render.** Keeps the best two moments per video (score 7 or more, never a high context risk),
   optionally checks each against your own editorial rule, downloads only those seconds of video,
   and builds the short: blurred backdrop, the video in the middle, your headline above, word-by-word
   captions below, a progress bar, and loudness evened out to -14 LUFS.
4. **Review.** A page at `http://localhost:8765` shows every clip. Edit the title and description,
   then Approve or Pass. Approved clips are copied to `approved/` with a text file holding the
   title, description and a link back to the source moment, ready to upload.

Every stage skips work it has already done, so you can rerun anything safely.

## Setup

You need [uv](https://docs.astral.sh/uv/) (it installs Python and every library on first run) and
[ffmpeg](https://ffmpeg.org/download.html). On a Mac: `brew install uv ffmpeg`. YouTube downloads
also want a JavaScript runtime for yt-dlp; if they fail, `brew install deno`.

```sh
git clone https://github.com/Noah-Airmet/clipfarm
cd clipfarm
cp .env.example .env                  # pick your AI and add its key
cp config.example.json config.json    # describe your channel, list your videos
uv run clipfarm.py run                # ingest, pick and render everything in config.json
uv run clipfarm.py review             # approve or pass at http://localhost:8765
```

You can also skip the list and pass links or files straight in:

```sh
uv run clipfarm.py run "https://www.youtube.com/watch?v=VIDEO_ID" ~/Movies/episode-12.mp4
```

Or run one stage at a time: `ingest`, `pick`, `render`.

## Plugging in your own AI

Set `LLM_PROVIDER` in `.env`:

| Provider | What it uses | Set |
|---|---|---|
| `anthropic` (default) | Claude through the Anthropic API | `ANTHROPIC_API_KEY`; `LLM_MODEL` defaults to `claude-opus-5-5` |
| `openai` | Any OpenAI-compatible server: OpenAI, OpenRouter, Groq, Together, Ollama, LM Studio, vLLM | `OPENAI_API_KEY`, `OPENAI_BASE_URL` (empty for OpenAI itself), `LLM_MODEL` |
| `claude-cli` | The `claude` command from [Claude Code](https://claude.com/claude-code), so it can run on a Claude subscription instead of API credit | nothing, if `claude` is installed and signed in |
| `command` | Any shell command that reads a prompt on stdin and prints the answer | `LLM_COMMAND`, e.g. `llm -m gpt-5` |

The picking step is where model quality shows. A strong model finds moments that really stand on
their own; a small local model will work but picks more fragments. On Claude Opus, picking costs
roughly ten to twenty cents per hour of video. `LLM_FAST_MODEL` lets the cheap yes/no editorial check
use a smaller model.

All the AI calls go through one function, `ask_json` in `clipfarm.py`. To add another provider, add
a branch there that sends the prompt and returns the text.

## Plugging in your own transcription

Set `TRANSCRIBER` in `.env`:

- `local` (default): [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on your own
  machine. Free and private. `WHISPER_MODEL` trades speed for accuracy: `small` does an hour of audio in
  about six minutes on a recent Mac; `large-v3` or `turbo` get names and jargon right more often.
- `openai`: any OpenAI-compatible transcription API. OpenAI's `whisper-1` works, and
  [Groq](https://console.groq.com) runs `whisper-large-v3-turbo` very fast and cheaply
  (`TRANSCRIBE_BASE_URL=https://api.groq.com/openai/v1`). Audio is squeezed to about 11 MB an hour
  before upload; very long episodes can still go over a provider's file size limit.

Add the names and slang your speakers use to `glossary` in `config.json`. They're handed to the
transcriber as hints, which helps a lot with names.

The transcriber lives in `transcribe` in `clipfarm.py`. Anything that returns a list of words with start
and end times can go there (Apple Silicon users may like `mlx-whisper` or `parakeet-mlx`, which are much
faster on a Mac).

## Making it yours

Everything in `config.json`:

- `channel`: the name shown at the top of each short and the handle in the footer ("FULL EPISODE ·
  @yourshow"). Leave `handle` empty to drop the footer.
- `about` and `what_makes_a_good_clip`: plain sentences about your channel and audience. This is the
  most useful thing to tune. The AI picks very differently for a comedy podcast than for a lecture series.
- `videos`: links, YouTube ids, or file paths. An entry can be `{"url": "...", "note": "the part where
  they argue about salt"}`, and the AI will make sure that moment is one of its picks.
- `candidates_per_video`, `clips_per_video`, `min_score`, `min_seconds`, `max_seconds`: how many
  moments to consider, how many to render, the score bar, and the length range.
- `gate`: an optional editorial rule in plain English, checked against each clip with a few minutes of
  surrounding transcript before it renders. For example: "Drop it if a guest could look like they're
  saying the opposite of what they mean" or "Keep only clips where the host's point is answered in the
  clip." Leave it empty to skip the check. Verdicts are printed during `render` and cached in
  `work/<video>/gate.json`.
- `style`: the accent color (the lit caption word and the progress bar), text color and fonts. The
  bundled fonts are [Geist](https://vercel.com/font) (SIL Open Font License, see `fonts/OFL-Geist.txt`);
  drop any `.ttf` into `fonts/` and name it here, or give a full path.

The layout itself (sizes, positions, the blurred backdrop) is in `chrome_png`, `caption_frames` and
`render_one`.

## Redoing things

Everything a stage makes lives in `work/<video id>/`:

- delete `moments.json` to pick that video again (after changing `about`, say)
- delete a clip's folder under `clips/` to render it again
- delete `gate.json` after changing your `gate` rule
- delete `transcript.json` to transcribe again with a different model

## Ideas for going further

This is a stripped-down version of a bigger setup. Things worth adding if you need them:

- **Uploading on approve.** The YouTube Data API can upload each approved short as private with a
  scheduled publish time, so approving is posting and there's still a day to pull a mistake.
- **A submit page and a queue,** so other people can paste a link and a small worker runs it overnight.
- **Long-form segments.** The same transcript can yield 8 to 25 minute topical chapters for their own
  uploads, each with a generated thumbnail.
- **A second caption pass.** Transcribe just the clip again with a different model, and have a cheap AI
  fix only the words the two transcripts disagree on.
- **Speaker-following crops** that fill the 9:16 frame instead of letterboxing the 16:9 shot.
- **Learning from results:** pull each upload's views and retention back in and feed the winners to the
  picker as examples.

## A note on permission

Only clip videos you own or have permission to repost. Credit the source in the description; the
text files in `approved/` include a timestamped link back to the original moment.

## License

MIT. See `LICENSE`.
