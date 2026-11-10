#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10", "yt-dlp[default]", "faster-whisper>=1.1", "anthropic", "openai"]
# ///
"""Clip farm: turn long videos into captioned vertical shorts, with a person approving each one.

    uv run clipfarm.py run [url ...]   ingest, pick and render (the videos in config.json if no urls)
    uv run clipfarm.py ingest [url ...]   download audio and audience signals, transcribe
    uv run clipfarm.py pick [url ...]     an AI reads the transcript and signals and proposes moments
    uv run clipfarm.py render [url ...]   cut the best moments into 1080x1920 shorts
    uv run clipfarm.py review             open the review page at http://localhost:8765

Every stage skips work already done, so rerunning is safe. See README.md for setup.
"""
import hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
APPROVED = HERE / "approved"
FONTS = HERE / "fonts"


# ---------------------------------------------------------------- setup

def load_env():
    """Read KEY=value lines from .env into the environment (real environment variables win)."""
    f = HERE / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def env(key, default=""):
    return (os.environ.get(key) or default).strip()


def config():
    f = HERE / "config.json"
    if not f.exists():
        sys.exit("No config.json. Copy config.example.json to config.json and fill it in.")
    return json.loads(f.read_text())


def run(*cmd, timeout=3600):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"{Path(str(cmd[0])).name} failed: {(r.stderr or r.stdout).strip()[-600:]}")
    return r.stdout


def ytdlp(*args):
    return run(sys.executable, "-m", "yt_dlp", "--no-warnings", *args)


def jload(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def jsave(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def hms(t):
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


YT_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|live/|embed/)([\w-]{11})")


def source(entry):
    """A video from config.json or the command line: a YouTube link or id, another site yt-dlp
    supports, or a file on disk. Entries can be a string or {"url": ..., "note": ...}."""
    s = entry if isinstance(entry, str) else entry["url"]
    note = "" if isinstance(entry, str) else entry.get("note", "")
    p = Path(s).expanduser()
    if p.exists():
        vid = re.sub(r"[^\w-]+", "-", p.stem)[:40] + "-" + hashlib.sha1(str(p.resolve()).encode()).hexdigest()[:6]
        return {"id": vid, "file": str(p.resolve()), "note": note}
    m = YT_ID.search(s)
    if m or re.fullmatch(r"[\w-]{11}", s):
        vid = m.group(1) if m else s
        return {"id": vid, "url": f"https://www.youtube.com/watch?v={vid}", "note": note}
    return {"id": "v" + hashlib.sha1(s.encode()).hexdigest()[:10], "url": s, "note": note}


def videos(args):
    return [source(e) for e in (args or config().get("videos", []))]


# ---------------------------------------------------------------- AI (swap in your own here)

JSON_TAIL = "\n\nAnswer with only a JSON object that matches this schema, no other text:\n{schema}"


def ask_json(prompt, schema, fast=False):
    """Send a prompt, get back a dict matching `schema`. LLM_PROVIDER picks the backend:

    anthropic   Claude through the Anthropic API (ANTHROPIC_API_KEY)
    openai      any OpenAI-compatible server: OpenAI, OpenRouter, Groq, Ollama, LM Studio, vLLM...
                (OPENAI_API_KEY, OPENAI_BASE_URL)
    claude-cli  the `claude` CLI (Claude Code), which can run on a Claude subscription
    command     any shell command that reads the prompt on stdin and prints the answer (LLM_COMMAND)
    """
    provider = env("LLM_PROVIDER", "anthropic")
    model = (fast and env("LLM_FAST_MODEL")) or env("LLM_MODEL")
    if provider == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=model or "claude-opus-5-5", max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}})
        if r.stop_reason == "refusal":
            raise RuntimeError("the model declined this request")
        text = next(b.text for b in r.content if b.type == "text")
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=env("OPENAI_API_KEY", "none"), base_url=env("OPENAI_BASE_URL") or None)
        if not model:
            sys.exit("Set LLM_MODEL for the openai provider (e.g. gpt-5, or a model name your server knows).")
        r = client.chat.completions.create(model=model, messages=[
            {"role": "user", "content": prompt + JSON_TAIL.format(schema=json.dumps(schema))}])
        text = r.choices[0].message.content or ""
    elif provider == "claude-cli":
        cmd = ["claude", "-p", "--output-format", "json", "--json-schema", json.dumps(schema), "--tools", ""]
        r = subprocess.run(cmd + (["--model", model] if model else []), input=prompt,
                           capture_output=True, text=True, timeout=1800)
        if r.returncode:
            raise RuntimeError(f"claude failed: {(r.stderr or r.stdout).strip()[-400:]}")
        res = json.loads(r.stdout)
        if res.get("structured_output") is not None:
            return res["structured_output"]
        text = res.get("result", "")
    elif provider == "command":
        r = subprocess.run(env("LLM_COMMAND"), shell=True, input=prompt + JSON_TAIL.format(schema=json.dumps(schema)),
                           capture_output=True, text=True, timeout=1800)
        if r.returncode:
            raise RuntimeError(f"LLM_COMMAND failed: {(r.stderr or r.stdout).strip()[-400:]}")
        text = r.stdout
    else:
        sys.exit(f"Unknown LLM_PROVIDER {provider!r}: use anthropic, openai, claude-cli or command.")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # some local models think out loud
    return json.loads(text[text.find("{"): text.rfind("}") + 1])


def obj(props, required=None):
    """A JSON schema object with every property required and nothing extra (what strict modes want)."""
    return {"type": "object", "properties": props, "required": required or list(props), "additionalProperties": False}


# ---------------------------------------------------------------- transcription (swap in your own here)

def transcribe(audio, prompt=""):
    """Word-level transcript as [{"w": word, "s": start, "e": end}], punctuation attached to words.
    TRANSCRIBER picks the backend:

    local   faster-whisper on your own machine, free (WHISPER_MODEL: tiny, base, small, medium,
            large-v3, turbo; bigger is slower and more accurate)
    openai  any OpenAI-compatible transcription API: OpenAI whisper-1, Groq whisper-large-v3-turbo...
            (TRANSCRIBE_API_KEY, TRANSCRIBE_BASE_URL, TRANSCRIBE_MODEL)
    """
    backend = env("TRANSCRIBER", "local")
    if backend == "local":
        import numpy as np
        from faster_whisper import WhisperModel
        model = WhisperModel(env("WHISPER_MODEL", "small"), device="auto", compute_type="auto")
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(audio), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                             capture_output=True, check=True).stdout     # decode with ffmpeg, not PyAV
        samples = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
        segments, info = model.transcribe(samples, word_timestamps=True, vad_filter=True,
                                          initial_prompt=prompt or None)
        words, shown = [], 0
        for seg in segments:
            words += [{"w": w.word.strip(), "s": w.start, "e": w.end} for w in seg.words or [] if w.word.strip()]
            if seg.end - shown > 300:
                shown = seg.end
                print(f"  transcribed {hms(seg.end)} of {hms(info.duration)}", flush=True)
        return words
    if backend == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=env("TRANSCRIBE_API_KEY") or env("OPENAI_API_KEY", "none"),
                        base_url=env("TRANSCRIBE_BASE_URL") or None)
        small = Path(audio).with_name("upload.ogg")  # mono speech-quality audio: about 11 MB an hour
        run("ffmpeg", "-y", "-v", "error", "-i", audio, "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "24k", small)
        with open(small, "rb") as f:
            r = client.audio.transcriptions.create(
                model=env("TRANSCRIBE_MODEL", "whisper-1"), file=f, response_format="verbose_json",
                timestamp_granularities=["word", "segment"], prompt=prompt or None)
        small.unlink()
        r = r.model_dump() if hasattr(r, "model_dump") else r
        return punctuate([{"w": w["word"].strip(), "s": w["start"], "e": w["end"]} for w in r.get("words") or []],
                         r.get("segments") or [])
    sys.exit(f"Unknown TRANSCRIBER {backend!r}: use local or openai.")


def punctuate(words, segments):
    """Word timestamps from transcription APIs come without punctuation; borrow it from the segment
    text when the word counts line up, and end each segment with a period otherwise."""
    out, i = [], 0
    for seg in segments:
        mine = []
        while i < len(words) and words[i]["s"] < seg["end"] + 0.05:
            mine.append(words[i])
            i += 1
        toks = seg["text"].split()
        if len(toks) == len(mine):
            for w, t in zip(mine, toks):
                w["w"] = t
        elif mine and mine[-1]["w"][-1:] not in ".?!":
            mine[-1]["w"] += "."
        out += mine
    return out + words[i:] if out else words


def sentences(words, max_words=45):
    """Group words into sentences. Clips start and end on these, so cuts never land mid-sentence."""
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        gap = words[i + 1]["s"] - w["e"] if i + 1 < len(words) else 9
        if w["w"].rstrip("\"')]»”’")[-1:] in ".?!" or len(cur) >= max_words or gap > 2:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [{"start": c[0]["s"], "end": c[-1]["e"], "text": " ".join(w["w"] for w in c), "words": c} for c in out]


# ---------------------------------------------------------------- ingest

def ingest_one(v, cfg):
    d = WORK / v["id"]
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "meta.json").exists():
        if "file" in v:
            run("ffmpeg", "-y", "-v", "error", "-i", v["file"], "-vn", "-ac", "1", "-c:a", "aac", "-b:a", "128k",
                d / "source.m4a")
            dur = float(run("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", v["file"]))
            meta = {"title": Path(v["file"]).stem, "duration": dur, "comments": [], "heatmap": [], "file": v["file"]}
        else:
            ytdlp("-f", "bestaudio", "-x", "--audio-format", "m4a", "--write-info-json", "--write-comments",
                  "--extractor-args", "youtube:max_comments=500,all,100,0;comment_sort=top",
                  "-o", str(d / "source.%(ext)s"), v["url"])
            info = json.loads((d / "source.info.json").read_text())
            meta = {k: info.get(k) for k in ("title", "channel", "duration", "view_count", "upload_date")}
            meta["url"] = v["url"]
            meta["heatmap"] = info.get("heatmap") or []   # YouTube's "most replayed" graph, when it has one
            meta["comments"] = [{"text": c.get("text", ""), "likes": c.get("like_count") or 0}
                                for c in info.get("comments") or []]
            (d / "source.info.json").unlink()
        jsave(d / "meta.json", meta)
    if not (d / "transcript.json").exists():
        print("  transcribing", flush=True)
        t0 = time.time()
        words = transcribe(d / "source.m4a", ", ".join(cfg.get("glossary", [])))
        jsave(d / "transcript.json", {"sentences": sentences(words)})
        print(f"  transcribed in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- audience signals

TS = re.compile(r"(?<![\d:])(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?![\d:])")


def comment_signals(meta, bucket=30):
    """Timestamps people typed in comments, weighted by likes, grouped in 30-second buckets."""
    hits = {}
    for c in meta["comments"]:
        seen = set()
        for h, m, s in TS.findall(c["text"]):
            t = int(h or 0) * 3600 + int(m) * 60 + int(s)
            b = t // bucket * bucket
            if t >= (meta["duration"] or 0) or b in seen:
                continue
            seen.add(b)
            e = hits.setdefault(b, {"t": b, "weight": 0, "comments": []})
            e["weight"] += 1 + min(c["likes"], 200) / 10
            e["comments"].append(c)
    out = sorted(hits.values(), key=lambda e: -e["weight"])
    for e in out:
        e["comments"].sort(key=lambda c: -c["likes"])
    return out


def replay_peaks(meta, n=6):
    hm = meta["heatmap"]
    peaks = [h for i, h in enumerate(hm) if i and h["value"] >= hm[i - 1]["value"]
             and (i == len(hm) - 1 or h["value"] >= hm[i + 1]["value"])]
    return sorted(peaks, key=lambda h: -h["value"])[:n]


def signals_text(meta):
    lines = []
    cs = comment_signals(meta)
    if cs:
        lines.append("Timestamps viewers typed in comments (weight = mentions plus likes):")
        for e in cs[:12]:
            top = e["comments"][0]
            lines.append(f"- around {hms(e['t'])}: weight {e['weight']:.1f}, e.g. ({top['likes']} likes) {top['text'][:200]!r}")
    peaks = replay_peaks(meta)
    if peaks:
        lines.append("YouTube 'most replayed' peaks (1.0 = the most rewatched point):")
        lines += [f"- {hms(p['start_time'])}: {p['value']:.2f}" for p in peaks]
    top = sorted(meta["comments"], key=lambda c: -c["likes"])[:25]
    if top:
        lines.append("Most-liked comments (they often name a favorite moment without a timestamp):")
        lines += [f"- ({c['likes']} likes) {c['text'][:220]!r}" for c in top]
    return "\n".join(lines) or "No audience signals for this video. Judge from the transcript alone."


# ---------------------------------------------------------------- pick

PICK_SCHEMA = obj({"moments": {"type": "array", "items": obj({
    "first": {"type": "integer", "description": "number of the clip's first sentence"},
    "last": {"type": "integer", "description": "number of the clip's last sentence"},
    "title": {"type": "string", "description": "title for the short, under 70 characters"},
    "hook": {"type": "string", "description": "on-screen headline, 3 to 8 words"},
    "why": {"type": "string", "description": "one or two plain sentences on why this will hold a scroller"},
    "evidence": {"type": "array", "items": {"type": "string"},
                 "description": "audience signals that back this pick, quoted from the input; empty if none"},
    "score": {"type": "integer", "description": "1 to 10: how likely the clip is to do well"},
    "context_risk": {"type": "string", "enum": ["low", "medium", "high"]},
    "context_note": {"type": "string", "description": "how this could be misread out of context, or 'none'"},
    "description": {"type": "string", "description": "2 or 3 sentence description for the post; no links"},
    "requested": {"type": "boolean", "description": "true only for the moment that answers the request, if any"},
})}})

PICK_PROMPT = """You pick short-form vertical clips (YouTube Shorts, TikTok, Reels) from long videos.

About the channel: {about}
Video: "{title}" ({length})

Find the {n} best moments for shorts. A good moment:
- stands alone: a stranger scrolling understands it with no setup, and it starts on a strong first line
- lands something: a laugh, a sharp argument, a surprising fact, a clean answer to a common question, a story
- runs {min_s} to {max_s} seconds (check the timestamps of your first and last sentence)
- ends on a complete thought, not mid-argument
{taste}
The audience already told us what they liked. Weigh these signals heavily, and cite them in `evidence`:

{signals}

Be careful with context. A clip that makes someone look like they said something they didn't hurts them.
Rate context_risk honestly and explain it in context_note.

Score each moment 1-10 for how likely it is to do well. Give the moments best first.
{request}
The transcript is numbered by sentence as [#n mm:ss]. It is machine transcription, so names may be
misspelled; fix spelling in titles where you are sure.

{transcript}
"""

REQUEST = """
Someone asked for this video to be clipped and wrote: "{note}"
If that points at a moment that exists, it must be one of yours: find it, cut it so it stands alone, and mark
it requested. Mark nothing requested if it isn't there. Fill the rest of the list as usual.
"""


def pick_one(v, cfg):
    d = WORK / v["id"]
    if (d / "moments.json").exists():
        return
    meta, sents = jload(d / "meta.json"), jload(d / "transcript.json")["sentences"]
    transcript = "\n".join(f"[#{i} {hms(s['start'])}] {s['text']}" for i, s in enumerate(sents))
    taste = cfg.get("what_makes_a_good_clip", "")
    prompt = PICK_PROMPT.format(
        about=cfg.get("about") or cfg["channel"]["name"], title=meta["title"], length=hms(meta["duration"] or 0),
        n=cfg.get("candidates_per_video", 6), min_s=cfg.get("min_seconds", 20), max_s=cfg.get("max_seconds", 75),
        taste=f"- and, for this channel: {taste}\n" if taste else "", signals=signals_text(meta),
        request=REQUEST.format(note=v["note"].replace('"', "'")) if v["note"] else "", transcript=transcript)
    res = ask_json(prompt, PICK_SCHEMA)
    moments = []
    for m in res["moments"]:
        a, b = max(0, m["first"]), min(len(sents) - 1, m["last"])
        if b < a:
            continue
        m |= {"first": a, "last": b, "start": sents[a]["start"], "end": sents[b]["end"],
              "text": " ".join(s["text"] for s in sents[a:b + 1])}
        moments.append(m)
    jsave(d / "moments.json", moments)
    print(f"  {len(moments)} moments: " + "; ".join(f"{m['score']} {m['hook']!r}" for m in moments), flush=True)


# ---------------------------------------------------------------- editorial check (optional)

GATE_SCHEMA = obj({"verdict": {"type": "string", "enum": ["keep", "drop"]},
                   "reason": {"type": "string", "description": "one short sentence"}})

GATE_PROMPT = """A short clip has been proposed from "{title}". Below is the stretch around it, numbered by sentence,
with the clip marked between >>> CLIP START and <<< CLIP END. There are no speaker labels: work out who is
speaking from what they say.

Decide whether to keep the clip, watched on its own by a stranger scrolling, under this rule from the channel:

{rule}

{window}
"""


def gate(v, i, m, cfg, sents):
    """Ask the AI whether a clip passes the channel's own rule (config "gate"); cached in gate.json."""
    cache = WORK / v["id"] / "gate.json"
    got = jload(cache, {})
    if str(i) not in got:
        a, b = max(0, m["first"] - 25), min(len(sents), m["last"] + 25)
        lines = []
        for k in range(a, b):
            lines += [">>> CLIP START"] if k == m["first"] else []
            lines.append(f"[#{k} {hms(sents[k]['start'])}] {sents[k]['text']}")
            lines += ["<<< CLIP END"] if k == m["last"] else []
        title = jload(WORK / v["id"] / "meta.json")["title"]
        got[str(i)] = ask_json(GATE_PROMPT.format(title=title, rule=cfg["gate"], window="\n".join(lines)),
                               GATE_SCHEMA, fast=True)
        jsave(cache, got)
    return got[str(i)]


def select(v, cfg):
    """The best moments of a video: score at or above min_score (or requested), not high context risk,
    the right length, and past the gate if there is one."""
    moments = jload(WORK / v["id"] / "moments.json", [])
    sents = jload(WORK / v["id"] / "transcript.json")["sentences"]
    lo, hi = cfg.get("min_seconds", 20) * 0.75, cfg.get("max_seconds", 75) * 1.3
    pool = [(i, m) for i, m in enumerate(moments) if m["context_risk"] != "high" and lo <= m["end"] - m["start"] <= hi
            and (m["score"] >= cfg.get("min_score", 7) or m.get("requested"))]
    pool.sort(key=lambda x: (not x[1].get("requested"), -x[1]["score"], x[1]["context_risk"] != "low"))
    picked = []
    for i, m in pool:
        if len(picked) >= cfg.get("clips_per_video", 2):
            break
        if cfg.get("gate"):
            g = gate(v, i, m, cfg, sents)
            print(f"  gate #{i} {g['verdict']}: {g['reason']}", flush=True)
            if g["verdict"] != "keep":
                continue
            m = m | {"gate": g}
        picked.append(m | {"id": f"{v['id']}-{i}", "video": v["id"]})
    return picked


# ---------------------------------------------------------------- render

W, H = 1080, 1920
VIDEO_Y, VIDEO_H = 660, 608        # the 16:9 video band
CAP_Y, CAP_H = 1320, 375           # the caption band below it


def rgb(hexstr):
    h = hexstr.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def font(name, size):
    from PIL import ImageFont
    return ImageFont.truetype(str(FONTS / name) if (FONTS / name).exists() else name, size)


def wrap(draw, text, f, width):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=f) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + [cur] if cur else lines


def chrome_png(path, cfg, hook):
    """What stays on screen the whole clip: channel name, headline, footer."""
    from PIL import Image, ImageDraw
    st, ch = cfg.get("style", {}), cfg["channel"]
    accent, ink = rgb(st.get("accent", "#D3B26A")), rgb(st.get("text", "#F6F3EC"))
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, VIDEO_Y), fill=(10, 10, 9, 150))          # darken the blurred backdrop so text reads
    d.rectangle((0, VIDEO_Y + VIDEO_H, W, H), fill=(10, 10, 9, 150))
    tag, x = font(st.get("tag_font", "GeistMono-Medium.ttf"), 32), 78
    for c in ch["name"].upper():
        d.text((x, 177), c, font=tag, fill=accent)
        x += d.textlength(c, font=tag) + 4.8
    size = 87
    while True:
        hf = font(st.get("headline_font", "GeistSerif.ttf"), size)
        lines = wrap(d, hook, hf, W - 156)
        if len(lines) <= 3 or size <= 60:
            break
        size -= 6
    y = VIDEO_Y - 66 - len(lines) * size * 1.12
    for ln in lines:
        d.text((78, y), ln, font=hf, fill=ink)
        y += size * 1.12
    if ch.get("handle"):
        d.text((78, 1794), f"FULL EPISODE  ·  {ch['handle']}", font=font(st.get("tag_font", "GeistMono-Medium.ttf"), 28),
               fill=(*ink, 150))
    im.save(path)


def caption_chunks(words):
    chunks, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        gap = words[i + 1]["s"] - w["e"] if i + 1 < len(words) else 9
        if len(cur) >= 4 or len(" ".join(x["w"] for x in cur)) >= 20 or w["w"][-1] in ".,?!;:" or gap > 0.6:
            chunks.append(cur)
            cur = []
    return chunks + [cur] if cur else chunks


def caption_frames(folder, words, clip_start, clip_len, cfg):
    """One transparent PNG per spoken word (its phrase, with that word lit) and an ffmpeg concat list."""
    from PIL import Image, ImageDraw
    st = cfg.get("style", {})
    accent, ink = rgb(st.get("accent", "#D3B26A")), rgb(st.get("text", "#F6F3EC"))
    folder.mkdir(exist_ok=True)
    f = font(st.get("caption_font", "Geist-Black.ttf"), 81)
    blank = folder / "blank.png"
    Image.new("RGBA", (W, CAP_H), (0, 0, 0, 0)).save(blank)
    entries, t = [], 0.0
    chunks = caption_chunks(words)
    for ci, chunk in enumerate(chunks):
        chunk_end = chunks[ci + 1][0]["s"] if ci + 1 < len(chunks) else chunk[-1]["e"] + 0.4
        chunk_end = min(chunk_end, chunk[-1]["e"] + 0.8)
        for wi, w in enumerate(chunk):
            s = max(w["s"] - clip_start, 0)
            if s > t + 0.02:
                entries.append((blank, s - t))
                t = s
            e = min((chunk[wi + 1]["s"] if wi + 1 < len(chunk) else chunk_end) - clip_start, clip_len)
            if e <= t:
                continue
            im = Image.new("RGBA", (W, CAP_H), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            toks = [x["w"] for x in chunk]
            lines, line = [], []
            for k in range(len(toks)):                 # wrap, remembering word indexes
                if line and d.textlength(" ".join(toks[j] for j in line + [k]), font=f) > W - 156:
                    lines.append(line)
                    line = []
                line.append(k)
            lines.append(line)
            y = 45
            for ln in lines:
                x = (W - d.textlength(" ".join(toks[j] for j in ln), font=f)) / 2
                for j in ln:
                    d.text((x + 4, y + 6), toks[j], font=f, fill=(0, 0, 0, 150))
                    d.text((x, y), toks[j], font=f, fill=accent if j == wi else ink, stroke_width=3,
                           stroke_fill=(0, 0, 0, 200))
                    x += d.textlength(toks[j] + " ", font=f)
                y += 99
            p = folder / f"c{ci:03d}_{wi:02d}.png"
            im.save(p)
            entries.append((p, e - t))
            t = e
    if t < clip_len:
        entries.append((blank, clip_len - t))
    lines = ["ffconcat version 1.0"]
    for p, dur in entries:
        lines += [f"file '{p.name}'", f"duration {dur:.3f}"]
    lines.append(f"file '{blank.name}'")
    (folder / "list.txt").write_text("\n".join(lines) + "\n")
    return folder / "list.txt"


def render_one(m, meta, cfg):
    out = WORK / m["video"] / "clips" / m["id"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "short.mp4").exists():
        return
    start, end = max(0, m["start"] - 0.15), m["end"] + 0.35
    clip_len = end - start
    if meta.get("file"):
        src, offset = meta["file"], start
    else:                                        # download only the seconds we need
        sec0 = max(0, start - 2)
        if not (out / "section.mp4").exists():
            ytdlp("-f", "bv*[height<=1080]+ba/b[height<=1080]", "--download-sections", f"*{sec0:.2f}-{end + 1:.2f}",
                  "--force-keyframes-at-cuts", "--merge-output-format", "mp4", "-o", str(out / "section.%(ext)s"), meta["url"])
        src, offset = out / "section.mp4", start - sec0
    words = [w for s in jload(WORK / m["video"] / "transcript.json")["sentences"] for w in s["words"] if start <= w["s"] < end]
    caps = caption_frames(out / "caps", words, start, clip_len, cfg)
    chrome_png(out / "chrome.png", cfg, m["hook"])
    from PIL import Image
    Image.new("RGB", (W, 9), rgb(cfg.get("style", {}).get("accent", "#D3B26A"))).save(out / "bar.png")
    src_h = int(run("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height",
                    "-of", "csv=p=0", src).split()[0])
    fit = f"scale={W}:{VIDEO_H}:force_original_aspect_ratio=decrease:flags=lanczos"
    fg = fit if src_h >= 1000 else f"hqdn3d=1.5:1.5:0:0,{fit},cas=0.5"   # tidy up sub-1080p sources
    graph = (
        f"[0:v]split[a][b];"
        f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},gblur=sigma=38,eq=brightness=-0.32:saturation=0.8[bg];"
        f"[b]{fg}[fg];"
        f"[bg][fg]overlay=(main_w-overlay_w)/2:{VIDEO_Y}+({VIDEO_H}-overlay_h)/2[v1];"
        f"[v1][1:v]overlay=0:0[v2];"
        f"[2:v]fps=30,format=rgba[cap];"
        f"[v2][cap]overlay=0:{CAP_Y}:eof_action=pass[v3];"
        f"[v3][3:v]overlay=x='-{W}+{W}*t/{clip_len:.3f}':y={H - 9}:eval=frame,fps=30,format=yuv420p[v];"
        f"[0:a]loudnorm=I=-14:TP=-1.5:LRA=11,afade=t=in:d=0.12,afade=t=out:st={clip_len - 0.25:.3f}:d=0.25[a]"
    )
    run("ffmpeg", "-y", "-v", "error", "-ss", f"{offset:.3f}", "-t", f"{clip_len:.3f}", "-i", src,
        "-loop", "1", "-i", out / "chrome.png", "-f", "concat", "-safe", "0", "-i", caps,
        "-loop", "1", "-i", out / "bar.png", "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
        "-t", f"{clip_len:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-profile:v", "high",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out / "short.mp4", timeout=1800)
    run("ffmpeg", "-y", "-v", "error", "-ss", f"{min(clip_len * 0.4, 6):.2f}", "-i", out / "short.mp4",
        "-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "4", out / "poster.jpg")
    shutil.rmtree(out / "caps")
    for f in ("chrome.png", "bar.png", "section.mp4"):
        (out / f).unlink(missing_ok=True)
    link = f"{meta['url']}&t={int(m['start'])}s" if meta.get("url", "").startswith("https://www.youtube.com") else meta.get("url", "")
    jsave(out / "clip.json", {k: m.get(k) for k in ("id", "video", "title", "hook", "description", "why", "evidence",
                                                     "score", "context_risk", "context_note", "requested", "gate", "text")}
          | {"start": round(m["start"], 2), "end": round(m["end"], 2), "source_title": meta["title"], "source_link": link})


# ---------------------------------------------------------------- review page

def clips():
    reviews = jload(WORK / "reviews.json", {})
    out = []
    for f in sorted(WORK.glob("*/clips/*/clip.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        c = jload(f)
        if (f.parent / "short.mp4").exists():
            c["media"] = str(f.parent.relative_to(HERE))
            c["review"] = reviews.get(c["id"], {"status": "new"})
            out.append(c)
    return out


def save_review(body):
    reviews = jload(WORK / "reviews.json", {})
    cid, status = body["id"], body["status"]
    reviews[cid] = {"status": status, "title": body.get("title", ""), "description": body.get("description", ""),
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    jsave(WORK / "reviews.json", reviews)
    APPROVED.mkdir(exist_ok=True)
    folder = next(WORK.glob(f"*/clips/{cid}"))
    if status == "approved":                     # approved shorts land in approved/, ready to upload
        shutil.copy(folder / "short.mp4", APPROVED / f"{cid}.mp4")
        c = jload(folder / "clip.json")
        (APPROVED / f"{cid}.txt").write_text(f"{body.get('title', '')}\n\n{body.get('description', '')}\n\n"
                                             f"From: {c['source_title']} {c['source_link']}\n")
    else:
        for ext in ("mp4", "txt"):
            (APPROVED / f"{cid}.{ext}").unlink(missing_ok=True)


def review(port=8765):
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(HERE), **kw)

        def log_message(self, *a):
            pass

        def send_json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/api/clips":
                return self.send_json(clips())
            if path in ("/", "/review.html"):
                self.path = path = "/review.html"
            elif not Path(self.translate_path(path)).resolve().is_relative_to(WORK.resolve()):
                return self.send_error(404)              # serve the review page and work/ only
            rng = self.headers.get("Range")
            file = Path(self.translate_path(path))
            if rng and file.suffix == ".mp4" and file.exists():   # Safari needs byte ranges to play video
                size = file.stat().st_size
                a, _, b = rng.removeprefix("bytes=").partition("-")
                a, b = int(a or 0), min(int(b) if b else size - 1, size - 1)
                self.send_response(206)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
                self.send_header("Content-Length", str(b - a + 1))
                self.end_headers()
                with open(file, "rb") as f:
                    f.seek(a)
                    self.wfile.write(f.read(b - a + 1))
                return
            super().do_GET()

        def do_POST(self):
            if self.path != "/api/review":
                return self.send_error(404)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            save_review(body)
            self.send_json({"ok": True})

    print(f"Review page: http://localhost:{port}  (Ctrl-C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


# ---------------------------------------------------------------- main

def stage(name, fn, vids, cfg):
    for v in vids:
        print(f"{name} {v['id']}", flush=True)
        try:
            fn(v, cfg)
        except Exception as e:
            print(f"  failed: {e}", flush=True)


def ready(v, file):
    return (WORK / v["id"] / file).exists()


def do_pick(v, cfg):
    if ready(v, "transcript.json"):
        pick_one(v, cfg)
    else:
        print("  not ingested yet", flush=True)


def do_render(v, cfg):
    if not ready(v, "moments.json"):
        return print("  not picked yet", flush=True)
    meta = jload(WORK / v["id"] / "meta.json")
    for m in select(v, cfg):
        print(f"  render {m['id']} {hms(m['start'])}-{hms(m['end'])} {m['hook']!r}", flush=True)
        render_one(m, meta, cfg)


def main():
    load_env()
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "help"), sys.argv[2:]
    if cmd == "review":
        return review(int(env("PORT", "8765")))
    if cmd not in ("run", "ingest", "pick", "render"):
        return print(__doc__)
    cfg, vids = config(), videos(args)
    if not vids:
        sys.exit("No videos: add some to config.json or pass links on the command line.")
    WORK.mkdir(exist_ok=True)
    if cmd in ("run", "ingest"):
        stage("ingest", ingest_one, vids, cfg)
    if cmd in ("run", "pick"):
        stage("pick", do_pick, vids, cfg)
    if cmd in ("run", "render"):
        stage("render", do_render, vids, cfg)
    if cmd == "run":
        print("\nDone. Review the clips with: uv run clipfarm.py review", flush=True)


if __name__ == "__main__":
    main()
