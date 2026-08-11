"""Export the speaker script to Markdown with a derived timing table.

The eleven content slides are on the clock; the Thank You slide is not — its
notes hold the Q&A material, which is prepared but not spoken. Time cues are
computed from the text, never written by hand — see paper/_slide_script.py.

Usage:
    python -m paper._make_slide_script_md
Output:
    paper/FlyAda_WRC_SARA_2026_script.md
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper._slide_script import (SCRIPT, WPM, mmss, spoken_words,   # noqa: E402
                                 timings)

OUT = ROOT / "paper" / "FlyAda_WRC_SARA_2026_script.md"

VIDEO_SLIDES = {3, 5, 8, 9, 10}


def main():
    rows = timings()
    main_rows = [r for r in rows if r[0] != "thanks"]
    total = sum(r[3] for r in main_rows)
    talk = total / WPM * 60.0

    lines = [
        "# FlyAda — WRC SARA 2026 oral presentation script",
        "",
        "**Paper:** WRC26_0024 · *FlyAda: Belief-State Adaptation for Diffusion "
        "Policies under Partial Observation*  ",
        "**Speaker:** Lingxue Lyu, University of Pennsylvania  ",
        "**Slot:** 9 min presentation + 3 min Q&A · video channel: 8–10 min  ",
        "**Deck:** `paper/FlyAda_WRC_SARA_2026_oral.pptx` — 11 content slides + "
        "Thank You (12 pages)",
        "",
        "> **Delivery notes.** One question, one failure, one method, three "
        f"results — {len(main_rows)} content slides, ≈ {mmss(talk)} at "
        f"{WPM:.0f} words per minute. Cues below are derived from the word "
        "counts, not hand-written. Five slides carry video (3, 5, 8, 9, 10) and "
        "every one autoplays and loops on slide entry: **keep talking over it, "
        "do not wait for it to finish**. Bullets and takeaway bars fade in "
        "automatically — nothing needs clicking mid-slide.",
        "",
        "> **There are no backup slides.** The Q&A material — anticipated "
        "questions, implementation detail, the Task B and probe numbers, and the "
        "full caveat list — is in the **Thank You slide's speaker notes**, where "
        "presenter view keeps it in front of you.",
        "",
        "---",
        "",
    ]

    for key, num, title, w, start, end, _bk in main_rows:
        vid = "   ·   **video, autoplays + loops**" if num in VIDEO_SLIDES else ""
        lines += [f"### Slide {num} — {title}", "",
                  f"*[{mmss(start)} – {mmss(end)}]   ·   {w} words{vid}*", "",
                  SCRIPT[key].strip(), "", "---", ""]

    lines += [f"### Slide {rows[-1][1]} — Thank you  ·  Q&A prep", "",
              SCRIPT["thanks"].strip(), "", "---", ""]

    lines += [
        "## Timing budget",
        "",
        "| Slide | Section | Words | Start | ≈ sec |",
        "|---:|---|---:|---:|---:|",
    ]
    for key, num, title, w, start, end, _bk in main_rows:
        lines.append(f"| {num} | {title} | {w} | {mmss(start)} | {end - start:.0f} |")

    lines += [
        "",
        f"**Total: {total} words ≈ {mmss(talk)} at {WPM:.0f} wpm.**",
        "",
        f"Faster delivery (150 wpm) lands near {mmss(total / 150 * 60)}; slower "
        f"(115 wpm) near {mmss(total / 115 * 60)}. The 9-minute slot and the "
        "8–10 minute video window are both comfortable at the scripted pace — "
        "there is roughly a minute of headroom for pauses on the videos and for "
        "the questions that always take longer than expected.",
        "",
        f"The Thank You slide's notes add "
        f"{spoken_words(SCRIPT['thanks'])} words of prepared Q&A answers that "
        "are not spoken unless asked.",
        "",
    ]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved -> {OUT}")
    print(f"main line: {total} words ≈ {mmss(talk)} at {WPM:.0f} wpm "
          f"({len(main_rows)} slides)")
    for key, num, title, w, start, end, _bk in main_rows:
        print(f"  slide {num:>2}  {w:>4} words  {mmss(start):>5}  {title}")
    print(f"Q&A prep in notes: {spoken_words(SCRIPT['thanks'])} words "
          f"(not spoken)")


if __name__ == "__main__":
    main()
