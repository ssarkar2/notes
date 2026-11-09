"""
Check that the math in markdown files will render on GitHub.

GitHub renders math in two stages, and each can break it:

  1. Markdown runs first. Backslash + punctuation (\\, \\; \\{ \\} \\! \\|) loses its backslash,
     and `<` in inline math gets double-escaped (shows as "&lt;").
     -> Check: send each file to GitHub's markdown API (public, no auth, 60 req/hour) and verify
        every math segment comes back byte-identical.
  2. MathJax runs in the browser and fails on bad/unsupported TeX.
     -> Check: parse every segment with MathJax 3 locally (tools/tex_check.mjs; needs
        `npm install` in tools/ once).

Safe style that passes both:
  * display math in ```math fences (markdown leaves them alone)
  * inline math as $...$ using \\lt / \\gt instead of < / >, and no backslash+punctuation
    (\\thinspace instead of \\,; \\lbrace/\\rbrace instead of \\{ \\}), or the $`...`$ form

Usage:  python tools/check_math.py [files or dirs ...]      (default: whole tutorial)
        --offline   skip the GitHub API stage
Exit code 1 if anything is broken.
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent


@dataclass
class Seg:
    file: Path
    line: int
    tex: str
    display: bool
    form: str          # "fence" | "$$" | "$`" | "$"


def extract(path: Path) -> list[Seg]:
    """Find math segments roughly the way GitHub does (fences, $$..$$, $`..`$, $..$)."""
    text = path.read_text()
    segs: list[Seg] = []
    lines = text.split("\n")
    i = 0
    in_code = False
    buf: list[str] = []
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            if not in_code and s[3:].strip() == "math":
                start = i + 1
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith("```"):
                    j += 1
                segs.append(Seg(path, start, "\n".join(lines[i + 1:j]), True, "fence"))
                i = j + 1
                continue
            in_code = not in_code
            i += 1
            continue
        if in_code:
            i += 1
            continue
        buf.append((i + 1, ln))
        i += 1

    # rebuild non-code text with line numbers to find $$..$$ (may span lines) and inline math
    joined = ""
    line_at = []
    for n, ln in buf:
        for _ in ln + "\n":
            line_at.append(n)
        joined += ln + "\n"
    joined_nocode = re.sub(r"`[^`\n]*`", lambda m: " " * len(m.group()), joined)
    # $`...`$ first (uses backticks, so search in the raw text)
    for m in re.finditer(r"\$`(.+?)`\$", joined):
        segs.append(Seg(path, line_at[m.start()], m.group(1), False, "$`"))
        joined_nocode = joined_nocode[:m.start()] + " " * (m.end() - m.start()) + joined_nocode[m.end():]
    for m in re.finditer(r"\$\$(.+?)\$\$", joined_nocode, flags=re.S):
        segs.append(Seg(path, line_at[m.start()], joined[m.start() + 2:m.end() - 2], True, "$$"))
        joined_nocode = joined_nocode[:m.start()] + " " * (m.end() - m.start()) + joined_nocode[m.end():]
    for m in re.finditer(r"(?<![\\$])\$(?!\s)([^$\n]+?)(?<![\s\\])\$(?!\$)", joined_nocode):
        segs.append(Seg(path, line_at[m.start()], joined[m.start() + 1:m.end() - 1], False, "$"))
    segs.sort(key=lambda s: s.line)
    return segs


def github_render(text: str) -> str:
    req = urllib.request.Request(
        "https://api.github.com/markdown",
        data=json.dumps({"text": text, "mode": "gfm"}).encode(),
        headers={"Accept": "application/vnd.github+json", "User-Agent": "math-check"})
    return urllib.request.urlopen(req, timeout=30).read().decode()


def rendered_segments(html_out: str) -> list[str]:
    out = []
    for m in re.finditer(r'<math-renderer class="js-(inline|display)-math"[^>]*>(.*?)</math-renderer>',
                         html_out, flags=re.S):
        body = html.unescape(m.group(2)).strip()
        if body.startswith("$$") and body.endswith("$$"):
            body = body[2:-2]
        elif body.startswith("$") and body.endswith("$"):
            body = body[1:-1]
        out.append(body)
    return out


def norm(s: str) -> str:
    return " ".join(s.split())


def main(argv: list[str]) -> int:
    offline = "--offline" in argv
    args = [a for a in argv if not a.startswith("--")] or [str(ROOT)]
    files: list[Path] = []
    for a in args:
        p = Path(a).resolve()
        files += sorted(p.rglob("*.md")) if p.is_dir() else [p]
    files = [f for f in files if "node_modules" not in f.parts and f.name != "PLAN.md"]

    problems = 0
    all_segs: list[Seg] = []
    for f in files:
        segs = extract(f)
        all_segs += segs
        if offline or not segs:
            continue
        got = rendered_segments(github_render(f.read_text()))
        rel = f.relative_to(ROOT) if f.is_relative_to(ROOT) else f
        if len(got) != len(segs):
            print(f"{rel}: GitHub found {len(got)} math segments, checker found {len(segs)} "
                  f"(a stray or unmatched $?)")
            problems += 1
        # GitHub keeps document order; our list is sorted by line (same-line order may differ)
        got_set = [norm(g) for g in got]
        for s in segs:
            if norm(s.tex) not in got_set:
                cand = min(got, key=lambda g: _dist(norm(g), norm(s.tex))) if got else ""
                print(f"{rel}:{s.line}: [markdown] mangled ({s.form})\n    source:   {norm(s.tex)}\n"
                      f"    rendered: {norm(cand)}")
                problems += 1

    # stage 2: MathJax parse
    if not (TOOLS / "node_modules" / "mathjax-full").exists():
        print("skipping MathJax stage: run `npm install` in tools/ first")
    else:
        payload = [{"id": k, "tex": s.tex, "display": s.display} for k, s in enumerate(all_segs)]
        res = subprocess.run(["node", str(TOOLS / "tex_check.mjs")], input=json.dumps(payload),
                             capture_output=True, text=True, cwd=TOOLS)
        if res.returncode != 0:
            print(res.stderr)
            return 2
        for fail in json.loads(res.stdout or "[]"):
            s = all_segs[fail["id"]]
            rel = s.file.relative_to(ROOT) if s.file.is_relative_to(ROOT) else s.file
            print(f"{rel}:{s.line}: [mathjax] {fail['error']}\n    source: {norm(s.tex)}")
            problems += 1

    print(f"\n{len(files)} files, {len(all_segs)} math segments, {problems} problem(s)")
    return 1 if problems else 0


def _dist(a: str, b: str) -> int:
    """Cheap similarity: length of differing prefix/suffix (good enough to pick the culprit)."""
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return -i


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
