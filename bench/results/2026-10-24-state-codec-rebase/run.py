#!/usr/bin/env python3
"""model_validation.py - V100 regression/benchmark harness for the Strata BF16 checkpoint PR rebase.

Three arms, each its own server on its own port, one at a time:
  upstream  build-upstream/strata        default session codec (fp32), no --session-state-codec flag
  fp32      build-rebased/strata         --session-state-codec fp32
  bf16      build-rebased/strata         --session-state-codec bf16

Per size, per arm, per round: a deterministic long document with three code words, a SAVE to disk, a disjoint
request that parks the conversation, a parked-resume continuation, a mid-document rewind that selects an earlier
retained checkpoint, a disk RESTORE of the saved file, and - after a restore this harness could verify - a 512-token
greedy continuation whose first-token logits are dumped by the engine (STRATA_DUMP_FIRST_LOGITS). Logits and session
files are compared between arms afterwards.

Nothing here kills or restarts anything it did not start: it refuses to run while any GPU compute process exists,
and cleanup stops only this harness's own Popen tree. Root stops and restores production (port 8080) around it.

Standard library only. See model_validation-usage.txt for the CLI.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRODUCTION_CONFIG = Path(os.environ.get("STRATA_BENCH_CONFIG", "strata-bench.json"))
VENV_PY = os.environ.get("STRATA_BENCH_PYTHON", sys.executable)
GPU_SMI = "/usr/bin/nvidia-smi"
CHARS_PER_TOKEN = 3.2                     # the repo's own needle_bench estimate, only used to size a first draft
CODES = ("lantern", "cobalt", "juniper")  # fixed code words; positions are decided by token counts, not RNG
PINNED = ["--adapt-swaps", "0", "--pcie-frac", "0", "--spec", "4", "--suffix-draft", "0"]
PROMPT_CACHE_EVERY = "8192"               # the spacing bench_capture_codec.py measured at; keeps rewind targets real
ARMS = ("upstream", "fp32", "bf16")
ARM_SOURCE = {"upstream": Path(os.environ.get("STRATA_BENCH_UPSTREAM", ".")),
              "fp32": Path(os.environ.get("STRATA_BENCH_REBASED", ".")),
              "bf16": Path(os.environ.get("STRATA_BENCH_REBASED", "."))}
ARM_CODEC = {"upstream": None, "fp32": "fp32", "bf16": "bf16"}
SESSION_FILE_VERSIONS = {1: "v1 (running state fp32 in every checkpoint)",
                         2: "v2 (each checkpoint says how its running state is stored)"}


class HarnessError(Exception):
    pass


# ------------------------------------------------------------------ preflight
def gpu_compute_processes():
    """Every process the driver reports as holding GPU context, plus what each one is. Read-only nvidia-smi call."""
    try:
        out = subprocess.run([GPU_SMI, "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise HarnessError(f"cannot run {GPU_SMI} to check for GPU compute processes: {e}") from None
    if out.returncode != 0:
        raise HarnessError(f"{GPU_SMI} said: {(out.stderr or out.stdout).strip()[:400]}")
    procs = []
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            procs.append({"pid": int(parts[0]), "process": parts[1],
                          "used_mib": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None})
    return procs


def port_in_use(port):
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", int(port))) == 0


def git_read_only(source_root, *args):
    """A read-only git query (rev-parse / status --porcelain). No mutation of anyone's repository."""
    try:
        r = subprocess.run(["git", "-C", str(source_root), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def gpu_info():
    try:
        r = subprocess.run([GPU_SMI, "--query-gpu=index,name,driver_version,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


# ------------------------------------------------------------------ own process tree only
def children_of(pid):
    """Direct children from /proc, so a kill never needs a process-name pattern."""
    out = []
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat", "rb") as f:
                    fields = f.read().decode(errors="replace").rsplit(") ", 1)
                ppid = int(fields[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            if ppid == pid:
                out.append(int(entry))
    except OSError:
        pass
    return out


def tree_of(root_pid):
    seen, queue, order = set(), [root_pid], []
    while queue:
        pid = queue.pop(0)
        if pid in seen:
            continue
        seen.add(pid)
        order.append(pid)
        queue.extend(children_of(pid))
    return order


# ------------------------------------------------------------------ the engine's own log lines
# Each pattern is anchored on a phrase the engine actually prints (src/program/generate.cpp). If main changes the
# wording, the measurement is reported missing - never invented.
LOG_PATTERNS = {
    # Named groups throughout: the saved and restored session lines carry different counts, so a positional index
    # could silently read a path as a timing. Each event reads its own fields by name.
    "park": re.compile(r"conversation cache: (?P<kind>parked|skipped) (?P<tokens>\d+) tokens in (?P<park_ms>[\d.]+)"
                       r" ms; parked=(?P<parked>\d+) bytes=(?P<cache_bytes>\d+)(?: evictions=(?P<evictions>\d+))?"
                       r"(?:.*?snapshot_bytes=(?P<snapshot_bytes>\d+))?"),
    "resume": re.compile(r"conversation cache: restored (?P<tokens>\d+) tokens \((?P<from>live|checkpoint)\) in "
                         r"(?P<resume_ms>[\d.]+) ms; parked=(?P<parked>\d+) bytes=(?P<cache_bytes>\d+)"),
    "session_saved": re.compile(r"session saved (?P<tokens>\d+) tokens, (?P<checkpoints>\d+) of \d+ checkpoints, "
                                r"(?P<bytes>\d+) bytes to (?P<file>\S+) in (?P<engine_ms>[\d.]+) ms"
                                r"(?: \(state (?P<state_ms>[\d.]+) ms\))?"),
    "session_restored": re.compile(r"session restored (?P<tokens>\d+) tokens, (?P<checkpoints>\d+) checkpoints, "
                                   r"(?P<bytes>\d+) bytes from (?P<file>\S+) in (?P<engine_ms>[\d.]+) ms"),
    "checkpoint_selected": re.compile(r"conversation cache: resumed through the checkpoint at (?P<at_tokens>\d+) "
                                      r"tokens \((?P<note>.*?)\) in (?P<restore_ms>[\d.]+) ms; the other "
                                      r"(?P<others_untouched>\d+)"),
    "retained_coded": re.compile(r"conversation cache: retained (?P<retained>\d+) checkpoints? beside the live state,"
                                 r" (?P<retained_bytes>\d+) bytes of running state and token lists kept as stored "
                                 r"\((?P<coded>\d+) coded, (?P<coded_bytes>\d+) bytes of it\)"),
    # engine request line; bench_lib's PROMPT regex stops before the generated clause, this one keeps it
    "prompt": re.compile(r"prompt (?P<tokens>\d+) tokens = (?P<reused>\d+) reused \+ (?P<read>\d+) read in "
                         r"(?P<prompt_ms>[\d.]+) ms.*?(?P<generated>\d+) generated in (?P<generated_ms>[\d.]+) ms"
                         r".*?(?P<checkpoints_after>\d+) checkpoints"),
}


def parse_log_events(lines):
    """All engine log events of every known kind in `lines`, keyed by pattern name."""
    found = {k: [] for k in LOG_PATTERNS}
    for line in lines:
        for name, rx in LOG_PATTERNS.items():
            m = rx.search(line)
            if not m:
                continue
            out = {}
            for key, val in m.groupdict().items():
                if val is None:
                    out[key] = None
                elif key in ("kind", "from", "note", "file"):
                    out[key] = val
                elif key.endswith("_ms"):
                    out[key] = float(val)
                else:
                    out[key] = int(val)
            found[name].append(out)
    return found


# ------------------------------------------------------------------ VmRSS sampling
class RssSampler:
    """Polls one pid's VmRSS every 20 ms. The window max is the operation's peak only to that resolution, and most
    parked state is pageable host memory the kernel may already have written out: increments are observed lower
    bounds, never true high-water marks."""

    def __init__(self, pid, interval_s=0.02):
        self.pid, self.interval_s, self.stop = pid, interval_s, False
        self.samples, self.window = [], None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop:
            try:
                with open(f"/proc/{self.pid}/status") as f:
                    v = next((int(l.split()[1]) * 1024 for l in f if l.startswith("VmRSS:")), None)
            except (OSError, ValueError, StopIteration):
                v = None
            if v is not None:
                self.samples.append(v)
                if self.window is not None:
                    self.window.append(v)
            time.sleep(self.interval_s)

    def baseline(self, n=25):
        vals = self.samples[-n:]
        return statistics.median(vals) if vals else 0

    def measure(self, fn):
        base = self.baseline()
        self.window = []
        try:
            out = fn()
        finally:
            peak = max(self.window) if self.window else base
            self.window = None
        return {"result": out, "rss_peak_increment_mib": round((peak - base) / 2 ** 20, 1),
                "baseline_mib": round(base / 2 ** 20, 1), "samples": len(self.samples)}

    def stop_now(self):
        self.stop = True
        self.thread.join(timeout=2)


# ------------------------------------------------------------------ prompts (deterministic, shared by every arm)
def filler_chars(source_root: Path, n_chars: int) -> str:
    """Deterministic repository text in a fixed order (the shape tools/needle_bench.py uses), read from the upstream
    tree so every arm sees byte-identical documents. Built once per size."""
    pairs = [("docs", "*.md"), ("src", "*.cpp"), ("include", "*.hpp"), ("serve", "*.py"), ("tools", "*.py")]
    parts, total = [], 0
    while total < n_chars:
        added = False
        for d, pat in pairs:
            base = source_root / d
            if not base.is_dir():
                continue
            for f in sorted(base.rglob(pat)):
                try:
                    t = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                block = f"\n\n=== {d}/{f.name} ===\n{t}"
                parts.append(block)
                total += len(block)
                added = True
                if total >= n_chars:
                    break
            if total >= n_chars:
                break
        if not added:
            raise HarnessError(f"no filler text found under {source_root}")
    return "".join(parts)[:n_chars]


def split_at_words(text: str, frac: float) -> int:
    """A character index near frac*len that lands on a word boundary."""
    target = max(0, min(len(text) - 1, int(len(text) * frac)))
    idx = text.find(" ", target)
    return idx if idx > 0 else target


def build_doc(body: str, marks, tail="\n\n[End of document.]"):
    """A document from `body` with a code word inserted at each (char_index, code_word) in `marks`, which must be
    sorted ascending. Each needle goes in as its own line, so it never hides inside filler noise."""
    pieces, last = [], 0
    for at, code in marks:
        pieces.append(body[last:at])
        pieces.append(f"\n\n The magic code word for this passage is {code}. It appears exactly once here.\n\n")
        last = at
    pieces.append(body[last:])
    head = ("[Case marker - read the whole document before answering. This conversation belongs to one test case "
            "only; do not carry anything over from any other conversation.]\n\n")
    return head + "".join(pieces) + tail


def documents_for(size: int, source_root: Path, rewind_tokens=None):
    """(marker, docs) for one target size. Deterministic: same marker, same body, same needle positions in every arm.

    A_EARLY is the same body cut at `rewind_tokens` (a target, never claimed as exact) and closed with its own end
    marker. It is what a mid-document edit sends, so the engine has to select a checkpoint well behind A's live end
    rather than simply continuing it. With --prompt-cache-every 8192 pinned, aim it just above a multiple of 8192:
    below the first checkpoint there is nothing retained to select."""
    marker = f"MV{size}"
    chars = max(1024, int(size * CHARS_PER_TOKEN))
    body = filler_chars(source_root, chars)
    cut_points = [split_at_words(body, f) for f in (0.10, 0.50, 0.90)]
    doc_a = f"[Validation case {size} A]\n" + build_doc(body, list(zip(cut_points, CODES)))
    target = rewind_tokens if rewind_tokens else min(max(256,size-256),max(9216,size//2+1024))
    frac = min(0.90, max(0.02, target / max(1, size)))
    early_end = split_at_words(body, frac)
    doc_early = f"[Validation case {size} A]\n" + build_doc(body[:early_end], [(cut_points[0], CODES[0])],
                          tail="\n\n[End of the part shown so far.]")
    other = filler_chars(source_root, max(2048, chars // 3))
    doc_b = (f"[Validation case {size} B]\n[Case marker - unrelated conversation. Ignore every other conversation.]\n\n" + other[:len(other) // 2] +
             "\n\nIn one short sentence, without quoting: what kind of text is this?")
    return marker, {"A": doc_a, "A_EARLY": doc_early, "B": doc_b}


def requests_for(marker):
    """The request bodies for one case. Fixed wording, no RNG, no timestamps: the same list in every arm, so
    cross-arm comparisons only ever line up where the prompts are identical."""
    ask = lambda code: (f"Question {code}: what is the magic code word you were told near one part of the document? "
                        f"Reply with that single word only.")
    cont_q = ("Continue the document in the same style, as closely as you can to what actually follows the part you "
              "were shown. Write at least 400 words.")
    return [
        {"name": "prefill_A", "messages": [{"role": "user", "content": "@A"}], "max_tokens": 8, "dump": False},
        {"name": "needle_10", "messages": [{"role": "user", "content": ask(CODES[0])}], "max_tokens": 8, "dump": False},
        {"name": "needle_50", "messages": [{"role": "user", "content": ask(CODES[1])}], "max_tokens": 8, "dump": False},
        {"name": "needle_90", "messages": [{"role": "user", "content": ask(CODES[2])}], "max_tokens": 8, "dump": False},
        # B is disjoint from A: it parks A in the conversation cache. Its answer never counts as recall.
        {"name": "park_B", "messages": [{"role": "user", "content": "@B"}], "max_tokens": 16, "dump": False},
        # resumes A: same marker + needles, then a factual question that is not about the needles
        {"name": "resume_A", "messages": [{"role": "user", "content": "@A"},
                                          {"role": "assistant", "content": "Understood. I read the whole document."},
                                          {"role": "user", "content": "Without rereading it, state how many magic code "
                                                                      "words this document contains and name each one in "
                                                                      "the order it appears. One line per word."}],
         "max_tokens": 64, "dump": True},
        # mid-document edit: same body, cut at the first needle, so an earlier retained checkpoint must be selected
        {"name": "rewind_mid", "messages": [{"role": "user", "content": "@A_EARLY"},
                                           {"role": "assistant", "content": "Understood. I read that part."},
                                           {"role": "user", "content": "From the part you were shown first, name the "
                                                                      "magic code word. Reply with one word only."}],
         "max_tokens": 16, "dump": True},
        # the continuation that follows a RESTORE (or a fresh prompt when no restore was verified)
        {"name": "continuation", "messages": [{"role": "user", "content": "@A"},
                                             {"role": "assistant", "content": "Understood. I read the whole document."},
                                             {"role": "user", "content": cont_q}],
         "max_tokens": 512, "dump": True},
        # the warm-cycle prompts: one fixed wording reused by every rep, so reps are comparable with each other
        {"name": "cycle_turn", "messages": [{"role": "user", "content": "@A"},
            {"role": "assistant", "content": "Understood. I read the whole document."},
            {"role": "user", "content":
            "In one short sentence: how many magic code words were in the document I showed you?"}],
         "max_tokens": 24, "dump": False},
        {"name": "cycle_resume", "messages": [{"role": "user", "content": "@A"},
            {"role": "assistant", "content": "Understood. I read the whole document."},
            {"role": "user", "content":
            "Name each magic code word from the document, one per line."}], "max_tokens": 32, "dump": False},
    ]


def materialize(specs, docs):
    out = []
    for s in specs:
        msgs = [{"role": m["role"], "content": docs.get(k[1:], k) if (k := m["content"]).startswith("@") else k}
                for m in s["messages"]]
        out.append({**s, "messages": msgs})
    return out


def sha256_file(path: Path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def messages_sha(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ first-logit dumps
class DumpIndex:
    """Tracks the engine's STRATA_DUMP_FIRST_LOGITS counter. The engine keeps a process-global counter and writes
    <path>.<n> for each request that asks for logits, so the harness knows which file is whose only if it counts
    those requests itself."""

    def __init__(self, base: Path):
        self.base, self.n = base, 0

    def claim(self, case, name):
        path = Path(f"{self.base}.{self.n}")
        rec = {"dump_index": self.n, "path": str(path), "case": case, "request": name}
        self.n += 1
        return rec

    def settle(self, record):
        path = Path(record["path"])
        try:
            record["bytes"] = path.stat().st_size
            record["exists"] = True
        except OSError:
            record["bytes"] = None
            record["exists"] = False
        return record


def read_logits(path: Path):
    """Raw float32, whole vocabulary. Returns (values, stats) or (None, why_missing)."""
    try:
        size = path.stat().st_size
    except OSError:
        return None, {"reason": "dump file absent", "path": str(path)}
    if size == 0 or size % 4:
        return None, {"reason": f"unexpected dump size {size}", "path": str(path)}
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        return None, {"reason": f"cannot read dump: {e}", "path": str(path)}
    a = array.array("f")
    a.frombytes(raw)
    if array.array("f").itemsize != 4 or sys.byteorder != "little":
        return None, {"reason": "float32 little-endian assumption not met on this host", "path": str(path)}
    vals = list(a)
    finite = [v for v in vals if math.isfinite(v)]
    stats = {"path": str(path), "bytes": size, "vocab_floats": len(vals), "finite_count": len(finite),
             "nonfinite_count": len(vals) - len(finite)}
    if not finite:
        return None, {**stats, "reason": "no finite logits"}
    stats["max_abs_finite"] = round(max(abs(v) for v in finite), 6)
    return vals, stats


def kl_and_top1(p, q):
    """Full-vocab KL(P||Q) with a numerically stable logsumexp, plus top-1 ids and agreement. p and q are the raw
    logit lists of matched requests."""
    n = min(len(p), len(q))
    if n == 0:
        return None
    fin_p = sum(1 for v in p[:n] if math.isfinite(v))
    fin_q = sum(1 for v in q[:n] if math.isfinite(v))

    def logsumexp(vals):
        m = max(vals)
        return m + math.log(sum(math.exp(v - m) for v in vals))

    lp, lq = [], []
    try:
        sp, sq = logsumexp(p[:n]), logsumexp(q[:n])
        for i in range(n):
            lp.append(p[i] - sp if math.isfinite(p[i]) else float("-inf"))
            lq.append(q[i] - sq if math.isfinite(q[i]) else float("-inf"))
    except (ValueError, OverflowError) as e:
        return {"vocab_floats_compared": n, "error": f"logsumexp failed: {e}"}
    kl = 0.0
    for i in range(n):
        if lp[i] == float("-inf"):
            continue
        pi = math.exp(lp[i])
        if pi <= 0.0:
            continue
        if lq[i] == float("-inf"):
            kl = float("inf")
            break
        kl += pi * (lp[i] - lq[i])
    top_p = max(range(n), key=lambda i: p[i] if math.isfinite(p[i]) else float("-inf"))
    top_q = max(range(n), key=lambda i: q[i] if math.isfinite(q[i]) else float("-inf"))
    shared = [abs(p[i] - q[i]) for i in range(n) if math.isfinite(p[i]) and math.isfinite(q[i])]
    return {"vocab_floats_compared": n, "finite_a": fin_p, "finite_b": fin_q,
            "max_abs_logit_diff": round(max(shared), 6) if shared else None,
            "top1_a": top_p, "top1_b": top_q, "top1_agree": top_p == top_q,
            "kl_A_given_B": None if kl == float("inf") else round(kl, 9),
            "kl_is_infinite": kl == float("inf")}


def session_file_header(path: Path):
    """Reads the Strata session-file header (magic STRSESS\\x01, version u32 at offset 8) without touching anything else."""
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError as e:
        return {"exists": False, "reason": str(e)}
    if len(head) < 12:
        return {"exists": True, "magic_ok": False, "bytes": len(head)}
    magic_ok = head[:8] == b"STRSESS\x01"
    version = int.from_bytes(head[8:12], "little")
    # A file that is not a Strata session at all has no meaningful version: report the bytes it does have and stop.
    if not magic_ok:
        return {"exists": True, "magic_ok": False, "first_bytes": head[:12].hex()}
    return {"exists": True, "magic_ok": True, "version": version,
            "version_meaning": SESSION_FILE_VERSIONS.get(version, "unknown")}


# ------------------------------------------------------------------ the server under test
class ArmServer:
    """One server.py + one engine, started from a copy of production-config.json. Owns nothing outside outdir."""

    def __init__(self, arm, args, base_cfg, paths):
        self.arm = arm
        self.args = args
        self.paths = paths
        self.source_root = ARM_SOURCE[arm]
        self.exe = args.upstream_exe if arm == "upstream" else args.rebased_exe
        self.proc = None
        self.engine_pid = None
        self.server_log_path = paths["server_log"]
        self.engine_log_path = paths["engine_log"]
        self.url = f"http://127.0.0.1:{args.port}"
        self.cfg = self._build_config(base_cfg)

    def _build_config(self, base):
        cfg = json.loads(json.dumps(base))                     # a deep copy: never mutate production's dict
        args = list(cfg.get("args") or [])
        args = [a for a in args if a != "--vision"]
        for flag in ("--session-state-codec", "--spec", "--suffix-draft", "--adapt-swaps", "--pcie-frac",
                     "--prompt-cache-every"):                  # ours always wins, no duplicate flags
            while flag in args:
                i = args.index(flag)
                del args[i:i + 2]
        codec = ARM_CODEC[self.arm]
        if codec:
            args += ["--session-state-codec", codec]
        args += PINNED + ["--prompt-cache-every", PROMPT_CACHE_EVERY]
        cfg["args"] = args
        cfg.pop("vision", None)                                # text-only comparisons: no encoder process
        cfg["exe"] = str(self.exe)
        cfg["cwd"] = str(self.source_root)
        cfg["port"] = self.args.port                           # never 8080, which belongs to production
        cfg["host"] = "127.0.0.1"
        cfg["gpu"] = 0
        cfg["log"] = str(self.engine_log_path)
        cfg["slot_save_path"] = str(self.paths["sessions"])
        env = dict(cfg.get("env") or {})                       # config-carried env, not a global mutation
        env["STRATA_DUMP_FIRST_LOGITS"] = str(self.paths["logits"])
        if "STRATA_PREFILL_CPU_SHARE" in os.environ:
            env["STRATA_PREFILL_CPU_SHARE"] = os.environ["STRATA_PREFILL_CPU_SHARE"]
        cfg["env"] = env
        for key in ("tokenizer", "model_name", "cuda"):        # kept as production has them
            if base.get(key) is not None:
                cfg[key] = base[key]
        return cfg

    def write_config(self):
        self.paths["config"].write_text(json.dumps(self.cfg, indent=1), encoding="utf-8")

    def start(self, deadline_s):
        self.write_config()
        with open(self.server_log_path, "w", encoding="utf-8") as out:
            self.proc = subprocess.Popen(
                [VENV_PY, str(self.source_root / "serve" / "server.py"), "--engine", "strata",
                 "--config", str(self.paths["config"]), "--host", "127.0.0.1", "--port", str(self.args.port)],
                cwd=str(self.source_root), stdout=out, stderr=subprocess.STDOUT)
        self._wait_loaded(deadline_s)
        self.engine_pid = self._find_engine_pid()
        if self.engine_pid is None:
            raise HarnessError(f"arm {self.arm}: the server is loaded but no engine child is visible. "
                               f"Server log tail:\n{self.tail()}")

    def _wait_loaded(self, deadline_s):
        end = time.time() + deadline_s
        while time.time() < end:
            if self.proc.poll() is not None:
                raise HarnessError(f"arm {self.arm}: server exited ({self.proc.returncode}) before the model loaded. "
                                   f"Server log tail:\n{self.tail()}")
            try:
                st = self.get("/v1/status", 5)
                if st.get("loaded"):
                    return
            except Exception:
                pass
            time.sleep(3)
        raise HarnessError(f"arm {self.arm}: model not loaded within {deadline_s}s. "
                           f"Server log tail:\n{self.tail()}")

    def _find_engine_pid(self):
        """The engine child of our own server only: never a pgrep over the whole process table."""
        for pid in tree_of(self.proc.pid)[1:]:
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace")
            except OSError:
                continue
            if "--serve" in cmd and "server.py" not in cmd:
                return pid
        return None

    def tail(self, chars=2000):
        out = ""
        for p in (self.server_log_path, self.engine_log_path):
            try:
                with open(p, errors="replace") as f:
                    f.seek(max(0, os.path.getsize(p) - chars))
                    out += f"\n--- tail of {p.name} ---\n{f.read()}"
            except OSError:
                pass
        return out or "(no output)"

    # HTTP
    def _req(self, method, path, body=None, timeout=7200):
        data = json.dumps(body).encode() if body is not None else b"{}"
        req = urllib.request.Request(self.url + path, data=data,
                                     headers={"Content-Type": "application/json"}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as e:
            detail = ""
            if isinstance(e, urllib.error.HTTPError):
                try:
                    detail = e.read().decode(errors="replace")[:1000]
                except OSError:
                    pass
            raise HarnessError(f"arm {self.arm}: {method} {path} failed ({e}) {detail}\n{self.tail()}") from None

    def get(self, path, timeout=60):
        req = urllib.request.Request(self.url + path, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def log_pos(self):
        try:
            return os.path.getsize(self.engine_log_path)
        except OSError:
            return 0

    def log_since(self, pos):
        """New engine log lines since `pos`. Read-only; the engine owns the file."""
        try:
            with open(self.engine_log_path, "rb") as f:
                f.seek(pos)
                return f.read().decode(errors="replace").splitlines()
        except OSError:
            return []

    def stop(self):
        """Terminate and reap only this server and the children observed beneath it."""
        if not self.proc:
            return
        tracked = set(tree_of(self.proc.pid))
        if self.proc.poll() is None:
            self.proc.terminate()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            tracked.update(tree_of(self.proc.pid))
            self.proc.poll()  # reap the server; zombies must not hold up the next arm
            if not any(_pid_alive(pid) for pid in tracked):
                break
            time.sleep(0.25)
        for pid in tracked:
            if _pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.proc.wait(timeout=15)


def _pid_alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def _alive_pids():
    return {int(e) for e in os.listdir("/proc") if e.isdigit() and _pid_alive(int(e))}


# ------------------------------------------------------------------ the per-arm case run
class CaseRunner:
    def __init__(self, srv: ArmServer, arm, size, round_no, outdir, args):
        self.srv, self.arm, self.size, self.round = srv, arm, size, round_no
        self.outdir, self.args = outdir, args
        self.pos = 0
        case_dir = outdir / f"{arm}-r{round_no}-{size}"
        case_dir.mkdir(parents=True, exist_ok=True)
        self.case_dir = case_dir
        self.dumps = DumpIndex(case_dir / "first-logits")
        self.sampler = RssSampler(srv.engine_pid)

    def close(self):
        self.sampler.stop_now()

    # --- requests
    def chat(self, msgs, max_tokens):
        body = {"model": "strata", "max_tokens": max_tokens, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}, "messages": msgs}
        t0 = time.time()
        r = self.srv._req("POST", "/v1/chat/completions", body)
        wall = round(time.time() - t0, 3)
        msg = (r.get("choices") or [{}])[0].get("message", {})
        text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        t, u = r.get("timings") or {}, r.get("usage") or {}
        return {"answer": text, "id": r.get("id"), "finish_reason": (r.get("choices") or [{}])[0].get("finish_reason"),
                "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
                "decode_ms": t.get("predicted_ms"), "decode_tok_s": t.get("predicted_per_second"),
                "client_wall_s": wall, "messages_sha": messages_sha(msgs),
                "raw_timings_missing": t == {}, "usage_missing": u == {}}

    def slot(self, action, filename):
        t0 = time.time()
        r = self.srv._req("POST", f"/slots/0?action={action}", {"filename": filename})
        r["client_wall_s"] = round(time.time() - t0, 3)
        return r

    def metrics_cache(self):
        try:
            m = self.srv.get("/metrics", 20)
            cc = m.get("conversation_cache") or {}
            return {k: cc.get(k) for k in ("parked", "bytes", "parks", "restores", "evictions")}
        except Exception as e:
            return {"unavailable": f"{type(e).__name__}: {e}"}

    def run_request(self, spec, docs, tag):
        before = self.pos
        # The engine increments its dump counter for EVERY chat request, including prefill and park requests.
        claimed = self.dumps.claim(tag, spec["name"])
        measured = self.sampler.measure(lambda: self.chat(spec["messages"], spec["max_tokens"]))
        r = measured["result"]
        lines = self.srv.log_since(before)
        self.pos = self.srv.log_pos()
        r.update({"name": spec["name"], "kind": tag, "log_events": parse_log_events(lines),
                  "rss_peak_increment_mib": measured["rss_peak_increment_mib"],
                  "rss_baseline_mib": measured["baseline_mib"]})
        if claimed:
            r["first_logits_dump"] = self.dumps.settle(claimed)
        return r

    # --- the case itself
    def run(self, docs):
        specs = materialize(requests_for(self.size), docs)
        by_name = {s["name"]: s for s in specs}
        rec = {"arm": self.arm, "round": self.round, "target_prompt_tokens": self.size,
               "started_at": int(time.time()), "steps": [], "session_files": [], "save_restore_reps": []}

        def emit(step):
            rec["steps"].append(step)
            self._write(rec)                                  # incremental: a later failure keeps this evidence

        # Save the long A session itself; unrelated single-question prompts would replace it before SAVE.
        for name in ("prefill_A",):
            step = self.run_request(by_name[name], docs, "fresh_prompt")
            step["needle_expected"] = {"needle_10": CODES[0], "needle_50": CODES[1],
                                       "needle_90": CODES[2]}.get(name)
            if step.get("needle_expected"):
                step["needle_found"] = step["needle_expected"] in (step["answer"] or "").lower()
                step["recall_class"] = "fresh_prompt_needle"   # never counted as restored-state recall
            emit(step)
        prefill = next(s for s in rec["steps"] if s["name"] == "prefill_A")
        rec["actual_prompt_tokens_prefill"] = prefill.get("prompt_tokens")

        # 2. first SAVE, before anything else can park the conversation
        first_name = f"{self.arm}-r{self.round}-{self.size}-A-first.json"
        save = self.sampler.measure(lambda: {"api": self.slot("save", first_name),
                                             "log_events": self._logs_since_save()})
        path = self.srv.paths["sessions"] / first_name
        sf = {"filename": first_name, "api_response": save["result"]["api"],
              "file_bytes": path.stat().st_size if path.exists() else None,
              "sha256": sha256_file(path), "header": session_file_header(path),
              "log_events": save["result"]["log_events"],
              "rss_peak_increment_mib": save["rss_peak_increment_mib"], "kind": "save_first"}
        rec["session_files"].append(sf)
        emit(sf)

        # 3. disjoint B parks A; then resume A and check the parked-state evidence explicitly
        park = self.run_request(by_name["park_B"], docs, "park")
        emit(park)
        resume = self.run_request(by_name["resume_A"], docs, "parked_resume")
        ev = self._restore_evidence(resume["log_events"])
        resume["restore_evidence"] = ev
        resume["verified_restored"] = bool(ev["evidence_lines"]) and (resume.get("cache_n") or 0) > 0
        if not resume["verified_restored"]:
            resume["verification_failed_because"] = ev["missing"]
        resume["expected_words"] = list(CODES)
        resume["words_found"] = [c for c in CODES if c in (resume["answer"] or "").lower()]
        emit(resume)

        # 4. mid-document edit/rewind: select an earlier retained checkpoint
        emit(self.run_request(by_name["park_B"], docs, "park_before_mid_document_rewind"))
        rewind = self.run_request(self._rewind_spec(specs), docs, "mid_document_rewind")
        ck = rewind["log_events"]["checkpoint_selected"]
        rewind["coded_checkpoint_restore_exercised"] = bool(ck)
        rewind["retained_coded_events"] = rewind["log_events"]["retained_coded"]
        # where the edit aimed, as a target only: the API's prompt_tokens for this request is the actual figure
        rewind["rewind_target_tokens"] = self.args.rewind_tokens or min(max(256,self.size-256),max(9216,self.size//2+1024))
        if not ck:
            rewind["note"] = ("no 'resumed through the checkpoint at N tokens' line for this request. That phrase is "
                              "only printed when a retained checkpoint holds coded state, so an fp32 arm legitimately "
                              "has none; treat it as unmeasured, not as a failure. The engine also keeps only the last "
                              f"{PROMPT_CACHE_EVERY}-checkpoint-spacing prefixes of the current conversation, so a cut "
                              "below its first checkpoint has nothing retained to select.")
        emit(rewind)

        # 5. disk restore: switch away, RESTORE the saved file, ask the same continuation question
        away = self.run_request(by_name["park_B"], docs, "switch_away")
        emit(away)
        restore = self._disk_restore(first_name, "disk_restore_first")
        rec["session_files"][-1]["restore_evidence"] = restore["api_response"]
        emit(restore)
        verified_disk = bool(restore["log_events"]["session_restored"]) and \
            (restore["api_response"].get("n_restored") or 0) > 0
        rec["disk_restore_verified"] = verified_disk

        # 6. long continuation after the restore (or a fresh prompt when the restore could not be verified)
        cont_kind = "after_verified_restore" if verified_disk else "fresh_prompt_no_verified_restore"
        cont = self.run_request(by_name["continuation"], docs, cont_kind)
        cont["restore_evidence_present"] = verified_disk
        (self.case_dir / "continuation.txt").write_text(cont.get("answer") or "", encoding="utf-8")
        emit(cont)

        # 7. warm SAVE/RESTORE and short turn/park/resume reps, preserving every run
        for i in range(self.args.reps):
            rec["save_restore_reps"].append(self._rep_cycle(by_name, first_name, i))
            self._write(rec)

        rec["conversation_cache_final"] = self.metrics_cache()
        rec["finished_at"] = int(time.time())
        self._write(rec)
        return rec

    def _logs_since_save(self):
        before = self.pos
        lines = self.srv.log_since(before)
        self.pos = self.srv.log_pos()
        return parse_log_events(lines)

    def _disk_restore(self, filename, kind):
        before = self.pos
        measured = self.sampler.measure(lambda: {"api": self.slot("restore", filename)})
        lines = self.srv.log_since(before)
        self.pos = self.srv.log_pos()
        path = self.srv.paths["sessions"] / filename
        return {"kind": kind, "filename": filename, "api_response": measured["result"]["api"],
                "log_events": parse_log_events(lines),
                "file_bytes": path.stat().st_size if path.exists() else None, "sha256": sha256_file(path),
                "header": session_file_header(path),
                "rss_peak_increment_mib": measured["rss_peak_increment_mib"],
                "rss_baseline_mib": measured["baseline_mib"]}

    def _rep_cycle(self, by_name, filename, i):
        """One warm cycle: a short turn on the current conversation, SAVE, park with B, RESTORE, short resume."""
        filename = f"warm-{i}-{filename}"  # preserve the original SAVE bytes for cross-build comparison
        out = {"rep": i}
        out["turn"] = self.run_request(by_name["cycle_turn"], {}, f"warm_cycle_{i}")
        before = self.pos
        save = self.sampler.measure(lambda: {"api": self.slot("save", filename)})
        if save["result"]["api"].get("n_saved", 0) < self.size // 2:
            raise HarnessError("warm SAVE did not contain the long conversation; refusing a mislabeled measurement")
        lines = self.srv.log_since(before)
        self.pos = self.srv.log_pos()
        path = self.srv.paths["sessions"] / filename
        out["save_warm"] = {"api_response": save["result"]["api"], "log_events": parse_log_events(lines),
                            "file_bytes": path.stat().st_size if path.exists() else None,
                            "sha256": sha256_file(path), "header": session_file_header(path),
                            "kind": "save_warm", "rss_peak_increment_mib": save["rss_peak_increment_mib"]}
        out["park"] = self.run_request(by_name["park_B"], {}, f"warm_park_{i}")
        out["restore"] = self._disk_restore(filename, f"restore_warm_{i}")
        continued = dict(by_name["cycle_resume"])
        continued["messages"] = list(by_name["cycle_turn"]["messages"]) + [
            {"role":"assistant", "content":out["turn"]["answer"]},
            {"role":"user", "content":"Name each magic code word from the document, one per line."}]
        r = self.run_request(continued, {}, f"warm_cycle_{i}")
        ev = self._restore_evidence(r["log_events"])
        r["verified_restored"] = bool(out["restore"]["log_events"]["session_restored"]) and \
            out["restore"]["api_response"].get("n_restored", 0) >= self.size // 2 and (r.get("cache_n") or 0) > 0
        r["restore_source"] = "disk"
        r["restore_evidence"] = ev
        out["resume"] = r
        return out

    def _rewind_spec(self, specs):
        """The mid-document edit request. Its document is A_EARLY, the same body cut well behind A's live end, so an
        earlier retained checkpoint has to be selected rather than the conversation simply continued."""
        return next(s for s in specs if s["name"] == "rewind_mid")

    def _restore_evidence(self, events):
        """What the engine's own log says about a restore/borrow for this request. Missing evidence is listed, never
        assumed."""
        lines, missing = [], []
        for kind in ("resume", "session_restored", "checkpoint_selected"):
            for e in events.get(kind, []):
                lines.append({"kind": kind, **e})
        if not lines:
            missing.append("no conversation-cache restore, session RESTORE, or checkpoint-selection line in this "
                           "request's log window")
        return {"evidence_lines": lines, "missing": missing}

    def _write(self, rec):
        p = self.case_dir / "case.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        os.replace(tmp, p)


# ------------------------------------------------------------------ evidence shared across arms
def collect_evidence(outdir):
    """Every case.json under outdir, keyed (arm, round, size)."""
    found = {}
    for p in sorted(outdir.glob("*/case.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            found[f"unreadable:{p}"] = {"error": str(e)}
            continue
        key = f"{rec.get('arm')}-r{rec.get('round')}-{rec.get('target_prompt_tokens')}"
        found[key] = rec
    return found


# ------------------------------------------------------------------ cross-arm comparisons
def logit_comparisons(outdir, arms_present):
    """Compares the first-token dumps of matched requests between arm pairs. Returns one entry per pair/size/round."""
    cases = collect_evidence(outdir)
    wanted = [("upstream", "fp32"), ("fp32", "bf16")] if len(arms_present) > 1 else []
    out = []
    for arm_a, arm_b in wanted:
        if arm_a not in arms_present or arm_b not in arms_present:
            continue
        for key, rec_a in sorted(cases.items()):
            # iterate only the left arm's records: walking every case would emit each pair several times
            if "unreadable" in key or not isinstance(rec_a, dict) or rec_a.get("arm") != arm_a:
                continue
            steps_a = {s.get("name"): s for s in rec_a.get("steps", []) if isinstance(s, dict)}
            size, rnd = rec_a.get("target_prompt_tokens"), rec_a.get("round")
            rec_b = cases.get(f"{arm_b}-r{rnd}-{size}")     # the same case in the right arm
            if not isinstance(rec_b, dict):
                continue
            steps_b = {s.get("name"): s for s in rec_b.get("steps", []) if isinstance(s, dict)}
            for name in ("resume_A", "rewind_mid", "continuation"):
                sa, sb = steps_a.get(name), steps_b.get(name)
                if not sa or not sb:
                    continue
                entry = {"pair": f"{arm_a}|{arm_b}", "case_key": key, "request": name, "round": rnd, "size": size}
                if sa.get("messages_sha") != sb.get("messages_sha"):
                    entry["matched"] = False
                    entry["why_not_comparable"] = ("prompt message hashes differ "
                                                   f"({sa.get('messages_sha')} vs {sb.get('messages_sha')}); the two "
                                                   "arms did not send an identical prompt")
                    out.append(entry)
                    continue
                da, db = sa.get("first_logits_dump"), sb.get("first_logits_dump")
                if not da or not db:
                    entry["matched"] = False
                    entry["why_not_comparable"] = ("this request did not ask for a first-token dump on both arms, so "
                                                   "there is nothing to compare")
                    out.append(entry)
                    continue
                va, sta = read_logits(Path(da["path"]))
                vb, stb = read_logits(Path(db["path"]))
                entry["dump_a"], entry["dump_b"] = sta, stb
                if va is None or vb is None:
                    entry["matched"] = False
                    entry["why_not_comparable"] = ("one or both dump files could not be read as raw float32; they are "
                                                   "kept on disk untouched for inspection")
                    out.append(entry)
                    continue
                entry["matched"] = True
                entry["comparison"] = kl_and_top1(va, vb)
                entry["restored_state_verified"] = bool(
                    (sa.get("verified_restored") or sa.get("restore_evidence_present")) and
                    (sb.get("verified_restored") or sb.get("restore_evidence_present")))
                entry["scope_note"] = ("first generated token of this request only, at the position the engine actually "
                                       "dumped; not a teacher-forced comparison of the whole continuation")
                out.append(entry)
    return out


def session_file_comparisons(outdir):
    """Where the saved histories are identical, compare the default-fp32 session files byte for byte."""
    cases = collect_evidence(outdir)
    out = []
    for key, rec in sorted(cases.items()):
        if not isinstance(rec, dict) or rec.get("arm") != "upstream":
            continue
        size, rnd = rec.get("target_prompt_tokens"), rec.get("round")
        partner = cases.get(f"fp32-r{rnd}-{size}")
        if not isinstance(partner, dict):
            continue
        for fa in rec.get("session_files", []):
            match = next((fb for fb in partner.get("session_files", [])
                          if fb.get("filename", "").endswith(f"-A-first.json")), None)
            if not match:
                continue
            entry = {"case_key": key, "round": rnd, "size": size, "kind": "first_save",
                     "upstream": {k: fa.get(k) for k in ("filename", "file_bytes", "sha256", "header")},
                     "rebased_fp32": {k: match.get(k) for k in ("filename", "file_bytes", "sha256", "header")}}
            entry["same_sha256"] = (fa.get("sha256") is not None and fa.get("sha256") == match.get("sha256"))
            entry["same_bytes"] = (fa.get("file_bytes") is not None and
                                   fa.get("file_bytes") == match.get("file_bytes"))
            entry["expected_versions"] = {"upstream": 1, "rebased_fp32": 1}
            entry["caveat"] = ("identical bytes are only evidence of compatibility when the two prompts were "
                               "identical; check the case's prompt hashes and answers before reading this as a match")
            out.append(entry)
    return out


# ------------------------------------------------------------------ summary
def stats(values):
    v = [x for x in values if isinstance(x, (int, float))]
    if not v:
        return {"n": 0}
    return {"n": len(v), "median": round(statistics.median(v), 3), "min": round(min(v), 3), "max": round(max(v), 3)}


def build_summary(outdir, meta):
    cases = collect_evidence(outdir)
    arms_present = sorted({r["arm"] for r in cases.values() if isinstance(r, dict) and r.get("arm")})
    per_arm, recall = {}, {}
    for arm in arms_present:
        mem, lat, saves, restores = [], [], [], []
        verified_recall = fresh_needle = 0
        for key, rec in sorted(cases.items()):
            if not isinstance(rec, dict) or rec.get("arm") != arm:
                continue
            for s in rec.get("steps", []):
                if s.get("rss_peak_increment_mib") is not None:
                    mem.append(s["rss_peak_increment_mib"])
                if s.get("prompt_ms") is not None:
                    lat.append(s["prompt_ms"])
                if s.get("recall_class") == "fresh_prompt_needle":
                    fresh_needle += int(bool(s.get("needle_found")))
                if s.get("name") == "resume_A" and s.get("verified_restored"):
                    verified_recall += len(s.get("words_found") or [])
            for rep in rec.get("save_restore_reps", []):
                for k, bucket in (("save_warm", saves), ("restore", restores)):
                    api = (rep.get(k) or {}).get("api_response") or {}
                    ms = ((api.get("timings") or {}).get("save_ms") if k == "save_warm"
                          else (api.get("timings") or {}).get("restore_ms"))
                    if ms is not None:
                        bucket.append(ms)
        per_arm[arm] = {"cases": sum(1 for r in cases.values() if isinstance(r, dict) and r.get("arm") == arm),
                        "rss_peak_increment_mib": stats(mem), "prompt_ms": stats(lat),
                        "warm_save_ms": stats(saves), "warm_restore_ms": stats(restores)}
        recall[arm] = {"verified_restored_needle_hits": verified_recall,
                       "fresh_prompt_needle_hits_not_restored_state": fresh_needle}
    return {
        "metadata": meta,
        "arms_present": arms_present,
        "per_arm_measurements": per_arm,
        "recall_counts": {"note": "counted only where this harness verified a restore (engine log evidence plus "
                                  "cache_n > 0); fresh-prompt needle hits are reported separately and never as "
                                  "restored-state recall", **recall},
        "logit_comparisons": logit_comparisons(outdir, arms_present),
        "session_file_comparisons": session_file_comparisons(outdir),
        "not_run_or_not_measured": [
            "no invented pass/fail threshold: numbers only, root judges them",
            "first-token KL covers only the positions actually dumped (the first token of resume_A, rewind_mid and "
            "continuation); it is not a teacher-forced comparison of the full continuation",
            "RSS figures are observed peak increments sampled every 20 ms from outside the process - lower bounds, "
            "not true high-water marks; pageable parked state may already be written out",
            "the BF16 prefix-borrow/donor path is exercised only if the engine chose it during these requests; the "
            "engine logs no dedicated donor line in the phrases this harness parses, so donor retention is covered by "
            "the repository's own unit tests rather than claimed here",
            "a missing measurement is reported as missing: log/metrics wording changes on main are not guessed at",
        ],
        "written_at": int(time.time()),
    }


# ------------------------------------------------------------------ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream-exe", required=True, type=Path)
    ap.add_argument("--rebased-exe", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--sizes", default="32768,65536", help="comma-separated target prompt tokens (smaller for smoke)")
    ap.add_argument("--reps", type=int, default=5, help="warm SAVE/RESTORE cycles per case")
    ap.add_argument("--rounds", type=int, default=2, help="arm order alternates forward then reverse")
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--arms", default="upstream,fp32,bf16")
    ap.add_argument("--load-deadline-s", type=int, default=1800, help="how long to wait for the model to load")
    ap.add_argument("--rewind-tokens", type=int, default=None, metavar="N",
                    help=f"where the mid-document edit cuts (a target). Aim it just above a multiple of "
                         f"--prompt-cache-every ({PROMPT_CACHE_EVERY}) so an earlier retained checkpoint exists to "
                         f"select; below the first checkpoint there is nothing retained. Default: near the middle "
                         f"of each size, above the first retained periodic checkpoint where possible.")
    args = ap.parse_args(argv)
    args.sizes = [int(x) for x in str(args.sizes).split(",") if x.strip()]
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in args.arms if a not in ARMS]
    if bad:
        ap.error(f"--arms accepts {', '.join(ARMS)}; got {bad}")
    if args.port == 8080:
        ap.error("--port 8080 belongs to production; this harness refuses to touch it")
    for p in (args.upstream_exe, args.rebased_exe):
        if not Path(p).is_file():
            ap.error(f"engine binary not built yet: {p}")
    return args


def main(argv=None):
    args = parse_args(argv)
    outdir = Path(args.out_dir)
    if outdir.exists() and any(outdir.iterdir()):
        raise SystemExit("Use a new empty output directory to preserve previous evidence")
    outdir.mkdir(parents=True, exist_ok=True)

    base_cfg = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    meta_path = HERE / "provenance.json"
    provenance = json.loads(meta_path.read_text()) if meta_path.is_file() else {}

    gpu_procs = gpu_compute_processes()
    if gpu_procs:
        raise SystemExit("refusing to run: GPU compute process(es) already present, and this harness kills nothing:\n  "
                         + "\n  ".join(f"pid {p['pid']} {p['process']} ({p['used_mib']} MiB)" for p in gpu_procs)
                         + "\nStop production first (root's job), then rerun.")
    if port_in_use(args.port):
        raise SystemExit(f"refusing to run: something is already listening on 127.0.0.1:{args.port}")

    meta = {"harness": "model_validation.py", "started_at": int(time.time()), "argv": sys.argv[1:],
            "cli": vars(args) | {"upstream_exe": str(args.upstream_exe), "rebased_exe": str(args.rebased_exe),
                                 "out_dir": str(outdir)},
            "production_config_sha256": sha256_file(PRODUCTION_CONFIG), "provenance": provenance,
            "pinned_engine_args": PINNED, "prompt_cache_every": PROMPT_CACHE_EVERY,
            "python_interpreter": VENV_PY, "gpu_info": gpu_info(),
            "sources": {a: {"root": str(ARM_SOURCE[a]),
                            "head": git_read_only(ARM_SOURCE[a], "rev-parse", "HEAD"),
                            "dirty": git_read_only(ARM_SOURCE[a], "status", "--porcelain") != ""}
                        for a in args.arms},
            "engines": {"upstream": str(args.upstream_exe), "rebased": str(args.rebased_exe)},
            "arm_log_patterns": {k: v.pattern for k, v in LOG_PATTERNS.items()},
            "cases": []}
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    runs = [(r, list(args.arms) if r % 2 == 0 else list(reversed(args.arms))) for r in range(args.rounds)]
    errors = []
    try:
        for round_no, order in runs:
            for arm in order:
                for size in args.sizes:
                    tag = f"round {round_no} arm {arm} size {size}"
                    print(f"[harness] starting {tag}", flush=True)
                    paths = {"config": outdir / f"{arm}-r{round_no}-{size}.json",
                             "server_log": outdir / f"{arm}-r{round_no}-{size}.server.log",
                             "engine_log": outdir / f"{arm}-r{round_no}-{size}.engine.log",
                             "sessions": outdir / "sessions" / f"{arm}-r{round_no}-{size}",
                             "logits": outdir / f"{arm}-r{round_no}-{size}" / "first-logits"}
                    paths["sessions"].mkdir(parents=True, exist_ok=True)
                    (outdir / f"{arm}-r{round_no}-{size}").mkdir(parents=True, exist_ok=True)
                    srv = ArmServer(arm, args, base_cfg, paths)
                    runner = None
                    try:
                        srv.start(args.load_deadline_s)
                        marker, docs = documents_for(size, ARM_SOURCE["upstream"], args.rewind_tokens)
                        runner = CaseRunner(srv, arm, size, round_no, outdir, args)
                        rec = runner.run(docs)
                        rec["case_marker"] = marker
                        meta["cases"].append({"round": round_no, "arm": arm, "size": size,
                                              "order_in_round": order.index(arm),
                                              "engine_pid": srv.engine_pid,
                                              "actual_prompt_tokens": rec.get("actual_prompt_tokens_prefill"),
                                              "disk_restore_verified": rec.get("disk_restore_verified"),
                                              "case_json": str(runner.case_dir / "case.json")})
                        print(f"[harness] {tag}: prompt_tokens={rec.get('actual_prompt_tokens_prefill')} "
                              f"disk_restore_verified={rec.get('disk_restore_verified')}", flush=True)
                    except (HarnessError, Exception) as e:      # keep the evidence, say what broke, continue
                        errors.append(f"{tag}: {type(e).__name__}: {e}")
                        print(f"[harness] FAILED {tag}: {e}", flush=True)
                    finally:
                        if runner:
                            runner.close()
                        srv.stop()
                        time.sleep(2)                           # own child tree has exited; let the driver release its contexts
                        (outdir / "metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    finally:
        meta["errors"] = errors
        (outdir / "metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        summary = build_summary(outdir, meta)
        summary["errors"] = errors
        (outdir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"[harness] wrote {outdir / 'summary.json'}; {len(errors)} failed run(s)", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
