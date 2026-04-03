"""
CoT Data Generation System for NVIDIA Nemotron Competition
- Agent 1: Solve problems with retry until correct
- Agent 2: Distill reasoning into concise CoT + extract techniques
- Concurrent execution with ThreadPoolExecutor
- Resilient: resume from checkpoint, retry on failure
"""

import os
import json
import time
import logging
import signal
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import requests

# === Config ===
load_dotenv(Path(__file__).parent / ".env")

API_BASE_URL = os.getenv("API_BASE_URL")
API_MODEL = os.getenv("API_MODEL")
API_KEY = os.getenv("API_KEY")

TRAIN_CSV = Path(__file__).parent / "train.csv"
OUTPUT_DIR = Path(__file__).parent / "cot_output"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
RAW_COT_DIR = OUTPUT_DIR / "raw_cot"
DISTILLED_COT_DIR = OUTPUT_DIR / "distilled_cot"
TECHNIQUES_DIR = OUTPUT_DIR / "techniques"

MAX_SOLVE_RETRIES = 10  # max retries per problem to get correct answer
MAX_API_RETRIES = 5     # max API call retries on network error
RETRY_DELAY = 2         # base delay in seconds for exponential backoff

for d in [OUTPUT_DIR, RAW_COT_DIR, DISTILLED_COT_DIR, TECHNIQUES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "generation.log"),
    ],
)
log = logging.getLogger(__name__)

API_ENDPOINT = API_BASE_URL.rstrip("/") + "/v1/chat/completions"
API_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# Thread-safe checkpoint lock
ckpt_lock = threading.Lock()


# === Graceful shutdown ===
shutdown_requested = False

def handle_signal(signum, frame):
    global shutdown_requested
    log.info("Shutdown requested, finishing current tasks...")
    shutdown_requested = True

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


# === Robust API call ===
def api_call(messages: list[dict], temperature: float = 0.7, max_tokens: int = 8192) -> str:
    """Call API with exponential backoff retry."""
    payload = {
        "model": API_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(MAX_API_RETRIES):
        if shutdown_requested:
            raise RuntimeError("Shutdown requested")
        try:
            resp = requests.post(API_ENDPOINT, headers=API_HEADERS, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or ""
            return reasoning + ("\n\n" if reasoning and content else "") + content
        except Exception as e:
            delay = RETRY_DELAY * (2 ** attempt)
            log.warning(f"API error (attempt {attempt+1}/{MAX_API_RETRIES}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError(f"API call failed after {MAX_API_RETRIES} retries")


# === Load / Save checkpoint (thread-safe) ===
def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"solved": {}, "distilled": {}, "techniques_extracted": {}}


def save_checkpoint(ckpt: dict):
    with ckpt_lock:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(ckpt, f, indent=2)


# === Load training data ===
def load_train_data():
    import csv
    data = []
    with open(TRAIN_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({"id": row["id"], "prompt": row["prompt"], "answer": row["answer"]})
    return data


# === Agent 1: Solver ===
SOLVER_SYSTEM = """You are a math and logic reasoning expert. Solve the given problem step by step.

Rules:
1. Think carefully and show your complete reasoning process
2. Put your final answer inside \\boxed{...}
3. Show all intermediate steps and calculations
4. If you make a mistake, you will be told to try again without the correct answer
"""

SOLVER_RETRY = """Your previous answer is INCORRECT. The correct answer is NOT what you got.

Please reconsider the problem from scratch. Try a different approach if needed.
Show your complete reasoning and put the final answer inside \\boxed{...}.

Your previous attempt:
{previous}"""


def extract_boxed_answer(text: str) -> str | None:
    import re
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r'\\boxed\{(.*?)\}(?=\s*$|\s*<)', text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def normalize_answer(ans: str) -> str:
    ans = str(ans).strip()
    try:
        f = float(ans)
        if f == int(f):
            return str(int(f))
        return str(round(f, 4))
    except (ValueError, OverflowError):
        return ans.lower().strip()


def solve_problem(prompt: str, correct_answer: str) -> dict | None:
    messages = [
        {"role": "system", "content": SOLVER_SYSTEM},
        {"role": "user", "content": f"{prompt}\n\nPut your final answer inside \\boxed{{}}."},
    ]

    for attempt in range(MAX_SOLVE_RETRIES):
        if shutdown_requested:
            return None

        response = api_call(messages, temperature=0.7 if attempt == 0 else 0.9 - 0.1 * min(attempt, 5))
        extracted = extract_boxed_answer(response)

        if extracted and normalize_answer(extracted) == normalize_answer(correct_answer):
            log.info(f"  ✅ Correct answer obtained on attempt {attempt + 1}")
            return {"full_reasoning": response, "answer": extracted}

        log.info(f"  ❌ Wrong answer: got '{extracted}', expected '{correct_answer}'")
        messages.append({"role": "assistant", "content": response})
        messages.append({
            "role": "user",
            "content": SOLVER_RETRY.format(previous=response[-500:] if len(response) > 500 else response),
        })

    log.warning(f"  Failed to solve after {MAX_SOLVE_RETRIES} attempts")
    return None


# === Agent 2: Distiller ===
DISTILLER_SYSTEM = """You are a reasoning compression expert. Your job is to take a detailed problem-solving process and create a concise, high-quality Chain-of-Thought (CoT).

Requirements:
1. Keep ONLY the essential reasoning steps that lead to the correct answer
2. Remove false starts, redundant calculations, and verbose explanations
3. Keep the logic clear and easy to follow
4. The CoT should be complete enough that someone could reproduce the answer from it
5. Put the final answer inside \\boxed{...}

Output format: concise step-by-step reasoning followed by \\boxed{answer}"""

TECHNIQUE_SYSTEM = """You are a math education expert. Based on the detailed solution process provided, extract:

1. Problem-solving technique(s) used
2. Key insights or shortcuts
3. Common pitfalls to avoid
4. A general approach that would work for similar problems

Be concise and actionable. Focus on transferable skills, not problem-specific details."""


def distill_reasoning(prompt: str, full_reasoning: str) -> str | None:
    if shutdown_requested:
        return None
    messages = [
        {"role": "system", "content": DISTILLER_SYSTEM},
        {"role": "user", "content": f"Problem: {prompt}\n\nFull reasoning process:\n{full_reasoning}"},
    ]
    return api_call(messages, temperature=0.3, max_tokens=2048)


def extract_technique(prompt: str, full_reasoning: str) -> str | None:
    if shutdown_requested:
        return None
    messages = [
        {"role": "system", "content": TECHNIQUE_SYSTEM},
        {"role": "user", "content": f"Problem: {prompt}\n\nFull reasoning process:\n{full_reasoning}"},
    ]
    return api_call(messages, temperature=0.3, max_tokens=1024)


# === Worker: process one problem end-to-end ===
def process_problem(item: dict, ckpt: dict, progress: dict) -> str:
    """Process a single problem: solve -> distill -> extract technique. Returns pid."""
    pid = item["id"]
    prompt = item["prompt"]
    answer = item["answer"]

    try:
        # === Phase 1: Solve ===
        if pid not in ckpt["solved"]:
            log.info(f"[{pid}] Solving: {prompt[:80]}...")
            result = solve_problem(prompt, answer)

            if result:
                raw_path = RAW_COT_DIR / f"{pid}.json"
                with open(raw_path, "w") as f:
                    json.dump({"id": pid, "prompt": prompt, "answer": answer, "full_reasoning": result["full_reasoning"]}, f, ensure_ascii=False, indent=2)
                with ckpt_lock:
                    ckpt["solved"][pid] = {"raw_path": str(raw_path), "attempts": 1}
                save_checkpoint(ckpt)
                log.info(f"[{pid}] ✅ Saved raw CoT")
            else:
                with ckpt_lock:
                    ckpt["solved"][pid] = {"raw_path": None, "attempts": MAX_SOLVE_RETRIES, "failed": True}
                save_checkpoint(ckpt)
                log.warning(f"[{pid}] ❌ Failed to solve")
                with ckpt_lock:
                    progress["done"] += 1
                return pid

        # Skip if solve failed
        if ckpt["solved"][pid].get("failed"):
            with ckpt_lock:
                progress["done"] += 1
            return pid

        # Load raw reasoning
        raw_path = ckpt["solved"][pid]["raw_path"]
        if not raw_path or not os.path.exists(raw_path):
            with ckpt_lock:
                progress["done"] += 1
            return pid

        with open(raw_path) as f:
            full_reasoning = json.load(f)["full_reasoning"]

        # === Phase 2a: Distill ===
        if pid not in ckpt["distilled"]:
            log.info(f"[{pid}] Distilling reasoning...")
            distilled = distill_reasoning(prompt, full_reasoning)
            if distilled:
                distill_path = DISTILLED_COT_DIR / f"{pid}.json"
                with open(distill_path, "w") as f:
                    json.dump({"id": pid, "prompt": prompt, "answer": answer, "distilled_cot": distilled}, f, ensure_ascii=False, indent=2)
                with ckpt_lock:
                    ckpt["distilled"][pid] = str(distill_path)
                save_checkpoint(ckpt)
                log.info(f"[{pid}] ✅ Distilled CoT saved")

        # === Phase 2b: Extract technique ===
        if pid not in ckpt["techniques_extracted"]:
            log.info(f"[{pid}] Extracting technique...")
            technique = extract_technique(prompt, full_reasoning)
            if technique:
                tech_path = TECHNIQUES_DIR / f"{pid}.json"
                with open(tech_path, "w") as f:
                    json.dump({"id": pid, "prompt": prompt, "answer": answer, "technique": technique}, f, ensure_ascii=False, indent=2)
                with ckpt_lock:
                    ckpt["techniques_extracted"][pid] = str(tech_path)
                save_checkpoint(ckpt)
                log.info(f"[{pid}] ✅ Technique saved")

        with ckpt_lock:
            progress["done"] += 1
        return pid

    except Exception as e:
        log.error(f"[{pid}] Unhandled error: {e}")
        with ckpt_lock:
            progress["done"] += 1
        return pid


# === Main pipeline ===
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Only process first N problems (0=all)")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent workers (default=8)")
    args = parser.parse_args()

    if not all([API_BASE_URL, API_MODEL, API_KEY]):
        print("Please fill in .env file with API_BASE_URL, API_MODEL, API_KEY")
        sys.exit(1)

    log.info(f"Loading training data from {TRAIN_CSV}")
    train_data = load_train_data()
    if args.limit > 0:
        train_data = train_data[:args.limit]
    log.info(f"Loaded {len(train_data)} problems (limit={args.limit}, workers={args.workers})")

    ckpt = load_checkpoint()
    log.info(f"Checkpoint: {len(ckpt['solved'])} solved, {len(ckpt['distilled'])} distilled, {len(ckpt['techniques_extracted'])} techniques")

    total = len(train_data)
    progress = {"done": 0}

    # Filter to only unsolved or incomplete problems
    todo = [item for item in train_data if (
        item["id"] not in ckpt["solved"]
        or ckpt["solved"][item["id"]].get("failed")
        or (not ckpt["solved"][item["id"]].get("failed") and (
            item["id"] not in ckpt["distilled"]
            or item["id"] not in ckpt["techniques_extracted"]
        ))
    )]
    log.info(f"Remaining to process: {len(todo)}/{total}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_problem, item, ckpt, progress): item["id"] for item in todo}

        for future in as_completed(futures):
            pid = futures[future]
            if shutdown_requested:
                log.info("Shutdown requested, cancelling remaining...")
                for f in futures:
                    f.cancel()
                break
            try:
                future.result()
                done = progress["done"]
                log.info(f"Progress: {done}/{total} ({done/total*100:.1f}%)")
            except Exception as e:
                log.error(f"[{pid}] Future error: {e}")

    # === Final merge ===
    log.info("Merging all outputs...")
    merge_outputs(ckpt)
    log.info("Done!")


def merge_outputs(ckpt: dict):
    full_data = []
    for pid, info in ckpt["solved"].items():
        if info.get("failed") or not info.get("raw_path"):
            continue
        with open(info["raw_path"]) as f:
            full_data.append(json.load(f))

    distilled_data = []
    for pid, path in ckpt["distilled"].items():
        if not path or not os.path.exists(path):
            continue
        with open(path) as f:
            distilled_data.append(json.load(f))

    with open(OUTPUT_DIR / "full_cot_dataset.json", "w") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "distilled_cot_dataset.json", "w") as f:
        json.dump(distilled_data, f, ensure_ascii=False, indent=2)

    log.info(f"Merged: {len(full_data)} raw CoT, {len(distilled_data)} distilled CoT")


if __name__ == "__main__":
    main()
