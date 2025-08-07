#!/usr/bin/env python3
"""
Usage:
  python scripts/simulate_pdp_eval.py --base-url https://digitalfuelcapital.github.io/dfc-public-pdp-test-pages/ --runs 100

Summary:
- Crawls local PDP file list from pdp_pages/active (HTML files) and maps them to full GitHub Pages URLs using --base-url.
- Loads shoppers and prompts from data/shoppers.csv with columns: name,memory,prompt.
- Runs N simulations (default 100). Each run:
  * Selects a single shopper row (name + memory + a specific prompt) at random unless --shopper-name is provided.
  * Samples up to --max-pdp pages (default: all found) and shuffles order if --shuffle.
  * Calls OpenAI Responses API (model default 'gpt-4o') with a single instruction: choose the best PDP for the prompt and provide a short justification + feature importance.
  * Expects JSON output with fields: chosen_url, justification, features (list of {name, weight, note}).
- Writes per-run records to data/output/raw_runs.jsonl and aggregates to:
  * data/output/summary.csv (win counts per PDP)
  * data/output/feature_importance.csv (average feature weights per PDP and feature)
  * data/output/log.csv (run metadata)

Environment variables:
- OPENAI_API_KEY (required)
- OPENAI_BASE_URL (optional, defaults to https://api.openai.com/v1)

CLI options:
- --base-url: Required base URL where GitHub Pages hosts the PDPs.
- --runs: Number of simulations (default 100)
- --model: Model name (default 'gpt-4o')
- --temperature: Decoding temperature (default 0.2)
- --timeout: HTTP timeout seconds per call (default 60)
- --shuffle: Shuffle PDP list before each run (flag)
- --max-pdp: Limit number of PDPs sampled per run (default: use all found)
- --shopper-name: Filter to a specific shopper by name (optional)
- --dry-run: Do not call the API; emit mocked results (flag)

Requirements:
- Python 3.9+
- pip install openai pandas python-dotenv (optional for local env)

Notes:
- The shoppers.csv must include multiple prompt rows per shopper to represent likely ways a shopper might query ChatGPT.
- Shopper "memory" should include only background info the model could reasonably know (general bio/history), not hidden preferences.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, try to load .env manually
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    os.environ[key] = value

# Lazy import so the script can run --dry-run without openai installed
OpenAI = None  # type: ignore


DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "output"
SHOPPERS_CSV = DATA_DIR / "shoppers.csv"
PDP_DIR = Path("pdp_pages") / "active"


@dataclass
class ShopperPrompt:
    name: str
    memory: str
    prompt: str


def load_shoppers(csv_path: Path, filter_name: Optional[str] = None) -> List[ShopperPrompt]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing shoppers CSV at {csv_path}")
    shoppers: List[ShopperPrompt] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"name", "memory", "prompt"}
        if not required.issubset({(k or "").strip().lower() for k in reader.fieldnames or []}):
            raise ValueError(f"shoppers.csv must have columns: name,memory,prompt. Got: {reader.fieldnames}")
        for row in reader:
            name = row.get("name", "").strip()
            memory = row.get("memory", "").strip()
            prompt = row.get("prompt", "").strip()
            if not name or not prompt:
                continue
            if filter_name and name.lower() != filter_name.lower():
                continue
            shoppers.append(ShopperPrompt(name=name, memory=memory, prompt=prompt))
    if not shoppers:
        raise ValueError("No shopper prompts loaded after filtering.")
    return shoppers


def discover_pdp_files(pdp_dir: Path) -> List[Path]:
    if not pdp_dir.exists():
        return []
    # Include only .html files
    files = sorted([p for p in pdp_dir.iterdir() if p.is_file() and p.suffix.lower() == ".html"])
    return files


def map_local_to_urls(files: List[Path], base_url: str) -> List[str]:
    base = base_url.rstrip("/") + "/"
    urls = []
    for p in files:
        # Files are expected to be deployed at the repo root on GitHub Pages; mirror relative path under base URL
        rel = p.as_posix()
        urls.append(base + rel)
    return urls


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PDP_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def build_prompt(shopper: ShopperPrompt, pdp_urls: List[str]) -> str:
    # Construct a constrained instruction to produce JSON
    pdp_lines = "\n".join(f"- {u}" for u in pdp_urls)
    return (
        "You are assisting an online shopper to evaluate product detail pages (PDPs) for an Oxford shirt.\n"
        "Context about the shopper (background the model could reasonably know):\n"
        f"{shopper.memory}\n\n"
        "Task:\n"
        f"{shopper.prompt}\n\n"
        "Available PDPs (URLs):\n"
        f"{pdp_lines}\n\n"
        "Instructions:\n"
        "- Examine only the information that would be visible by visiting those URLs.\n"
        "- Choose exactly one best PDP URL for the shopper's needs.\n"
        "- Provide a short justification.\n"
        "- Provide a list of features and a numeric weight for each (0-1) indicating importance for your decision.\n"
        "- Respond ONLY as a minified JSON object with keys: chosen_url, justification, features.\n"
        "- features must be an array of objects with keys: name, weight, note.\n"
        "Example JSON: {\"chosen_url\":\"https://example.com/p1\",\"justification\":\"...\",\"features\":[{\"name\":\"fabric\",\"weight\":0.35,\"note\":\"...\"},{\"name\":\"fit\",\"weight\":0.25,\"note\":\"...\"}]}\n"
    )


def parse_model_json(s: str) -> Dict[str, Any]:
    """Parse JSON from model response with debug logging."""
    print(f"DEBUG: parse_model_json - Input string length: {len(s)}")
    print(f"DEBUG: parse_model_json - First 200 chars: {repr(s[:200])}")
    
    # Try to extract JSON even if model adds text
    # Use a simpler regex pattern that works with Python's re module
    try:
        # First try: look for balanced braces using a simpler approach
        candidates = []
        brace_count = 0
        start_pos = -1
        
        for i, char in enumerate(s):
            if char == '{':
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    candidates.append(s[start_pos:i + 1])
        
        print(f"DEBUG: parse_model_json - Found {len(candidates)} JSON candidates")
        
        # If no balanced braces found, fallback: naive first/last brace
        if not candidates:
            print("DEBUG: parse_model_json - No balanced braces found, trying fallback")
            first = s.find("{")
            last = s.rfind("}")
            if first != -1 and last != -1 and last > first:
                candidates = [s[first:last + 1]]
                print(f"DEBUG: parse_model_json - Fallback candidate: {repr(candidates[0][:100])}")
        
        best: Optional[Dict[str, Any]] = None
        for i, c in enumerate(candidates[::-1]):
            print(f"DEBUG: parse_model_json - Trying candidate {i}: {repr(c[:100])}")
            try:
                obj = json.loads(c)
                if isinstance(obj, dict):
                    best = obj
                    print(f"DEBUG: parse_model_json - Successfully parsed JSON with keys: {list(obj.keys())}")
                    break
            except Exception as e:
                print(f"DEBUG: parse_model_json - Failed to parse candidate {i}: {e}")
                continue
        
        if best is None:
            print("DEBUG: parse_model_json - No valid JSON found in any candidate")
            raise ValueError("Failed to parse JSON from model response.")
        
        return best
        
    except Exception as e:
        print(f"DEBUG: parse_model_json - Exception in parsing: {e}")
        raise


def init_openai():
    global OpenAI
    from openai import OpenAI as _OpenAI  # local import
    OpenAI = _OpenAI  # type: ignore


def call_model(
    api_key: str,
    base_url: Optional[str],
    model: str,
    prompt: str,
    temperature: float,
    timeout: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    print(f"DEBUG: call_model - Starting with dry_run={dry_run}, model={model}")
    
    if dry_run:
        print("DEBUG: call_model - Running in dry-run mode")
        # produce a small mocked plausible JSON
        # Randomly choose a URL from list within prompt
        urls = re.findall(r"- (https?://\S+)", prompt)
        print(f"DEBUG: call_model - Found {len(urls)} URLs in prompt")
        chosen = random.choice(urls) if urls else (urls[0] if urls else "UNKNOWN")
        print(f"DEBUG: call_model - Chosen URL: {chosen}")
        features = [
            {"name": "fabric", "weight": round(random.uniform(0.2, 0.5), 2), "note": "Oxford cotton noted on PDP."},
            {"name": "fit", "weight": round(random.uniform(0.1, 0.4), 2), "note": "Size chart and fit guidance present."},
            {"name": "price", "weight": round(random.uniform(0.05, 0.3), 2), "note": "Competitive pricing displayed."},
        ]
        total = sum(f["weight"] for f in features)
        if total > 0:
            for f in features:
                f["weight"] = round(f["weight"] / total, 2)
        result = {
            "chosen_url": chosen,
            "justification": "Based on fabric quality, fit details, and value.",
            "features": features,
        }
        print(f"DEBUG: call_model - Returning dry-run result: {result}")
        return result

    print("DEBUG: call_model - Making actual API call")
    if OpenAI is None:
        print("DEBUG: call_model - Initializing OpenAI client")
        init_openai()

    try:
        client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=timeout)
        print(f"DEBUG: call_model - Created OpenAI client with base_url={base_url}")
        
        # Use Responses API for structured output tendency
        system_msg = (
            "You are a precise evaluator that outputs valid minified JSON only. "
            "Do not include markdown code fences or additional commentary."
        )
        
        print(f"DEBUG: call_model - Sending request to model {model}")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        
        text = response.choices[0].message.content or ""
        print(f"DEBUG: call_model - Received response length: {len(text)}")
        print(f"DEBUG: call_model - Raw response: {repr(text[:200])}")
        
        parsed = parse_model_json(text)
        print(f"DEBUG: call_model - Successfully parsed JSON")
        return parsed
        
    except Exception as e:
        print(f"DEBUG: call_model - Exception occurred: {type(e).__name__}: {e}")
        raise


def aggregate_results(raw_records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # summary: win counts per PDP
    win_counts: Dict[str, int] = {}
    # feature importance: average weights per PDP and feature
    feature_accum: Dict[Tuple[str, str], List[float]] = {}

    for rec in raw_records:
        chosen = rec.get("result", {}).get("chosen_url")
        if chosen:
            win_counts[chosen] = win_counts.get(chosen, 0) + 1
        features = rec.get("result", {}).get("features") or []
        for f in features:
            name = str(f.get("name", "")).strip().lower()
            w = coerce_float(f.get("weight"), 0.0)
            if chosen and name:
                feature_accum.setdefault((chosen, name), []).append(w)

    summary_rows = [{"pdp_url": url, "wins": wins} for url, wins in sorted(win_counts.items(), key=lambda x: -x[1])]
    feat_rows: List[Dict[str, Any]] = []
    for (url, feat), weights in sorted(feature_accum.items()):
        avg = sum(weights) / len(weights) if weights else 0.0
        feat_rows.append({"pdp_url": url, "feature": feat, "avg_weight": round(avg, 4), "n": len(weights)})

    return summary_rows, feat_rows


def write_jsonl(path: Path, records: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv_dicts(path: Path, rows: List[Dict[str, Any]], field_order: Optional[List[str]] = None):
    if not rows:
        # Create empty file with no rows but leave header unknown
        path.write_text("", encoding="utf-8")
        return
    fields = field_order or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def run_simulations(
    base_url: str,
    runs: int,
    model: str,
    temperature: float,
    timeout: int,
    shuffle: bool,
    max_pdp: Optional[int],
    shopper_name: Optional[str],
    dry_run: bool,
) -> Dict[str, Path]:
    print(f"DEBUG: run_simulations - Starting with {runs} runs, dry_run={dry_run}")
    
    ensure_dirs()
    print("DEBUG: run_simulations - Directories ensured")
    
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"DEBUG: run_simulations - API key present: {bool(api_key)}")
    if not api_key and not dry_run:
        raise EnvironmentError("OPENAI_API_KEY not set. Set it or use --dry-run for a no-API test.")

    base_api_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    print(f"DEBUG: run_simulations - Base API URL: {base_api_url}")

    # Discover PDP pages and map to URLs
    print(f"DEBUG: run_simulations - Discovering PDP files in {PDP_DIR}")
    files = discover_pdp_files(PDP_DIR)
    print(f"DEBUG: run_simulations - Found {len(files)} PDP files: {[f.name for f in files]}")
    if not files:
        print(f"Warning: No PDP files found in {PDP_DIR}.", file=sys.stderr)
    
    pdp_urls = map_local_to_urls(files, base_url)
    print(f"DEBUG: run_simulations - Mapped to {len(pdp_urls)} URLs")
    for i, url in enumerate(pdp_urls):
        print(f"DEBUG: run_simulations - URL {i+1}: {url}")
    
    if not pdp_urls:
        raise ValueError("No PDP URLs available to evaluate.")

    # Load shoppers
    print(f"DEBUG: run_simulations - Loading shoppers from {SHOPPERS_CSV}")
    shoppers = load_shoppers(SHOPPERS_CSV, filter_name=shopper_name)
    print(f"DEBUG: run_simulations - Loaded {len(shoppers)} shoppers")

    # Prepare outputs
    raw_path = OUTPUT_DIR / "raw_runs.jsonl"
    summary_path = OUTPUT_DIR / "summary.csv"
    features_path = OUTPUT_DIR / "feature_importance.csv"
    log_path = OUTPUT_DIR / "log.csv"
    print(f"DEBUG: run_simulations - Output paths prepared")

    raw_records: List[Dict[str, Any]] = []
    start_ts = int(time.time())

    for i in range(1, runs + 1):
        print(f"\nDEBUG: run_simulations - Starting run {i}/{runs}")
        
        shopper = random.choice(shoppers)
        print(f"DEBUG: run_simulations - Selected shopper: {shopper.name}")
        
        sample_urls = list(pdp_urls)
        if shuffle:
            random.shuffle(sample_urls)
            print("DEBUG: run_simulations - URLs shuffled")
        if max_pdp is not None and max_pdp > 0:
            sample_urls = sample_urls[:max_pdp]
            print(f"DEBUG: run_simulations - Limited to {len(sample_urls)} URLs")
        
        prompt = build_prompt(shopper, sample_urls)
        print(f"DEBUG: run_simulations - Built prompt length: {len(prompt)}")

        try:
            print("DEBUG: run_simulations - Calling model...")
            result = call_model(
                api_key=api_key or "",
                base_url=base_api_url,
                model=model,
                prompt=prompt,
                temperature=temperature,
                timeout=timeout,
                dry_run=dry_run,
            )
            print(f"DEBUG: run_simulations - Model call successful")
            
            # Basic validation
            chosen = result.get("chosen_url")
            print(f"DEBUG: run_simulations - Chosen URL: {chosen}")
            if chosen not in sample_urls:
                print("DEBUG: run_simulations - Chosen URL not in candidates, attempting coercion")
                # attempt to coerce by loose match
                matches = [u for u in sample_urls if chosen and chosen.strip().lower() in u.lower()]
                if matches:
                    print(f"DEBUG: run_simulations - Found match: {matches[0]}")
                    result["chosen_url"] = matches[0]
                else:
                    print("DEBUG: run_simulations - No matches found for coercion")
        except Exception as e:
            print(f"DEBUG: run_simulations - Exception in run {i}: {type(e).__name__}: {e}")
            result = {"error": str(e)}
            
        rec = {
            "run_index": i,
            "timestamp": int(time.time()),
            "shopper_name": shopper.name,
            "shopper_memory": shopper.memory,
            "shopper_prompt": shopper.prompt,
            "pdp_candidates": sample_urls,
            "model": model,
            "temperature": temperature,
            "result": result,
        }
        raw_records.append(rec)
        print(f"DEBUG: run_simulations - Run {i} completed")

    print(f"\nDEBUG: run_simulations - All runs completed, writing outputs...")
    
    # Write raw
    write_jsonl(raw_path, raw_records)
    print(f"DEBUG: run_simulations - Raw data written to {raw_path}")

    # Aggregate
    summary_rows, feat_rows = aggregate_results(raw_records)
    write_csv_dicts(summary_path, summary_rows, field_order=["pdp_url", "wins"])
    write_csv_dicts(features_path, feat_rows, field_order=["pdp_url", "feature", "avg_weight", "n"])
    print(f"DEBUG: run_simulations - Summary and features written")

    # Log
    log_rows = [{
        "start_ts": start_ts,
        "end_ts": int(time.time()),
        "runs": runs,
        "n_pdp": len(pdp_urls),
        "n_shopper_rows": len(shoppers),
        "model": model,
        "temperature": temperature,
        "base_url": base_url,
        "shuffle": shuffle,
        "max_pdp": max_pdp if max_pdp is not None else "",
        "dry_run": dry_run,
    }]
    write_csv_dicts(log_path, log_rows)
    print(f"DEBUG: run_simulations - Log written")

    return {
        "raw": raw_path,
        "summary": summary_path,
        "features": features_path,
        "log": log_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Simulate PDP evaluations with OpenAI.")
    parser.add_argument("--base-url", required=True, help="GitHub Pages base URL, e.g., https://digitalfuelcapital.github.io/dfc-public-pdp-test-pages/")
    parser.add_argument("--runs", type=int, default=100, help="Number of simulations to run")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name (default: gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle PDP list per run")
    parser.add_argument("--max-pdp", type=int, default=None, help="Limit number of PDPs sampled per run")
    parser.add_argument("--shopper-name", default=None, help="Filter to a specific shopper name")
    parser.add_argument("--dry-run", action="store_true", help="Do not call the API; generate mocked results")
    args = parser.parse_args()

    outputs = run_simulations(
        base_url=args.base_url,
        runs=args.runs,
        model=args.model,
        temperature=args.temperature,
        timeout=args.timeout,
        shuffle=args.shuffle,
        max_pdp=args.max_pdp,
        shopper_name=args.shopper_name,
        dry_run=args.dry_run,
    )
    print("Wrote outputs:")
    for k, v in outputs.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()