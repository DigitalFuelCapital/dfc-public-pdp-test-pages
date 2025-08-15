#!/usr/bin/env python3
"""
Execute multiple PDP evaluation tests with different PDP sets.

Usage:
  python scripts/execute_multiple_tests.py --base-url https://digitalfuelcapital.github.io/dfc-public-pdp-test-pages/ --config pdp_pages/multi_test_config.json

Summary:
- Loads test configuration from multi_test_config.json which defines multiple test scenarios
- Each test has a test_name and list of active_pdps to evaluate
- Runs the simulate_pdp_eval functionality for each test
- All outputs go to the same raw_runs.jsonl file with test_name field added
- Aggregated outputs (summary.csv, feature_importance.csv) include all tests

Configuration format:
{
  "tests": [
    {
      "test_name": "baseline_test",
      "description": "Test baseline PDPs without UVP variants",
      "active_pdps": ["pdp_pages/active/pdp_oxford_alpine_and_oak.html", ...]
    }
  ],
  "default_settings": {
    "runs": 100,
    "model": "gpt-4o",
    ...
  }
}

Environment variables:
- OPENAI_API_KEY (required)
- OPENAI_BASE_URL (optional, defaults to https://api.openai.com/v1)

CLI options:
- --config: Path to multi-test configuration file (default: pdp_pages/multi_test_config.json)
- --base-url: Required base URL where GitHub Pages hosts the PDPs
- --runs: Number of simulations per test (overrides config default)
- --model: Model name (overrides config default)
- --timeout: HTTP timeout seconds per call (overrides config default)
- --shuffle: Shuffle PDP list before each run (overrides config default)
- --max-pdp: Limit number of PDPs sampled per run (overrides config default)
- --shopper-name: Filter to a specific shopper by name (optional)
- --threads: Number of parallel threads for API calls (overrides config default)
- --dry-run: Do not call the API; emit mocked results (flag)
- --test-name: Run only a specific test by name (optional)

Requirements:
- Python 3.9+
- pip install openai pandas python-dotenv

Notes:
- All results are written to the same output files with test_name field distinguishing runs
- Tests run sequentially, not in parallel
- Each test uses the same shoppers.csv file
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import the simulation functions from the existing script
from simulate_pdp_eval import (
    run_simulations,
    ensure_dirs,
    OUTPUT_DIR,
    append_csv_dicts,
)

def load_multi_test_config(config_path: Path) -> Dict[str, Any]:
    """
    Load multi-test configuration from JSON file.
    
    Expected format:
    {
        "tests": [
            {
                "test_name": "test1",
                "description": "...",
                "active_pdps": ["path1.html", "path2.html", ...]
            }
        ],
        "default_settings": {
            "runs": 100,
            "model": "gpt-4o",
            ...
        }
    }
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Multi-test config file not found at {config_path}")
    
    try:
        with config_path.open('r', encoding='utf-8') as f:
            config = json.load(f)
        
        if "tests" not in config:
            raise ValueError("Config must contain 'tests' array")
        
        tests = config["tests"]
        if not isinstance(tests, list) or not tests:
            raise ValueError("Config 'tests' must be a non-empty list")
        
        # Validate each test
        for i, test in enumerate(tests):
            if not isinstance(test, dict):
                raise ValueError(f"Test {i} must be an object")
            
            if "test_name" not in test:
                raise ValueError(f"Test {i} missing 'test_name'")
            
            if "active_pdps" not in test:
                raise ValueError(f"Test {i} missing 'active_pdps'")
            
            if not isinstance(test["active_pdps"], list):
                raise ValueError(f"Test {i} 'active_pdps' must be a list")
        
        default_settings = config.get("default_settings", {})
        if not isinstance(default_settings, dict):
            raise ValueError("Config 'default_settings' must be an object")
        
        return config
        
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Invalid multi-test config file: {e}")


def validate_pdp_files(pdp_paths: List[str]) -> List[str]:
    """
    Validate that PDP files exist and filter to valid ones.
    Returns list of valid file paths.
    """
    valid_files = []
    for file_path in pdp_paths:
        full_path = Path(file_path)
        if full_path.exists() and full_path.suffix.lower() == ".html":
            valid_files.append(file_path)
        else:
            print(f"Warning: PDP file not found or not HTML: {file_path}")
    
    return valid_files


def execute_multiple_tests(
    config_path: Path,
    base_url: str,
    runs: Optional[int] = None,
    model: Optional[str] = None,
    timeout: Optional[int] = None,
    shuffle: Optional[bool] = None,
    max_pdp: Optional[int] = None,
    shopper_name: Optional[str] = None,
    threads: Optional[int] = None,
    dry_run: bool = False,
    test_name_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute multiple tests based on configuration file.
    
    Returns summary of all tests executed.
    """
    print(f"DEBUG: execute_multiple_tests - Loading config from {config_path}")
    config = load_multi_test_config(config_path)
    
    tests = config["tests"]
    default_settings = config.get("default_settings", {})
    
    # Apply CLI overrides to default settings
    effective_settings = {
        "runs": runs if runs is not None else default_settings.get("runs", 30),
        "model": model if model is not None else default_settings.get("model", "gpt-5-mini"),
        "timeout": timeout if timeout is not None else default_settings.get("timeout", 60),
        "shuffle": shuffle if shuffle is not None else default_settings.get("shuffle", False),
        "max_pdp": max_pdp if max_pdp is not None else default_settings.get("max_pdp", None),
        "threads": threads if threads is not None else default_settings.get("threads", 10),
    }
    
    print(f"DEBUG: execute_multiple_tests - Effective settings: {effective_settings}")
    
    # Filter tests if specific test requested
    if test_name_filter:
        tests = [t for t in tests if t["test_name"] == test_name_filter]
        if not tests:
            raise ValueError(f"Test '{test_name_filter}' not found in configuration")
        print(f"DEBUG: execute_multiple_tests - Filtered to test: {test_name_filter}")
    
    print(f"DEBUG: execute_multiple_tests - Will execute {len(tests)} tests")
    
    # Ensure output directories exist
    ensure_dirs()
    
    # Track summary across all tests
    all_results = []
    
    for i, test in enumerate(tests):
        test_name = test["test_name"]
        description = test.get("description", "")
        active_pdps = test["active_pdps"]
        
        print(f"\n{'='*60}")
        print(f"EXECUTING TEST {i+1}/{len(tests)}: {test_name}")
        print(f"Description: {description}")
        print(f"PDPs: {len(active_pdps)} files")
        print(f"{'='*60}")
        
        # Validate PDP files
        valid_pdps = validate_pdp_files(active_pdps)
        if not valid_pdps:
            print(f"ERROR: No valid PDP files found for test '{test_name}'. Skipping.")
            continue
        
        if len(valid_pdps) != len(active_pdps):
            print(f"WARNING: Only {len(valid_pdps)}/{len(active_pdps)} PDP files are valid for test '{test_name}'")
        
        try:
            # Run simulations for this test
            outputs = run_simulations(
                base_url=base_url,
                runs=effective_settings["runs"],
                model=effective_settings["model"],
                timeout=effective_settings["timeout"],
                shuffle=effective_settings["shuffle"],
                max_pdp=effective_settings["max_pdp"],
                shopper_name=shopper_name,
                dry_run=dry_run,
                threads=effective_settings["threads"],
                test_name=test_name,
                pdp_file_paths=valid_pdps,
            )
            
            result = {
                "test_name": test_name,
                "status": "completed",
                "pdp_count": len(valid_pdps),
                "runs_completed": effective_settings["runs"],
                "outputs": outputs,
            }
            
            print(f"✅ Test '{test_name}' completed successfully")
            
        except Exception as e:
            print(f"❌ Test '{test_name}' failed: {type(e).__name__}: {e}")
            result = {
                "test_name": test_name,
                "status": "failed",
                "error": str(e),
                "pdp_count": len(valid_pdps),
                "runs_completed": 0,
            }
        
        all_results.append(result)
    
    # Summary
    completed_tests = [r for r in all_results if r["status"] == "completed"]
    failed_tests = [r for r in all_results if r["status"] == "failed"]
    
    print(f"\n{'='*60}")
    print(f"EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {len(tests)}")
    print(f"Completed: {len(completed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if completed_tests:
        total_runs = sum(r["runs_completed"] for r in completed_tests)
        print(f"Total simulation runs: {total_runs}")
        print(f"\nOutput files (aggregated across all tests):")
        if completed_tests:
            # Use outputs from first completed test as they're all the same files
            for output_type, path in completed_tests[0]["outputs"].items():
                print(f"- {output_type}: {path}")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for result in failed_tests:
            print(f"- {result['test_name']}: {result['error']}")
    
    return {
        "total_tests": len(tests),
        "completed": len(completed_tests),
        "failed": len(failed_tests),
        "results": all_results,
        "settings": effective_settings,
    }


def main():
    parser = argparse.ArgumentParser(description="Execute multiple PDP evaluation tests.")
    parser.add_argument("--config", default="pdp_pages/multi_test_config.json", help="Path to multi-test configuration file")
    parser.add_argument("--base-url", default="https://digitalfuelcapital.github.io/dfc-public-pdp-test-pages/", help="GitHub Pages base URL, e.g., https://digitalfuelcapital.github.io/dfc-public-pdp-test-pages/")
    parser.add_argument("--runs", type=int, help="Number of simulations per test (overrides config)")
    parser.add_argument("--model", help="OpenAI model name (overrides config)")
    parser.add_argument("--timeout", type=int, help="HTTP timeout seconds (overrides config)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle PDP list per run (overrides config)")
    parser.add_argument("--max-pdp", type=int, help="Limit number of PDPs sampled per run (overrides config)")
    parser.add_argument("--shopper-name", help="Filter to a specific shopper name")
    parser.add_argument("--threads", type=int, help="Number of parallel threads for API calls (overrides config)")
    parser.add_argument("--dry-run", action="store_true", help="Do not call the API; generate mocked results")
    parser.add_argument("--test-name", help="Run only a specific test by name")
    args = parser.parse_args()

    config_path = Path(args.config)
    
    try:
        summary = execute_multiple_tests(
            config_path=config_path,
            base_url=args.base_url,
            runs=args.runs,
            model=args.model,
            timeout=args.timeout,
            shuffle=args.shuffle if args.shuffle else None,
            max_pdp=args.max_pdp,
            shopper_name=args.shopper_name,
            threads=args.threads,
            dry_run=args.dry_run,
            test_name_filter=args.test_name,
        )
        
        if summary["failed"] > 0:
            sys.exit(1)
            
    except Exception as e:
        print(f"FATAL ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()