"""
Oracle Evaluation Framework for Pandas Mutant Testing.

This framework evaluates whether property-based tests can detect mutants
introduced into pandas functions. It runs baseline and IR-generated tests
against each mutant and reports kill rates.

Usage:
    python evaluate_mutants.py [--function FUNCTION] [--mutant MUTANT_ID] [--runs N]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

BASE_DIR = Path("/Users/xiaofanlu/Documents/github_repos/property_based_test_agent")
ORACLE_GEN_DIR = BASE_DIR / "experiments/oracle_generation/pandas/DataFrame"
ORACLE_EVAL_DIR = BASE_DIR / "experiments/oracle_evaluation"
PANDAS_BUG_FINDING_DIR = BASE_DIR / "pandas_bug_finding"


@dataclass
class MutantResult:
    function: str
    mutant_id: str
    mutant_name: str
    baseline_passed: bool
    baseline_tests_run: int
    baseline_tests_failed: int
    mutant_passed: bool
    mutant_tests_run: int
    mutant_tests_failed: int
    killed: bool
    kill_rate: float
    execution_time: float
    winner: str
    failing_tests: list


@dataclass
class EvaluationResult:
    function: str
    timestamp: str
    total_mutants: int
    killed_mutants: int
    survival_rate: float
    kill_rate: float
    mutant_results: list


class MutantEvaluator:
    def __init__(self, function_dir: Path, function_name: str):
        self.function_dir = function_dir
        self.function_name = function_name
        self.baseline_test = function_dir / "baseline_test.py"
        self.ir_test = function_dir / "ir_generated_test.py"
        self.mutant_wrapper = function_dir / "mutant_wrapper.py"
        self.results = []

    def run_tests(self, test_file: Path, env_vars: dict = None) -> tuple:
        """Run tests and return (passed, failed, output)."""
        if not test_file.exists():
            return (True, 0, 0, f"Test file not found: {test_file}")

        cmd = ["uv", "run", "pytest", str(test_file), "-v", "--tb=short"]

        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(ORACLE_GEN_DIR.parent.parent),
                capture_output=True,
                text=True,
                env=env,
                timeout=300,
            )

            passed = "passed" in result.stdout.lower()
            failed = "failed" in result.stdout.lower()

            lines = result.stdout.split("\n")
            for line in lines:
                if " passed" in line.lower() and " failed" in line.lower():
                    parts = line.lower().split()
                    for i, p in enumerate(parts):
                        if "failed" in p and i > 0:
                            try:
                                failed_count = int(parts[i - 1])
                            except:
                                failed_count = 0
                        if "passed" in p and i > 0:
                            try:
                                passed_count = int(parts[i - 1])
                            except:
                                passed_count = 0
                    break
            else:
                if result.returncode == 0:
                    passed_count = 1
                    failed_count = 0
                else:
                    passed_count = 0
                    failed_count = 1

            test_passed = result.returncode == 0 and failed_count == 0

            return (
                test_passed,
                passed_count,
                failed_count,
                result.stdout + result.stderr,
            )

        except subprocess.TimeoutExpired:
            return (False, 0, 0, "Test timed out after 300 seconds")
        except Exception as e:
            return (False, 0, 0, f"Error running tests: {str(e)}")

    def load_mutant_info(self) -> dict:
        """Load mutant info from the wrapper file."""
        if not self.mutant_wrapper.exists():
            return {}

        mutant_info = {}
        with open(self.mutant_wrapper) as f:
            content = f.read()
            if "MUTANT_INFO" in content:
                try:
                    start_marker = "MUTANT_INFO = {"
                    start_idx = content.find(start_marker)
                    if start_idx != -1:
                        brace_start = start_idx + len(start_marker) - 1
                        brace_count = 0
                        end_idx = brace_start
                        for i, c in enumerate(content[brace_start:]):
                            if c == "{":
                                brace_count += 1
                            elif c == "}":
                                brace_count -= 1
                            if brace_count == 0:
                                end_idx = brace_start + i + 1
                                break
                        mutant_str = content[start_idx:end_idx]
                        local_ns = {}
                        exec(mutant_str, {}, local_ns)
                        mutant_info = local_ns.get("MUTANT_INFO", {})
                except Exception:
                    pass
        return mutant_info

    def evaluate_mutant(self, mutant_id: str, runs: int = 3) -> MutantResult:
        """Evaluate a single mutant against both baseline and IR tests."""
        print(f"\n  Evaluating {self.function_name} with mutant {mutant_id}...")

        mutant_info = self.load_mutant_info()
        mutant_name = mutant_info.get(mutant_id, {}).get("name", "unknown")

        baseline_passed = True
        baseline_run = 0
        baseline_failed = 0
        mutant_passed = True
        mutant_run = 0
        mutant_failed = 0
        all_output = []

        start_time = time.time()

        for run in range(runs):
            print(f"    Run {run + 1}/{runs}")

            env_no_mutant = {}
            env_with_mutant = {"MUTANT_ID": mutant_id}

            baseline_result = self.run_tests(self.baseline_test, env_no_mutant)
            if run == 0:
                baseline_passed = baseline_result[0]
                baseline_run = baseline_result[1]
                baseline_failed = baseline_result[2]

            mutant_result = self.run_tests(self.baseline_test, env_with_mutant)
            if run == 0:
                mutant_passed = mutant_result[0]
                mutant_run = mutant_result[1]
                mutant_failed = mutant_result[2]

            all_output.append(
                {"run": run + 1, "baseline": baseline_result, "mutant": mutant_result}
            )

        execution_time = time.time() - start_time

        killed = not mutant_passed and baseline_passed

        if killed:
            winner = "test"
            kill_rate = 1.0
        elif mutant_passed and baseline_passed:
            winner = "mutant"
            kill_rate = 0.0
        else:
            winner = "invalid"
            kill_rate = 0.0

        failing_tests = []
        if killed:
            for output in all_output:
                if not output["mutant"][0]:
                    failing_tests.append(output["mutant"][3])

        return MutantResult(
            function=self.function_name,
            mutant_id=mutant_id,
            mutant_name=mutant_name,
            baseline_passed=baseline_passed,
            baseline_tests_run=baseline_run,
            baseline_tests_failed=baseline_failed,
            mutant_passed=mutant_passed,
            mutant_tests_run=mutant_run,
            mutant_tests_failed=mutant_failed,
            killed=killed,
            kill_rate=kill_rate,
            execution_time=execution_time,
            winner=winner,
            failing_tests=failing_tests[:3],
        )

    def evaluate_all_mutants(self, runs: int = 3) -> EvaluationResult:
        """Evaluate all mutants for this function."""
        print(f"\n{'=' * 60}")
        print(f"Evaluating function: {self.function_name}")
        print(f"{'=' * 60}")

        if not self.mutant_wrapper.exists():
            print(f"  No mutant wrapper found for {self.function_name}")
            return None

        mutant_info = self.load_mutant_info()
        mutant_ids = list(mutant_info.keys())

        all_results = []
        killed_count = 0

        for mutant_id in mutant_ids:
            result = self.evaluate_mutant(mutant_id, runs)
            all_results.append(asdict(result))
            if result.killed:
                killed_count += 1

        total_mutants = len(mutant_ids)
        survival_rate = (
            (total_mutants - killed_count) / total_mutants if total_mutants > 0 else 0
        )
        kill_rate = killed_count / total_mutants if total_mutants > 0 else 0

        eval_result = EvaluationResult(
            function=self.function_name,
            timestamp=datetime.now().isoformat(),
            total_mutants=total_mutants,
            killed_mutants=killed_count,
            survival_rate=survival_rate,
            kill_rate=kill_rate,
            mutant_results=all_results,
        )

        self.results = all_results
        return eval_result

    def print_results(self, result: EvaluationResult):
        """Print evaluation results."""
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {result.function}")
        print(f"{'=' * 60}")
        print(f"Total Mutants: {result.total_mutants}")
        print(f"Killed: {result.killed_mutants}")
        print(f"Survived: {result.total_mutants - result.killed_mutants}")
        print(f"Kill Rate: {result.kill_rate:.2%}")
        print()

        print(f"{'Mutant ID':<10} {'Mutant Name':<30} {'Killed':<10} {'Winner':<10}")
        print("-" * 60)

        for mr in result.mutant_results:
            status = "KILLED" if mr["killed"] else "SURVIVED"
            print(
                f"{mr['mutant_id']:<10} {mr['mutant_name']:<30} {status:<10} {mr['winner']:<10}"
            )

        print()

        killed_mutants = [mr for mr in result.mutant_results if mr["killed"]]
        survived_mutants = [mr for mr in result.mutant_results if not mr["killed"]]

        if killed_mutants:
            print("KILLED MUTANTS:")
            for mr in killed_mutants:
                print(f"  - {mr['mutant_id']} ({mr['mutant_name']})")
                if mr["failing_tests"]:
                    print(f"    Failing test output (first 500 chars):")
                    for ft in mr["failing_tests"]:
                        print(f"    {ft[:500]}")

        if survived_mutants:
            print("\nSURVIVED MUTANTS (tests did not detect):")
            for mr in survived_mutants:
                print(f"  - {mr['mutant_id']} ({mr['mutant_name']})")

        winner = "test" if result.kill_rate >= 0.5 else "mutant"
        print(f"\nWINNER: {winner.upper()}")
        if winner == "test":
            print("  The tests successfully detected most mutants.")
        else:
            print("  Most mutants survived - tests need improvement.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate mutant kill rates for pandas functions"
    )
    parser.add_argument(
        "--function",
        "-f",
        type=str,
        default=None,
        help="Function to evaluate (groupby, reindex, to_json)",
    )
    parser.add_argument(
        "--mutant", "-m", type=str, default=None, help="Specific mutant ID to evaluate"
    )
    parser.add_argument(
        "--runs", "-r", type=int, default=3, help="Number of evaluation runs per mutant"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output JSON file for results"
    )

    args = parser.parse_args()

    functions = ["groupby", "reindex", "to_json"]
    if args.function:
        if args.function not in functions:
            print(f"Unknown function: {args.function}")
            print(f"Available: {', '.join(functions)}")
            sys.exit(1)
        functions = [args.function]

    all_results = []

    for func_name in functions:
        func_dir = ORACLE_GEN_DIR / func_name
        if not func_dir.exists():
            print(f"Function directory not found: {func_dir}")
            continue

        evaluator = MutantEvaluator(func_dir, func_name)
        result = evaluator.evaluate_all_mutants(args.runs)

        if result:
            evaluator.print_results(result)
            all_results.append(asdict(result))

    if args.output and all_results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)

    total_mutants = sum(r["total_mutants"] for r in all_results)
    total_killed = sum(r["killed_mutants"] for r in all_results)
    overall_kill_rate = 0.0

    if total_mutants > 0:
        overall_kill_rate = total_killed / total_mutants
        print(f"Total Mutants: {total_mutants}")
        print(f"Total Killed: {total_killed}")
        print(f"Overall Kill Rate: {overall_kill_rate:.2%}")

        winner = "tests" if overall_kill_rate >= 0.5 else "mutants"
        print(f"\nOverall Winner: {winner.upper()}")

    return 0 if overall_kill_rate >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
