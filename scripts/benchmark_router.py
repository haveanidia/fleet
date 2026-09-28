from fleet.benchmark import run_benchmark

if __name__ == "__main__":
    report = run_benchmark(
        "artifacts/fleet-bank",
        output_path="artifacts/fleet-bank/benchmark_results.json",
    )

    for case in report["cases"]:
        print("=" * 100)
        print("EXPECTED:", case["expected"])
        print(case["prompt"])
        print("\nOUTPUT\n", case["output"])
        if case["expected"] is not None:
            print("\nTOP PROMPT ROUTE:", case.get("top_prompt_route"))
            print("SEMANTIC TOP:", case.get("semantic_top"))
        else:
            print("\nPROMPT NULL RATE:", case.get("prompt_null_rate"))
        print("GATING:", case["gating"])
        print("ROUTE OK:", case["route_ok"])

    print("\nPASS:", report["passed"])
    if report["failures"]:
        print("FAILURES:")
        for failure in report["failures"]:
            print("-", failure)

    if not report["passed"]:
        raise SystemExit(1)
