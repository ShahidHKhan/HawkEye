"""
A ~30-line test runner, so these tests need no dependencies at all.

pytest isn't in this project's dependency groups, and adding it would mean
relocking uv.lock -- which `uv sync --frozen` in both workflows would reject if
it ever drifted. So the tests are written as plain `test_*()` functions using
bare `assert`: this runner executes them today, and pytest would collect them
unchanged if it's ever added.
"""
import traceback


def run_tests(namespace: dict) -> int:
    """Run every test_* callable in `namespace`. Returns a process exit code."""
    tests = sorted(
        (name, fn) for name, fn in namespace.items()
        if name.startswith("test_") and callable(fn)
    )
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception:
            failures.append(name)
            print(f"  [FAIL] {name}")
            print("".join("         " + ln for ln in traceback.format_exc().splitlines(keepends=True)))

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print("failed: " + ", ".join(failures))
    return 1 if failures else 0
