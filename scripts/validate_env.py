"""Validate that all required packages and environment settings are present.

Usage
-----
    python scripts/validate_env.py

Exit code 0 means all checks passed; non-zero means at least one failed.
Run this after ``uv sync`` to confirm a clean install on a new machine.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Outcome of a single environment check."""

    name: str
    passed: bool
    message: str = ""


@dataclass
class Report:
    """Aggregated results from all checks."""

    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    def print_summary(self) -> None:
        if sys.platform == "win32":
            ok, fail = "[OK]", "[FAIL]"
        else:
            ok, fail = "\033[32m✓\033[0m", "\033[31m✗\033[0m"
        for r in self.results:
            symbol = ok if r.passed else fail
            detail = f"  {r.message}" if r.message else ""
            print(f"  {symbol}  {r.name}{detail}")

        total = len(self.results)
        passed = total - len(self.failed)
        print(f"\n  {passed}/{total} checks passed.")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_python_version(report: Report) -> None:
    """Require Python >= 3.11."""
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 11)
    report.add(
        CheckResult(
            name="Python version",
            passed=ok,
            message=f"found {major}.{minor}" + ("" if ok else " — need >=3.11"),
        )
    )


def _import_check(
    report: Report,
    package: str,
    min_version: str | None = None,
    attr: str = "__version__",
    optional: bool = False,
) -> None:
    """Try importing a package and optionally compare its version."""
    try:
        mod = importlib.import_module(package)
        version = getattr(mod, attr, "?")
        if min_version:
            from packaging.version import Version  # noqa: PLC0415

            ok = Version(str(version)) >= Version(min_version)
            msg = f"{version}" + ("" if ok else f" - need >={min_version}")
        else:
            ok, msg = True, str(version)
        report.add(CheckResult(name=f"import {package}", passed=ok, message=msg))
    except ImportError as exc:
        if optional:
            report.add(
                CheckResult(
                    name=f"import {package} (optional)",
                    passed=True,
                    message="not installed (optional nlp feature)",
                )
            )
        else:
            report.add(
                CheckResult(name=f"import {package}", passed=False, message=str(exc))
            )


def check_required_packages(report: Report) -> None:
    """Verify all pipeline dependencies are importable."""
    packages: list[tuple[str, str | None, bool]] = [
        ("pandas", "2.2", False),
        ("numpy", "1.26", False),
        ("scipy", "1.13", False),
        ("sklearn", "1.5", False),
        ("torch", "2.2", False),
        ("mlflow", "2.18", False),
        ("dvc", "3.56", False),
        ("httpx", "0.27", False),
        ("pydantic_settings", "2.6", False),
        ("pyarrow", "18.0", False),
        ("bertopic", "0.16", True),
        ("sentence_transformers", "3.0", True),
        ("yaml", None, False),  # pyyaml
    ]
    for pkg, min_ver, is_opt in packages:
        _import_check(report, pkg, min_ver, optional=is_opt)


def check_cuda(report: Report) -> None:
    """Report CUDA availability (optional — does not fail validation)."""
    try:
        import torch  # noqa: PLC0415

        available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if available else "N/A"
        report.add(
            CheckResult(
                name="CUDA (optional)",
                passed=True,
                message=(
                    "available — " + device_name
                    if available
                    else "not available (CPU mode)"
                ),
            )
        )
    except ImportError:
        report.add(
            CheckResult(
                name="CUDA (optional)",
                passed=True,
                message="torch not installed — skip",
            )
        )


def check_paths(report: Report) -> None:
    """Verify that expected project directories exist."""
    root = Path(__file__).resolve().parents[1]
    dirs = [
        root / "data",
        root / "data" / "raw",
        root / "data" / "processed",
        root / "data" / "raw" / "external_metadata",
        root / "data" / "logs",
        root / "models",
        root / "configs",
        root / "src",
    ]
    for d in dirs:
        report.add(
            CheckResult(
                name=f"path {d.relative_to(root)}",
                passed=d.exists(),
                message="" if d.exists() else "missing — run `mkdir -p`",
            )
        )


def check_env_file(report: Report) -> None:
    """Verify that a .env file (or .env.example) exists."""
    root = Path(__file__).resolve().parents[1]
    env = root / ".env"
    example = root / ".env.example"
    if env.exists():
        report.add(CheckResult(name=".env file", passed=True, message="found"))
    elif example.exists():
        report.add(
            CheckResult(
                name=".env file",
                passed=True,
                message=".env not found but .env.example exists — copy it to .env",
            )
        )
    else:
        report.add(
            CheckResult(
                name=".env file",
                passed=False,
                message="neither .env nor .env.example found",
            )
        )


def check_settings_load(report: Report) -> None:
    """Try loading Pydantic Settings without errors."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "configs"))
        from settings import load_settings  # noqa: PLC0415

        load_settings()
        report.add(CheckResult(name="Settings load", passed=True, message="OK"))
    except Exception as exc:  # noqa: BLE001
        report.add(CheckResult(name="Settings load", passed=False, message=str(exc)))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all_checks() -> Report:
    """Execute every check and return a consolidated report."""
    report = Report()
    checks: list[Callable[[Report], None]] = [
        check_python_version,
        check_required_packages,
        check_cuda,
        check_paths,
        check_env_file,
        check_settings_load,
    ]
    for check in checks:
        check(report)
    return report


def main() -> int:
    """Entry point."""
    print("\n=== validate_env.py - environment sanity check ===\n")
    report = run_all_checks()
    report.print_summary()
    if report.failed:
        print("\nValidation FAILED. Fix the items above and re-run.\n")
        return 1
    print("\nAll checks passed. Environment is ready.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
