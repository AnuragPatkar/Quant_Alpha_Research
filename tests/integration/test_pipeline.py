"""
INTEGRATION TEST: Pipeline Orchestration
========================================
Tests the run_pipeline.py script to ensure it correctly orchestrates
the sequence of sub-scripts based on command-line arguments.

This test mocks the actual execution of sub-scripts (subprocess.run)
to avoid running the full multi-hour pipeline, focusing on the
control logic and argument passing.
"""

import sys
import warnings
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, call

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Import guard — catches ImportError AND transitive ModuleNotFoundError /
# syntax errors that surface as ImportError.
# ---------------------------------------------------------------------------
try:
    # Try importing as a module from scripts package (preferred)
    import scripts.run_pipeline as run_pipeline
except (ImportError, ModuleNotFoundError):
    try:
        # Fallback: try importing directly (since SCRIPTS_DIR is in sys.path)
        import run_pipeline
    except (ImportError, ModuleNotFoundError) as exc:
        warnings.warn(f"Could not import run_pipeline: {exc}")
        run_pipeline = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_script_name(mock_call) -> str:
    """
    Safely extract the script filename from a subprocess.run mock call.

    subprocess.run is expected to be called as:
        subprocess.run([sys.executable, "/path/to/script.py", ...], ...)

    Raises AssertionError with context on unexpected call shapes so failures
    are readable rather than crashing with IndexError / AttributeError.
    """
    positional_args, _ = mock_call
    cmd = positional_args[0]
    assert isinstance(cmd, list), (
        f"Expected subprocess.run to receive a list command, got {type(cmd)}: {cmd!r}"
    )
    assert len(cmd) >= 2, (
        f"Command list too short to contain a script path: {cmd!r}"
    )
    return Path(cmd[1]).name


def _get_script_cmd(calls, script_name: str) -> list:
    """
    Return the full command list for the first call matching *script_name*.
    Raises AssertionError with a descriptive message if the script was never called.
    """
    for c in calls:
        positional_args, _ = c
        cmd = positional_args[0]
        if isinstance(cmd, list) and len(cmd) >= 2 and Path(cmd[1]).name == script_name:
            return cmd
    executed = [_extract_script_name(c) for c in calls]
    raise AssertionError(
        f"{script_name!r} was not called. Scripts that ran: {executed}"
    )


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

@pytest.mark.skipif(run_pipeline is None, reason="run_pipeline.py could not be imported")
class TestPipelineOrchestrator:

    @pytest.fixture
    def mock_subprocess(self):
        """Mock subprocess.run to prevent actual script execution."""
        with patch("subprocess.run") as mock:
            mock.return_value.returncode = 0  # simulate success by default
            yield mock

    # ------------------------------------------------------------------
    # Smoke test: entry point is importable and prints help
    # ------------------------------------------------------------------

    @pytest.mark.skipif(
        not (SCRIPTS_DIR / "run_pipeline.py").exists(),
        reason="run_pipeline.py not found on disk — skipping filesystem smoke test",
    )
    def test_pipeline_help_command(self):
        """Verify the script is executable and prints help text."""
        cmd = [sys.executable, str(SCRIPTS_DIR / "run_pipeline.py"), "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"--help exited with code {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "Pipeline" in result.stdout, (
            f"Expected 'Pipeline' in --help output, got:\n{result.stdout}"
        )

    # ------------------------------------------------------------------
    # Default execution flow
    # ------------------------------------------------------------------

    def test_default_execution_flow(self, mock_subprocess):
        """
        Verify default flags:
          update_data      → YES  (maps to update_data.py)
          validate         → NO   (maps to validate_factors.py)
          train            → NO   (maps to train_models.py)
          deploy           → NO   (maps to deploy_model.py)
          inference        → YES  (maps to generate_predictions.py)
          backtest         → NO   (maps to run_backtest.py)
          optimize         → YES  (maps to optimize_portfolio.py)
          report           → YES  (maps to create_report.py)
        """
        with patch.object(sys, "argv", ["run_pipeline.py"]):
            run_pipeline.run()

        calls = mock_subprocess.call_args_list
        executed_scripts = [_extract_script_name(c) for c in calls]

        # --- scripts that MUST run ---
        assert "update_data.py" in executed_scripts
        assert "generate_predictions.py" in executed_scripts
        assert "optimize_portfolio.py" in executed_scripts
        assert "create_report.py" in executed_scripts

        # --- scripts that must NOT run by default ---
        assert "validate_factors.py" not in executed_scripts
        assert "train_models.py" not in executed_scripts
        assert "deploy_model.py" not in executed_scripts
        assert "run_backtest.py" not in executed_scripts

    def test_default_execution_order(self, mock_subprocess):
        """
        Data must be fetched before predictions are generated;
        predictions must exist before portfolio is optimised;
        the report must be the final step.
        """
        with patch.object(sys, "argv", ["run_pipeline.py"]):
            run_pipeline.run()

        calls = mock_subprocess.call_args_list
        executed_scripts = [_extract_script_name(c) for c in calls]

        def pos(name):
            assert name in executed_scripts, f"{name!r} was not called"
            return executed_scripts.index(name)

        assert pos("update_data.py") < pos("generate_predictions.py"), (
            "update_data.py must run before generate_predictions.py"
        )
        assert pos("generate_predictions.py") < pos("optimize_portfolio.py"), (
            "generate_predictions.py must run before optimize_portfolio.py"
        )
        assert pos("optimize_portfolio.py") < pos("create_report.py"), (
            "optimize_portfolio.py must run before create_report.py"
        )

    # ------------------------------------------------------------------
    # Full-rebuild flags
    # ------------------------------------------------------------------

    def test_full_rebuild_flags(self, mock_subprocess):
        """Verify --full-rebuild triggers training, validation, deployment, and backtest."""
        with patch.object(
            sys, "argv",
            ["run_pipeline.py", "--full-rebuild", "--validate", "--deploy", "--backtest"],
        ):
            run_pipeline.run()

        calls = mock_subprocess.call_args_list
        executed_scripts = [_extract_script_name(c) for c in calls]

        assert "validate_factors.py" in executed_scripts
        assert "train_models.py" in executed_scripts
        assert "deploy_model.py" in executed_scripts
        assert "run_backtest.py" in executed_scripts

        # --force-rebuild must be forwarded to train_models.py
        train_cmd = _get_script_cmd(calls, "train_models.py")
        assert "--force-rebuild" in train_cmd, (
            f"Expected '--force-rebuild' in train_models.py call, got: {train_cmd}"
        )

    def test_full_rebuild_execution_order(self, mock_subprocess):
        """Training must precede deployment; validation must precede training."""
        with patch.object(
            sys, "argv",
            ["run_pipeline.py", "--full-rebuild", "--validate", "--deploy"],
        ):
            run_pipeline.run()

        calls = mock_subprocess.call_args_list
        executed_scripts = [_extract_script_name(c) for c in calls]

        def pos(name):
            assert name in executed_scripts, f"{name!r} was not called"
            return executed_scripts.index(name)

        assert pos("validate_factors.py") < pos("train_models.py"), (
            "validate_factors.py must run before train_models.py"
        )
        assert pos("train_models.py") < pos("deploy_model.py"), (
            "train_models.py must run before deploy_model.py"
        )

    # ------------------------------------------------------------------
    # --skip-data flag
    # ------------------------------------------------------------------

    def test_skip_data_flag_suppresses_update(self, mock_subprocess):
        """--skip-data must prevent update_data.py from running."""
        with patch.object(sys, "argv", ["run_pipeline.py", "--skip-data"]):
            run_pipeline.run()

        calls = mock_subprocess.call_args_list
        executed_scripts = [_extract_script_name(c) for c in calls]

        assert "update_data.py" not in executed_scripts, (
            "--skip-data was passed but update_data.py still ran"
        )
        # Downstream steps should still run
        assert "generate_predictions.py" in executed_scripts
        assert "optimize_portfolio.py" in executed_scripts

    # ------------------------------------------------------------------
    # Optimization argument forwarding
    # ------------------------------------------------------------------

    def test_optimization_arguments(self, mock_subprocess):
        """Verify capital, opt-method, and target-vol are forwarded correctly."""
        with patch.object(
            sys, "argv",
            [
                "run_pipeline.py",
                "--skip-data",
                "--capital", "500000",
                "--opt-method", "risk_parity",
                "--target-vol", "0.12",
            ],
        ):
            run_pipeline.run()

        calls = mock_subprocess.call_args_list
        opt_cmd = _get_script_cmd(calls, "optimize_portfolio.py")

        assert "--capital" in opt_cmd, f"Missing --capital in: {opt_cmd}"
        # argparse may serialise as int or float; accept either
        capital_idx = opt_cmd.index("--capital") + 1
        assert opt_cmd[capital_idx] in ("500000", "500000.0"), (
            f"Unexpected capital value: {opt_cmd[capital_idx]!r}"
        )

        assert "--method" in opt_cmd, f"Missing --method in: {opt_cmd}"
        method_idx = opt_cmd.index("--method") + 1
        assert opt_cmd[method_idx] == "risk_parity", (
            f"Unexpected method value: {opt_cmd[method_idx]!r}"
        )

        assert "--target-vol" in opt_cmd, f"Missing --target-vol in: {opt_cmd}"
        vol_idx = opt_cmd.index("--target-vol") + 1
        assert opt_cmd[vol_idx] == "0.12", (
            f"Unexpected target-vol value: {opt_cmd[vol_idx]!r}"
        )

    def test_optimization_arguments_skip_data_respected(self, mock_subprocess):
        """When --skip-data is combined with optimization flags, data step is skipped."""
        with patch.object(
            sys, "argv",
            ["run_pipeline.py", "--skip-data", "--capital", "500000",
             "--opt-method", "risk_parity", "--target-vol", "0.12"],
        ):
            run_pipeline.run()

        calls = mock_subprocess.call_args_list
        executed_scripts = [_extract_script_name(c) for c in calls]
        assert "update_data.py" not in executed_scripts

    # ------------------------------------------------------------------
    # Subprocess failure propagation
    # ------------------------------------------------------------------

    def test_pipeline_aborts_on_subprocess_failure(self, mock_subprocess):
        """
        If a sub-script returns a non-zero exit code, the pipeline should
        abort rather than silently continue to downstream steps.
        """
        # FIX: run_pipeline uses subprocess.run(..., check=True).
        # A mock return_value with returncode=1 does NOT trigger the check=True exception logic.
        # We must explicitly raise CalledProcessError via side_effect to simulate failure.
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["mock_script.py"]
        )

        with patch.object(sys, "argv", ["run_pipeline.py"]):
            with pytest.raises((SystemExit, RuntimeError, subprocess.CalledProcessError)):
                run_pipeline.run()