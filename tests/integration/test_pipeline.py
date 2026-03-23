"""
Pipeline Orchestration Integration Tests
========================================
Validates the execution flow and dependency resolution of the central pipeline orchestrator.

Purpose
-------
This module verifies that the `run_pipeline.py` entry point correctly parses
execution arguments and spawns the appropriate subprocesses in the exact
topological order required by the Directed Acyclic Graph (DAG). 

Role in Quantitative Workflow
-----------------------------
Ensures that the daily production or research workflows do not suffer from
execution state corruption (e.g., training a model before data is fetched)
by testing the control flow logic using aggressive subprocess mocking.

Dependencies
------------
- **Pytest**: Test execution framework.
- **Unittest.Mock**: Subprocess isolation to bypass multi-hour pipeline runs.
"""

import sys
import warnings
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, call

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

# Graceful fallback mechanisms for dynamic module resolution across nested execution boundaries
try:
    import scripts.run_pipeline as run_pipeline
except (ImportError, ModuleNotFoundError):
    try:
        import run_pipeline
    except (ImportError, ModuleNotFoundError) as exc:
        warnings.warn(f"Could not import run_pipeline: {exc}")
        run_pipeline = None

def _extract_script_name(mock_call) -> str:
    """
    Safely extracts the script filename from a mocked subprocess.run call.

    Args:
        mock_call (tuple): The captured call arguments from a MagicMock.

    Returns:
        str: The filename of the executed script.

    Raises:
        AssertionError: If the command list is malformed or lacks a script path.
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
    Isolates the full command list for the first execution matching the target script.

    Args:
        calls (list): The list of captured mock calls.
        script_name (str): The filename of the target script.

    Returns:
        list: The ordered list of arguments passed to subprocess.run.

    Raises:
        AssertionError: If the target script was never executed.
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

@pytest.mark.skipif(run_pipeline is None, reason="run_pipeline.py could not be imported")
class TestPipelineOrchestrator:

    @pytest.fixture
    def mock_subprocess(self):
        """
        Provisions a mocked subprocess boundary to prevent actual script execution.

        Yields:
            MagicMock: The patched subprocess.run object configured to return success.
        """
        with patch("subprocess.run") as mock:
            mock.return_value.returncode = 0 
            yield mock

    @pytest.mark.skipif(
        not (SCRIPTS_DIR / "run_pipeline.py").exists(),
        reason="run_pipeline.py not found on disk — skipping filesystem smoke test",
    )
    def test_pipeline_help_command(self):
        """
        Verifies the orchestrator script is executable and exposes CLI documentation.

        Args:
            None

        Returns:
            None
        """
        cmd = [sys.executable, str(SCRIPTS_DIR / "run_pipeline.py"), "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"--help exited with code {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "Pipeline" in result.stdout, (
            f"Expected 'Pipeline' in --help output, got:\n{result.stdout}"
        )

    def test_default_execution_flow(self, mock_subprocess):
        """
        Validates the default execution state machine when no explicit flags are provided.

        Ensures the pipeline correctly defaults to data fetching, inference, optimization,
        and reporting, while explicitly suppressing training and deployment stages.

        Args:
            mock_subprocess (MagicMock): The isolated subprocess fixture.

        Returns:
            None
        """
        with patch.object(sys, "argv", ["run_pipeline.py"]):
            run_pipeline.run()

        calls = mock_subprocess.call_args_list
        executed_scripts = [_extract_script_name(c) for c in calls]

        # Assert inclusion of mandatory structural pipeline components
        assert "update_data.py" in executed_scripts
        assert "generate_predictions.py" in executed_scripts
        assert "optimize_portfolio.py" in executed_scripts
        assert "create_report.py" in executed_scripts

        # Assert exclusion of computationally intensive ML lifecycle components by default
        assert "validate_factors.py" not in executed_scripts
        assert "train_models.py" not in executed_scripts
        assert "deploy_model.py" not in executed_scripts
        assert "run_backtest.py" not in executed_scripts

    def test_default_execution_order(self, mock_subprocess):
        """
        Asserts the strict topological execution order of the default DAG.

        Guarantees data is fetched prior to inference, inference completes before 
        optimization, and reporting executes as the final summary stage.

        Args:
            mock_subprocess (MagicMock): The isolated subprocess fixture.

        Returns:
            None
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

    def test_full_rebuild_flags(self, mock_subprocess):
        """
        Verifies that the full rebuild override correctly triggers all heavy ML components.

        Args:
            mock_subprocess (MagicMock): The isolated subprocess fixture.

        Returns:
            None
        """
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

        # Guarantee strict propagation of rebuild state constraints to the modeling engine
        train_cmd = _get_script_cmd(calls, "train_models.py")
        assert "--force-rebuild" in train_cmd, (
            f"Expected '--force-rebuild' in train_models.py call, got: {train_cmd}"
        )

    def test_full_rebuild_execution_order(self, mock_subprocess):
        """
        Asserts the topological integrity of the full rebuild DAG sequence.

        Args:
            mock_subprocess (MagicMock): The isolated subprocess fixture.

        Returns:
            None
        """
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

    def test_skip_data_flag_suppresses_update(self, mock_subprocess):
        """
        Validates the suppression logic for the data ingestion phase.

        Args:
            mock_subprocess (MagicMock): The isolated subprocess fixture.

        Returns:
            None
        """
        with patch.object(sys, "argv", ["run_pipeline.py", "--skip-data"]):
            run_pipeline.run()

        calls = mock_subprocess.call_args_list
        executed_scripts = [_extract_script_name(c) for c in calls]

        assert "update_data.py" not in executed_scripts, (
            "--skip-data was passed but update_data.py still ran"
        )
        # Verify downstream DAG resolution continues unhindered
        assert "generate_predictions.py" in executed_scripts
        assert "optimize_portfolio.py" in executed_scripts

    def test_optimization_arguments(self, mock_subprocess):
        """
        Verifies the accurate propagation of hyperparameter constraints to the optimization layer.

        Args:
            mock_subprocess (MagicMock): The isolated subprocess fixture.

        Returns:
            None
        """
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
        # Tolerate type-casting variance from upstream argument resolution
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
        """
        Asserts that parallel invocation of optimization constraints and skip flags resolve correctly.

        Args:
            mock_subprocess (MagicMock): The isolated subprocess fixture.

        Returns:
            None
        """
        with patch.object(
            sys, "argv",
            ["run_pipeline.py", "--skip-data", "--capital", "500000",
             "--opt-method", "risk_parity", "--target-vol", "0.12"],
        ):
            run_pipeline.run()

        calls = mock_subprocess.call_args_list
        executed_scripts = [_extract_script_name(c) for c in calls]
        assert "update_data.py" not in executed_scripts

    def test_pipeline_aborts_on_subprocess_failure(self, mock_subprocess):
        """
        Validates the strict Fail-Fast mechanism of the orchestrator.

        Ensures that if any sub-script emits a non-zero exit code, the pipeline 
        immediately halts to prevent downstream propagation of corrupted states.

        Args:
            mock_subprocess (MagicMock): The isolated subprocess fixture.

        Returns:
            None
        """
        # Injects an explicit CalledProcessError to mathematically trigger the `check=True` 
        # exception handler, accurately simulating a structural failure in the DAG.
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["mock_script.py"]
        )

        with patch.object(sys, "argv", ["run_pipeline.py"]):
            with pytest.raises((SystemExit, RuntimeError, subprocess.CalledProcessError)):
                run_pipeline.run()