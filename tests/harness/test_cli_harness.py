from typer.testing import CliRunner
from bakery.cli import app

runner = CliRunner()


def test_harness_run_default_config(tmp_path):
    result = runner.invoke(app, [
        "harness-run", "experiments/gwangyo_default.yaml",
        "--out", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "gwangyo_default" / "predictions.csv").exists()
    assert (tmp_path / "out" / "gwangyo_default" / "metrics.json").exists()
