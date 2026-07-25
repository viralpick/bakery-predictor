from typer.testing import CliRunner
from bakery.cli import app

runner = CliRunner()


def test_harness_run_default_config(tmp_path):
    result = runner.invoke(app, [
        "harness-run", "experiments/gwangyo_default.yaml",
        "--out", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0, result.output
    d = tmp_path / "out" / "gwangyo_default"
    assert (d / "comparison.csv").exists()
    assert (d / "category_total" / "predictions.csv").exists()
    assert (d / "category_total" / "metrics.json").exists()

    report = tmp_path / "out" / "gwangyo_default" / "report.html"
    assert report.exists()
    html = report.read_text(encoding="utf-8")
    assert "전체매진 위험" in html
    assert "품목별 매진" in html          # gwangyo_default는 store_gw01 → 매진 섹션 포함
