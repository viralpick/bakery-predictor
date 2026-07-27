"""bakery build-multistore CLI이 multistore_daily.parquet를 생성한다."""
from unittest.mock import patch
from typer.testing import CliRunner
from bakery.cli import app

runner = CliRunner()


def test_build_multistore_command_invokes_builder(tmp_path):
    fake = tmp_path / "multistore_daily.parquet"
    with patch("bakery.data.bonavi_loader_v2.build_multistore") as mock_build:
        mock_build.return_value = fake
        result = runner.invoke(app, ["build-multistore"])
    assert result.exit_code == 0
    mock_build.assert_called_once()
