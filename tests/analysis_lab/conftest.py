"""핸들러 테스트 공용 — 실 parquet IO 없이 AnalysisInputs 속성을 주입한다.

AnalysisInputs의 입력 속성은 functools.cached_property라서 __dict__에 값을 직접
넣으면 IO 없이 그 값이 쓰인다. 핸들러별로 필요한 속성만 주면 된다.
"""
import pytest

from bakery.analysis.lab.inputs import AnalysisInputs


@pytest.fixture
def stub_inputs():
    def _make(*, store="store_gw01", alpha=0.8, params=None, **attributes):
        inputs = AnalysisInputs(store=store, alpha=alpha, params=params or {})
        for name, value in attributes.items():
            inputs.__dict__[name] = value       # cached_property 사전 채우기
        return inputs
    return _make
