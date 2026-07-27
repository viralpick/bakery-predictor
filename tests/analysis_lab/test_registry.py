import pytest

from bakery.analysis.lab.registry import (
    DATA_ANALYSES,
    HYPOTHESES,
    all_names,
    load_handlers,
    resolve,
)
from bakery.analysis.lab.result import KIND_DATA, KIND_HYPOTHESIS
from bakery.analysis.lab.spec import DEPRECATED_ANALYSES

# DATA_ANALYSES/HYPOTHESES를 모듈 레벨에서 직접 참조하는 테스트도 채워지도록
# import 시점에 한 번 로드한다(Task 4 시점엔 HANDLER_MODULES가 비어 있어 no-op).
load_handlers()


def test_kinds_are_tagged_per_section():
    for name, handler in DATA_ANALYSES.items():
        assert handler.kind == KIND_DATA, name
        assert handler.name == name
    for name, handler in HYPOTHESES.items():
        assert handler.kind == KIND_HYPOTHESIS, name
        assert handler.name == name


def test_no_deprecated_name_is_registered():
    assert all_names() & DEPRECATED_ANALYSES == frozenset()


def test_data_and_hypothesis_namespaces_do_not_collide():
    assert set(DATA_ANALYSES) & set(HYPOTHESES) == set()


def test_every_handler_has_korean_title():
    for handler in list(DATA_ANALYSES.values()) + list(HYPOTHESES.values()):
        assert handler.title != ""
        assert handler.title != handler.name        # 제목은 이름 재사용 금지(한국어 산문)


# 아래 두 테스트는 Task 6(category_mix)·Task 7(demand_absorption)에서 핸들러가
# 등록되기 전까지 KeyError로 실패한다. 두 이름을 각각 별도 테스트로 분리해
# xfail을 걸어야, Task 6에서 category_mix만 등록됐을 때 그 테스트만 정상적으로
# xfail 해제될 수 있다(하나로 묶으면 demand_absorption 미등록 탓에 계속 xfail로
# 남아 category_mix 쪽 회귀를 못 알아챈다).
@pytest.mark.xfail(reason="핸들러는 Task 6/7에서 등록", strict=True)
def test_resolve_returns_handler_for_data_section():
    assert resolve("category_mix").kind == KIND_DATA


@pytest.mark.xfail(reason="핸들러는 Task 6/7에서 등록", strict=True)
def test_resolve_returns_handler_for_hypothesis_section():
    assert resolve("demand_absorption").kind == KIND_HYPOTHESIS


def test_resolve_raises_on_unknown():
    # match=: KeyError 메시지 전체를 고정하면 registry.py의 안내 문구를 조금만
    # 바꿔도 테스트가 깨진다. 여기서 검증할 계약은 "요청한 이름이 메시지에
    # 나온다"는 것뿐이라 부분 매치가 정확 비교보다 적절하다.
    with pytest.raises(KeyError, match="nope"):
        resolve("nope")


def test_all_names_is_union_of_two_sections():
    assert all_names() == frozenset(DATA_ANALYSES) | frozenset(HYPOTHESES)
