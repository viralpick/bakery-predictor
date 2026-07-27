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


def test_resolve_returns_handler_for_data_section():
    assert resolve("category_mix").kind == KIND_DATA


# demand_absorption은 Task 7에서 등록된다 — 그 전까지 xfail 유지.
@pytest.mark.xfail(reason="핸들러는 Task 7에서 등록", strict=True)
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


def test_load_handlers_is_idempotent():
    """load_handlers는 all_names/resolve/_handlers_in_order에서 반복 호출된다.
    sys.modules 캐시로 데코레이터가 재실행되지 않아야 한다(재실행되면 중복 등록으로 터짐)."""
    from bakery.analysis.lab.registry import DATA_ANALYSES, HYPOTHESES, load_handlers

    load_handlers()
    before = (len(DATA_ANALYSES), len(HYPOTHESES))
    load_handlers()
    load_handlers()
    assert (len(DATA_ANALYSES), len(HYPOTHESES)) == before


def test_duplicate_registration_is_rejected():
    """같은 이름을 두 섹션 중 어디에든 다시 등록하면 즉시 실패해야 한다."""
    from bakery.analysis.lab.registry import load_handlers, register_data, register_hypothesis

    load_handlers()
    # 이미 등록된 category_mix를 재등록 시도 — 같은 섹션
    with pytest.raises(ValueError, match="category_mix"):   # 메시지 문구는 고정 계약이 아니라 이름 포함만 확인
        register_data("category_mix", "중복")(lambda inputs: None)
    # 다른 섹션에 같은 이름을 등록해도 거부돼야 한다(두 딕셔너리 모두 검사)
    with pytest.raises(ValueError, match="category_mix"):
        register_hypothesis("category_mix", "중복")(lambda inputs: None)
