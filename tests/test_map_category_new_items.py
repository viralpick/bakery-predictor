"""map_category 키워드 보강 회귀 테스트 (신규 데이터 편입 2026-07-23).

신규 파일 광교 당일폐기=Y 품목 중 옛 분류기가 etc로 놓쳤던 12품목이 제자리
카테고리로 가고, 진짜 샐러드/델리 6품목은 salad(타깃 제외 대상)로 가는지 검증.
"""

import pytest

from bakery.data.bonavi_loader import map_category


@pytest.mark.parametrize(
    "name,expected",
    [
        # 진짜 샐러드/델리 → salad (타깃 제외)
        ("머쉬룸 그린빈 샐러드", "salad"),
        ("반숙란 베이컨 시저 샐러드", "salad"),
        ("이탈리안햄 포테이토 샐러드", "salad"),
        ("치킨 시저 샐러드 보울 (2EA/BOX)", "salad"),
        ("쉬림프&퀴노아 샐러드 보울 (2EA/BOX)", "salad"),
        # 옛 분류기가 etc로 놓친 진짜 베이커리 → 제자리
        ("반숙란 에그마요 샌드위치", "sandwich"),
        ("크렌베리 크림치즈 샌드위치", "sandwich"),
        ("멀티 그레인 크랜베리 치킨 샌드위치", "sandwich"),
        ("미니 소시지(메이플)_24", "pastry"),
        ("포테이토 소시지_24", "pastry"),
        ("아티제 멀티그레인 롤_8개입", "bread"),
        ("제주 유자 브라우니", "sweets"),
    ],
)
def test_new_keywords_categorize(name, expected):
    assert map_category(name) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        # 기존 분류 회귀 — 보강이 기존 매핑을 깨지 않는지
        ("우유 식빵", "bread"),          # 식빵 guard
        ("초코 크루아상", "pastry"),
        ("딸기 생크림 케이크", "cake"),
        ("아메리카노", "beverage"),
        ("클럽샌드위치", "sandwich"),      # 기존 클럽샌드 + 신규 샌드위치 둘 다 sandwich
    ],
)
def test_existing_categories_unchanged(name, expected):
    assert map_category(name) == expected
