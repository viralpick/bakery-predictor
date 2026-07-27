"""분석/가설 핸들러 모듈 — import만으로 registry에 자기 등록된다.

registry.load_handlers()가 이 목록을 순회한다. 새 핸들러 모듈 추가 시 여기에 이름을 넣는다.
"""

HANDLER_MODULES: tuple[str, ...] = ("sales", "absorption", "calendar_bias", "model_bias", "waste")
