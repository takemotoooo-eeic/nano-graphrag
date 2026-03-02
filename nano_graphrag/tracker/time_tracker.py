"""経過時間の計測用トラッカー。インデクシング・クエリなどのフェーズごとに秒数を記録する。"""

import time
from typing import Dict, Optional


class TimeTracker:
    """インデクシングや回答など、フェーズごとの経過時間を記録する。"""

    def __init__(self):
        self.reset()

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self)

    def reset(self) -> None:
        """計測結果をリセットする。"""
        self._elapsed: Dict[str, float] = {}
        self._starts: Dict[str, float] = {}

    def start(self, phase: str) -> None:
        """フェーズの計測を開始する。"""
        self._starts[phase] = time.perf_counter()

    def stop(self, phase: str) -> float:
        """フェーズの計測を止め、経過秒数を加算して返す。"""
        start = self._starts.pop(phase, None)
        if start is None:
            return 0.0
        elapsed = time.perf_counter() - start
        self._elapsed[phase] = self._elapsed.get(phase, 0.0) + elapsed
        return elapsed

    def add_elapsed(self, phase: str, seconds: float) -> None:
        """フェーズに経過秒数を加算する。"""
        self._elapsed[phase] = self._elapsed.get(phase, 0.0) + seconds

    def get_elapsed(self, phase: Optional[str] = None) -> float:
        """指定フェーズの合計秒数。phase が None の場合は全フェーズの合計。"""
        if phase is not None:
            return self._elapsed.get(phase, 0.0)
        return sum(self._elapsed.values())

    def get_all(self) -> Dict[str, float]:
        """全フェーズの経過時間（秒）を返す。"""
        return dict(self._elapsed)

    def __str__(self) -> str:
        parts = [f"{k}: {v:.2f}s" for k, v in sorted(self._elapsed.items())]
        total = sum(self._elapsed.values())
        return "TimeTracker(" + ", ".join(parts) + f", total: {total:.2f}s)"
