"""
percept_slam.py — SLAM / VIO 前端(感知層)
===========================================
公開介面(對外承諾,ORB-SLAM3 接上後簽名不變):
    update(frame, imu_data=None) -> SlamState
    get_state() -> SlamState
    reset()
    set_backend(backend)
不對外開放:ORB-SLAM3 內部地圖點、關鍵幀管理、特徵抽取實作。

--------------------------------------------------------------------
★ 誠實聲明(報告與交接都必須照抄,不可含糊):
  本檔第一版 **不是 SLAM**。它是一個「介面 + 特徵健康度監測」的過渡實作,
  存在的唯一理由是:信心階梯 L2 的定義依賴「SLAM 特徵點 > 80」這個數字
  (§4.2),而這個數字在 ORB-SLAM3 綁上來之前,可以先用 cv2.ORB 對每幀
  抽特徵點數量來提供——它衡量的是「這個畫面紋理夠不夠支撐視覺追蹤」,
  這一點與 SLAM 追蹤是否穩定高度相關,但它 **不產生位姿、不產生地圖**。

  因此第一版:
    state.tracking     : 有意義(特徵夠多 = 有機會追得住)
    state.n_features   : 有意義(真的數出來的)
    state.pose         : 永遠 None
    state.translation_mm: 永遠 None → memory_zone_state 的「位移 4–5 公尺
                          降級」(§4.6)在第一版沒有訊號來源,只能靠時間降級。
  ORB-SLAM3 綁上來後,把 backend 換成 Orbslam3Backend,以上三項就會有值,
  其他檔案一行都不用改——這正是模組化判準(§3.1)要驗證的事。
"""
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from shared_store import cfg, now_or

try:
    import cv2
except Exception:
    cv2 = None


@dataclass
class SlamState:
    tracking: bool = False
    n_features: int = 0
    lost_duration: float = 0.0          # 連續追蹤失敗持續秒數
    pose: Optional[np.ndarray] = None   # 4x4,tag/世界座標系;第一版恆 None
    translation_mm: Optional[Tuple[float, float, float]] = None
    backend: str = "none"
    ts: float = 0.0

    def to_dict(self):
        return {"tracking": self.tracking, "n_features": self.n_features,
                "lost_duration": round(self.lost_duration, 2),
                "has_pose": self.pose is not None, "backend": self.backend}


# ============================================================
# 後端
# ============================================================
class _FeatureHealthBackend:
    """v1:cv2.ORB 特徵點健康度。純 CPU,縮圖後抽,對主迴圈負擔可控。"""
    name = "orb_feature_health"

    def __init__(self):
        self._orb = None
        if cv2 is not None:
            try:
                self._orb = cv2.ORB_create(nfeatures=int(cfg("SLAM_ORB_NFEATURES", 500)))
            except Exception as e:
                print(f"⚠️ [slam] ORB 初始化失敗: {e}")

    def process(self, frame, imu_data):
        if self._orb is None or frame is None:
            return 0, None, None
        try:
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scale = float(cfg("SLAM_DOWNSCALE", 0.5))
            if 0 < scale < 1.0:
                gray = cv2.resize(gray, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_AREA)
            kps = self._orb.detect(gray, None)
            return len(kps), None, None
        except Exception:
            return 0, None, None


class _NullBackend:
    """完全停用(省算力 / 單元測試)。"""
    name = "null"

    def process(self, frame, imu_data):
        return 0, None, None


class Orbslam3Backend:
    """
    ORB-SLAM3 綁定的預留位置。實作時只需在 process() 內:
        self.slam.TrackRGBD(frame, depth, ts) / TrackMonocularInertial(...)
    回傳 (特徵點數, 4x4 位姿, (tx,ty,tz) mm)。其餘檔案不用動。
    """
    name = "orbslam3"

    def __init__(self, vocab_path=None, settings_path=None):
        self.ready = False
        self.vocab_path = vocab_path
        self.settings_path = settings_path
        print("ℹ️ [slam] Orbslam3Backend 尚未綁定(第一版預留),自動回退特徵健康度")

    def process(self, frame, imu_data):
        return 0, None, None


# ============================================================
# 前端
# ============================================================
class SlamFrontend:
    def __init__(self, backend=None):
        self._backend = backend or (_FeatureHealthBackend() if cv2 is not None else _NullBackend())
        self._state = SlamState(backend=self._backend.name, ts=time.time())
        self._lost_since = None
        self._interval = float(cfg("SLAM_UPDATE_INTERVAL", 0.2))   # 5 Hz 足夠餵信心階梯
        self._last_run = 0.0

    def set_backend(self, backend):
        self._backend = backend
        self._state.backend = getattr(backend, "name", "custom")

    def update(self, frame, imu_data=None, now=None) -> SlamState:
        now = now_or(now)
        if now - self._last_run < self._interval:
            return self._state                      # 降頻,不每幀都算
        self._last_run = now

        n, pose, trans = self._backend.process(frame, imu_data)
        min_feat = int(cfg("SLAM_MIN_FEATURES", 80))     # §4.2 L2 定義的 80
        tracking = bool(n >= min_feat or pose is not None)

        if tracking:
            self._lost_since = None
            lost = 0.0
        else:
            self._lost_since = self._lost_since or now
            lost = now - self._lost_since

        self._state = SlamState(tracking=tracking, n_features=int(n),
                                lost_duration=lost, pose=pose,
                                translation_mm=trans,
                                backend=self._backend.name, ts=now)
        return self._state

    def get_state(self) -> SlamState:
        return self._state

    def reset(self):
        self._lost_since = None
        self._state = SlamState(backend=self._backend.name, ts=time.time())


_singleton = None


def frontend() -> SlamFrontend:
    global _singleton
    if _singleton is None:
        _singleton = SlamFrontend()
    return _singleton


def update(frame, imu_data=None, now=None) -> SlamState:
    return frontend().update(frame, imu_data, now)


def get_state() -> SlamState:
    return frontend().get_state()


def reset():
    frontend().reset()


if __name__ == "__main__":
    f = SlamFrontend()
    rich = (np.random.rand(480, 640) * 255).astype(np.uint8)     # 雜訊 = 特徵多
    blank = np.full((480, 640), 128, np.uint8)                   # 白牆 = 特徵少
    print("紋理豐富:", f.update(rich, now=100.0).to_dict())
    print("白牆:", f.update(blank, now=101.0).to_dict())
    print("白牆 3 秒後:", f.update(blank, now=104.0).to_dict())
