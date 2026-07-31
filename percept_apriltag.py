"""
percept_apriltag.py — AprilTag / QR 分區錨點偵測(感知層)
==========================================================
公開介面(對外承諾):
    detect_tags(frame) -> List[TagDetection]
    set_camera_intrinsics(fx, fy, cx, cy)
    load_tag_zone_map(path=None) / bind_tag(tag_id, zone_id, room_type)
    backend_name() -> str
不對外開放:tag 解碼演算法細節、後端選擇、位姿求解。

--------------------------------------------------------------------
後端三選一(自動降級,任何一層失敗都不 raise,回空 list):
  ① pupil_apriltags   : 官方 C 實作,唯一提供真正的 decision_margin
  ② cv2.aruco(36h11) : OpenCV 內建,無 decision_margin
  ③ cv2.QRCodeDetector: QR 分區身分(§4.1 `ZONE:<room_type>:<zone_id>`)
①②③ 可同時啟用:AprilTag 負責分區內精確定位,QR 負責分區身分(自帶語意)。

★ decision_margin 的誠實處理(信心階梯 L1 依賴它,不能造假):
  只有 pupil_apriltags 有這個數字。後端②③沒有時 `decision_margin=None`、
  `margin_source="unavailable"`,memory_zone_state 收到 None 時不會直接給 L1,
  改為要求「連續兩次偵測到同一 tag」才升 L1(見 memory_zone_state 內註解)。
  不用假的銳利度指標去冒充 decision_margin——量綱不同,門檻 35 會失去意義。

★ 距離估計兩條路:
  有內參(set_camera_intrinsics)→ solvePnP 得完整位姿,距離取 |t|。
  無內參 → 針孔近似 d ≈ fx_est × tag_size / 像素寬,fx_est 由 CAM_HFOV 推。
  後者只給「大概多遠」,不給方位角,適合 QR 這種只需身分的用途。
"""
import math
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from shared_store import cfg, data_dir, read_json, write_json_atomic

try:
    import cv2
except Exception:                                    # 沒有 cv2 也不能讓整包掛掉
    cv2 = None

try:
    import pupil_apriltags as _pupil
except Exception:
    _pupil = None


# ============================================================
# 資料結構
# ============================================================
@dataclass
class TagDetection:
    """一次偵測到的分區錨點。座標系:影像像素;pose 為 tag→camera。"""
    tag_id: int                      # AprilTag 數字 id;QR 用 payload hash 出穩定負數
    family: str                      # "tag36h11" / "qr"
    center: Tuple[float, float]
    corners: np.ndarray              # (4,2) float32,順時針
    decision_margin: Optional[float] = None
    margin_source: str = "unavailable"   # "apriltag" | "unavailable"
    payload: Optional[str] = None        # QR 原始字串
    zone_id: Optional[str] = None
    room_type: Optional[str] = None
    distance_mm: Optional[float] = None
    pose_R: Optional[np.ndarray] = None  # (3,3)
    pose_t: Optional[np.ndarray] = None  # (3,1) mm
    backend: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def has_pose(self) -> bool:
        return self.pose_t is not None

    @property
    def resolved(self) -> bool:
        """是否已解析出分區身分(有 zone_id 才能拿去更新 zone state)。"""
        return self.zone_id is not None

    def to_dict(self):
        return {
            "tag_id": self.tag_id, "family": self.family,
            "center": [float(self.center[0]), float(self.center[1])],
            "decision_margin": self.decision_margin,
            "margin_source": self.margin_source,
            "zone_id": self.zone_id, "room_type": self.room_type,
            "distance_mm": self.distance_mm, "backend": self.backend,
        }


# ============================================================
# 相機內參 / tag 尺寸
# ============================================================
_INTRINSICS = {"fx": None, "fy": None, "cx": None, "cy": None}


def set_camera_intrinsics(fx, fy, cx, cy):
    """OAK 校正資料(`getCameraIntrinsics`)拿到後呼叫一次即可。"""
    _INTRINSICS.update({"fx": float(fx), "fy": float(fy),
                        "cx": float(cx), "cy": float(cy)})


def _fx_estimate(frame_w):
    if _INTRINSICS["fx"]:
        return _INTRINSICS["fx"]
    hfov = float(cfg("TAG_CAM_HFOV_DEG", 95.0))       # OAK-D Pro W 彩色約 95°
    return (frame_w / 2.0) / max(1e-6, math.tan(math.radians(hfov / 2.0)))


def _tag_size_mm():
    return float(cfg("TAG_SIZE_MM", 130.0))           # §4.1 建議 12–15cm


# ============================================================
# tag_id / QR → 分區身分 對照表
# ============================================================
_ZONE_MAP = {}                                        # {"7": {"zone_id":..., "room_type":...}}
_ZONE_MAP_PATH = None
_QR_RE = re.compile(r"^\s*ZONE:([A-Za-z0-9_\-]+):([A-Za-z0-9_\-]+)\s*$", re.I)


def zone_map_path():
    global _ZONE_MAP_PATH
    if _ZONE_MAP_PATH is None:
        _ZONE_MAP_PATH = os.path.join(data_dir(), "tag_zone_map.json")
    return _ZONE_MAP_PATH


def load_tag_zone_map(path=None):
    """讀 tag_id → 分區對照表(建置模式 proc_setup_wizard 產生)。"""
    global _ZONE_MAP, _ZONE_MAP_PATH
    _ZONE_MAP_PATH = path or zone_map_path()
    _ZONE_MAP = read_json(_ZONE_MAP_PATH, default={}) or {}
    return dict(_ZONE_MAP)


def bind_tag(tag_id, zone_id, room_type, persist=True):
    """建置模式用:把一個實體 tag 綁到一個分區。"""
    _ZONE_MAP[str(tag_id)] = {"zone_id": zone_id, "room_type": room_type}
    if persist:
        write_json_atomic(zone_map_path(), _ZONE_MAP)
    return dict(_ZONE_MAP[str(tag_id)])


def _resolve_identity(det: TagDetection):
    """QR payload 優先(自帶語意),其次查對照表。"""
    if det.payload:
        m = _QR_RE.match(det.payload)
        if m:
            det.room_type, det.zone_id = m.group(1).lower(), m.group(2)
            return det
    entry = _ZONE_MAP.get(str(det.tag_id))
    if entry:
        det.zone_id = entry.get("zone_id")
        det.room_type = entry.get("room_type")
    return det


# ============================================================
# 後端
# ============================================================
_pupil_detector = None
_aruco_detector = None
_qr_detector = None
_backends = []


def _init_backends():
    global _pupil_detector, _aruco_detector, _qr_detector
    if _backends:
        return _backends
    if _pupil is not None:
        try:
            _pupil_detector = _pupil.Detector(families="tag36h11", nthreads=2,
                                              quad_decimate=float(cfg("TAG_QUAD_DECIMATE", 1.5)))
            _backends.append("pupil_apriltags")
        except Exception as e:
            print(f"⚠️ [apriltag] pupil_apriltags 初始化失敗: {e}")
    if cv2 is not None and hasattr(cv2, "aruco") and "pupil_apriltags" not in _backends:
        try:
            d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            params = cv2.aruco.DetectorParameters()
            _aruco_detector = cv2.aruco.ArucoDetector(d, params)
            _backends.append("cv2.aruco")
        except Exception as e:
            print(f"⚠️ [apriltag] cv2.aruco 初始化失敗: {e}")
    if cv2 is not None and bool(cfg("TAG_QR_ENABLED", True)):
        try:
            _qr_detector = cv2.QRCodeDetector()
            _backends.append("qr")
        except Exception:
            pass
    if not _backends:
        _backends.append("none")
        print("⚠️ [apriltag] 無可用後端(缺 cv2/pupil_apriltags),detect_tags 一律回空")
    return _backends


def backend_name():
    return "+".join(_init_backends())


def available():
    return _init_backends() != ["none"]


# ============================================================
# 位姿 / 距離
# ============================================================
def _solve_pose(corners, frame_w):
    """有內參 → solvePnP;無內參 → 只用像素寬近似距離。"""
    s = _tag_size_mm() / 2.0
    if cv2 is not None and _INTRINSICS["fx"]:
        obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)
        K = np.array([[_INTRINSICS["fx"], 0, _INTRINSICS["cx"]],
                      [0, _INTRINSICS["fy"], _INTRINSICS["cy"]],
                      [0, 0, 1]], dtype=np.float32)
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float32), K, None,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                R, _ = cv2.Rodrigues(rvec)
                return R, tvec.reshape(3, 1), float(np.linalg.norm(tvec))
        except Exception:
            pass
    # 針孔近似
    w_px = max(np.linalg.norm(corners[0] - corners[1]),
               np.linalg.norm(corners[1] - corners[2]))
    if w_px < 1e-3:
        return None, None, None
    d = _fx_estimate(frame_w) * _tag_size_mm() / float(w_px)
    return None, None, float(d)


def _mk(tag_id, family, corners, frame_w, backend, margin=None, payload=None):
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    R, t, dist = _solve_pose(corners, frame_w)
    det = TagDetection(
        tag_id=int(tag_id), family=family,
        center=(float(corners[:, 0].mean()), float(corners[:, 1].mean())),
        corners=corners,
        decision_margin=(float(margin) if margin is not None else None),
        margin_source=("apriltag" if margin is not None else "unavailable"),
        payload=payload, distance_mm=dist, pose_R=R, pose_t=t, backend=backend,
    )
    return _resolve_identity(det)


# ============================================================
# 主介面
# ============================================================
def detect_tags(frame) -> List[TagDetection]:
    """
    對一張影像做分區錨點偵測。

    frame: BGR 或灰階 ndarray(灰階可直接吃 → 與夜視 IR/mono 管線天生相容)
    回傳:List[TagDetection],無偵測回 []。任何內部例外都吞掉回 [],
          因為定位失效必須優雅退回「純避障」,不能讓主迴圈崩掉。
    """
    if frame is None or cv2 is None:
        return []
    backends = _init_backends()
    if backends == ["none"]:
        return []
    try:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        return []
    H, W = gray.shape[:2]
    out: List[TagDetection] = []

    if _pupil_detector is not None:
        try:
            for r in _pupil_detector.detect(gray):
                out.append(_mk(r.tag_id, "tag36h11", r.corners, W,
                               "pupil_apriltags", margin=r.decision_margin))
        except Exception as e:
            print(f"⚠️ [apriltag] pupil 偵測失敗: {e}")
    elif _aruco_detector is not None:
        try:
            corners, ids, _ = _aruco_detector.detectMarkers(gray)
            if ids is not None:
                for c, i in zip(corners, ids.flatten()):
                    out.append(_mk(int(i), "tag36h11", c.reshape(4, 2), W, "cv2.aruco"))
        except Exception as e:
            print(f"⚠️ [apriltag] aruco 偵測失敗: {e}")

    if _qr_detector is not None:
        try:
            ok, infos, pts, _ = _qr_detector.detectAndDecodeMulti(gray)
            if ok and pts is not None:
                for txt, quad in zip(infos, pts):
                    if not txt:
                        continue
                    out.append(_mk(_qr_pseudo_id(txt), "qr", quad.reshape(4, 2), W,
                                   "qr", payload=txt))
        except Exception:
            pass                                     # QR 是輔助,失敗靜默
    return out


def _qr_pseudo_id(payload: str) -> int:
    """QR 沒有數字 id,用 payload 產生穩定負數 id,避免與 AprilTag id 撞號。"""
    h = 0
    for ch in payload:
        h = (h * 131 + ord(ch)) & 0x7FFFFFFF
    return -(h % 1000000 + 1)


def best_tag(dets: List[TagDetection]) -> Optional[TagDetection]:
    """
    多個 tag 同時入鏡時挑一個當定位依據:
      已解析身分 > 有 decision_margin 且較大 > 距離較近。
    """
    if not dets:
        return None
    def key(d):
        return (0 if d.resolved else 1,
                -(d.decision_margin if d.decision_margin is not None else 0.0),
                d.distance_mm if d.distance_mm is not None else 1e9)
    return sorted(dets, key=key)[0]


# ============================================================
# 自我檢查(python percept_apriltag.py)
# ============================================================
if __name__ == "__main__":
    print("後端:", backend_name())
    load_tag_zone_map()
    if cv2 is not None and hasattr(cv2, "aruco"):
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        img = np.full((480, 640), 255, np.uint8)
        marker = cv2.aruco.generateImageMarker(d, 7, 200)
        img[140:340, 220:420] = marker
        bind_tag(7, "living_01", "living_room", persist=False)
        res = detect_tags(img)
        for r in res:
            print("偵測:", r.to_dict())
        print("最佳:", best_tag(res))
