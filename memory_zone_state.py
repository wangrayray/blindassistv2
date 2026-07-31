"""
memory_zone_state.py — 分區狀態 + 信心階梯 L1–L5(狀態層)
==========================================================
公開介面(對外承諾):
    on_tag_seen(tag, now=None)
    on_slam_update(slam_state, now=None)
    on_displacement(mm, now=None)          # §4.6 位移降級(第一版無訊號源,預留)
    on_door_seen(now=None)                 # §4.6 YOLO door 當輔助訊號
    report_scene_vote(vote, now=None)      # §4.8 落差偵測 + §5.3 延伸場景四/五
    get_confidence_level(now=None) -> ConfLevel
    get_current_zone(now=None) -> dict|None
    get_scene_hint(now=None) -> dict|None
    get_status(now=None) -> dict
    pop_events() -> List[dict]
    register_zone(zone_id, room_type, name=None)
    camera_to_zone_local(xyz_mm, now=None) -> (xyz, quality) | None
不對外開放:信心階梯門檻常數、分區檔案格式、落差偵測連續計數。

--------------------------------------------------------------------
信心階梯(§4.2)
  L1  看到 tag 且 decision_margin > 35            → 精確座標
  L2  tag 消失 < 2 秒 且 SLAM 特徵點 > 80          → VIO 外推
  L3  tag 消失 2–10 秒                             → 僅回報分區
  L4  SLAM 丟失,但 30 秒內曾有 L1/L2               → 「最後已知分區」假設
  L5  超過 30 秒無定位                             → 純避障

★ decision_margin 為 None 時怎麼辦(percept_apriltag 後端②③沒有這個數字):
  不假造分數。改為要求「TAG_REPEAT_WINDOW 秒內連續看到同一 tag 兩次」才承認
  L1。理由:AprilTag 的誤判多是單幀偶發,連兩次同 id 的機率極低,這是與
  decision_margin 不同量綱但同目的的替代把關。狀態會標 `margin_source`,
  報告需誠實揭露兩種把關方式並存。

★ 備用場景猜測絕不與 L1–L3 混用措辭(§5.3 延伸場景四):
  scene hint 只在 L4/L5 期間存在,一旦 tag 回來立刻清空;`get_current_zone()`
  永遠不會回傳猜的房型,猜測只從 `get_scene_hint()` 出去,措辭固定保守。
"""
import os
import time
from enum import IntEnum
from typing import List, Optional

import numpy as np

from shared_store import (cfg, data_dir, log_divergence, now_or, read_json,
                          write_json_atomic)


class ConfLevel(IntEnum):
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    L5 = 5


LEVEL_DESC = {
    ConfLevel.L1: "精確定位",
    ConfLevel.L2: "推估定位",
    ConfLevel.L3: "僅分區",
    ConfLevel.L4: "最後已知分區",
    ConfLevel.L5: "無定位",
}


class ZoneState:
    def __init__(self, path=None, autoload=True):
        self.path = path or os.path.join(data_dir(), "zones.json")
        self.zones = {}                 # zone_id → {room_type, name, visits, last_seen, created}
        self._events: List[dict] = []

        # tag 狀態
        self._last_tag = None           # 最近一次被「接受」的 tag
        self._last_tag_ts = 0.0
        self._pending_tag = None        # 無 margin 時的連續確認暫存
        self._pending_ts = 0.0
        self._margin_source = "unavailable"
        self._had_high_conf_ts = 0.0    # 最後一次達到 L1/L2 的時間

        # SLAM
        self._slam_tracking = False
        self._slam_features = 0

        # §4.6
        self._disp_since_tag = 0.0
        self._door_flag_ts = 0.0

        # §4.8 / §5.3
        self._gap_streak = 0
        self._gap_scene = None
        self._scene_hint = None         # {"scene", "scene_zh", "score", "ts"}
        self._last_level = ConfLevel.L5
        self._current_zone_id = None

        if autoload:
            self.load()

    # ============================================================
    # 分區登錄(建置模式 / 重訪)
    # ============================================================
    def load(self):
        self.zones = read_json(self.path, default={}) or {}
        return self.zones

    def save(self):
        return write_json_atomic(self.path, self.zones)

    def register_zone(self, zone_id, room_type, name=None, persist=True):
        z = self.zones.get(zone_id, {"visits": 0, "created": time.time()})
        z.update({"room_type": room_type, "name": name or z.get("name") or zone_id})
        self.zones[zone_id] = z
        if persist:
            self.save()                 # ★ 註冊即落檔
        return dict(z)

    def _touch(self, zone_id, now):
        """
        造訪計數 +1 並立即存檔。
        ★ 過去版本的 bug:touch() 只改記憶體不 save(),重開機造訪次數全部歸零。
          這裡固定寫回,並用 visit cooldown 避免同一次停留被重複計數。
        """
        z = self.zones.get(zone_id)
        if z is None:
            return None
        cd = float(cfg("ZONE_VISIT_COOLDOWN", 60.0))
        if "last_seen" not in z or now - float(z["last_seen"]) > cd:
            z["visits"] = int(z.get("visits", 0)) + 1
        z["last_seen"] = now
        self.zones[zone_id] = z
        self.save()
        return dict(z)

    # ============================================================
    # 輸入事件
    # ============================================================
    def on_tag_seen(self, tag, now=None):
        """
        tag: percept_apriltag.TagDetection,或含同名欄位的 dict。
        未解析出 zone_id(沒綁分區)的 tag 直接忽略——不知道是哪一區的錨點
        對定位沒有意義,硬用只會製造假信心。
        """
        now = now_or(now)
        if tag is None:
            return self.get_confidence_level(now)
        get = (lambda k: tag.get(k)) if isinstance(tag, dict) else (lambda k: getattr(tag, k, None))
        zone_id = get("zone_id")
        if not zone_id:
            return self.get_confidence_level(now)

        margin = get("decision_margin")
        min_margin = float(cfg("TAG_MIN_MARGIN", 35.0))
        accepted = False
        if margin is not None:
            accepted = margin > min_margin
            self._margin_source = "apriltag"
        else:
            # 無 margin → 連續兩次同 id 才承認
            win = float(cfg("TAG_REPEAT_WINDOW", 1.5))
            tid = get("tag_id")
            if self._pending_tag == tid and (now - self._pending_ts) <= win:
                accepted = True
            self._pending_tag, self._pending_ts = tid, now
            self._margin_source = "repeat_confirm"

        if not accepted:
            return self.get_confidence_level(now)

        prev_zone = self._current_zone_id
        self._last_tag = tag
        self._last_tag_ts = now
        self._had_high_conf_ts = now
        self._disp_since_tag = 0.0
        self._door_flag_ts = 0.0
        self._current_zone_id = zone_id

        # 分區沒登錄過 → 自動用 tag 帶的 room_type 補登(QR 自帶語意)
        if zone_id not in self.zones:
            self.register_zone(zone_id, get("room_type") or "unknown")
        self._touch(zone_id, now)

        if prev_zone != zone_id:
            self._gap_streak = 0
            self._emit({"type": "zone_change", "zone_id": zone_id,
                        "room_type": self.zones[zone_id].get("room_type"),
                        "text": f"進入{self._zone_name(zone_id)}"})

        # tag 回來 → 立刻清掉備用猜測(§5.3:避免舊的模糊猜測干擾新的精確定位)
        if self._scene_hint is not None:
            self._scene_hint = None
        return self.get_confidence_level(now)

    def on_slam_update(self, slam_state, now=None):
        now = now_or(now)
        if slam_state is None:
            return self.get_confidence_level(now)
        get = (lambda k: slam_state.get(k)) if isinstance(slam_state, dict) \
            else (lambda k: getattr(slam_state, k, None))
        self._slam_tracking = bool(get("tracking"))
        self._slam_features = int(get("n_features") or 0)
        trans = get("translation_mm")
        if trans is not None:
            try:
                self.on_displacement(float(np.linalg.norm(np.asarray(trans, dtype=float))), now)
            except Exception:
                pass
        return self.get_confidence_level(now)

    def on_displacement(self, mm, now=None):
        """§4.6:離開最後一次 tag 位置太遠 → 分區假設失效。"""
        now = now_or(now)
        self._disp_since_tag += max(0.0, float(mm))
        return self.get_confidence_level(now)

    def on_door_seen(self, now=None):
        """YOLO 偵測到 door → 可能剛換分區,當輔助訊號讓信心降一級。"""
        self._door_flag_ts = now_or(now)
        return self.get_confidence_level(now)

    # ============================================================
    # 信心階梯
    # ============================================================
    def get_confidence_level(self, now=None) -> ConfLevel:
        now = now_or(now)
        level = self._raw_level(now)

        # §4.6 位移 / 換門降級:分區身分變得不可靠,最多只能給 L4
        suspicious = (self._disp_since_tag > float(cfg("ZONE_CHANGE_DIST_MM", 4500.0))
                      or (self._door_flag_ts and now - self._door_flag_ts
                          < float(cfg("DOOR_SUSPECT_WINDOW", 5.0))))
        if suspicious and level < ConfLevel.L4:
            level = ConfLevel.L4

        if level != self._last_level:
            self._emit({"type": "level_change", "from": int(self._last_level),
                        "to": int(level), "desc": LEVEL_DESC[level]})
            # 信心回升到 L3 以上 → 備用猜測立即失效
            if level <= ConfLevel.L3:
                self._scene_hint = None
            self._last_level = level
        return level

    def _raw_level(self, now):
        if self._last_tag_ts <= 0:
            return ConfLevel.L5
        since = now - self._last_tag_ts
        if since <= float(cfg("TAG_FRESH_SEC", 0.5)):
            return ConfLevel.L1
        if since < float(cfg("L2_MAX_SEC", 2.0)) and self._slam_tracking \
                and self._slam_features >= int(cfg("SLAM_MIN_FEATURES", 80)):
            return ConfLevel.L2
        if since <= float(cfg("L3_MAX_SEC", 10.0)):
            return ConfLevel.L3
        if (now - self._had_high_conf_ts) <= float(cfg("L4_MAX_SEC", 30.0)):
            return ConfLevel.L4
        return ConfLevel.L5

    # ============================================================
    # 分區查詢
    # ============================================================
    def _zone_name(self, zone_id):
        z = self.zones.get(zone_id) or {}
        from percept_pseudo_vlm import SCENE_ZH
        return z.get("name") or SCENE_ZH.get(z.get("room_type"), z.get("room_type") or zone_id)

    def get_current_zone(self, now=None) -> Optional[dict]:
        """
        L1/L2 → 分區 + 座標可信;L3 → 只有分區;L4 → 分區但標 assumed;
        L5 → None。永遠不會回傳偽VLM猜的房型。
        """
        now = now_or(now)
        level = self.get_confidence_level(now)
        if level >= ConfLevel.L5 or self._current_zone_id is None:
            return None
        z = dict(self.zones.get(self._current_zone_id) or {})
        z.update({
            "zone_id": self._current_zone_id,
            "level": int(level),
            "assumed": level >= ConfLevel.L4,
            "coords_valid": level <= ConfLevel.L2,
            "display_name": self._zone_name(self._current_zone_id),
        })
        return z

    def get_scene_hint(self, now=None) -> Optional[dict]:
        """
        §5.3 延伸場景四/五:只在 L4/L5 期間有效的弱訊號猜測。
        回傳 dict(含固定保守措辭)或 None。
        """
        now = now_or(now)
        if self.get_confidence_level(now) < ConfLevel.L4 or self._scene_hint is None:
            return None
        ttl = float(cfg("SCENE_HINT_TTL", 20.0))
        if now - self._scene_hint["ts"] > ttl:
            self._scene_hint = None
            return None
        h = dict(self._scene_hint)
        h["text"] = f"可能還在{h['scene_zh']}附近,但無法確定精確位置"
        h["below_l4"] = True
        return h

    # ============================================================
    # 偽VLM 場景投票進來(§4.8 + §5.3)
    # ============================================================
    def report_scene_vote(self, vote, now=None):
        """
        vote: percept_pseudo_vlm.SceneVote(或 dict)。回傳觸發的事件 list。
        兩種用途,依當下信心等級分流,絕不混用:
          L1–L3(知道自己在哪) → 拿投票做記憶—現實落差偵測
          L4/L5(不知道在哪)   → 拿投票當備用弱訊號猜測
        """
        now = now_or(now)
        if vote is None:
            return []
        get = (lambda k: vote.get(k)) if isinstance(vote, dict) else (lambda k: getattr(vote, k, None))
        if not get("passed"):
            return []                        # 沒過雙重門檻 → 一律不採用
        scene = get("scene")
        margin = float(get("margin") or 0.0)
        score = float(get("score") or 0.0)
        level = self.get_confidence_level(now)
        fired = []

        if level >= ConfLevel.L4:
            self._scene_hint = {"scene": scene, "scene_zh": get("scene_zh"),
                                "score": score, "ts": now}
            return fired

        zone = self.get_current_zone(now)
        if not zone or not zone.get("room_type"):
            return fired
        if margin < float(cfg("GAP_MARGIN_THRESH", 5.0)):
            return fired

        if scene == zone["room_type"]:
            self._gap_streak = 0
            self._gap_scene = None
            return fired

        if self._gap_scene != scene:
            self._gap_scene, self._gap_streak = scene, 0
        self._gap_streak += 1
        if self._gap_streak >= int(cfg("GAP_CONFIRM_SCANS", 3)):
            ev = {
                "type": "memory_gap", "zone_id": zone["zone_id"],
                "registered": zone["room_type"], "observed": scene,
                "score": score, "margin": margin,
                "text": f"這裡的樣子和記錄的{zone['display_name']}不太一樣",
            }
            self._emit(ev)
            fired.append(ev)
            log_divergence({"kind": "memory_gap", "mode": get("mode"),
                            "system1": scene, "system2": zone["room_type"],
                            "detail": {"score": score, "margin": margin,
                                       "zone_id": zone["zone_id"]}})
            self._gap_streak = 0
        return fired

    # ============================================================
    # 物件座標(§4.3)
    # ============================================================
    def camera_to_zone_local(self, xyz_mm, now=None):
        """
        相機座標系 3D 點 → 分區局部座標。
        回傳 ((x,y,z), quality) 或 None(L3 以下不給座標,§4.3 L4/L5 不寫入)。
          quality="tag_pose"     : 有 solvePnP 位姿,座標掛在 tag 錨點上
          quality="camera_frame" : 沒有內參/位姿,只能存「當下相機座標」,
                                   僅供粗略方向參考,不可宣稱精確
        """
        now = now_or(now)
        if self.get_confidence_level(now) > ConfLevel.L2:
            return None
        tag = self._last_tag
        if tag is None:
            return None
        R = getattr(tag, "pose_R", None)
        t = getattr(tag, "pose_t", None)
        p = np.asarray(xyz_mm, dtype=float).reshape(3, 1)
        if R is not None and t is not None:
            local = R.T @ (p - np.asarray(t, dtype=float).reshape(3, 1))
            return (tuple(float(v) for v in local.flatten()), "tag_pose")
        return (tuple(float(v) for v in p.flatten()), "camera_frame")

    # ============================================================
    # 事件 / 狀態
    # ============================================================
    def _emit(self, ev):
        ev.setdefault("ts", time.time())
        self._events.append(ev)
        if len(self._events) > 50:
            self._events = self._events[-50:]

    def pop_events(self):
        ev, self._events = self._events, []
        return ev

    def get_status(self, now=None):
        now = now_or(now)
        level = self.get_confidence_level(now)
        return {
            "level": int(level), "level_desc": LEVEL_DESC[level],
            "zone": self.get_current_zone(now),
            "scene_hint": self.get_scene_hint(now),
            "since_tag_sec": round(now - self._last_tag_ts, 1) if self._last_tag_ts else None,
            "margin_source": self._margin_source,
            "slam_tracking": self._slam_tracking,
            "slam_features": self._slam_features,
            "disp_since_tag_mm": round(self._disp_since_tag, 1),
            "n_zones": len(self.zones),
        }


# ============================================================
# 模組層單例
# ============================================================
_singleton = None


def state() -> ZoneState:
    global _singleton
    if _singleton is None:
        _singleton = ZoneState()
    return _singleton


def on_tag_seen(tag, now=None):
    return state().on_tag_seen(tag, now)


def on_slam_update(slam_state, now=None):
    return state().on_slam_update(slam_state, now)


def report_scene_vote(vote, now=None):
    return state().report_scene_vote(vote, now)


def get_confidence_level(now=None):
    return state().get_confidence_level(now)


def get_current_zone(now=None):
    return state().get_current_zone(now)


def get_scene_hint(now=None):
    return state().get_scene_hint(now)


def get_status(now=None):
    return state().get_status(now)


def pop_events():
    return state().pop_events()
