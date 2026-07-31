"""
proc_gen3_pipeline.py — 第三代背景基礎設施層的整合門面(流程層)
================================================================
公開介面(對外承諾):
    Gen3Pipeline(hardware=None, publish_fn=None)
        .step(frame, dets=None, mode="daily", imu_data=None, depth_frame=None, now=None) -> dict
        .note_object(cls, cam_xyz_mm, conf, now=None) -> dict|None
        .pixel_to_camera_xyz(u, v, depth_mm) -> (x, y, z) | None
        .on_safety_alert(now=None)
        .status()
不對外開放:各子模組的呼叫順序、降頻策略。

存在理由:§3.4 的資料流向要求 main.py 依序呼叫
    percept_apriltag → percept_slam → memory_zone_state → 觸覺層
如果讓 main.py 直接串這四個,main.py 就得知道四個模組的內部節奏(誰要降頻、
誰要先誰後),等於把模組邊界打破。本檔把這串包成單一 `step()`,
main.py 的改動只有三行:

    from proc_gen3_pipeline import Gen3Pipeline          # ① import
    gen3 = Gen3Pipeline(hardware=hw)                     # ② 初始化
    g = gen3.step(frame, dets=dets, mode=mode)           # ③ 主迴圈每幀
    for s in g["speeches"]: say(s)

★ 待機模式(§2.2):mode="idle" 時,tag 偵測與記憶衰減照跑(衰減永遠運作),
  偽VLM 場景投票與物件登記暫停,符合「背景程序降頻」的既有設計初衷。
★ 安全優先:主迴圈發出避障/樓梯警告時呼叫 `on_safety_alert()`,
  信心震動會安靜讓路。
"""
from typing import List

import memory_object
import memory_zone_state as zone_state
import percept_apriltag as apriltag
import percept_pseudo_vlm as pseudo_vlm
import percept_slam as slam
from hw_confidence import ConfidenceHaptic
from shared_store import cfg, now_or


class Gen3Pipeline:
    def __init__(self, hardware=None, publish_fn=None, zone=None, objects=None):
        self.zone = zone or zone_state.state()
        self.objects = objects or memory_object.memory()
        self.pvlm = pseudo_vlm.instance()
        self.slam = slam.frontend()
        self.haptic = ConfidenceHaptic(hardware=hardware, publish_fn=publish_fn)
        apriltag.load_tag_zone_map()

        self._t_tag = 0.0
        self._t_vote = 0.0
        self._t_decay = 0.0
        self._last_hint_say = 0.0

    # ============================================================
    def step(self, frame, dets=None, mode="daily", imu_data=None,
             depth_frame=None, now=None) -> dict:
        now = now_or(now)
        idle = (mode == "idle")
        speeches: List[str] = []

        # ---- ① AprilTag / QR ----
        tags = []
        if frame is not None and now - self._t_tag >= float(cfg("TAG_DETECT_INTERVAL", 0.15)):
            self._t_tag = now
            tags = apriltag.detect_tags(frame)
            best = apriltag.best_tag([t for t in tags if t.resolved]) or apriltag.best_tag(tags)
            if best is not None and best.resolved:
                self.zone.on_tag_seen(best, now)

        # ---- ② SLAM(特徵健康度 → L2 判準) ----
        st = self.slam.update(frame, imu_data, now) if frame is not None else self.slam.get_state()
        self.zone.on_slam_update(st, now)

        # ---- ③ door 當分區改變輔助訊號(§4.6) ----
        if dets and _has_class(dets, "door"):
            self.zone.on_door_seen(now)

        # ---- ④ 偽VLM 場景投票(待機暫停) ----
        vote = None
        if not idle and dets and now - self._t_vote >= float(cfg("SCENE_VOTE_INTERVAL", 1.0)):
            self._t_vote = now
            vote = self.pvlm.infer_scene(dets, mode=mode, now=now)
            if vote is not None:
                self.zone.report_scene_vote(vote, now)

        # ---- ⑤ 信心等級 → 觸覺(§7) ----
        level = self.zone.get_confidence_level(now)
        self.haptic.set_confidence_vibration(int(level), now=now)
        self.haptic.tick(now)

        # ---- ⑥ 記憶衰減計時器(永遠運作,含待機) ----
        decayed = []
        if now - self._t_decay >= float(cfg("DECAY_TICK_INTERVAL", 60.0)):
            self._t_decay = now
            decayed = self.objects.decay_tick(now)

        # ---- ⑦ 事件 → 語音 ----
        events = self.zone.pop_events()
        announce = (vote.announce_allowed if vote is not None else True) and not idle
        for ev in events:
            if ev["type"] == "zone_change" and announce:
                speeches.append(ev["text"])
            elif ev["type"] == "memory_gap" and announce:
                speeches.append(ev["text"])
        for d in decayed[:1]:                       # 一次最多提醒一筆,不轟炸
            if announce:
                speeches.append(d["text"])

        # ---- ⑧ L4/L5 備用猜測(§5.3 延伸場景四/五) ----
        hint = self.zone.get_scene_hint(now)
        if (hint and announce
                and now - self._last_hint_say > float(cfg("SCENE_HINT_SAY_INTERVAL", 20.0))):
            self._last_hint_say = now
            speeches.append(hint["text"])

        return {
            "level": int(level),
            "zone": self.zone.get_current_zone(now),
            "scene_hint": hint,
            "vote": vote.to_dict() if vote is not None else None,
            "tags": [t.to_dict() for t in tags],
            "slam": st.to_dict(),
            "events": events,
            "speeches": speeches,
            "voice_modifier": self.haptic.pending_voice(),   # L4 的「可能」
        }

    # ============================================================
    # 物件登記
    # ============================================================
    def pixel_to_camera_xyz(self, u, v, depth_mm):
        """像素 + 深度 → 相機座標系 3D 點(mm)。沒有內參就回 None。"""
        fx = apriltag._INTRINSICS["fx"]
        if not fx or depth_mm is None:
            return None
        fy = apriltag._INTRINSICS["fy"] or fx
        cx = apriltag._INTRINSICS["cx"]
        cy = apriltag._INTRINSICS["cy"]
        z = float(depth_mm)
        return ((float(u) - cx) * z / fx, (float(v) - cy) * z / fy, z)

    def note_object(self, cls, cam_xyz_mm, conf, now=None):
        """
        把一次物件觀測寫進分區記憶。內部自動:
          相機座標 → 分區局部座標(§4.3)→ 依信心等級決定收不收(L3+ 一律拒絕)。
        回傳寫入的 instance 或 None。
        """
        now = now_or(now)
        local = self.zone.camera_to_zone_local(cam_xyz_mm, now)
        if local is None:
            return None
        xyz, quality = local
        z = self.zone.get_current_zone(now)
        if not z:
            return None
        return self.objects.register_detection(
            cls, xyz, z["zone_id"], conf, self.zone.get_confidence_level(now),
            coord_quality=quality, now=now)

    def query_object(self, cls, now=None, limit=3):
        now = now_or(now)
        z = self.zone.get_current_zone(now)
        return self.objects.get_nearest(cls, current_zone=(z or {}).get("zone_id"),
                                        current_xyz=(0.0, 0.0, 0.0), now=now, limit=limit)

    def on_safety_alert(self, now=None):
        self.haptic.hold_for_safety(now=now)

    def status(self, now=None):
        now = now_or(now)
        s = self.zone.get_status(now)
        s.update({"objects": self.objects.stats(now),
                  "apriltag_backend": apriltag.backend_name(),
                  "slam_backend": self.slam.get_state().backend})
        return s


def _has_class(dets, name):
    for d in dets or []:
        try:
            c = d.get("cls") if isinstance(d, dict) else (d[4] if len(d) >= 6 else None)
            if c and str(c).lower() == name:
                return True
        except Exception:
            continue
    return False
