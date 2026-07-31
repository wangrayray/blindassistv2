"""AI 推論 worker (獨立進程) — 整合版 v10
三組新功能 (Search REACH 距離 / 遮蔽超時 / 燈號穩定化)
+ 斑馬線 angle 斜率判斷 (轉頭 vs 平移)
"""
import os
import time
import queue

import cv2
import numpy as np
from ultralytics import YOLO

from shared_config import CONFIG, SIG_RETURN_IDLE
from shared_utils import (zh, horizontal_to_motor_7way, hysteretic_motor_7way,
                   zone_to_motor, ratio_to_clock, memory_fresh,
                   get_object_distance_mm, get_finger_distance_mm)
from percept_trackers import StabilityLockTracker, HandLockTracker, DepthAnomalyDetector
from percept_hands import HandTracker, is_user_own_hand
from percept_crosswalk import get_crosswalk_direction_hough, CrosswalkStabilizer
from percept_stairs_fusion import StairsFusion
from percept_light_color import classify_light_color, find_red_blobs


# 手部遮蔽判定: 手深度與目標最後深度差 < 此值 = 手擋住
OCCLUSION_DEPTH_TOL_MM = 200


def ai_worker_process(in_queue, out_queue, command_queue):
    print("🧠 [AI 核心] 啟動中...")

    # 解析 YOLO 推論裝置:強制用 GPU (此子進程的 ultralytics 自動偵測
    # 在 spawn + 雙顯卡環境下會誤判掉到 CPU,故明確指定 cuda:0)
    import torch
    if torch.cuda.is_available():
        YOLO_DEVICE = getattr(CONFIG, "YOLO_DEVICE", 0)
        print(f"🎮 [AI 核心] YOLO 推論裝置: cuda:{YOLO_DEVICE} "
              f"({torch.cuda.get_device_name(0)})")
    else:
        YOLO_DEVICE = "cpu"
        print("⚠️ [AI 核心] CUDA 不可用,YOLO 退回 CPU")

    model_instance = None
    current_model_key = None
    traffic_models = None

    hand_tracker = None
    hand_lock = HandLockTracker()
    lock_tracker = StabilityLockTracker(
        tolerance=CONFIG.SEARCH_LOCK_PIXEL_TOL,
        duration=CONFIG.SEARCH_LOCK_DURATION,
    )
    depth_detector = DepthAnomalyDetector(CONFIG.FRAME_W, CONFIG.FRAME_H)
    stairs_fusion = StairsFusion()
    crosswalk_stab = CrosswalkStabilizer()

    # 背景全物件掃描:YOLO11 官方 COCO 權重 (廣角幀);失敗不致命
    scan_model = None
    if CONFIG.SCAN_BG_ENABLED:
        try:
            scan_model = YOLO(CONFIG.MODEL_SCAN)
            scan_model.to(f"cuda:{YOLO_DEVICE}" if YOLO_DEVICE != "cpu" else "cpu")
            print(f"🛰️ [背景掃描] 載入 {CONFIG.MODEL_SCAN} → cuda:{YOLO_DEVICE}")
        except Exception as e:
            print(f"⚠️ [背景掃描] 模型載入失敗,停用: {e}")
    # 物件記憶表: { label(lower): {"ts":, "clock":, "zone":, "conf":} }
    object_memory = {}
    # VLM 語意推理結果暫存 + 本次搜尋是否已請求過推理
    reason_hint = None             # {"target":, "zone":, "clock":}
    reason_requested_target = None # 已對哪個目標發過推理請求 (避免重複請求)
    reason_request_out = None      # 待回傳給 main 的推理請求 (target)
    last_dets = []                 # 最近一次 YOLO 偵測 (回傳給 main 餵偽VLM 場景投票)

    mode = "idle"
    target_obj = None
    yolo_paused = False            # VLM 執行期間暫停 YOLO 推論 (讓出 GPU/VRAM)

    # ---- Search 預掃冷啟動 (廣角 VLM 標籤) ----
    prescan_tags = {}              # label -> {"zone", "ts"} (由 main 經 command 推入)
    search_last_prescan_hint = 0.0 # 粗略引導冷卻
    search_oak_seen = False        # OAK 是否已實際看到目標 (看到後不再給粗略引導)

    # ---- Daily 計時 ----
    last_daily_warn = 0.0
    last_high_priority_say = {}

    # ---- Traffic 計時 ----
    last_light_say = 0.0
    last_crossing_guide = 0.0
    last_red_warn = 0.0
    crossing_started_at = 0.0
    arrived_counter = 0

    # ---- Traffic 燈號穩定化 ----
    light_streak = {"green": 0, "red": 0, "none": 0}
    stable_light_status = None    # 經過確認的燈號狀態

    # ---- Search 計時 ----
    search_state = "STAGE_FIND"
    search_last_seen = 0.0
    search_last_guidance = 0.0
    search_last_too_far = 0.0      # 「對準但太遠」獨立冷卻
    search_last_vert = 0.0         # 垂直(抬頭/低頭)提示獨立冷卻

    # ---- Search 遮蔽記憶 (應對手擋住目標) ----
    last_known_target_cx = None
    last_known_target_cy = None
    last_known_target_depth = None
    last_known_target_box = None
    memory_start_time = None       # 開始用記憶的時間,用於超時兜底

    # ---- Search 多目標追蹤 + 馬達遲滯 ----
    search_track_box = None        # 目前鎖定追蹤的目標框 (在多個同類間維持同一個)
    search_last_motor = None       # 上次方位馬達 (遲滯用)

    traffic_state = "WAITING"

    def get_hand_tracker():
        nonlocal hand_tracker
        if hand_tracker is None:
            print("🖐️ 初始化 MediaPipe Hands...")
            hand_tracker = HandTracker()
            print("✅ MediaPipe 就緒")
        return hand_tracker

    def clear_last_known():
        nonlocal last_known_target_cx, last_known_target_cy
        nonlocal last_known_target_depth, last_known_target_box
        nonlocal memory_start_time
        nonlocal search_track_box, search_last_motor
        last_known_target_cx = None
        last_known_target_cy = None
        last_known_target_depth = None
        last_known_target_box = None
        memory_start_time = None
        search_track_box = None      # 目標真的遺失 → 釋放鎖定,下次重新選最近
        search_last_motor = None

    def _lookup_prescan_zone(target_label, now):
        """
        從 prescan_tags 查 target 是否在有效期內被廣角預掃看到。
        命中回傳 zone ("左"/"正前"/"右"),否則 None。
        prescan_tags 結構: {label: {"zone":..., "ts":...}}
        """
        if not target_label or not prescan_tags:
            return None
        t = target_label.lower()
        entry = prescan_tags.get(t)
        if entry is None:
            for k, v in prescan_tags.items():
                if t in k or k in t:
                    entry = v
                    break
        if entry is None:
            return None
        if now - entry.get("ts", 0) > CONFIG.PRESCAN_TAG_TTL:
            return None
        return entry.get("zone")

    def reset_all_timers(now):
        nonlocal search_state, search_last_seen, search_last_guidance
        nonlocal search_last_too_far
        nonlocal search_last_vert
        nonlocal traffic_state, last_light_say, last_crossing_guide, last_red_warn
        nonlocal arrived_counter, crossing_started_at, last_daily_warn
        nonlocal last_high_priority_say
        nonlocal light_streak, stable_light_status
        nonlocal search_last_prescan_hint, search_oak_seen
        nonlocal reason_hint, reason_requested_target, reason_request_out
        search_state = "STAGE_FIND"
        search_last_seen = now
        search_last_guidance = now
        search_last_too_far = 0.0
        search_last_vert = 0.0
        search_last_prescan_hint = 0.0
        search_oak_seen = False
        # 新搜尋會話 → 清推理暫存 (object_memory 是背景世界模型,不清)
        reason_hint = None
        reason_requested_target = None
        reason_request_out = None
        traffic_state = "WAITING"
        last_light_say = 0.0
        last_crossing_guide = 0.0
        last_red_warn = 0.0
        crossing_started_at = 0.0
        arrived_counter = 0
        last_daily_warn = 0.0
        last_high_priority_say = {}
        light_streak = {"green": 0, "red": 0, "none": 0}
        stable_light_status = None
        lock_tracker.reset()
        hand_lock.reset()
        depth_detector.reset()
        stairs_fusion.reset()
        crosswalk_stab.reset()
        clear_last_known()

    def _bg_scan_update(wide_frame, now):
        """對廣角幀跑 YOLO11,更新 object_memory(物件→時間戳/鐘向/方位)。"""
        if scan_model is None or wide_frame is None:
            return
        if yolo_paused:
            return                      # 背景物件掃描不是安全關鍵,讓 VLM 先跑
        try:
            ww = wide_frame.shape[1]
            res = scan_model(wide_frame, verbose=False, conf=CONFIG.SCAN_BG_CONF, device=YOLO_DEVICE)[0]
            # 同類取信心最高的一個代表
            best = {}
            for box in res.boxes:
                cls = scan_model.names[int(box.cls)].lower()
                cf = float(box.conf[0]) if box.conf is not None else 0.0
                x1, _, x2, _ = map(int, box.xyxy[0])
                cx = (x1 + x2) / 2.0
                if cls not in best or cf > best[cls][0]:
                    best[cls] = (cf, cx)
            for cls, (cf, cx) in best.items():
                clock, zone = ratio_to_clock(cx / ww)
                object_memory[cls] = {"ts": now, "clock": clock,
                                      "zone": zone, "conf": cf}
            # debug: 印這次掃到什麼 + 目前記憶表大小
            if best:
                seen = ", ".join(f"{c}@{object_memory[c]['clock']}點"
                                 for c in best)
                print(f"🛰️ [背景掃描] 偵測 {len(best)} 類: {seen} "
                      f"| 記憶表共 {len(object_memory)} 物件")
            else:
                print(f"🛰️ [背景掃描] 本幀無偵測 | 記憶表共 "
                      f"{len(object_memory)} 物件")
            # 控制表大小:超量就丟最舊的
            if len(object_memory) > CONFIG.SCAN_BG_MAX_ITEMS:
                oldest = sorted(object_memory.items(), key=lambda kv: kv[1]["ts"])
                for k, _ in oldest[:len(object_memory) - CONFIG.SCAN_BG_MAX_ITEMS]:
                    object_memory.pop(k, None)
        except Exception as e:
            print(f"⚠️ [背景掃描] 推論失敗: {e}")

    def _query_object_memory(target, now):
        """查記憶表,命中且未過期 → 回 entry;否則 None。寬鬆比對。"""
        if not target:
            return None
        t = target.lower()
        entry = object_memory.get(t)
        if entry is None:
            for k, v in object_memory.items():
                if t in k or k in t:
                    entry = v
                    break
        if entry and memory_fresh(entry["ts"], now, CONFIG.OBJECT_MEMORY_TTL):
            return entry
        return None

    def switch_model(new_mode):
        nonlocal model_instance, current_model_key, traffic_models
        if new_mode == current_model_key:
            return
        if model_instance is not None:
            del model_instance
            model_instance = None
        if traffic_models is not None:
            del traffic_models
            traffic_models = None

        if new_mode == "traffic":
            light_path = CONFIG.MODEL_TRAFFIC_LIGHT
            seg_path = CONFIG.MODEL_CROSSWALK
            if not (os.path.exists(light_path) and os.path.exists(seg_path)):
                print("⚠️ Traffic 模型缺失")
                current_model_key = None
                return
            print("🔄 載入 traffic 雙模型...")
            traffic_models = {
                "light": YOLO(light_path, task="detect"),
                "seg":   YOLO(seg_path, task="segment"),
            }
            for _m in traffic_models.values():
                _m.to(f"cuda:{YOLO_DEVICE}" if YOLO_DEVICE != "cpu" else "cpu")
            current_model_key = "traffic"
            print("✅ traffic 雙模型載入完成")
        else:
            path = CONFIG.MODEL_PATHS.get(new_mode)
            if path and os.path.exists(path):
                print(f"🔄 切換模型 → {new_mode}")
                model_instance = YOLO(path, task="detect")
                model_instance.to(f"cuda:{YOLO_DEVICE}" if YOLO_DEVICE != "cpu" else "cpu")
                current_model_key = new_mode
                print(f"✅ 模型 {new_mode} 載入完成")
            else:
                print(f"⚠️ 模型不存在: {path}")
                current_model_key = None

    def infer_detect(model, frame, conf, force=False):
        nonlocal last_dets
        """
        force=True 的呼叫端(Traffic 燈號/斑馬線)不受 VLM 暫停影響。
        其餘模式在 VLM 執行期間回空 list:語意標籤暫時缺席可以接受,
        但深度型安全判斷不經過這裡,所以不會一起停擺。
        """
        if model is None:
            return []
        if yolo_paused and not force:
            return []
        res = model(frame, verbose=False, conf=conf, device=YOLO_DEVICE)[0]
        out = []
        for box in res.boxes:
            cls = model.names[int(box.cls)]
            cf = float(box.conf[0]) if box.conf is not None else 0.0
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            out.append((x1, y1, x2, y2, cls, cf))
        last_dets = out
        return out

    # ============================================
    # Daily mode
    # ============================================
    def run_daily(frame, depth_frame, w, h, now):
        nonlocal last_daily_warn
        draws, speeches, vibs = [], [], []

        dets = infer_detect(model_instance, frame, CONFIG.CONF_DAILY)

        general_items = []
        high_priority_items = []
        vib_candidates = []   # (score, motor_id, vib_mode);最後只取最高危一顆震動
        stairs_yolo_box = None   # daily 樓梯面板用:YOLO 看到的樓梯真實框

        for (x1, y1, x2, y2, cls, conf) in dets:
            if cls == "stairs":
                color = (0, 165, 255)
                stairs_yolo_box = (int(x1), int(y1), int(x2), int(y2))
            elif cls == "kettle":
                color = (0, 100, 255)
            elif cls == "cone":
                color = (0, 255, 255)
            elif cls == "hole":
                color = (128, 0, 128)
            elif cls in CONFIG.DAILY_WARN_CLASSES:
                color = (0, 0, 255)
            else:
                color = (128, 128, 128)

            label = f"{zh(cls)} {conf:.2f}"
            box_motor = None   # 有算方位的物件 → 框上標 ->Mx

            if cls in CONFIG.DAILY_WARN_CLASSES and cls != "stairs":
                dist_mm = get_object_distance_mm(depth_frame, x1, y1, x2, y2,
                                                 conservative=True)
                cx = (x1 + x2) // 2
                motor_id, dir_name, clock = horizontal_to_motor_7way(cx, w)
                box_motor = motor_id
                severity = CONFIG.DAILY_CLASS_SEVERITY.get(cls, "medium")
                prefix = CONFIG.DAILY_CLASS_PREFIX.get(cls, "")
                # 電報式方位:改用時鐘方向 (與觸覺馬達的物理配置一致 —
                # M4 本來就對應「1 點」方向的馬達,語音講出來的時鐘要跟身上震動的位置一致)
                zone = clock.replace(" ", "") + "鐘方向"

                if dist_mm:
                    cm = int(dist_mm / 10)
                    label += f" {cm}cm"
                    # 距離 >1m 講公尺,否則公分
                    if dist_mm >= 1000:
                        dist_txt = f"{dist_mm/1000:.1f}公尺"
                    else:
                        dist_txt = f"{cm}公分"
                    # 電報式:物件 方位 距離 (前綴如「小心燙」放最前)
                    text = (f"{prefix} {zh(cls)} {zone} {dist_txt}" if prefix
                            else f"{zh(cls)} {zone} {dist_txt}")
                else:
                    text = (f"{prefix} {zh(cls)} {zone}" if prefix
                            else f"{zh(cls)} {zone}")

                if severity == "high":
                    high_priority_items.append((cls, text, motor_id, dist_mm))
                else:
                    general_items.append(text)

                danger_dist = (CONFIG.HIGH_PRIORITY_NEAR_MM if severity == "high"
                               else CONFIG.DAILY_NEAR_DIST_MM)
                if dist_mm and dist_mm < danger_dist:
                    vib_mode = "warn" if severity == "high" else "guide"
                    rank = 2 if severity == "high" else 1
                    # 分數:先比 severity,同級再比近 (越近分越高)
                    score = rank * 10_000_000 - dist_mm
                    vib_candidates.append((score, motor_id, vib_mode))

            if box_motor is not None:
                draws.append(("BOX_M", x1, y1, x2, y2, label, color, box_motor))
            else:
                draws.append(("BOX", x1, y1, x2, y2, label, color))

        for cls, text, motor_id, dist_mm in high_priority_items:
            last = last_high_priority_say.get(cls, 0)
            if now - last > CONFIG.HIGH_PRIORITY_INTERVAL:
                speeches.append(text)
                last_high_priority_say[cls] = now

        if general_items and (now - last_daily_warn > CONFIG.SCAN_INTERVAL):
            speeches.append("、".join(general_items))
            last_daily_warn = now

        # v19: 深度圖傳第二參數;depth_anomaly 只當 cliff 保底 (具名參數)。
        # 修 v17 bug: 舊版把 depth_anomaly 誤傳成深度圖 → 樓梯永遠量不到距離。
        depth_anomaly = depth_detector.update(depth_frame, now) if depth_frame is not None else None
        stairs_result = stairs_fusion.update(dets, depth_frame, now, anomaly=depth_anomaly)
        # 自動身高:樓梯地面模型反推的相機高度同步給深度偵測器,
        # 兩者共用同一個高度基準,不會各算各的。
        depth_detector.set_cam_height(stairs_fusion.cam_height_mm)
        for msg in stairs_fusion.pop_height_warnings():
            speeches.append(msg)

        if stairs_result:
            speeches.append(stairs_result["msg"])
            severity = stairs_result["severity"]
            # 靠近(high)→warn 長重;有(medium/low)→guide,讓觸覺也反映近遠
            vib_mode = "warn" if severity == "high" else "guide"
            rank = 2 if severity == "high" else 1
            s_dist = stairs_result.get("distance_mm") or 9999
            vib_candidates.append((rank * 10_000_000 - s_dist, 3, vib_mode))

            # ---- daily 樓梯偵測狀態面板 (方案 A) ----
            near = False
            if stairs_result["has_depth"]:
                # depth 觸發:severity high = 腳邊(近);medium = 前方(遠)
                near = severity == "high"
            draws.append(("STAIRS_PANEL", {
                "severity": severity,
                "type": stairs_result.get("type"),
                "has_yolo": stairs_result["has_yolo"],
                "has_depth": stairs_result["has_depth"],
                "near": near,
                "yolo_box": stairs_yolo_box,
                "distance_mm": stairs_result.get("distance_mm"),
                "conf": stairs_result.get("conf"),
                "dist_src": stairs_result.get("dist_src"),
                "edge_row": stairs_result.get("edge_row"),
                "msg": stairs_result.get("msg", ""),
            }))

        # 全 daily 一個 tick 內,震動只取「最高危」那一顆 (其餘危險物仍照常語音)。
        if vib_candidates:
            vib_candidates.sort(key=lambda c: c[0], reverse=True)
            _, best_motor, best_mode = vib_candidates[0]
            vibs.append((best_motor, best_mode))

        return draws, speeches, vibs

    # ============================================
    # Search mode
    # ============================================
    def _find_target(dets, w, depth_frame):
        """
        從偵測結果挑出要引導的目標框。
          - 尚未鎖定:在所有符合的候選中選「最近」(深度最小);
            深度全拿不到時退而選「最置中」(離畫面中心最近)。
          - 已鎖定:選離上次追蹤框中心最近的候選 → 在多個同類物間維持同一個,
            不會逐幀亂跳。若最近的也離太遠 (超過 TRACK_MAX_JUMP),視為原目標已離開,
            改回「選最近」重新鎖定。
        """
        nonlocal search_track_box
        draws = []
        t = (target_obj or "").lower()

        candidates = []
        for (x1, y1, x2, y2, cls, conf) in dets:
            cls_l = cls.lower()
            is_target = bool(t) and (t in cls_l or cls_l in t)
            label = f"{zh(cls)} {conf:.2f}"
            if is_target:
                candidates.append((x1, y1, x2, y2))
            else:
                draws.append(("BOX", x1, y1, x2, y2, label, (128, 128, 128)))

        if not candidates:
            return None, draws

        def box_cx_cy(b):
            return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0

        chosen = None
        if search_track_box is not None:
            # 已鎖定 → 選離上次框中心最近的候選
            tcx, tcy = box_cx_cy(search_track_box)
            chosen = min(candidates,
                         key=lambda b: (box_cx_cy(b)[0] - tcx) ** 2
                                     + (box_cx_cy(b)[1] - tcy) ** 2)
            ccx, ccy = box_cx_cy(chosen)
            jump = ((ccx - tcx) ** 2 + (ccy - tcy) ** 2) ** 0.5
            if jump > CONFIG.SEARCH_TRACK_MAX_JUMP * w:
                chosen = None   # 跳太遠 → 原目標可能已離開,重新選最近

        if chosen is None:
            # 未鎖定 (或剛失鎖) → 選最近;深度全無則選最置中
            scored = []
            for b in candidates:
                d = get_object_distance_mm(depth_frame, *b)
                scored.append((d, b))
            with_depth = [(d, b) for d, b in scored if d is not None]
            if with_depth:
                chosen = min(with_depth, key=lambda p: p[0])[1]
            else:
                chosen = min(candidates,
                             key=lambda b: abs(box_cx_cy(b)[0] - w / 2.0))

        search_track_box = chosen
        # 被選中的目標畫綠框,其餘符合的畫灰框 (跟非目標一樣,避免混淆)
        for b in candidates:
            color = (0, 255, 0) if b == chosen else (128, 128, 128)
            tag = "TARGET" if b == chosen else zh(target_obj)
            draws.append(("BOX", b[0], b[1], b[2], b[3], tag, color))

        return chosen, draws

    def run_search(frame, depth_frame, w, h, now):
        nonlocal search_state, search_last_seen, search_last_guidance
        nonlocal search_last_too_far
        nonlocal search_last_vert
        nonlocal last_known_target_cx, last_known_target_cy
        nonlocal last_known_target_depth, last_known_target_box
        nonlocal memory_start_time
        nonlocal search_last_prescan_hint, search_oak_seen
        nonlocal search_last_motor
        nonlocal reason_requested_target, reason_request_out

        dets = infer_detect(model_instance, frame, CONFIG.CONF_SEARCH)
        target_box, draws = _find_target(dets, w, depth_frame)
        speeches, vibs = [], []

        # ----- STAGE_FIND -----
        if search_state == "STAGE_FIND":
            if target_box:
                # OAK 實際看到目標 → 交棒給精確系統,不再給粗略引導
                search_oak_seen = True
                search_last_seen = now
                search_state = "STAGE_CENTER"
                print("🔍 search: FIND → CENTER")
            else:
                # OAK 還沒看到 → 依優先序給粗略引導
                hinted = False
                cooled = (now - search_last_prescan_hint
                          > CONFIG.SEARCH_PRESCAN_HINT_COOLDOWN)

                # (1) 背景 YOLO 記憶表:最準,有鐘向
                mem = _query_object_memory(target_obj, now)
                if not search_oak_seen and mem is not None and cooled:
                    speeches.append(
                        f"剛才在你的{mem['clock']}鐘方向看到{zh(target_obj)},"
                        f"請往{mem['zone']}邊轉頭找找")
                    vibs.append((zone_to_motor(mem["zone"]), "guide"))
                    search_last_prescan_hint = now
                    hinted = True

                # (2) VLM 預掃標籤 (備援)
                if (not hinted and CONFIG.SEARCH_PRESCAN_HINT_ENABLED
                        and not search_oak_seen and prescan_tags):
                    zone = _lookup_prescan_zone(target_obj, now)
                    if zone is not None and cooled:
                        speeches.append(
                            f"剛才在你的{zone}方有看到{zh(target_obj)},"
                            f"請往{zone}邊轉頭找找")
                        vibs.append((zone_to_motor(zone), "guide"))
                        search_last_prescan_hint = now
                        hinted = True

                # (3) 都查不到 → 用 VLM 語意推理
                if not hinted and not search_oak_seen:
                    rt = (target_obj or "").lower()
                    if reason_hint and reason_hint.get("target") == rt:
                        # 已有推理結果 → 用它的方位引導 (完整推理語句由 main 念過)
                        if cooled:
                            z = reason_hint.get("zone") or "正前"
                            speeches.append(f"請往{z}邊找找看")
                            vibs.append((zone_to_motor(z), "guide"))
                            search_last_prescan_hint = now
                            hinted = True
                    elif CONFIG.SEARCH_REASON_VLM and reason_requested_target != rt:
                        # 尚未對此目標請求過 → 發一次推理請求給 main
                        # ★ 必須檢查 SEARCH_REASON_VLM:這個開關預設 False
                        #   (搜尋改為只靠 YOLO + 查表),但舊版這裡沒檢查,
                        #   仍會持續產生請求送給 main → main 再丟給 VLM worker
                        #   → VLM 端才被丟掉。整條鏈路白跑,還佔住 VLM busy 狀態。
                        #   開關的兩端行為必須一致,不能只有一端關。
                        reason_request_out = rt
                        reason_requested_target = rt

                # (4) 全都沒有 → 退回即時引導
                if not hinted:
                    if now - search_last_guidance > CONFIG.SEARCH_LOST_TIMEOUT:
                        speeches.append(f"沒看到{zh(target_obj)},請左右轉頭")
                        search_last_guidance = now
            return draws, speeches, vibs

        # ----- STAGE_CENTER -----
        if search_state == "STAGE_CENTER":
            if target_box is None:
                if now - search_last_seen > CONFIG.SEARCH_LOST_TIMEOUT:
                    speeches.append("目標離開畫面,請左右轉頭")
                    search_state = "STAGE_FIND"
                    search_last_seen = now
                    clear_last_known()
                return draws, speeches, vibs

            search_last_seen = now
            target_cx = (target_box[0] + target_box[2]) // 2
            target_cy = (target_box[1] + target_box[3]) // 2
            target_dist = get_object_distance_mm(depth_frame, *target_box)

            if target_dist is not None:
                last_known_target_cx = target_cx
                last_known_target_cy = target_cy
                last_known_target_depth = target_dist
                last_known_target_box = target_box
                memory_start_time = None

            motor_id, _, clock = hysteretic_motor_7way(
                target_cx, w, search_last_motor,
                CONFIG.SEARCH_ZONE_HYS_MARGIN, CONFIG.SEARCH_ZONE_DEADZONE)
            search_last_motor = motor_id

            # 垂直對準判斷 (7 馬達無法表上下,用語音)。
            # cy 偏上緣 → 目標在使用者視野上方 → 頭太低,需抬頭;偏下緣反之。
            vert_ratio = target_cy / h
            vert_hint = None
            if vert_ratio < CONFIG.SEARCH_VERT_TOP_RATIO:
                vert_hint = "請把頭抬高一點"
            elif vert_ratio > CONFIG.SEARCH_VERT_BOTTOM_RATIO:
                vert_hint = "請把頭往下一點"

            # 水平已對準 (motor 3) 但垂直沒對準 → 優先給垂直提示,
            # 避免「水平 OK 就判定對準、但目標其實在畫面上下緣抓不到」。
            if motor_id == 3 and vert_hint is not None:
                if now - search_last_vert > CONFIG.SEARCH_VERT_INTERVAL:
                    speeches.append(vert_hint)
                    vibs.append((3, "point"))
                    search_last_vert = now
                return draws, speeches, vibs

            # 對準正前方 (12 點鐘方向)
            if motor_id == 3:
                # 對準了, 檢查距離是否進伸手範圍
                if target_dist and target_dist > CONFIG.SEARCH_REACH_DISTANCE_MM:
                    if now - search_last_too_far > CONFIG.SEARCH_TOO_FAR_INTERVAL:
                        cm = int(target_dist / 10)
                        speeches.append(
                            f"目標已對準, 但還在前方 {cm} 公分, 請先走近")
                        vibs.append((3, "guide"))
                        search_last_too_far = now
                    return draws, speeches, vibs

                speeches.append("對準了,請伸手")
                vibs.append((3, "arrive"))
                search_state = "STAGE_REACH"
                search_last_guidance = now
                lock_tracker.reset()
                hand_lock.reset()
                print("🔍 search: CENTER → REACH")
                return draws, speeches, vibs

            if now - search_last_guidance > CONFIG.SEARCH_GUIDE_INTERVAL:
                # 微偏左/右 (motor 2/4, 11點/1點) → 快對準了,用 point 輕點做微調
                if motor_id in (2, 4):
                    side = "左" if motor_id == 2 else "右"
                    speeches.append(f"快對準了,再往{side}一點點")
                    vibs.append((motor_id, "point"))
                else:
                    if target_dist:
                        cm = int(target_dist / 10)
                        speeches.append(f"目標在 {clock}鐘方向,約 {cm} 公分")
                    else:
                        speeches.append(f"目標在 {clock}鐘方向")
                    vibs.append((motor_id, "guide"))
                search_last_guidance = now
            return draws, speeches, vibs

        # ----- STAGE_REACH -----
        if search_state == "STAGE_REACH":
            using_last_known = False

            if target_box is None:
                if last_known_target_box is not None:
                    tracker_tmp = get_hand_tracker()
                    hand_check = tracker_tmp.detect(frame)

                    if hand_check is not None:
                        hand_depth_check = get_finger_distance_mm(
                            depth_frame, hand_check["all_landmarks"]
                        )
                        if (hand_depth_check is not None
                                and last_known_target_depth is not None
                                and abs(hand_depth_check - last_known_target_depth)
                                    < OCCLUSION_DEPTH_TOL_MM):
                            using_last_known = True
                            target_box = last_known_target_box
                            if memory_start_time is None:
                                memory_start_time = now

                if not using_last_known:
                    if now - search_last_seen > CONFIG.SEARCH_TARGET_LOST_REGRESS:
                        speeches.append("目標移動,請重新對準")
                        search_state = "STAGE_CENTER"
                        search_last_seen = now
                        search_last_guidance = now
                        lock_tracker.reset()
                        clear_last_known()
                    return draws, speeches, vibs

            # 用記憶超過 N 秒還沒抵達 → 兜底退出
            if using_last_known and memory_start_time is not None:
                if now - memory_start_time > CONFIG.SEARCH_MEMORY_TIMEOUT:
                    speeches.append("找不到目標,請把手收回重試")
                    search_state = "STAGE_CENTER"
                    search_last_seen = now
                    search_last_guidance = now
                    lock_tracker.reset()
                    clear_last_known()
                    return draws, speeches, vibs

            search_last_seen = now
            target_cx = (target_box[0] + target_box[2]) // 2
            target_cy = (target_box[1] + target_box[3]) // 2

            tracker = get_hand_tracker()
            hand = tracker.detect(frame)

            target_depth = get_object_distance_mm(depth_frame, *target_box)

            if not using_last_known and target_depth is not None:
                last_known_target_cx = target_cx
                last_known_target_cy = target_cy
                last_known_target_depth = target_depth
                last_known_target_box = target_box
                memory_start_time = None

            if hand is None:
                if now - search_last_guidance > CONFIG.SEARCH_HAND_PROMPT_INTERVAL:
                    if target_depth:
                        cm = int(target_depth / 10)
                        speeches.append(f"請伸出您的手,目標在前方 {cm} 公分")
                    else:
                        speeches.append("請伸出您的手")
                    vibs.append((3, "guide"))
                    search_last_guidance = now
                return draws, speeches, vibs

            is_own, _ = is_user_own_hand(hand["all_landmarks"], w, h)
            if is_own is False:
                if now - search_last_guidance > CONFIG.SEARCH_OWN_HAND_PROMPT_INTERVAL:
                    speeches.append("偵測到他人的手,請伸出您的手")
                    search_last_guidance = now
                return draws, speeches, vibs

            locked = hand_lock.update(hand["all_landmarks"], now)
            if locked is None:
                return draws, speeches, vibs

            tip = hand["tip"]
            palm = hand["palm"]
            draws.append(("BOX", tip[0]-12, tip[1]-12, tip[0]+12, tip[1]+12,
                          "TIP", (255, 0, 255)))
            draws.append(("LINE", palm[0], palm[1], tip[0], tip[1], (255, 0, 255)))

            target_color = (180, 180, 255) if using_last_known else (0, 255, 255)
            draws.append(("LINE", tip[0], tip[1], target_cx, target_cy, target_color))

            if using_last_known:
                draws.append(("BOX",
                    last_known_target_box[0], last_known_target_box[1],
                    last_known_target_box[2], last_known_target_box[3],
                    "OCCLUDED", (180, 180, 255)))

            hand_depth = get_finger_distance_mm(depth_frame, hand["all_landmarks"])

            if using_last_known:
                target_depth = last_known_target_depth

            dx = target_cx - tip[0]
            dy = target_cy - tip[1]
            pixel_offset = (dx**2 + dy**2) ** 0.5
            z_diff = (target_depth - hand_depth) if (target_depth and hand_depth) else None

            is_at = pixel_offset < CONFIG.SEARCH_LOCK_PIXEL_OK and (
                z_diff is None or abs(z_diff) < CONFIG.SEARCH_LOCK_Z_MM
            )

            if is_at:
                if lock_tracker.update(pixel_offset, now):
                    speeches.append("抵達了,任務完成")
                    speeches.append(SIG_RETURN_IDLE)
                    clear_last_known()
                    return draws, speeches, vibs
                if now - search_last_guidance > CONFIG.SEARCH_AT_TARGET_HOLD_INTERVAL:
                    # 對準了,但若深度上還差一截 → 報剩餘公分繼續引導伸手,
                    # 不要只說「保持不動」(使用者是要拿東西,不是停住)。
                    if z_diff is not None and z_diff > 30:
                        cm = int(z_diff / 10)
                        speeches.append(f"對準了,再往前 {cm} 公分")
                        vibs.append((3, "guide"))
                    elif z_diff is not None and z_diff < -30:
                        cm = int(abs(z_diff) / 10)
                        speeches.append(f"對準了,縮回 {cm} 公分")
                        vibs.append((3, "guide"))
                    else:
                        # 深度也幾乎貼合 → 真的快碰到,穩住即可
                        speeches.append("對準了,慢慢伸手碰")
                        vibs.append((3, "arrive"))
                    search_last_guidance = now
                return draws, speeches, vibs

            lock_tracker.reset()

            if now - search_last_guidance > CONFIG.SEARCH_GUIDE_INTERVAL:
                if pixel_offset > CONFIG.SEARCH_LOCK_PIXEL_OK * 1.5:
                    parts = []
                    target_motor = 3
                    if dx < -120:   parts.append("向左"); target_motor = 0
                    elif dx < -50:  parts.append("向左一點"); target_motor = 2
                    elif dx > 120:  parts.append("向右"); target_motor = 6
                    elif dx > 50:   parts.append("向右一點"); target_motor = 4
                    if dy < -50:    parts.append("向上")
                    elif dy > 50:   parts.append("向下")
                    if parts:
                        speeches.append(" ".join(parts))
                        vibs.append((target_motor, "guide"))
                else:
                    if z_diff is None:
                        speeches.append("快到了,慢慢前伸")
                        vibs.append((3, "guide"))
                    elif z_diff > 150:
                        cm = int(z_diff / 10)
                        speeches.append(f"再往前 {cm} 公分")
                        vibs.append((3, "guide"))
                    elif z_diff < -100:
                        cm = int(abs(z_diff) / 10)
                        speeches.append(f"手伸過頭,縮回 {cm} 公分")
                        vibs.append((3, "guide"))
                    else:
                        speeches.append("快碰到了")
                        vibs.append((3, "arrive"))
                search_last_guidance = now
            return draws, speeches, vibs

        return draws, speeches, vibs

    # ============================================
    # Traffic mode
    # ============================================
    def detect_traffic_light(frame, zoom=None):
        """單一影像紅綠燈:YOLO 低門檻「定位」→ 框內 HSV「判色」(主) → YOLO 類別 (備援)。
        低畫質下小綠人圖形先糊掉、發光體色度還在 → 網路降級為定位器,顏色由物理決定。
        zoom=(x0,x1,y0,y1) 比例:先裁該區餵 YOLO (等效變焦,號誌都在畫面上半)。
        回傳 (status, best_conf, draws);紅燈優先。"""
        if traffic_models is None:
            return None, 0.0, []

        H, W = frame.shape[:2]
        off_x = off_y = 0
        infer_img = frame
        if zoom is not None:
            zx0, zx1, zy0, zy1 = zoom
            off_x, off_y = int(W * zx0), int(H * zy0)
            infer_img = frame[off_y:int(H * zy1), off_x:int(W * zx1)]

        res = traffic_models["light"](
            infer_img, verbose=False,
            conf=getattr(CONFIG, "CONF_LIGHT_LOCALIZE", 0.25),
            device=YOLO_DEVICE
        )[0]
        status = None
        best_conf = 0.0
        draws = []
        for box in res.boxes:
            cls_name = traffic_models["light"].names[int(box.cls)]
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 += off_x; x2 += off_x; y1 += off_y; y2 += off_y

            # 主判:框內 HSV 色塊 (原生解析度 crop)
            color, score, _dbg = classify_light_color(frame, (x1, y1, x2, y2))
            if color is not None:
                eff_conf = 0.5 + min(0.5, score * 3)
                tag = f"{'紅' if color == 'red' else '綠'}(色) {score:.2f}"
            elif conf >= CONFIG.CONF_TRAFFIC_LIGHT:
                # 備援:YOLO 類別
                color = {"Red_light": "red", "Green_light": "green"}.get(cls_name)
                if color is None:
                    continue
                eff_conf = conf
                tag = f"{'紅' if color == 'red' else '綠'}(Y) {conf:.2f}"
            else:
                continue   # 判不出色 + 信心不足 → 不猜

            bcol = (0, 0, 255) if color == "red" else (0, 255, 0)
            draws.append(("BOX", x1, y1, x2, y2, tag, bcol))
            if color == "red" and (status != "red" or eff_conf > best_conf):
                status, best_conf = "red", eff_conf
            elif color == "green" and status != "red" and eff_conf > best_conf:
                status, best_conf = "green", eff_conf

        if status is None and getattr(CONFIG, "LIGHT_FULLFRAME_RED_ASSIST", False):
            if find_red_blobs(frame):
                status, best_conf = "red", 0.3
                draws.append(("BOX", 2, 20, 160, 42, "紅(全域色塊)", (0, 0, 255)))
        return status, best_conf, draws

    def detect_traffic_light_fused(oak_frame, wide_frame):
        """
        雙鏡頭紅綠燈融合:OAK 主鏡頭 + USB 廣角鏡頭各跑一次。
        OAK 有向下俯角(CAM_PITCH_DEG),遠處高掛的號誌容易超出視野上緣;
        USB 廣角水平擺放補這個盲區。融合規則(安全優先):
          - 任一鏡頭看到「紅燈」→ 一律採紅燈 (寧可誤停,不可誤闖)。
          - 都沒紅、有任一綠 → 取綠 (兩鏡頭都判綠才會是綠,符合保守)。
          - 都沒看到 → None。
        只回傳「OAK 視角」的 draws(疊在 debug 視窗的 OAK 畫面上);
        廣角的框座標屬於另一張圖,不混疊以免畫面錯亂。
        """
        oak_status, oak_conf, oak_draws = detect_traffic_light(oak_frame)

        wide_status, wide_conf = None, 0.0
        if wide_frame is not None:
            wide_status, wide_conf, _ = detect_traffic_light(
                wide_frame, zoom=getattr(CONFIG, "TRAFFIC_WIDE_ZOOM", None))

        # 安全優先:任一紅 → 紅
        if oak_status == "red" or wide_status == "red":
            fused = "red"
        elif oak_status == "green" or wide_status == "green":
            fused = "green"
        else:
            fused = None

        # debug 視窗標示廣角是否也有貢獻
        if wide_status is not None:
            src = {"red": "紅", "green": "綠"}.get(wide_status, "?")
            oak_draws.append(("BOX", 2, 44, 150, 70,
                              f"廣角:{src} {wide_conf:.2f}", (255, 200, 0)))
        return fused, oak_draws

    def detect_crosswalk(frame, w, h):
        if traffic_models is None:
            return None, []
        res = traffic_models["seg"](
            frame, verbose=False, conf=CONFIG.CONF_CROSSWALK,
            device=YOLO_DEVICE
        )[0]
        if res.masks is None or len(res.masks.data) == 0:
            return None, []

        best_idx, best_area = -1, 0
        for i, mask in enumerate(res.masks.data):
            area = float(mask.sum())
            if area > best_area:
                best_area, best_idx = area, i
        if best_idx < 0:
            return None, []

        m = res.masks.data[best_idx].cpu().numpy()
        m = cv2.resize(m, (w, h))
        crosswalk_mask = (m > 0.5).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        crosswalk_mask = cv2.morphologyEx(crosswalk_mask, cv2.MORPH_CLOSE, kernel)

        draws = []
        contours, _ = cv2.findContours(
            crosswalk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(biggest) > 500:
                poly = biggest.reshape(-1, 2).tolist()
                draws.append(("POLY", poly, (0, 255, 255)))
        return crosswalk_mask, draws

    def run_traffic(frame, depth_frame, w, h, now, wide_frame=None):
        nonlocal traffic_state, last_light_say, last_crossing_guide
        nonlocal last_red_warn, arrived_counter, crossing_started_at
        nonlocal stable_light_status   # light_streak 只就地改 dict 內容,不需 nonlocal

        if traffic_models is None:
            return [], [], []

        # 紅綠燈:OAK + USB 廣角雙鏡頭融合 (廣角補 OAK 俯角造成的上方盲區)
        raw_light_status, light_draws = detect_traffic_light_fused(frame, wide_frame)
        # 斑馬線:只用 OAK (有向下俯角,看地面斑馬線角度/距離才正確)
        crosswalk_mask, seg_draws = detect_crosswalk(frame, w, h)
        # Hough 條紋斜率法需要原始灰階 (在 mask 範圍內抓條紋邊緣);算一次共用
        crosswalk_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        draws = light_draws + seg_draws
        speeches, vibs = [], []

        # ---- 燈號穩定化: 連續 N 幀才確認狀態切換 ----
        if raw_light_status == "green":
            light_streak["green"] += 1
            light_streak["red"] = 0
            light_streak["none"] = 0
        elif raw_light_status == "red":
            light_streak["red"] += 1
            light_streak["green"] = 0
            light_streak["none"] = 0
        else:
            light_streak["none"] += 1
            light_streak["green"] = 0
            light_streak["red"] = 0

        if light_streak["red"] >= CONFIG.LIGHT_CONFIRM_RED:
            stable_light_status = "red"
        elif light_streak["green"] >= CONFIG.LIGHT_CONFIRM_GREEN:
            stable_light_status = "green"
        elif light_streak["none"] >= CONFIG.LIGHT_CONFIRM_NONE:
            stable_light_status = None

        light_status = stable_light_status

        # ---- 共用: 評估是否能進入 CROSSING (先角度後偏移) ----
        def evaluate_crossing_ready(offset, angle, light_word, extra):
            if light_word == "無號誌":
                severe_threshold = CONFIG.CROSSING_OFFSET_OK * 2
            else:
                severe_threshold = CONFIG.CROSSING_OFFSET_OK * 3

            if offset is None:
                return False, f"{light_word},但看不到斑馬線,請小心確認方向", True

            # 先處理面向歪斜 (轉頭)
            if angle is not None and abs(angle) > CONFIG.CROSSING_ANGLE_OK:
                turn_side = "右" if angle > 0 else "左"
                msg = f"{light_word},您身體朝向歪斜,請向{turn_side}轉頭面向斑馬線"
                return False, msg, True

            # 角度 OK 後再看偏移 (平移)
            if abs(offset) < CONFIG.CROSSING_OFFSET_OK:
                return True, f"{light_word},您已對準斑馬線,{extra}", False

            if abs(offset) < severe_threshold:
                side = "左" if offset > 0 else "右"
                opp = "右" if side == "左" else "左"
                return True, f"{light_word},您稍微偏{side},請邊走邊往{opp}修正", False

            side = "左" if offset > 0 else "右"
            opp = "右" if side == "左" else "左"
            if light_word == "綠燈":
                msg = f"綠燈了!但您嚴重偏{side},請快速平移到斑馬線中央"
            else:
                msg = f"無號誌,您嚴重偏{side},請先向{opp}平移對準斑馬線再通過"
            return False, msg, True

        # ============================================
        # WAITING 階段
        # ============================================
        if traffic_state == "WAITING":
            crosswalk_offset = None
            crosswalk_angle = None
            if crosswalk_mask is not None:
                direction = crosswalk_stab.update(
                    get_crosswalk_direction_hough(crosswalk_mask, h, w, crosswalk_gray))
                if direction and direction != "almost_arrived":
                    crosswalk_offset = direction["offset"]
                    crosswalk_angle = direction["angle"]
                    bx, by = direction["bottom"]
                    tx, ty = direction["top"]
                    draws.append(("LINE", bx, by, tx, ty, (0, 255, 255)))

            if light_status == "green":
                ready, msg, severe = evaluate_crossing_ready(
                    crosswalk_offset, crosswalk_angle, "綠燈", "可以過馬路"
                )
                if ready:
                    speeches.append(msg)
                    traffic_state = "CROSSING"
                    last_crossing_guide = now
                    crossing_started_at = now
                    arrived_counter = 0
                    print("🚦 WAITING → CROSSING (綠燈)")
                else:
                    if now - last_light_say > CONFIG.TRAFFIC_LIGHT_INTERVAL:
                        speeches.append(msg)
                        if severe:
                            vibs.append((3, "warn"))
                        last_light_say = now
                return draws, speeches, vibs

            if light_status is None and crosswalk_mask is not None:
                ready, msg, severe = evaluate_crossing_ready(
                    crosswalk_offset, crosswalk_angle, "無號誌", "請確認車流後通過"
                )
                if ready:
                    speeches.append(msg)
                    traffic_state = "CROSSING"
                    last_crossing_guide = now
                    crossing_started_at = now
                    arrived_counter = 0
                    print("🚦 WAITING → CROSSING (無號誌)")
                else:
                    if now - last_light_say > CONFIG.TRAFFIC_LIGHT_INTERVAL:
                        speeches.append(msg)
                        if severe:
                            vibs.append((3, "warn"))
                        last_light_say = now
                return draws, speeches, vibs

            if now - last_light_say <= CONFIG.TRAFFIC_LIGHT_INTERVAL:
                return draws, speeches, vibs

            if light_status == "red":
                # 紅燈時也持續引導 (先轉頭, 再平移)
                if crosswalk_offset is not None:
                    if (crosswalk_angle is not None
                            and abs(crosswalk_angle) > CONFIG.CROSSING_ANGLE_OK):
                        turn_side = "右" if crosswalk_angle > 0 else "左"
                        speeches.append(
                            f"紅燈等待中,您身體朝向歪斜,請向{turn_side}轉頭面向斑馬線")
                    elif abs(crosswalk_offset) < CONFIG.CROSSING_OFFSET_OK:
                        speeches.append("紅燈等待中,您已對準斑馬線")
                    elif crosswalk_offset < 0:
                        speeches.append("紅燈等待中,您站太右,請左移")
                    else:
                        speeches.append("紅燈等待中,您站太左,請右移")
                else:
                    speeches.append("紅燈,請等待")
                last_light_say = now

            elif light_status is None:
                if now - last_light_say > CONFIG.NO_LIGHT_TIMEOUT:
                    speeches.append("找不到號誌也找不到斑馬線,請左右轉頭尋找")
                    last_light_say = now

            return draws, speeches, vibs

        # ============================================
        # CROSSING 階段
        # ============================================
        if traffic_state == "CROSSING":
            if light_status == "red":
                if now - last_red_warn > CONFIG.RED_WARN_INTERVAL:
                    speeches.append("注意!號誌變紅,請快速通過")
                    vibs.append((3, "warn"))
                    last_red_warn = now

            if crosswalk_mask is None:
                arrived_counter += 1
                duration = now - crossing_started_at
                if (arrived_counter >= CONFIG.CROSSING_ARRIVED_FRAMES
                        and duration >= CONFIG.CROSSING_MIN_DURATION):
                    speeches.append("已通過馬路,任務完成")
                    speeches.append(SIG_RETURN_IDLE)
                    traffic_state = "WAITING"
                    arrived_counter = 0
                    print("🚦 CROSSING → ARRIVED → idle")
                    return draws, speeches, vibs
                if now - last_crossing_guide > CONFIG.CROSSING_NO_MASK_HINT_INTERVAL:
                    speeches.append("看不清楚斑馬線,請慢慢走")
                    last_crossing_guide = now
                return draws, speeches, vibs

            arrived_counter = 0
            direction = crosswalk_stab.update(
                get_crosswalk_direction_hough(crosswalk_mask, h, w, crosswalk_gray))

            if direction == "almost_arrived":
                if now - last_crossing_guide > CONFIG.CROSSING_NO_MASK_HINT_INTERVAL:
                    speeches.append("即將到達對岸")
                    last_crossing_guide = now
                return draws, speeches, vibs

            if direction is None:
                return draws, speeches, vibs

            bx, by = direction["bottom"]
            tx, ty = direction["top"]
            draws.append(("LINE", bx, by, tx, ty, (0, 255, 255)))

            if now - last_crossing_guide > CONFIG.CROSSING_GUIDE_INTERVAL:
                offset = direction["offset"]
                angle = direction["angle"]

                # 先處理面向歪斜 (轉頭)
                if abs(angle) > CONFIG.CROSSING_ANGLE_OK:
                    if angle > 0:
                        speeches.append("請向右轉頭")
                        vibs.append((6, "guide"))
                    else:
                        speeches.append("請向左轉頭")
                        vibs.append((0, "guide"))
                # 角度 OK 後看偏移 (平移)
                elif abs(offset) < CONFIG.CROSSING_OFFSET_OK:
                    speeches.append("直走")
                    vibs.append((3, "guide"))
                elif offset < 0:
                    speeches.append("偏左修正")
                    vibs.append((2, "point"))
                else:
                    speeches.append("偏右修正")
                    vibs.append((4, "point"))
                last_crossing_guide = now
            return draws, speeches, vibs

        return draws, speeches, vibs

    # ============================================
    # 主迴圈
    # ============================================
    print("🚀 [AI 核心] 準備就緒")
    while True:
        try:
            while True:
                cmd = command_queue.get_nowait()
                # 標籤快取更新 (與 mode 切換可同時或獨立到來)
                if "prescan_tags" in cmd:
                    prescan_tags = cmd["prescan_tags"] or {}
                # 背景全物件掃描:對廣角幀更新記憶表
                if "wide_scan_frame" in cmd:
                    _bg_scan_update(cmd["wide_scan_frame"], time.time())
                # VLM 語意推理結果回灌
                if "reason_hint" in cmd:
                    reason_hint = cmd["reason_hint"]
                # UI 手動輸入身高 → 相機離地高度(手動 = 校驗基準)
                if "cam_height_mm" in cmd:
                    stairs_fusion.set_cam_height(cmd["cam_height_mm"], manual=True)
                    depth_detector.set_cam_height(stairs_fusion.cam_height_mm)
                # 換人配戴 → 清掉手動基準,讓自動估計重新接管
                if cmd.get("reset_height"):
                    stairs_fusion.reset_height_calibration()
                    depth_detector.set_cam_height(stairs_fusion.cam_height_mm)
                # VLM 執行期間暫停 YOLO,讓出 GPU/VRAM;VLM 完成後恢復
                if "yolo_pause" in cmd:
                    yolo_paused = bool(cmd["yolo_pause"])
                    print(f"{'⏸️ [YOLO] 暫停 (VLM 執行中,讓出 GPU)' if yolo_paused else '▶️ [YOLO] 恢復'}")
                    continue
                if ("mode" not in cmd and "target" not in cmd
                        and not cmd.get("reset_timers")
                        and "cam_height_mm" not in cmd
                        and not cmd.get("reset_height")
                        and "prescan_tags" not in cmd
                        and "wide_scan_frame" not in cmd
                        and "reason_hint" not in cmd):
                    continue
                if ("mode" not in cmd and "target" not in cmd
                        and not cmd.get("reset_timers")):
                    # 只是背景資料更新,不動模式
                    continue
                new_mode = cmd.get("mode", mode)
                if new_mode != mode and new_mode != "idle":
                    switch_model(new_mode)
                mode = new_mode
                target_obj = cmd.get("target", target_obj)
                if cmd.get("reset_timers"):
                    reset_all_timers(time.time())
        except queue.Empty:
            pass

        try:
            data = in_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if data is None:
            break

        # ★ VLM 執行中的處理方式(這裡曾經有一個嚴重的安全問題)
        #   舊版:`if yolo_paused: continue` —— 整幀跳過。
        #   後果:VLM 一次推論可能跑好幾秒,這段期間避障、樓梯、深度落差、
        #         紅綠燈全部停擺,而使用者完全不知道系統「暫時瞎了」。
        #         使用者按下語意描述的當下往往正站在陌生環境,恰恰是最需要
        #         避障的時候。
        #   現在:不跳過整幀。只在 infer_detect() / _bg_scan_update() 內部
        #         跳過 YOLO 的 GPU 呼叫;深度型安全邏輯(DepthAnomalyDetector、
        #         stairs_fusion 的地面模型)完全不受影響,照常運作。
        #         Traffic 模式的燈號/斑馬線 YOLO 也刻意不受影響——過馬路
        #         中途讓出 GPU 是不可接受的。
        frame = data["frame"]
        frame_id = data["frame_id"]
        depth_frame = data.get("depth_frame")
        wide_frame = data.get("wide_frame")   # traffic 雙鏡頭用 (其他模式為 None)
        w, h = data["w"], data["h"]
        # IMU 即時俯角:跨 process 只能走 queue,由 main.py 每幀塞進 frame dict。
        # None = 沒有 IMU 或資料過期 → 下游自動退回 CONFIG.CAM_PITCH_DEG。
        pitch_deg = data.get("pitch_deg")
        depth_detector.set_pitch(pitch_deg)
        stairs_fusion.set_pitch(pitch_deg)
        now = time.time()

        draws, speeches, vibs = [], [], []

        if mode == "idle":
            time.sleep(0.05)
        elif mode == "daily" and model_instance is not None:
            draws, speeches, vibs = run_daily(frame, depth_frame, w, h, now)
        elif mode == "search" and model_instance is not None:
            draws, speeches, vibs = run_search(frame, depth_frame, w, h, now)
        elif mode == "traffic" and traffic_models is not None:
            draws, speeches, vibs = run_traffic(frame, depth_frame, w, h, now,
                                                wide_frame=wide_frame)

        if out_queue.full():
            try: out_queue.get_nowait()
            except queue.Empty: pass
        try:
            out_queue.put_nowait({
                "frame_id": frame_id,
                "draws": draws,
                "speech": speeches,
                "vib": vibs,
                "reason_request": reason_request_out,   # 需要 VLM 推理的目標 (或 None)
                "dets": last_dets,                      # 給第三代偽VLM 場景投票
            })
        except queue.Full:
            pass
        reason_request_out = None   # 送出後清掉,避免重複請求