"""
樓梯偵測 v20 — 多階量測:階數 + 每階階高 + 水平走行距離
========================================================
設計:
  ① 有沒有樓梯 = YOLO。沒看到 → 不報樓梯 (但落差仍可由②'保底)。
  ② 幾階 / 多高 / 多遠 = 「物理地面模型 + 逐階掃描」:
       平地上任一列的深度有封閉解:depth(r) = k / sin(ray_pitch(r))
       k = 相機離地高度,由 YOLO 框下方的平地帶最小平方反推
         (→ 順帶自我校正配戴高度/俯角偏差,不完全依賴使用者輸入的身高)
       殘差投影成垂直高度:h(r) = (預期深度 − 實測深度) × sin(俯角)
         平地 ≈ 0,第一階 ≈ 階高,第二階 ≈ 2×階高… 呈階梯狀
       由框底往框頂掃,每個「高度明顯跳升且連續 N 列維持」的位置 = 一個階緣
       相鄰階緣的高度差 = 該階的階高

       ★ 為什麼不用直線擬合 (v19 的做法):
         地面深度隨列號是 1/sin 的非線性關係。直線擬合外推到樓梯區有
         100~400mm 系統性偏差,平地區就會誤觸發、階高符號都可能算錯。
         改物理模型後殘差 σ < 1mm,合成測試階高誤差 ≈ 0mm。

       ★ 為什麼用「相對式」(全程沿用同一個地面模型),不是每階重新擬合:
         文獻 (Perez-Yus et al. 2017; Zhao et al. 2018) 的多階量測以 3D 點雲
         平面分割 + IMU 姿態校正為基礎,每個踏面都有足夠點雲。本系統
         640x480 @ 30° 俯角,第二階踏面僅約 5~10 列、第三階更少,重新擬合
         樣本量不足。相對式每一階都拿得到數字,代價是誤差隨階數累積
         → 只有前 STAIRS_RELIABLE_N 階標為可信,平均階高只取這幾階。

  ②' 突發低落差 = DepthAnomalyDetector 接回,但只採 cliff (下行/路緣落差)。
  ③ 近遠分級 = 到第一階的水平距離 < STAIRS_NEAR_MM → high;否則 medium。

停止掃描的條件 (任一成立):
  box_top             掃到 YOLO 框頂
  depth_lost          連續 STAIRS_DEAD_ROWS 列無有效深度 (太遠/遮擋/反光)
  direction_flip      方向翻轉 (一階上一階下 → 擬合已崩)
  max_steps           達 STAIRS_MAX_STEPS 硬上限
  slant_out_of_range  距離跑到合理範圍外

--------------------------------------------------------------------
相機俯角:改吃 IMU 即時值
  ray_pitch 過去是 CONFIG.CAM_PITCH_DEG 固定常數,走路時頭一晃,整個
  地面模型就跟著失準。有 IMU 後由 ai_worker 每幀呼叫 set_pitch();
  沒有 IMU / 資料過期時 set_pitch(None),自動退回 CONFIG 常數,
  行為與加 IMU 之前完全相同。

相機高度:手動輸入 = 校驗基準(這是刻意的優先權設計)
  地面模型擬合出的 k 就是「相機離地高度」,理論上可以完全取代使用者
  手動輸入的身高。但**一旦使用者手動設定過,自動估計就只做比對警告,
  不再靜默覆蓋**。理由:自動值與手動值差距大,最可能的原因是
  IMU_PITCH_OFFSET_DEG(機構夾角)沒校準好——這種情況下靜默採用自動值,
  等於把一個校準錯誤變成看不見的系統性偏差,而且使用者永遠不會發現。
  寧可出聲說「量到的高度和你設定的差很多」,讓人去查。
  `reset_height_calibration()` 提供換人配戴時重新交還給自動估計的入口。

★ 精度誠實聲明 (答辯用):
  文獻實測 (Wang et al., Sensors 2023, RGB-D + IMU) 指出階高估計誤差本就
  偏大 — 上行普遍低估、下行不穩定高估,主因是深度相機在階緣與反光面的
  點雲黑洞、以及姿態角誤差。本系統雖已加入 IMU 即時俯角,但機構夾角
  仍需實測校準,實機誤差必然大於合成測試。階高數值定位為「大致參考」,
  不宜宣稱精確。

語音:「樓梯 前方 2.3公尺 上行 3階 每階約16公分」
      「樓梯 就在腳邊 下行 4階 每階約18公分 注意」
      單階時退回:「樓梯 前方 1.2公尺 上行 階高約16公分」

回傳欄位契約(v19 → v20 只增不改):
  v19 既有:has_yolo / has_depth / type / severity / distance_mm / dist_src /
            edge_row / near / yolo_box / conf / msg
  v20 新增:steps(每階明細)/ cam_h_fit(反推的相機高度)/ stop_reason
  ★ 舊欄位一個都不能改名或拿掉——proc_ai_worker 用 `[]` 直接取 has_depth /
    has_yolo,少一個就是執行期 KeyError,而且只有在真的偵測到樓梯時才會炸。

ai_worker 接線:
    depth_anomaly = depth_detector.update(depth_frame, now) if depth_frame is not None else None
    stairs_result = stairs_fusion.update(dets, depth_frame, now, anomaly=depth_anomaly)
"""
import math
import time

from shared_config import CONFIG
from shared_utils import get_object_distance_mm

import numpy as np


def _cfg(name, default):
    return getattr(CONFIG, name, default)


def _dist_phrase(mm):
    if not mm or mm <= 0:
        return None
    if mm >= 1000:
        return f"{mm/1000:.1f}公尺"
    return f"{int(round(mm/10.0))}公分"


class StairsFusion:
    def __init__(self):
        self._last_say = 0.0
        self._last_cliff_say = 0.0
        # debug 給面板
        self.last_edge_row = None     # 第一階斷點列 (畫線用)
        self.last_dist_src = "-"      # steps / near5 / none
        self.last_steps = []          # 每階詳細 (面板畫多條線用)
        self.last_stop_reason = "-"

        # ---- 相機離地高度 ----
        self.cam_height_mm = float(_cfg("CAM_HEIGHT_MM", 1750.0))
        self._manual_height = False   # 使用者是否手動設定過 (= 校驗基準)
        self._auto_height = None      # 自動估計的 EMA
        self._auto_n = 0
        self._last_height_warn = 0.0
        self._height_warnings = []

        # ---- 相機俯角 (IMU) ----
        self._pitch_deg = None

    # ============================================================
    # 姿態 / 高度注入
    # ============================================================
    def set_pitch(self, pitch_deg):
        """注入 IMU 即時俯角;None = 沒資料 → 退回 CONFIG.CAM_PITCH_DEG。"""
        self._pitch_deg = float(pitch_deg) if pitch_deg is not None else None

    @property
    def pitch_deg(self):
        return self._pitch_deg if self._pitch_deg is not None else CONFIG.CAM_PITCH_DEG

    def set_cam_height(self, mm, manual=False):
        """
        更新相機離地高度 (mm)。
        manual=True  → 使用者從 UI 輸入,設為校驗基準,之後自動估計不再覆蓋。
        manual=False → 自動估計,只有在使用者從未手動設定過時才會被採用。
        """
        try:
            mm = float(mm)
        except (TypeError, ValueError):
            return
        lo = float(_cfg("AUTO_HEIGHT_MIN_MM", 1200.0))
        hi = float(_cfg("AUTO_HEIGHT_MAX_MM", 2100.0))
        if not (lo <= mm <= hi):
            return                                 # 合理範圍防呆
        if manual:
            self.cam_height_mm = mm
            self._manual_height = True
            print(f"📏 [Stairs] 相機高度(手動基準): {mm:.0f}mm")
        elif not self._manual_height:
            self.cam_height_mm = mm

    def reset_height_calibration(self):
        """換人配戴:清掉手動基準,交還給自動估計。"""
        self._manual_height = False
        self._auto_height = None
        self._auto_n = 0
        self.cam_height_mm = float(_cfg("CAM_HEIGHT_MM", 1750.0))
        print("📏 [Stairs] 身高校準已重設,改由自動估計接管")

    def _note_auto_height(self, k, now=None):
        """
        把這次地面擬合得到的 k 併入自動身高估計。
        單次擬合會被地面雜訊拉走,所以走 EMA + 最少樣本數才採信。
        """
        now = time.time() if now is None else now
        if not _cfg("AUTO_HEIGHT_ENABLED", True):
            return
        lo = float(_cfg("AUTO_HEIGHT_MIN_MM", 1200.0))
        hi = float(_cfg("AUTO_HEIGHT_MAX_MM", 2100.0))
        if not (lo <= k <= hi):
            return
        a = float(_cfg("AUTO_HEIGHT_EMA_ALPHA", 0.15))
        self._auto_height = k if self._auto_height is None else (1 - a) * self._auto_height + a * k
        self._auto_n += 1
        if self._auto_n < int(_cfg("AUTO_HEIGHT_MIN_SAMPLES", 5)):
            return

        if not self._manual_height:
            self.cam_height_mm = self._auto_height     # 沒有手動基準 → 自動接管
            return

        # 有手動基準 → 只比對、不覆蓋
        diff = abs(self._auto_height - self.cam_height_mm)
        if diff < float(_cfg("AUTO_HEIGHT_WARN_DIFF_MM", 150.0)):
            return
        if now - self._last_height_warn < float(_cfg("AUTO_HEIGHT_WARN_COOLDOWN", 60.0)):
            return
        self._last_height_warn = now
        self._height_warnings.append(
            f"量到的配戴高度約{self._auto_height/10:.0f}公分,"
            f"和設定的{self.cam_height_mm/10:.0f}公分差距較大,建議重新校準")
        print(f"⚠️ [Stairs] 自動身高 {self._auto_height:.0f}mm vs 手動 "
              f"{self.cam_height_mm:.0f}mm,差 {diff:.0f}mm "
              f"(常見原因:IMU_PITCH_OFFSET_DEG 未校準)")

    def pop_height_warnings(self):
        w, self._height_warnings = self._height_warnings, []
        return w

    # ============================================================
    # 幾何
    # ============================================================
    def _ray_sin(self, row, img_h):
        """該列射線相對水平面的俯角 sin (同 DepthAnomalyDetector._row_sin 定義)。"""
        v = row / max(1, img_h - 1)
        ang = self.pitch_deg + (v - 0.5) * CONFIG.CAM_VFOV_DEG
        ang = min(max(ang, 0.5), 89.5)
        return math.sin(math.radians(ang))

    def _slant_to_horizontal(self, d_mm):
        """
        斜距 d → 水平走行距離 L = sqrt(d² − H²)。
        d 是深度相機到階緣的直線距離;使用者真正要走的是水平分量。
        d <= H (量到的點幾乎就在腳下,或雜訊) → 回 0 (視為已到)。
        """
        if d_mm is None:
            return None
        H = self.cam_height_mm
        if d_mm <= H:
            return 0.0
        return math.sqrt(d_mm * d_mm - H * H)

    # ============================================================
    # 地面斷點法:多階偵測
    # ============================================================
    def _scan_steps(self, depth_frame, box):
        """
        從框底往框頂掃,找出「所有」階緣斷點,回傳每一階的距離與高度。

        回傳 dict 或 None:
          steps: [{"edge_row", "slant_mm", "cum_h_mm", "step_h_mm", "reliable"}, ...]
                 由近到遠 (第一階在前)
          cam_h_fit: 由平地帶反推的相機離地高度 (mm),可與使用者輸入對照
          stop_reason: 停止掃描的原因
        """
        if depth_frame is None:
            return None
        H, W = depth_frame.shape[:2]
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1)); x2 = min(W, int(x2))
        y1 = max(0, int(y1)); y2 = min(H, int(y2))
        if x2 - x1 < 16:
            return None

        def row_median(r):
            seg = depth_frame[r, x1:x2]
            v = seg[(seg > 200) & (seg < 8000)]
            return float(np.median(v)) if v.size >= 8 else None

        # ---- 平地帶:框底 → 畫面底 (使用者腳前),擬合相機高度 k ----
        gy0 = min(max(int(y2), 0), H)
        rows, deps = [], []
        for r in range(gy0, H):
            m = row_median(r)
            if m is not None:
                rows.append(r); deps.append(m)
        if len(rows) < _cfg("STAIRS_GROUND_MIN_ROWS", 10):
            return None                     # 樓梯已頂到腳邊,沒尺可用

        model = np.array([self._ray_sin(r, H) for r in rows], dtype=np.float32)
        deps_a = np.asarray(deps, dtype=np.float32)
        inv = 1.0 / np.maximum(model, 1e-6)          # 1/sin(俯角)
        # 最小平方解 k:min Σ(d − k·inv)²  →  k = Σ(d·inv)/Σ(inv²)
        denom = float(np.sum(inv * inv))
        if denom < 1e-6:
            return None
        k = float(np.sum(deps_a * inv) / denom)
        if not (800.0 <= k <= 2500.0):      # 反推的相機高度不合理 → 擬合壞掉
            return None

        pred_ground = k * inv
        sigma = float(np.std(deps_a - pred_ground))
        self._note_auto_height(k)

        def height_at(r, m):
            """
            該列實測深度 m 相對地面模型的垂直高度。
            resid = 預期 − 實測:實測較近(較小)→ resid 為正 → 隆起(上行)。
            乘 sin(該列射線俯角) 把斜距殘差投影成垂直高度。
            """
            s = self._ray_sin(r, H)
            return (k / max(s, 1e-6) - m) * s

        # ---- 由框底往框頂逐列掃描 ----
        min_jump = max(float(_cfg("STAIRS_STEP_MIN_MM", 90)), 3.0 * sigma)
        need_rows = int(_cfg("STAIRS_STEP_ROWS", 3))
        dead_limit = int(_cfg("STAIRS_DEAD_ROWS", 12))
        max_steps = int(_cfg("STAIRS_MAX_STEPS", 6))
        h_min = float(_cfg("STAIRS_STEP_H_MIN_MM", 90))
        h_max = float(_cfg("STAIRS_STEP_H_MAX_MM", 300))

        steps = []
        base_h = 0.0            # 目前已確認階緣的累積高度
        dead = 0
        streak = 0
        cand_row = None
        cand_h = None
        stop = "box_top"
        direction = 0           # +1 上行 / -1 下行,用來偵測方向翻轉

        for r in range(gy0 - 1, y1 - 1, -1):
            m = row_median(r)
            if m is None:
                dead += 1
                if dead > dead_limit:
                    stop = "depth_lost"
                    break
                continue
            dead = 0
            if not (200.0 < m < 8000.0):
                stop = "slant_out_of_range"
                break

            h = height_at(r, m)
            if abs(h - base_h) >= min_jump:
                # 候選階緣:需連續 need_rows 列維持同方向的跳升
                if cand_row is None or (h - base_h) * (cand_h - base_h) <= 0:
                    cand_row, cand_h, streak = r, h, 1
                else:
                    streak += 1
                    cand_h = (cand_h * (streak - 1) + h) / streak   # 平均掉單列雜訊
                if streak >= need_rows:
                    step_h = cand_h - base_h
                    d = 1 if step_h > 0 else -1
                    if direction != 0 and d != direction:
                        stop = "direction_flip"
                        break
                    if not (h_min <= abs(step_h) <= h_max):
                        # 高度不像一階 (雜訊或反光洞) → 不認,重新找
                        cand_row = cand_h = None
                        streak = 0
                        continue
                    direction = d
                    steps.append({
                        "edge_row": int(cand_row),
                        "slant_mm": float(m),
                        "cum_h_mm": float(cand_h),
                        "step_h_mm": float(step_h),
                        "reliable": len(steps) < int(_cfg("STAIRS_RELIABLE_N", 2)),
                    })
                    base_h = cand_h
                    cand_row = cand_h = None
                    streak = 0
                    if len(steps) >= max_steps:
                        stop = "max_steps"
                        break
            else:
                cand_row = cand_h = None
                streak = 0

        if not steps:
            return None
        return {"steps": steps, "cam_h_fit": k, "sigma": sigma,
                "stop_reason": stop, "direction": direction}

    # ============================================================
    # 主流程
    # ============================================================
    def update(self, yolo_dets, depth_frame, now, anomaly=None):
        """
        yolo_dets  : YOLO 偵測 [(x1,y1,x2,y2,cls,conf), ...]
        depth_frame: OAK 深度圖 (mm, 與 YOLO 同座標) 或 None
        anomaly    : DepthAnomalyDetector.update() 的回傳 (dict/None)。
                     只採 type=='cliff';YOLO 樓梯在場時讓位。
        回傳 dict 或 None。
        """
        # ① 視覺:找信心最高的樓梯框
        best = None
        for det in yolo_dets or []:
            x1, y1, x2, y2, cls, conf = det
            if cls in CONFIG.STAIRS_YOLO_CLASSES:
                if best is None or conf > best[5]:
                    best = det

        # ②' 沒有 YOLO 樓梯 → 落差保底 (cliff-only;obstacle 一律忽略)
        if best is None:
            self.last_steps = []
            self.last_edge_row = None
            self.last_dist_src = "none"
            if anomaly and anomaly.get("type") == "cliff":
                if now - self._last_cliff_say > CONFIG.DEPTH_WARN_COOLDOWN:
                    self._last_cliff_say = now
                    return {
                        # ★ 欄位契約必須與 v19 完全一致:proc_ai_worker 用
                        #   stairs_result["has_depth"] 這種 [] 取值,少一個 key
                        #   就是執行期 KeyError,而且只在真的偵測到樓梯時才炸,
                        #   平常測試根本踩不到。v20 新欄位一律用「加的」。
                        "has_yolo": False,
                        "has_depth": True,
                        "type": "cliff",
                        "severity": "high",
                        "distance_mm": anomaly.get("front_mm"),
                        "dist_src": "anomaly",
                        "edge_row": None,
                        "near": True,
                        "yolo_box": None,
                        "box": None,
                        "conf": None,
                        "msg": "前方落差 注意",
                        "steps": [],
                        "cam_h_fit": None,
                        "stop_reason": "cliff",
                    }
            return None

        if now - self._last_say < CONFIG.STAIRS_FUSION_INTERVAL:
            return None

        box = (int(best[0]), int(best[1]), int(best[2]), int(best[3]))
        scan = self._scan_steps(depth_frame, box)

        # ---- 距離 ----
        if scan:
            steps = scan["steps"]
            self.last_steps = steps
            self.last_edge_row = steps[0]["edge_row"]
            self.last_dist_src = "steps"
            self.last_stop_reason = scan["stop_reason"]
            horiz = self._slant_to_horizontal(steps[0]["slant_mm"])
        else:
            steps = []
            self.last_steps = []
            self.last_edge_row = None
            self.last_dist_src = "near5"
            self.last_stop_reason = "no_scan"
            # 退路:保守取近 (整框最近 5%),只給距離、不給階數
            d = get_object_distance_mm(depth_frame, *box, conservative=True)
            horiz = self._slant_to_horizontal(d)

        self._last_say = now
        near = horiz is not None and horiz < CONFIG.STAIRS_NEAR_MM

        # ---- 語音組裝 (電報式) ----
        parts = ["樓梯"]
        if horiz is None:
            parts.append("前方")
        elif near and horiz < 400:
            parts.append("就在腳邊")
        else:
            parts.append(f"前方 {_dist_phrase(horiz)}")

        if steps:
            parts.append("上行" if steps[0]["step_h_mm"] > 0 else "下行")
            reliable = [s for s in steps if s["reliable"]] or steps[:1]
            avg_h = sum(abs(s["step_h_mm"]) for s in reliable) / len(reliable)
            if len(steps) > 1:
                parts.append(f"{len(steps)}階")
                parts.append(f"每階約{int(round(avg_h/10.0))}公分")
            else:
                parts.append(f"階高約{int(round(avg_h/10.0))}公分")

        if near:
            parts.append("注意")

        return {
            # ---- v19 既有契約(proc_ai_worker / proc_overlay 依賴,不可少)----
            "has_yolo": True,
            "has_depth": horiz is not None,
            "type": "stairs",
            "severity": "high" if near else ("medium" if horiz is not None else "low"),
            "distance_mm": horiz,
            "dist_src": self.last_dist_src,
            "edge_row": self.last_edge_row,
            "near": near,
            "yolo_box": box,
            "conf": float(best[5]),
            "msg": " ".join(parts),
            # ---- v20 新增(舊呼叫端不會讀,不影響相容)----
            "box": box,                 # yolo_box 的別名
            "steps": steps,             # 每階 (overlay 可畫多條階緣線)
            "cam_h_fit": scan["cam_h_fit"] if scan else None,
            "stop_reason": self.last_stop_reason,
        }

    def reset(self):
        self._last_say = 0.0
        self._last_cliff_say = 0.0
        self.last_edge_row = None
        self.last_steps = []
        self.last_dist_src = "-"


# ============================================================
# 合成測試:python stairs_fusion.py
# 造一張「平地 + N 階樓梯」的深度圖,驗證物理模型能不能把階高量回來。
# 這是 v19→v20 改用物理模型的主要證據,改動演算法後請重跑。
# ============================================================
if __name__ == "__main__":
    H, W = 480, 640
    CAM_H, PITCH, STEP_H = 1700.0, 30.0, 160.0
    CONFIG.CAM_PITCH_DEG = PITCH
    sf = StairsFusion()
    sf.set_pitch(PITCH)

    edge_rows = [300, 262, 228]          # 三個階緣 (由近到遠)
    depth = np.zeros((H, W), dtype=np.uint16)
    for r in range(H):
        v = r / (H - 1)
        ang = math.radians(min(max(PITCH + (v - 0.5) * CONFIG.CAM_VFOV_DEG, 0.5), 89.5))
        n_below = sum(1 for er in edge_rows if r <= er)      # 這一列踩在第幾階上
        eff_h = CAM_H - n_below * STEP_H                     # 相機到該踏面的垂直高度
        depth[r, :] = int(max(200.0, eff_h / math.sin(ang)))

    res = sf._scan_steps(depth, (100, 180, 540, 330))
    print(f"擬合相機高度 k = {res['cam_h_fit']:.1f} mm (真值 {CAM_H})")
    print(f"地面殘差 σ = {res['sigma']:.2f} mm")
    print(f"停止原因: {res['stop_reason']}  偵測階數: {len(res['steps'])} (真值 {len(edge_rows)})")
    for i, s in enumerate(res["steps"], 1):
        print(f"  第{i}階 列{s['edge_row']} 階高 {s['step_h_mm']:.1f}mm "
              f"(真值 {STEP_H}) {'可信' if s['reliable'] else '參考'}")

    print("\n--- 手動 vs 自動身高優先權 ---")
    sf2 = StairsFusion()
    for _ in range(30):
        sf2._note_auto_height(1650.0)
    print(f"沒手動設定 → 自動接管: {sf2.cam_height_mm:.0f}mm")
    sf3 = StairsFusion()
    sf3.set_cam_height(1750.0, manual=True)
    for _ in range(30):
        sf3._note_auto_height(1500.0)
    print(f"有手動基準 → 不覆蓋: {sf3.cam_height_mm:.0f}mm")
    print(f"警告訊息: {sf3.pop_height_warnings()}")
