"""穩定鎖、手部鎖、深度突變偵測"""
from collections import deque
import numpy as np
from shared_config import CONFIG


class StabilityLockTracker:
    """通用穩定鎖:某個值連續 N 秒在 tolerance 內 → locked"""
    def __init__(self, tolerance, duration):
        self.tolerance = tolerance
        self.duration = duration
        self.samples = deque()

    def update(self, value, now):
        if value is None:
            self.samples.clear()
            return False
        self.samples.append((now, value))
        cutoff = now - self.duration
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        if len(self.samples) < 3:
            return False
        span = self.samples[-1][0] - self.samples[0][0]
        if span < self.duration - 0.1:
            return False
        values = [v for _, v in self.samples]
        return (max(values) - min(values)) <= self.tolerance

    def reset(self):
        self.samples.clear()


class HandLockTracker:
    """
    鎖定第一隻偵測到的手, 跳動拒絕。

    ★ 跳動門檻改成「畫面寬度比例」而非寫死像素:
      原本寫死 200px,在 640 寬時等於 31% 畫面、在 1280 寬時只剩 15%,
      換解析度就等於偷偷改了演算法行為。改成比例後解析度無關。
    """
    MAX_JUMP_RATIO = 0.31          # 對畫面寬度的比例(640px 時等同舊的 200px)

    def __init__(self, frame_w=None):
        self.last_wrist = None
        self.last_seen = 0.0
        self.frame_w = frame_w or CONFIG.FRAME_W
        self.LOST_TIMEOUT = 1.5

    @property
    def max_jump_px(self):
        return self.MAX_JUMP_RATIO * self.frame_w

    def update(self, landmarks, now, frame_w=None):
        if landmarks is None:
            if self.last_wrist and now - self.last_seen > self.LOST_TIMEOUT:
                self.last_wrist = None
            return None
        if frame_w:
            self.frame_w = frame_w
        wrist = landmarks[0]
        if self.last_wrist is None:
            self.last_wrist = wrist
            self.last_seen = now
            return landmarks
        dx = abs(wrist[0] - self.last_wrist[0])
        dy = abs(wrist[1] - self.last_wrist[1])
        if dx > self.max_jump_px or dy > self.max_jump_px:
            return None
        self.last_wrist = wrist
        self.last_seen = now
        return landmarks

    def reset(self):
        self.last_wrist = None
        self.last_seen = 0.0


class DepthAnomalyDetector:
    """
    地面高度校正法 (Cloix et al., EURASIP 2016) 的線上版本。
    用相機俯角把深度(斜距)校正成「相對相機水平面的地面高度」,
    平地校正後≈常數;扣窗內中位數基準後 → 平地≈0、落差<0、隆起>0。
    上下分兩半各算落差/隆起像素比例:
      下半(腳邊)落差 → cliff danger;只有上半(遠)落差 → cliff warning;
      任一半隆起 → obstacle(上行台階)。
    回傳給 StairsFusion 的中性結果:
      type: 'obstacle' / 'cliff' / None
      front_mm, severity_hint('danger'/'warning')
    """
    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._depth_buf = deque(maxlen=CONFIG.DEPTH_TEMPORAL_FRAMES)
        # 預先算好每列射線俯角的 sin (校正用),frame 尺寸固定故只算一次
        self._sin_p = None
        self.obstacle_streak = 0
        self.cliff_streak = 0
        self._cliff_hint = "warning"
        self._last_front = None
        self.last_warn = 0.0
        self.last_zones = (None, None, None)   # (far, front, near) 即時三區距離
        self.last_streak = 0                   # 即時觸發進度 (給面板進度條)
        self.last_ground_method = "-"          # ransac / pitch (給面板顯示)
        self.last_ground_sigma = 0.0           # 地面殘差σ (自適應門檻用)
        # ---- 動態姿態 (IMU) / 動態相機高度 (自動估計或 UI 手動輸入) ----
        # 沒有資料時一律退回 CONFIG 的固定值,行為與加 IMU 之前完全相同。
        self._pitch_deg = None
        self._cam_height_mm = None

    # ============================================================
    # 動態姿態注入(IMU / 自動身高)
    # ============================================================
    def set_pitch(self, pitch_deg):
        """
        注入 IMU 即時俯角。傳 None = 沒有資料 → 退回 CONFIG.CAM_PITCH_DEG。
        變化超過 0.5 度才讓快取失效:IMU 每幀都有微小抖動,若每幀重算
        sin 表,快取形同虛設,白白吃掉 CPU。
        """
        if pitch_deg is None:
            if self._pitch_deg is not None:
                self._pitch_deg = None
                self._sin_p = None
            return
        if self._pitch_deg is None or abs(pitch_deg - self._pitch_deg) > 0.5:
            self._sin_p = None
        self._pitch_deg = float(pitch_deg)

    def set_cam_height(self, height_mm):
        """注入相機離地高度(UI 手動輸入或自動估計)。None = 退回 CONFIG。"""
        self._cam_height_mm = float(height_mm) if height_mm else None

    @property
    def pitch_deg(self):
        return self._pitch_deg if self._pitch_deg is not None else CONFIG.CAM_PITCH_DEG

    @property
    def cam_height_mm(self):
        return (self._cam_height_mm if self._cam_height_mm is not None
                else CONFIG.CAM_HEIGHT_MM)

    # ---- 透視校正用的每列 sin(射線俯角),依當前 H 與俯角快取 ----
    def _row_sin(self, H):
        if self._sin_p is not None and self._sin_p.shape[0] == H:
            return self._sin_p
        rows = np.arange(H, dtype=np.float32)
        v_frac = rows / max(1, H - 1)                      # 0(頂/遠)~1(底/近)
        ang_from_center = (v_frac - 0.5) * CONFIG.CAM_VFOV_DEG
        ray_pitch = self.pitch_deg + ang_from_center
        ray_pitch = np.clip(ray_pitch, 0.5, 89.5)
        self._sin_p = np.sin(np.radians(ray_pitch)).reshape(H, 1)
        return self._sin_p

    def _temporal_median(self, depth_frame):
        """多幀中位數,洗掉單幀整片崩壞。"""
        self._depth_buf.append(depth_frame.astype(np.float32))
        if len(self._depth_buf) == 1:
            return self._depth_buf[0]
        return np.median(np.stack(self._depth_buf, axis=0), axis=0)

    def _zone_median(self, depth_mm, h_range, w_range):
        H, W = depth_mm.shape
        y0, y1 = int(H * h_range[0]), int(H * h_range[1])
        x0, x1 = int(W * w_range[0]), int(W * w_range[1])
        roi = depth_mm[y0:y1, x0:x1]
        valid = roi[(roi >= CONFIG.DEPTH_MIN_MM) & (roi <= CONFIG.DEPTH_MAX_MM)]
        if valid.size < 30:
            return None
        return float(np.median(valid))

    def _rectify(self, depth_mm):
        """[fallback] 固定俯角校正成地面高度 (RANSAC 失敗時用)。"""
        H, W = depth_mm.shape
        d = depth_mm
        valid = (d >= CONFIG.DEPTH_MIN_MM) & (d <= CONFIG.DEPTH_MAX_MM)
        vertical_drop = d * self._row_sin(H)          # 斜距投影到垂直
        height = self.cam_height_mm - vertical_drop
        height[~valid] = np.nan
        return height

    def _fit_ground_ransac(self, depth_mm, x0, x1, y0, y1):
        """
        在掃描窗內,用 RANSAC 擬合「地面深度 vs 列號」模型 (自適應,免俯角)。
        作法:
          1. 每列取窗內有效深度的中位數 → (row, median_depth) 一組點。
          2. RANSAC 擬合直線 depth = a*row + b (平地近似線性;近列深度小、遠列大)。
          3. 回傳每像素的「預期地面深度」圖 ground_pred[H,W] (僅該列線性外推)。
        擬合失敗 (有效列太少/退化) → 回 None,呼叫端退回固定俯角法。
        """
        H, W = depth_mm.shape
        rows = np.arange(y0, y1)
        med = np.full(rows.shape[0], np.nan, dtype=np.float32)
        for i, r in enumerate(rows):
            seg = depth_mm[r, x0:x1]
            v = seg[(seg >= CONFIG.DEPTH_MIN_MM) & (seg <= CONFIG.DEPTH_MAX_MM)]
            if v.size >= 8:
                med[i] = np.median(v)

        good = ~np.isnan(med)
        rg = rows[good].astype(np.float32)
        dg = med[good]
        if rg.size < CONFIG.GROUND_RANSAC_MIN_ROWS:
            return None

        best_inliers = -1
        best_ab = None
        n = rg.size
        rng = np.random.default_rng(0)   # 固定種子 → 結果可重現,debug 友善
        for _ in range(CONFIG.GROUND_RANSAC_ITERS):
            i, j = rng.integers(0, n, size=2)
            if rg[i] == rg[j]:
                continue
            a = (dg[i] - dg[j]) / (rg[i] - rg[j])
            b = dg[i] - a * rg[i]
            pred = a * rg + b
            inliers = int(np.sum(np.abs(dg - pred) <= CONFIG.GROUND_RANSAC_INLIER_MM))
            if inliers > best_inliers:
                best_inliers, best_ab = inliers, (a, b)

        if best_ab is None or best_inliers < CONFIG.GROUND_RANSAC_MIN_ROWS:
            return None

        # 用 inliers 做最小平方精修 (RANSAC 取穩健模型,再 LS 提精度)
        a, b = best_ab
        pred_all = a * rg + b
        inl = np.abs(dg - pred_all) <= CONFIG.GROUND_RANSAC_INLIER_MM
        if inl.sum() >= 2:
            a, b = np.polyfit(rg[inl], dg[inl], 1)

        # 地面預期深度圖:每列一個值,沿 W 廣播
        row_idx = np.arange(H, dtype=np.float32)
        ground_row = a * row_idx + b           # [H]
        ground_pred = np.repeat(ground_row.reshape(H, 1), W, axis=1)
        # 地面殘差標準差σ (只用 inlier 列估,代表平地本身的深度雜訊水平)
        inl_resid = dg[inl] - (a * rg[inl] + b)
        sigma = float(np.std(inl_resid)) if inl.sum() >= 3 else 0.0
        self.last_ground_sigma = sigma
        return ground_pred

    def _classify(self, depth_mm):
        """單幀(已做多幀中位數)→ (type, front_mm, severity_hint) 或 None。"""
        front = self._zone_median(depth_mm, CONFIG.DEPTH_ZONE_FRONT, CONFIG.DEPTH_H_BAND)

        H, W = depth_mm.shape
        x0, x1 = int(W * CONFIG.DEPTH_SCAN_BAND_W[0]), int(W * CONFIG.DEPTH_SCAN_BAND_W[1])
        y0, y1 = int(H * CONFIG.DEPTH_SCAN_BAND_H[0]), int(H * CONFIG.DEPTH_SCAN_BAND_H[1])

        valid_full = (depth_mm >= CONFIG.DEPTH_MIN_MM) & (depth_mm <= CONFIG.DEPTH_MAX_MM)

        use_ransac = False
        if CONFIG.GROUND_RANSAC_ENABLED:
            ground_pred = self._fit_ground_ransac(depth_mm, x0, x1, y0, y1)
            if ground_pred is not None:
                use_ransac = True
                self.last_ground_method = "ransac"

        if use_ransac:
            # 殘差 = 實際深度 − 該列預期地面深度。
            #   殘差 > +DROP → 比地面遠 → 往下掉 (落差/cliff)
            #   殘差 < −RISE → 比地面近 → 隆起 (上行台階/obstacle)
            resid = np.where(valid_full, depth_mm - ground_pred, np.nan)
            win = resid[y0:y1, x0:x1]
            # 自適應門檻:地面雜訊σ 越大門檻越高 (治平地遠處雜訊誤報下行)
            sigma = getattr(self, "last_ground_sigma", 0.0)
            DROP = max(CONFIG.GROUND_DROP_MM, CONFIG.GROUND_NOISE_K * sigma)
            RISE = max(CONFIG.GROUND_RISE_MM, CONFIG.GROUND_NOISE_K * sigma * 0.7)
            win_valid = win[~np.isnan(win)]
            if win_valid.size < CONFIG.DEPTH_SCAN_MIN_VALID:
                return None

            mid = win.shape[0] // 2
            upper, lower = win[:mid], win[mid:]

            def ratios(sub):
                v = sub[~np.isnan(sub)]
                if v.size < CONFIG.DEPTH_SCAN_MIN_VALID // 2:
                    return None, None
                drop = float(np.mean(v >  DROP))   # 正殘差大 = 落差
                rise = float(np.mean(v < -RISE))   # 負殘差大 = 隆起
                return drop, rise

            up_drop, up_rise = ratios(upper)
            lo_drop, lo_rise = ratios(lower)

            DR = CONFIG.GROUND_DROP_RATIO
            RR = CONFIG.GROUND_RISE_RATIO
            FAR = CONFIG.GROUND_FAR_STRICT     # 上半(遠)門檻加嚴倍數

            # 危險序:腳邊落差 > 腳邊隆起 > 遠處落差 > 遠處隆起。
            # 下半(腳邊)用標準比例;上半(遠處,雜訊大)比例×FAR 才採信。
            if lo_drop is not None and lo_drop >= DR:
                return ("cliff", front, "danger")
            if lo_rise is not None and lo_rise >= RR:
                return ("obstacle", front, "danger")
            if up_drop is not None and up_drop >= DR * FAR:
                return ("cliff", front, "warning")
            if up_rise is not None and up_rise >= RR * FAR:
                return ("obstacle", front, "warning")
            return None

        # ---- fallback: 固定俯角地面高度法 ----
        self.last_ground_method = "pitch"
        height = self._rectify(depth_mm)
        win = height[y0:y1, x0:x1]
        wv = win[~np.isnan(win)]
        if wv.size < CONFIG.DEPTH_SCAN_MIN_VALID:
            return None
        win = win - float(np.median(wv))   # 扣中位數基準 → 平地≈0
        DROP = CONFIG.DROP_MM
        RISE = CONFIG.RISE_MM

        mid = win.shape[0] // 2
        upper, lower = win[:mid], win[mid:]

        def ratios_h(sub):
            v = sub[~np.isnan(sub)]
            if v.size < CONFIG.DEPTH_SCAN_MIN_VALID // 2:
                return None, None
            drop = float(np.mean(v < -DROP))   # 低於地面 = 落差
            rise = float(np.mean(v >  RISE))   # 高於地面 = 隆起
            return drop, rise

        up_drop, up_rise = ratios_h(upper)
        lo_drop, lo_rise = ratios_h(lower)
        TH = CONFIG.RATIO_THRESH

        def hit(r):
            return r is not None and r >= TH

        if hit(lo_drop):
            return ("cliff", front, "danger")
        if hit(lo_rise):
            return ("obstacle", front, "danger")
        if hit(up_drop):
            return ("cliff", front, "warning")
        if hit(up_rise):
            return ("obstacle", front, "warning")
        return None

    def _zone_dists(self, depth_mm):
        """三區 (遠/前/近) 中位數距離,供 HUD 顯示 + 融合報距離。"""
        far   = self._zone_median(depth_mm, CONFIG.DEPTH_ZONE_FAR,   CONFIG.DEPTH_H_BAND)
        front = self._zone_median(depth_mm, CONFIG.DEPTH_ZONE_FRONT, CONFIG.DEPTH_H_BAND)
        near  = self._zone_median(depth_mm, CONFIG.DEPTH_ZONE_NEAR,  CONFIG.DEPTH_H_BAND)
        return far, front, near

    def update(self, depth_frame, now, pitch_deg=None, cam_height_mm=None):
        """
        pitch_deg / cam_height_mm 為選填:給了就用即時值,沒給就沿用上次注入的
        值或 CONFIG 常數。舊呼叫端 `update(depth, now)` 行為完全不變。
        """
        if pitch_deg is not None:
            self.set_pitch(pitch_deg)
        if cam_height_mm is not None:
            self.set_cam_height(cam_height_mm)
        if depth_frame is None:
            self.last_zones = (None, None, None)
            return None

        depth_mm = self._temporal_median(depth_frame)
        # 三區距離每幀都算 (給 debug 面板即時顯示,不受冷卻影響)
        self.last_zones = self._zone_dists(depth_mm)
        cls = self._classify(depth_mm)

        is_obstacle = cls is not None and cls[0] == "obstacle"
        is_cliff    = cls is not None and cls[0] == "cliff"
        if cls is not None:
            self._last_front = cls[1]
            if is_cliff:
                self._cliff_hint = cls[2]

        self.obstacle_streak = self.obstacle_streak + 1 if is_obstacle else 0
        self.cliff_streak    = self.cliff_streak + 1 if is_cliff else 0
        # 給面板看的即時觸發進度 (取兩者較大)
        self.last_streak = max(self.obstacle_streak, self.cliff_streak)

        if now - self.last_warn < CONFIG.DEPTH_WARN_COOLDOWN:
            return None

        far, front, near = self.last_zones
        if self.obstacle_streak >= CONFIG.DEPTH_CONFIRM_FRAMES:
            self.obstacle_streak = 0
            self.last_warn = now
            return {"type": "obstacle", "front_mm": self._last_front,
                    "near_mm": near, "far_mm": far,
                    "severity_hint": "danger", "color": (0, 0, 255)}

        if self.cliff_streak >= CONFIG.DEPTH_CONFIRM_FRAMES:
            self.cliff_streak = 0
            self.last_warn = now
            return {"type": "cliff", "front_mm": self._last_front,
                    "near_mm": near, "far_mm": far,
                    "severity_hint": self._cliff_hint, "color": (255, 0, 255)}

        return None

    def reset(self):
        self._depth_buf.clear()
        self.obstacle_streak = 0
        self.cliff_streak = 0
        self._cliff_hint = "warning"
        self._last_front = None
        self.last_warn = 0.0
        self.last_zones = (None, None, None)
        self.last_streak = 0
        self.last_ground_method = "-"