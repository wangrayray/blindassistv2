"""
斑馬線方向萃取與對準
====================
方法論參考 (assistive crosswalk alignment 主流文獻):
  [1] Mascetti, Ahmetovic, Gerino, Bernareggi (2016). ZebraRecognizer:
      Pedestrian crossing recognition for people with visual impairment or
      blindness. Pattern Recognition, 60, 405-419.
      → 透視校正 + 方向估計 + 量化相對位置;並提出跨幀時空追蹤穩定化。
  [2] Ahmetovic, Bernareggi, Gerino, Mascetti (2014). ZebraRecognizer:
      Efficient and precise localization of pedestrian crossings. ICPR 2014,
      2566-2571. → 以 line-segment (EDLines) 抓白條邊緣估方向。
  [3] Hough-space slope-dispersion 系列 → 對遮擋/夜間 robust 的直線方向估計。

本實作的取捨 (針對本系統架構):
  上游已有 YOLO-seg 的「斑馬線區域二值 mask」,而非原始白黑條紋影像。
  主方向採 **Hough 條紋斜率法** (在 mask 範圍內對原始灰階抓白條邊緣,
  取走道方向加權平均),對應 [2][3] 的線段/直線方向估計。

★ PCA 主軸法已移除 (原為 Hough 失敗時的正式備援):
  理論上「對填滿的二值區域取主成分」比兩點連線抗遮擋,但實測不準——
  斑馬線 mask 在正對時近似矩形,矩形的兩個特徵值接近,主軸會亂跳;
  而它偏偏就是「使用者正對斑馬線」這個最重要的情況。留著一個在關鍵情境
  失準的備援,比沒有備援更危險 (會安靜地把人導偏)。
  現在的退階順序:Hough → 質心連線 (_centroid_fallback),不再經過 PCA。
"""
import math
import numpy as np
import cv2

from shared_config import CONFIG


def _centroid_fallback(xs, ys, frame_h, frame_w):
    """
    上下端中位數連線:取遠端帶與近端帶各自的 x 中位數,兩點連線當主軸。
    現在是 Hough 失敗時唯一的退階路徑 (PCA 已移除)。
    中位數而非平均:對 mask 邊緣毛刺與零星誤分割不敏感。
    """
    bottom_idx = ys > frame_h * CONFIG.CROSSWALK_BOTTOM_BAND
    top_idx = ys < frame_h * CONFIG.CROSSWALK_TOP_BAND
    if bottom_idx.sum() < CONFIG.CROSSWALK_MIN_REGION_PX:
        return None
    if top_idx.sum() < CONFIG.CROSSWALK_MIN_REGION_PX:
        return "almost_arrived"
    bottom_cx = float(np.median(xs[bottom_idx]))
    bottom_cy = float(np.median(ys[bottom_idx]))
    top_cx = float(np.median(xs[top_idx]))
    top_cy = float(np.median(ys[top_idx]))
    mid_y = frame_h * 0.5
    if abs(top_cy - bottom_cy) > 1:
        t = (mid_y - bottom_cy) / (top_cy - bottom_cy)
        mid_cx = bottom_cx + t * (top_cx - bottom_cx)
    else:
        mid_cx = (bottom_cx + top_cx) / 2
    dx = top_cx - bottom_cx
    dy = bottom_cy - top_cy
    angle = math.degrees(math.atan2(dx, dy)) if dy > 1 else 0.0
    return {
        "bottom": (int(bottom_cx), int(bottom_cy)),
        "top": (int(top_cx), int(top_cy)),
        "offset": mid_cx - frame_w / 2,
        "angle": angle,
        "method": "centroid",
    }


def get_crosswalk_direction(mask, frame_h, frame_w):
    """
    從斑馬線二值 mask 萃取「行走主軸方向」與「左右偏移」。
    回傳:
      None              -> 沒抓到斑馬線
      "almost_arrived"  -> 斑馬線只剩近端,即將到對岸
      dict(offset, angle, bottom, top, method)
        offset: 主軸在畫面中央高度的水平偏移 (px),正=偏右(使用者偏左)
        angle:  主軸相對垂直的傾角 (度),0=正對,正=頂端往右斜(使用者面向左)

    ★ 本函式現在就是質心連線法本身 (PCA 已移除,見檔頭說明)。
      保留這個函式名是為了向後相容:ai_worker 與 overlay 都在呼叫它,
      改名會波及呼叫端,而它的語意 (mask → 方向 dict) 完全沒變。
    """
    if mask.sum() < CONFIG.CROSSWALK_MIN_MASK_PX:
        return None
    ys, xs = np.where(mask > 0)

    # almost_arrived:近端有、遠端幾乎空
    bottom_idx = ys > frame_h * CONFIG.CROSSWALK_BOTTOM_BAND
    top_idx = ys < frame_h * CONFIG.CROSSWALK_TOP_BAND
    if bottom_idx.sum() < CONFIG.CROSSWALK_MIN_REGION_PX:
        return None
    if top_idx.sum() < CONFIG.CROSSWALK_MIN_REGION_PX:
        return "almost_arrived"

    return _centroid_fallback(xs, ys, frame_h, frame_w)


def get_crosswalk_direction_hough(mask, frame_h, frame_w, gray=None):
    """
    Hough 條紋斜率法 (預設方向估計法)。
    在 mask 範圍內對原始灰階抓條紋邊緣,取走道方向加權平均,比對填滿區域
    算主軸的區域法穩 (區域主成分在正對/矩形時主軸會亂跳,已移除)。
    參考: Hough-space slope-dispersion;Tian et al. 2021;ZebraRecognizer。

    三層保險:
      條紋抓得到 → method="hough"
      gray 未給 / 條紋抓不到 / 方向太發散 → 退回 get_crosswalk_direction (質心連線)

    介面與 get_crosswalk_direction 相同,多一個 gray (原始灰階,與畫面同尺寸)。
    """
    # 沒有原圖灰階 → 無法抓條紋邊緣,退回質心連線
    if gray is None:
        return get_crosswalk_direction(mask, frame_h, frame_w)

    if mask.sum() < CONFIG.CROSSWALK_MIN_MASK_PX:
        return None

    ys, xs = np.where(mask > 0)

    # almost_arrived 判斷 (與質心版一致:近端有、遠端幾乎空)
    bottom_idx = ys > frame_h * CONFIG.CROSSWALK_BOTTOM_BAND
    top_idx = ys < frame_h * CONFIG.CROSSWALK_TOP_BAND
    if bottom_idx.sum() < CONFIG.CROSSWALK_MIN_REGION_PX:
        return None
    if top_idx.sum() < CONFIG.CROSSWALK_MIN_REGION_PX:
        return "almost_arrived"

    # ---- 只在 mask 範圍內取灰階,抓條紋邊緣 ----
    roi = np.zeros_like(gray)
    roi[mask > 0] = gray[mask > 0]

    edges = cv2.Canny(roi, CONFIG.CW_HOUGH_CANNY_LO, CONFIG.CW_HOUGH_CANNY_HI)
    edges = cv2.bitwise_and(edges, edges, mask=(mask > 0).astype(np.uint8))
    # 侵蝕 mask 邊界,去掉 ROI 外框造成的假線
    er = cv2.erode((mask > 0).astype(np.uint8),
                   np.ones((5, 5), np.uint8), iterations=2)
    edges = cv2.bitwise_and(edges, edges, mask=er)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=CONFIG.CW_HOUGH_THRESH,
        minLineLength=CONFIG.CW_HOUGH_MIN_LEN,
        maxLineGap=CONFIG.CW_HOUGH_MAX_GAP,
    )

    if lines is None or len(lines) < CONFIG.CW_HOUGH_MIN_LINES:
        return get_crosswalk_direction(mask, frame_h, frame_w)

    # ---- 收集條紋邊緣斜率 (單位向量平均,避免 ±180 環繞) ----
    # 條紋邊緣彼此平行;走道方向 = 條紋方向轉 90°。
    vxs, vys, weights = [], [], []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < 1:
            continue
        wx, wy = -dy, dx          # 轉 90°
        n = math.hypot(wx, wy)
        wx, wy = wx / n, wy / n
        if wy > 0:                # 統一半圓 (指向遠端/上)
            wx, wy = -wx, -wy
        vxs.append(wx * length)
        vys.append(wy * length)
        weights.append(length)

    if not weights:
        return get_crosswalk_direction(mask, frame_h, frame_w)

    mean_vx = sum(vxs)
    mean_vy = sum(vys)
    norm = math.hypot(mean_vx, mean_vy)
    if norm < 1e-6:
        return get_crosswalk_direction(mask, frame_h, frame_w)
    vx, vy = mean_vx / norm, mean_vy / norm
    if vy > 0:
        vx, vy = -vx, -vy

    # 一致性檢查:條紋方向太發散 → 退回 PCA
    cos_sum, wsum = 0.0, 0.0
    for x, y, wgt in zip(vxs, vys, weights):
        nx = math.hypot(x, y)
        if nx < 1e-6:
            continue
        cos_sum += (x / nx * vx + y / nx * vy) * wgt
        wsum += wgt
    consistency = cos_sum / wsum if wsum > 0 else 0.0
    if consistency < CONFIG.CW_HOUGH_MIN_CONSISTENCY:
        return get_crosswalk_direction(mask, frame_h, frame_w)

    # ---- 角度 (定義同 PCA 版:垂直向上=0,頂端往右斜=正) ----
    angle = math.degrees(math.atan2(vx, -vy))

    # ---- 偏移:沿走道方向把質心投影到畫面中央高度 ----
    mean_x = float(xs.mean())
    mean_y = float(ys.mean())
    mid_y = frame_h * 0.5
    if abs(vy) > 1e-3:
        t = (mid_y - mean_y) / vy
        mid_cx = mean_x + t * vx
    else:
        mid_cx = mean_x

    y_top, y_bot = float(ys.min()), float(ys.max())
    def x_at(y):
        if abs(vy) > 1e-3:
            return mean_x + (y - mean_y) / vy * vx
        return mean_x

    return {
        "bottom": (int(x_at(y_bot)), int(y_bot)),
        "top": (int(x_at(y_top)), int(y_top)),
        "offset": float(mid_cx - frame_w / 2),
        "angle": float(angle),
        "method": "hough",
        "n_lines": len(lines),
        "consistency": float(consistency),
    }


class CrosswalkStabilizer:
    """
    時序穩定化 (對應 ZebraRecognizer 的時空追蹤精神):
      - offset/angle 做指數移動平均 (EMA),抑制 seg mask 逐幀抖動造成的指示彈跳。
      - "almost_arrived" 需連續確認 N 幀才採信,避免遠端被車/人短暫遮住就誤判到站。
      - 連續多幀沒抓到 → 重置,避免用過時的平滑值。
    用法:每幀 raw = get_crosswalk_direction(...); out = stab.update(raw)
          out 的型別與 get_crosswalk_direction 相同,但 offset/angle 已平滑。
    """
    def __init__(self):
        self.s_offset = None
        self.s_angle = None
        self.almost_streak = 0
        self.miss = 0

    def update(self, direction):
        if direction is None:
            self.miss += 1
            self.almost_streak = 0
            if self.miss >= CONFIG.CROSSWALK_MISS_RESET:
                self.reset()
            return None

        if direction == "almost_arrived":
            self.almost_streak += 1
            self.miss = 0
            if self.almost_streak >= CONFIG.CROSSWALK_ALMOST_CONFIRM:
                return "almost_arrived"
            return None   # 尚未連續確認 → 這幀不給可靠方向

        # 正常 dict
        self.miss = 0
        self.almost_streak = 0
        o = direction["offset"]
        a = direction["angle"]
        k = CONFIG.CROSSWALK_EMA_ALPHA
        if self.s_offset is None:
            self.s_offset, self.s_angle = o, a
        else:
            self.s_offset = k * o + (1 - k) * self.s_offset
            self.s_angle = k * a + (1 - k) * self.s_angle

        out = dict(direction)
        out["offset"] = self.s_offset
        out["angle"] = self.s_angle
        return out

    def reset(self):
        self.s_offset = None
        self.s_angle = None
        self.almost_streak = 0
        self.miss = 0