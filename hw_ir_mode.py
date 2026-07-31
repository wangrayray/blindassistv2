"""
ir_mode.py — IR 主動/被動深度動態切換(硬體層)
================================================
公開介面:
    IrModeController(device)
        .update(depth_frame, now=None) -> str|None   # 有切換才回傳新模式
        .set_mode(mode)                              # "active" / "passive"
        .status() -> dict
不對外開放:門檻判斷、冷卻計時。

--------------------------------------------------------------------
問題:OAK-D Pro 的 IR 點陣投影器並非「開著永遠比較好」
  室內 / 夜間 / 白牆等無紋理表面 → 被動立體匹配找不到對應點,深度大片破洞,
    此時 IR 點陣投影人造紋理,深度品質大幅改善 → 該開。
  戶外強烈日照 → 太陽的寬頻紅外線能量遠大於投影器,點陣被蓋過,
    不但沒幫助,投影器還持續發熱耗電 → 該關,改用純被動可見光立體。
  官方文件也建議避免陽光直射時依賴主動投影。

指標:深度圖的「無效像素比例」
  不需要額外感測器、不需要環境光感測器,深度圖自己就是最直接的品質指標。
  無效比例高 = 匹配失敗多 = 需要幫忙。

為什麼要雙門檻遲滯 + 連續確認 + 冷卻(三層防抖):
  單一門檻會在臨界值附近來回切換(走進走出陰影、雲飄過),而每次切換都會
  讓深度串流短暫不穩。三層防抖:
    ① 雙門檻:開的門檻(IR_INVALID_HIGH)高於關的門檻(IR_INVALID_LOW)
    ② 連續確認:要連續 IR_CONFIRM_FRAMES 幀都符合才動作
    ③ 冷卻:切換後 IR_SWITCH_COOLDOWN 秒內不再切
  這三層與 DepthAnomalyDetector 的 DEPTH_CONFIRM_FRAMES、馬達方位的
  遲滯設計是同一套思路——本系統所有「狀態切換」都用這個模式。

★ API 選用:`setIrLaserDotProjectorBrightness(mA)`
  depthai 2.15.0.0 起提供,參數是實際電流(mA)。舊的 0~1 正規化版本
  (`setIrLaserDotProjectorIntensity`) 在不同版本間行為不一致,相容性較差。
  任何 API 呼叫失敗一律自動停用本控制器,不影響其他功能——深度還是有,
  只是少了自動切換。
"""
import time

from shared_config import CONFIG


def _cfg(name, default):
    return getattr(CONFIG, name, default)


class IrModeController:
    ACTIVE = "active"
    PASSIVE = "passive"

    def __init__(self, device, start_mode=None):
        """
        device: depthai Device 物件。傳 None(webcam fallback)時本控制器
                自動停用,update() 永遠回 None。
        """
        self.device = device
        self.enabled = bool(_cfg("IR_MODE_ENABLED", True)) and device is not None
        self.mode = start_mode or self.ACTIVE
        self._streak_active = 0
        self._streak_passive = 0
        self._last_switch = 0.0
        self._last_ratio = None
        if self.enabled:
            # 開機先套用一次初始模式,確保硬體狀態與 self.mode 一致
            self._apply(self.mode)

    # ------------------------------------------------------------
    def _apply(self, mode):
        """實際下 API。失敗就整個停用,不再嘗試(避免每幀噴錯)。"""
        if self.device is None:
            return False
        try:
            dot = float(_cfg("IR_DOT_BRIGHTNESS_MA", 800)) if mode == self.ACTIVE else 0.0
            flood = float(_cfg("IR_FLOOD_BRIGHTNESS_MA", 0))
            self.device.setIrLaserDotProjectorBrightness(dot)
            if flood > 0:
                self.device.setIrFloodLightBrightness(flood if mode == self.ACTIVE else 0.0)
            self.mode = mode
            return True
        except Exception as e:
            print(f"⚠️ [IR] 設定失敗,停用動態切換(深度仍可用): {e}")
            self.enabled = False
            return False

    def set_mode(self, mode):
        """手動指定模式(UI/debug 用),會重置連續確認計數。"""
        if not self.enabled or mode not in (self.ACTIVE, self.PASSIVE):
            return False
        self._streak_active = self._streak_passive = 0
        self._last_switch = time.time()
        return self._apply(mode)

    # ------------------------------------------------------------
    def update(self, depth_frame, now=None):
        """
        每幀呼叫。回傳新模式字串(剛切換時)或 None(沒動作)。
        depth_frame 為 None 時直接跳過——沒有深度就沒有判斷依據。
        """
        if not self.enabled or depth_frame is None:
            return None
        now = time.time() if now is None else now

        total = depth_frame.size
        if total == 0:
            return None
        invalid = float((depth_frame == 0).sum()) / total
        self._last_ratio = invalid

        hi = float(_cfg("IR_INVALID_HIGH", 0.45))
        lo = float(_cfg("IR_INVALID_LOW", 0.20))
        need = int(_cfg("IR_CONFIRM_FRAMES", 15))

        # 無效比例高 → 需要主動投影;低 → 可以純被動(雙門檻遲滯)
        if invalid >= hi:
            self._streak_active += 1
            self._streak_passive = 0
        elif invalid <= lo:
            self._streak_passive += 1
            self._streak_active = 0
        else:
            # 落在兩個門檻之間 = 灰色地帶,兩邊計數都不累積,維持現狀
            self._streak_active = self._streak_passive = 0
            return None

        if now - self._last_switch < float(_cfg("IR_SWITCH_COOLDOWN", 8.0)):
            return None

        if self._streak_active >= need and self.mode != self.ACTIVE:
            self._last_switch = now
            self._streak_active = 0
            if self._apply(self.ACTIVE):
                print(f"💡 [IR] 無效像素 {invalid:.0%} → 切換為主動投影")
                return self.ACTIVE
        elif self._streak_passive >= need and self.mode != self.PASSIVE:
            self._last_switch = now
            self._streak_passive = 0
            if self._apply(self.PASSIVE):
                print(f"☀️ [IR] 無效像素 {invalid:.0%} → 切換為純被動")
                return self.PASSIVE
        return None

    def status(self):
        return {"enabled": self.enabled, "mode": self.mode,
                "invalid_ratio": (round(self._last_ratio, 3)
                                  if self._last_ratio is not None else None)}


if __name__ == "__main__":
    import numpy as np

    class _FakeDevice:
        def setIrLaserDotProjectorBrightness(self, mA):
            print(f"   [fake] dot={mA}mA")

    ir = IrModeController(_FakeDevice(), start_mode=IrModeController.PASSIVE)
    bad = np.zeros((100, 100), dtype=np.uint16)          # 全部無效 → 100%
    good = np.full((100, 100), 1500, dtype=np.uint16)    # 全部有效 → 0%
    t = 100.0
    for i in range(20):
        r = ir.update(bad, now=t + i)
        if r:
            print(f"第 {i} 幀切換 → {r}")
    print("狀態:", ir.status())
    t += 100
    for i in range(20):
        r = ir.update(good, now=t + i)
        if r:
            print(f"第 {i} 幀切換 → {r}")
    print("狀態:", ir.status())
    # 灰色地帶不應觸發
    mid = np.full((100, 100), 1500, dtype=np.uint16)
    mid[:30] = 0                                          # 30% 無效
    print("灰色地帶 30%:", [ir.update(mid, now=t + 200 + i) for i in range(20)].count(None), "幀無動作")
