"""
hw_confidence.py — 信心等級觸覺編碼(硬體層)
==============================================
公開介面(對外承諾):
    ConfidenceHaptic(hardware=None, publish_fn=None, motor_id=None)
        .set_confidence_vibration(level)
        .tick(now=None)
        .hold_for_safety(seconds)
        .pending_voice()
        .stop()
不對外開放:PWM 細節、payload 格式、節奏排程。

--------------------------------------------------------------------
與 `hw_haptic.py`(既有 hardware.py 改名而來)的關係:
  hw_haptic 負責「送出震動」這件事本身(MQTT/連線/冷卻);
  本檔負責「定位信心要對應什麼震動節奏」這個決策,包在它外面:

      from hw_haptic import HardwareManager      # 改名前是 hardware.py
      from hw_confidence import ConfidenceHaptic
      conf_hap = ConfidenceHaptic(hardware=HardwareManager())

  拆開的理由:震動節奏是會被實測推翻的參數,信心編碼是會演化的決策邏輯,
  兩者變動頻率差很多,綁在同一個檔案裡每次調節奏都要動到連線程式碼。

震動對應(§7)
  L1 穩定 100%   L2 穩定 70%   L3 中等脈衝 70%
  L4 弱短脈衝 40% + 語音「可能」   L5 不發方位震動(只留避障震動)

★ 安全優先權:避障/樓梯警告永遠比信心回饋重要。主迴圈在發出安全震動時
  呼叫 `hold_for_safety()`,信心通道會安靜讓路,避免兩種震動疊在一起讓
  使用者分不清哪個是危險訊號。
"""
import time

from shared_store import cfg

# level → (intensity 0-100, freq 1-3, duration_ms, 重發週期秒)
# 「穩定」= 重發週期略短於 duration,觸覺上連成一片;「脈衝」= 週期遠大於 duration。
LEVEL_PATTERN = {
    1: (100, 1, 600, 0.55),
    2: (70,  1, 600, 0.55),
    3: (70,  2, 250, 1.00),
    4: (40,  3, 120, 2.00),
    5: None,                     # L5:不發方位震動
}

LEVEL_VOICE = {4: "可能"}        # §7:L4 語音加「可能」修飾詞


class ConfidenceHaptic:
    def __init__(self, hardware=None, publish_fn=None, motor_id=None):
        self.hardware = hardware
        self.publish_fn = publish_fn
        self.motor_id = int(motor_id if motor_id is not None
                            else cfg("CONF_HAPTIC_MOTOR", 3))
        self._level = 5
        self._last_emit = None
        self._hold_until = 0.0
        self._voice_pending = None
        self._enabled = bool(cfg("CONF_HAPTIC_ENABLED", True))

    # ------------------------------------------------------------
    def set_confidence_vibration(self, level, now=None):
        """
        設定目前的定位信心等級(1–5)。等級改變時立刻發一次,
        之後由 tick() 依節奏維持。回傳是否有實際送出震動。
        """
        now = time.time() if now is None else now
        try:
            level = int(level)
        except Exception:
            return False
        changed = (level != self._level)
        self._level = max(1, min(5, level))
        if changed:
            self._voice_pending = LEVEL_VOICE.get(self._level)
            self._last_emit = None           # 強制立刻重發
        return self.tick(now)

    def tick(self, now=None):
        """主迴圈每幀呼叫。非阻塞,只在該發的時候發。"""
        now = time.time() if now is None else now
        if not self._enabled or now < self._hold_until:
            return False
        pat = LEVEL_PATTERN.get(self._level)
        if pat is None:
            return False
        intensity, freq, dur_ms, period = pat
        if self._last_emit is not None and now - self._last_emit < period:
            return False
        self._last_emit = now
        return self._emit(self.motor_id, intensity, freq, dur_ms)

    def hold_for_safety(self, seconds=None, now=None):
        """安全震動期間讓路。"""
        now = time.time() if now is None else now
        self._hold_until = now + float(seconds if seconds is not None
                                       else cfg("CONF_HAPTIC_SAFETY_HOLD", 1.0))

    def pending_voice(self):
        """取出待加的語音修飾詞(L4 的「可能」),取過即清空。"""
        v, self._voice_pending = self._voice_pending, None
        return v

    def stop(self):
        self._level = 5
        self._voice_pending = None

    # ------------------------------------------------------------
    def _emit(self, motor_id, intensity, freq, duration_ms):
        payload = f"{motor_id}:{int(intensity)}:{int(freq)}:{int(duration_ms)}"
        # ① 呼叫端自帶送出函式(單元測試 / 未來 UART 直送 ESP32)
        if self.publish_fn is not None:
            try:
                self.publish_fn(payload)
                return True
            except Exception as e:
                print(f"⚠️ [hw_confidence] publish_fn 失敗: {e}")
                return False
        if self.hardware is None:
            return False
        # ② 既有 HardwareManager 的 MQTT client(可自訂強度)
        client = getattr(self.hardware, "client", None)
        topic = cfg("TOPIC_CTRL_VIB", "guide/control/vibrate")
        if client is not None and hasattr(client, "publish"):
            try:
                client.publish(topic, payload, qos=1)
                return True
            except Exception as e:
                print(f"⚠️ [hw_confidence] MQTT 送出失敗: {e}")
        # ③ 最後退路:用既有預設 pattern(強度不可調,只求有回饋)
        try:
            mode = "point" if intensity <= 70 else "guide"
            self.hardware.vibrate(motor_id, mode)
            return True
        except Exception:
            return False


if __name__ == "__main__":
    sent = []
    h = ConfidenceHaptic(publish_fn=sent.append)
    t = 1000.0
    for lv in (1, 2, 3, 4, 5):
        h.set_confidence_vibration(lv, now=t)
        t += 3.0
        h.tick(now=t)
        t += 3.0
    print("送出:", sent)
    h.set_confidence_vibration(1, now=t)
    h.hold_for_safety(2.0, now=t)
    print("安全讓路期間:", h.tick(now=t + 1.0))
    print("讓路結束:", h.tick(now=t + 3.0))
