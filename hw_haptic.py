"""
MQTT 與 ESP32 通訊(觸覺執行層)
================================
★ 斷線偵測:震動是視障使用者唯一的即時安全回饋管道,MQTT 斷線後
  `publish()` 不會報錯、訊息直接進虛空,使用者會以為「沒有障礙物」——
  這是靜默失效,比崩潰更危險。因此加上 on_disconnect + is_online(),
  由 main.py 在離線超過門檻時用語音明確告知。
★ paho-mqtt 2.0 的 Client() 建構式有 breaking change(必須指定
  CallbackAPIVersion),1.x 沒有這個參數。防呆:先試 2.x 寫法,
  TypeError 就退回裸呼叫,兩個版本都能跑。
"""
import time
import threading

import paho.mqtt.client as mqtt

from shared_config import CONFIG
from shared_utils import vibrate_payload


def _make_client():
    """建立 MQTT client,同時相容 paho-mqtt 1.x 與 2.x。"""
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except (AttributeError, TypeError):
        return mqtt.Client()


class HardwareManager:
    def __init__(self):
        self.last_vib_time = {}
        self.connected = False
        self.offline_since = time.time()      # 還沒連上前一律視為離線
        self.client = _make_client()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        threading.Thread(target=self._connect_loop, daemon=True).start()

    def _connect_loop(self):
        print(f"🔄 連線 MQTT ({CONFIG.MQTT_BROKER})...")
        while True:
            try:
                self.client.connect(CONFIG.MQTT_BROKER, CONFIG.MQTT_PORT, 60)
                self.client.loop_start()
                print("✅ MQTT 連線成功")
                return
            except Exception as e:
                print(f"❌ MQTT 連線失敗,5 秒後重試: {e}")
                time.sleep(5)

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        # properties 參數只有 paho 2.x 會傳,給預設值讓 1.x 也能呼叫
        self.connected = (rc == 0)
        if self.connected:
            self.offline_since = None
            print("✅ MQTT on_connect rc=0")
        else:
            self.offline_since = self.offline_since or time.time()
            print(f"⚠️ MQTT on_connect rc={rc}(非 0 = 未連上)")

    def _on_disconnect(self, client, userdata, rc, properties=None, reason=None):
        self.connected = False
        self.offline_since = time.time()
        print(f"⚠️ MQTT 斷線 rc={rc},震動回饋暫時失效(loop_start 會自動重連)")

    def _on_message(self, client, userdata, msg):
        # 目前無任何訂閱的感測器 topic;保留 callback 供未來擴充。
        pass

    # ============================================================
    # 狀態查詢(給 main.py 判斷要不要語音警告)
    # ============================================================
    def is_online(self):
        return bool(self.connected)

    def offline_duration(self):
        """已離線幾秒;線上回 0.0。"""
        if self.connected or self.offline_since is None:
            return 0.0
        return time.time() - self.offline_since

    # ============================================================
    def vibrate(self, motor_id, mode="point"):
        now = time.time()
        if now - self.last_vib_time.get(motor_id, 0) > CONFIG.HARDWARE_VIB_COOLDOWN:
            payload = vibrate_payload(motor_id, mode)
            self.client.publish(CONFIG.TOPIC_CTRL_VIB, payload, qos=1)
            self.last_vib_time[motor_id] = now

    def publish_raw(self, payload):
        """
        直接送出自訂 payload(motor:intensity:freq:duration)。
        給 hw_confidence.ConfidenceHaptic 用——信心震動需要 0-100 連續強度,
        不能只用 VIB_PATTERNS 的四個預設檔位。
        """
        try:
            self.client.publish(CONFIG.TOPIC_CTRL_VIB, payload, qos=1)
            return True
        except Exception as e:
            print(f"⚠️ MQTT 送出失敗: {e}")
            return False

    def shutdown(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass
