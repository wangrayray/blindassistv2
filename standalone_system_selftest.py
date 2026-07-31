"""
standalone_system_selftest.py — 系統層自我測試(不需硬體)
============================================================
執行:  python standalone_system_selftest.py

涵蓋本輪修復/新增的所有子系統。與 `standalone_gen3_selftest.py` 分工:
  gen3_selftest   → 第三代新模組(分區記憶、偽VLM、信心階梯)的內部行為
  system_selftest → 既有子系統的修復是否成立,以及**模組之間的欄位契約**

★ 為什麼要有「欄位契約」測試(第 1 組):
  重寫 stairs_fusion 到 v20 時,回傳 dict 少了 has_depth / has_yolo 兩個 key,
  而 proc_ai_worker 是用 `stairs_result["has_depth"]` 直接取值。
  這種錯誤 py_compile 不會抓、import 測試不會抓、pyflakes 也不會抓,
  只有在真的偵測到樓梯的那一刻才會 KeyError —— 也就是使用者站在樓梯前面
  的時候。所以契約必須有自動化測試守著。
"""
import os
import sys
import tempfile

import numpy as np

os.environ.setdefault("GEN3_DATA_DIR", tempfile.mkdtemp(prefix="sys_test_"))

from shared_config import CONFIG
import percept_crosswalk as crosswalk
from hw_ir_mode import IrModeController
from percept_stairs_fusion import StairsFusion
from percept_trackers import DepthAnomalyDetector, HandLockTracker

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f"  ({detail})" if detail and not cond else ""))


def synth_stairs_depth(cam_h=1700.0, pitch=30.0, step_h=160.0,
                       edge_rows=(300, 262, 228), H=480, W=640):
    """合成「平地 + N 階」深度圖:每一列依物理模型 depth = k / sin(ray_pitch)。"""
    import math
    d = np.zeros((H, W), dtype=np.uint16)
    for r in range(H):
        v = r / (H - 1)
        ang = math.radians(min(max(pitch + (v - 0.5) * CONFIG.CAM_VFOV_DEG, 0.5), 89.5))
        n_below = sum(1 for er in edge_rows if r <= er)
        d[r, :] = int(max(200.0, (cam_h - n_below * step_h) / math.sin(ang)))
    return d


# ============================================================
print("\n── 1. 樓梯回傳欄位契約(跨模組,最容易靜默壞掉)──")
V19_KEYS = {"has_yolo", "has_depth", "type", "severity", "distance_mm",
            "dist_src", "edge_row", "near", "yolo_box", "conf", "msg"}
V20_KEYS = {"steps", "cam_h_fit", "stop_reason"}

sf = StairsFusion()
sf.set_pitch(30.0)
depth = synth_stairs_depth()
dets = [(100, 180, 540, 330, "stairs", 0.88)]
res = sf.update(dets, depth, now=1000.0)
check("偵測到樓梯時有回傳", res is not None)
missing = V19_KEYS - set(res or {})
check("v19 既有欄位一個不缺", not missing, f"缺 {missing}")
check("v20 新欄位存在", not (V20_KEYS - set(res or {})))
check("has_yolo / has_depth 為布林", isinstance(res["has_yolo"], bool)
      and isinstance(res["has_depth"], bool))
check("yolo_box 為四元組", isinstance(res["yolo_box"], tuple) and len(res["yolo_box"]) == 4)
check("conf 來自 YOLO 信心", abs(res["conf"] - 0.88) < 1e-6)

sf2 = StairsFusion()
cliff = sf2.update([], depth, now=2000.0, anomaly={"type": "cliff", "front_mm": 800})
check("落差保底也回傳完整契約", cliff is not None and not (V19_KEYS - set(cliff)))
check("落差分支 has_yolo=False", cliff["has_yolo"] is False)

# 模擬 proc_ai_worker 的實際取值方式(用 [] 而非 .get)
try:
    _ = res["has_depth"], res["has_yolo"], res["msg"], res["severity"]
    ok = True
except KeyError as e:
    ok, err = False, str(e)
check("proc_ai_worker 的 [] 取值不會 KeyError", ok, "" if ok else err)

# ============================================================
print("\n── 2. 樓梯 v20 多階量測精度 ──")
scan = StairsFusion()
scan.set_pitch(30.0)
r = scan._scan_steps(depth, (100, 180, 540, 330))
check("反推相機高度誤差 < 5mm", abs(r["cam_h_fit"] - 1700.0) < 5.0,
      f"{r['cam_h_fit']:.1f}")
check("地面殘差 σ < 2mm", r["sigma"] < 2.0, f"{r['sigma']:.2f}")
check("階數正確 (3 階)", len(r["steps"]) == 3, str(len(r["steps"])))
err = max(abs(s["step_h_mm"] - 160.0) for s in r["steps"])
check("每階階高誤差 < 5mm", err < 5.0, f"{err:.1f}")
check("只有前 N 階標為可信", sum(1 for s in r["steps"] if s["reliable"])
      == CONFIG.STAIRS_RELIABLE_N)

# ============================================================
print("\n── 3. 手動 vs 自動身高優先權 ──")
a = StairsFusion()
for _ in range(30):
    a._note_auto_height(1650.0)
check("沒手動設定 → 自動估計接管", abs(a.cam_height_mm - 1650.0) < 20.0,
      f"{a.cam_height_mm:.0f}")
b = StairsFusion()
b.set_cam_height(1750.0, manual=True)
for _ in range(30):
    b._note_auto_height(1500.0)
check("有手動基準 → 自動不覆蓋", abs(b.cam_height_mm - 1750.0) < 1e-6,
      f"{b.cam_height_mm:.0f}")
w = b.pop_height_warnings()
check("差距過大時出聲警告", len(w) == 1 and "重新校準" in w[0])
check("警告只發一次", b.pop_height_warnings() == [])
b.reset_height_calibration()
check("重設後交還自動估計", b._manual_height is False)
c = StairsFusion()
c.set_cam_height(50.0, manual=True)
check("不合理身高被擋下", c.cam_height_mm != 50.0)

# ============================================================
print("\n── 4. IMU 動態俯角 ──")
d0 = DepthAnomalyDetector(640, 480)
check("沒注入時用 CONFIG 常數", abs(d0.pitch_deg - CONFIG.CAM_PITCH_DEG) < 1e-6)
d0.set_pitch(45.0)
check("注入後改用即時值", abs(d0.pitch_deg - 45.0) < 1e-6)
s45 = float(d0._row_sin(480)[240][0])
d0.set_pitch(20.0)
s20 = float(d0._row_sin(480)[240][0])
check("俯角改變會改變射線幾何(快取有失效)", abs(s45 - s20) > 0.1,
      f"{s45:.3f} vs {s20:.3f}")
d0.set_pitch(None)
check("傳 None 退回 CONFIG 常數", abs(d0.pitch_deg - CONFIG.CAM_PITCH_DEG) < 1e-6)

st = StairsFusion()
st.set_pitch(30.0)
k30 = st._scan_steps(depth, (100, 180, 540, 330))["cam_h_fit"]
st.set_pitch(20.0)
k20 = st._scan_steps(depth, (100, 180, 540, 330))["cam_h_fit"]
check("俯角錯誤會讓擬合高度偏掉(證明真的有吃進去)", abs(k30 - k20) > 50.0,
      f"{k30:.0f} vs {k20:.0f}")

# ============================================================
print("\n── 5. HandLockTracker 解析度無關 ──")
h1 = HandLockTracker(frame_w=640)
h2 = HandLockTracker(frame_w=1280)
check("跳動門檻隨畫面寬度縮放", abs(h2.max_jump_px - 2 * h1.max_jump_px) < 1e-6)
check("640 寬時等同舊的 200px", abs(h1.max_jump_px - 200.0) < 2.0,
      f"{h1.max_jump_px:.1f}")
lm = [(100, 100)] + [(0, 0)] * 20
h1.update(lm, 0.0)
far = [(100 + int(h1.max_jump_px) + 50, 100)] + [(0, 0)] * 20
check("超過門檻的跳動被拒絕", h1.update(far, 0.1) is None)
near = [(120, 110)] + [(0, 0)] * 20
check("小幅移動被接受", h1.update(near, 0.2) is not None)

# ============================================================
print("\n── 6. 斑馬線:PCA 已移除 ──")
mask = np.zeros((480, 640), dtype=np.uint8)
mask[100:450, 280:360] = 1
out = crosswalk.get_crosswalk_direction(mask, 480, 640)
check("方向估計仍可用", isinstance(out, dict))
check("退階路徑是質心連線,不是 PCA", out.get("method") == "centroid",
      str(out.get("method")))
check("原始碼已無 PCA 實作", "np.linalg.eigh" not in
      open(crosswalk.__file__, encoding="utf-8").read())
check("config 已無 PCA 參數", not hasattr(CONFIG, "CROSSWALK_PCA_MIN_RATIO"))

# ============================================================
print("\n── 7. IR 主動/被動遲滯 ──")
class _FakeDev:
    def __init__(self): self.calls = []
    def setIrLaserDotProjectorBrightness(self, mA): self.calls.append(mA)

dev = _FakeDev()
ir = IrModeController(dev, start_mode=IrModeController.PASSIVE)
bad = np.zeros((100, 100), dtype=np.uint16)
sw = [ir.update(bad, now=100.0 + i) for i in range(20)]
check("無效像素高 → 切主動", IrModeController.ACTIVE in sw)
check("切換前需連續確認,不是第一幀就切",
      sw.index(IrModeController.ACTIVE) >= CONFIG.IR_CONFIRM_FRAMES - 1)
good = np.full((100, 100), 1500, dtype=np.uint16)
sw2 = [ir.update(good, now=300.0 + i) for i in range(20)]
check("無效像素低 → 切被動", IrModeController.PASSIVE in sw2)
mid = np.full((100, 100), 1500, dtype=np.uint16)
mid[:30] = 0
check("灰色地帶不切換(雙門檻遲滯)",
      all(x is None for x in [ir.update(mid, now=500.0 + i) for i in range(30)]))
ir2 = IrModeController(None)
check("沒有 device 時自動停用", ir2.enabled is False and ir2.update(bad) is None)

# ============================================================
print("\n── 8. MQTT 斷線偵測 ──")
import hw_haptic
class _FakeClient:
    def __init__(self): self.sent = []
    def publish(self, t, p, qos=1): self.sent.append(p)
    def connect(self, *a, **k): pass
    def loop_start(self): pass
    def loop_stop(self): pass
    def disconnect(self): pass

hwm = hw_haptic.HardwareManager.__new__(hw_haptic.HardwareManager)
hwm.last_vib_time = {}
hwm.connected = False
hwm.offline_since = 1000.0
hwm.client = _FakeClient()
check("初始為離線狀態", hwm.is_online() is False)
check("離線時間可查詢", hwm.offline_duration() > 0)
hwm._on_connect(None, None, None, 0)
check("連上後 is_online() 為 True", hwm.is_online() is True)
check("連上後離線時間歸零", hwm.offline_duration() == 0.0)
hwm._on_disconnect(None, None, 1)
check("斷線後回報離線", hwm.is_online() is False)
check("publish_raw 可送自訂強度", hwm.publish_raw("3:55:2:200")
      and hwm.client.sent[-1] == "3:55:2:200")

# ============================================================
print("\n── 9. Flask API ──")
import proc_flask_app as fa

class _Q:
    def __init__(self): self.items = []
    def put(self, x): self.items.append(x)

fa.State.ai_cmd_q = _Q()
cli = fa.app.test_client()
r1 = cli.post("/api/height", json={"height_cm": 175})
check("設定合理身高成功", r1.status_code == 200)
check("身高換算為公釐送進 ai_worker",
      fa.State.ai_cmd_q.items[-1].get("cam_height_mm") == 1750.0)
check("超出範圍的身高被拒絕",
      cli.post("/api/height", json={"height_cm": 300}).status_code == 400)
check("非數字身高被拒絕",
      cli.post("/api/height", json={"height_cm": "abc"}).status_code == 400)
check("重設身高會送 reset_height",
      cli.post("/api/height/reset").status_code == 200
      and fa.State.ai_cmd_q.items[-1].get("reset_height") is True)
z = cli.get("/api/zone").get_json()
check("第三代未啟用時分區 API 回 enabled=False", z.get("enabled") is False)
fa.State.gen3_status = {"level": 1, "zone": {"display_name": "廚房", "assumed": False}}
z2 = cli.get("/api/zone").get_json()
check("分區 API 只回名稱與等級", set(z2) == {"enabled", "level", "zone_name", "assumed"}
      and z2["zone_name"] == "廚房")

# ============================================================
print("\n" + "=" * 52)
print(f"通過 {len(PASS)} 項,失敗 {len(FAIL)} 項")
for f in FAIL:
    print("  ❌", f)
sys.exit(1 if FAIL else 0)
