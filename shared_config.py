"""
全域設定 + 訊號常數 + 中文標籤
所有時間 / 間隔 / 距離 / 震動參數都集中在這裡
"""
import os


class CONFIG:
    # ============================================
    # 相機
    # ============================================
    CAM_ID_HEAD = 0
    FRAME_W = 640
    FRAME_H = 480

    # ============================================
    # 廣角輔助鏡頭 (第二顆 USB cam, 無深度)
    # 用途:每隔 N 秒擷取一張存進環形 buffer,VLM 背景預掃,
    #       建立「物品 + 粗略方位 (左/正前/右) + 時間戳」標籤快取,
    #       供 Search 模式冷啟動時先給粗略引導。
    # ============================================
    WIDE_CAM_ENABLED = True              # 總開關 (沒接第二顆鏡頭就設 False)
    WIDE_CAM_ID = 2                     # USB device id (筆電內建通常 0,外接多為 1)
    WIDE_CAM_W = 1280                    # 640→1280:判色 crop 像素數 ×4 (YOLO 推論成本不變)
    WIDE_CAM_H = 720
    WIDE_CAM_EXPOSURE = None             # 手動曝光 (None=自動)。判色逆光失效時試 -7
    WIDE_CAPTURE_INTERVAL = 2.0          # 每隔幾秒擷取一張存入 buffer (一般/省電)
    WIDE_CAPTURE_INTERVAL_FAST = 0.10    # traffic 模式廣角加速擷取 (紅綠燈需即時)
    WIDE_BUFFER_SIZE = 5                 # 環形 buffer 保留最近幾張

    # ============================================
    # VLM 預掃 (背景, 低頻, 永遠讓位給使用者主動請求)
    # ============================================
    PRESCAN_ENABLED = False              # 總開關 (已停用:VLM 不再背景空轉,改由背景 YOLO 掃描負責物件記憶)
    PRESCAN_INTERVAL = 5.0               # 每隔幾秒做一次背景預掃 (必須 > VLM 單次耗時)
    PRESCAN_USE_CLOUD = False            # 預掃固定走本地, 不燒雲端配額 (True 才用雲端)
    PRESCAN_TAG_TTL = 15.0               # 標籤有效期 (秒);超過視為過期,Search 不採用

    # ============================================
    # 主迴圈
    # ============================================
    MAIN_LOOP_FPS = 30
    SHOW_DEBUG_WINDOW = True
    SHOW_DEPTH_WINDOW = True              # 額外顯示 depth heatmap
    SHOW_WIDE_WINDOW = True               # 額外顯示廣角鏡頭畫面 (紅綠燈主鏡頭,最頭上那顆)

    # ============================================
    # MQTT
    # ============================================
    MQTT_BROKER = "127.0.0.1"
    MQTT_PORT = 1883
    TOPIC_CTRL_VIB  = "guide/control/vibrate"

    # ============================================
    # 模型路徑
    # ============================================
    MODEL_PATHS = {
        "daily":  r"weights/dailywaring_fixed.pt",
        "search": r"weights/yolo26m.pt",
    }
    MODEL_TRAFFIC_LIGHT = r"weights/trafficlight.pt"
    MODEL_CROSSWALK     = r"weights/crosswalk.pt"

    # ============================================
    # 背景全物件掃描 (廣角相機 + YOLO11 官方 COCO 權重)
    # 不限模式,持續建一張「物件 → (時間戳, 鐘向, 方位)」記憶表;
    # 搜尋時先查表,查不到再用 VLM 語意推理。
    # ============================================
    WIDE_CAM_ROTATE_180 = True   # USB 廣角鏡頭實體裝反 → 抓幀時旋轉 180° 校正

    SCAN_BG_ENABLED   = True
    SEARCH_REASON_VLM = False            # 搜尋模式找不到目標時是否用 VLM 推測方位 (已停用:太久,只靠 YOLO+查表)
    MODEL_SCAN        = r"weights/yolo26m.pt"   # 官方 COCO 權重 (n=最輕,可換 s/m)
    SCAN_BG_INTERVAL  = 1.5                      # 多久對廣角幀掃一次 (秒)
    SCAN_BG_CONF      = 0.40                     # 背景掃描信心門檻
    OBJECT_MEMORY_TTL = 60.0                     # 記憶表項目有效期 (秒);超過視為過時不採用
    SCAN_BG_MAX_ITEMS = 40                       # 記憶表最多保留幾種物件 (防無限長)

    CONF_DAILY = 0.5
    CONF_SEARCH = 0.5
    CONF_TRAFFIC_LIGHT = 0.4          # YOLO 類別當備援時的信心門檻 (判色失敗才用)
    CONF_CROSSWALK = 0.5

    # ---- 紅綠燈:YOLO 定位 + HSV 判色 (light_color.py) ----
    CONF_LIGHT_LOCALIZE = 0.25        # 定位用低門檻 (顏色會覆核,框到就好)
    LIGHT_COLOR_MIN_PX = 6            # 框內該色最少像素
    LIGHT_COLOR_MIN_RATIO = 0.02      # 該色佔 crop 面積最低比例
    TRAFFIC_WIDE_ZOOM = (0.10, 0.90, 0.00, 0.65)  # 廣角裁上半中央餵 YOLO = 等效變焦;None 關閉
    LIGHT_FULLFRAME_RED_ASSIST = False  # 全畫面紅塊輔助:只拉紅、永不給綠 (預設關)

    # ============================================
    # 震動模式 (intensity 0-100, freq 1-3, duration_ms)
    # ============================================
    VIB_PATTERNS = {
        "point":  (60,  1, 300),    # 方向提示 (輕)
        "guide":  (75,  2, 400),    # 持續引導 (中)
        "arrive": (100, 3, 200),    # 抵達/對準 (短促重)
        "warn":   (100, 3, 800),    # 警告 (長重)
    }

    # 同一馬達最低觸發間隔 (秒) - 防 MQTT 連發
    HARDWARE_VIB_COOLDOWN = 0.3

    # HUD 馬達指示器亮多久 (秒)
    MOTOR_DISPLAY_DURATION = 0.5

    # 馬達分區視覺化 (debug 視窗畫出 7 馬達對應畫面區間 + 物件→馬達標註)
    SHOW_MOTOR_ZONES = True
    MOTOR_ZONE_ALPHA = 0.16

    # ============================================
    # Daily mode
    # ============================================
    DAILY_WARN_CLASSES = {
        "chair", "sharp", "table", "hole",
         "kettle", "stairs", "cone",
    }

    # 一般類別播報冷卻
    SCAN_INTERVAL = 8.0
    # 一般類別觸發震動的距離
    DAILY_NEAR_DIST_MM = 1500

    # 高優先類別 (立刻警告, 不等冷卻)
    DAILY_CLASS_SEVERITY = {
        "kettle": "high",     # 燙傷
        "cone":   "medium",   # 工地
        "hole":   "high",     # 坑洞
        "sharp":  "high",     # 尖銳
        "stairs": "medium",   # 由 stairs_fusion 接管
    }
    DAILY_CLASS_PREFIX = {
        "kettle": "小心燙",
        "cone":   "施工區",
        "hole":   "注意",
        "sharp":  "注意",
    }
    HIGH_PRIORITY_INTERVAL = 2.0      # 高優先類別自身冷卻 (與樓梯一致 2 秒,不互相打斷)
    HIGH_PRIORITY_NEAR_MM = 2000      # 高優先觸發震動的距離

    # ============================================
    # Depth Anomaly (樓梯) — 地面高度校正法
    # 參考 Cloix et al., EURASIP 2016。用相機俯角把深度校正成
    # 「相對相機水平面的地面高度」,平地校正後≈常數;
    # 落差(下行/路緣)為負、隆起(上行台階)為正。
    # 取捨:視障安全下,漏報下行落差代價遠大於誤報 → 門檻寧可早講。
    # ============================================
    # ★ 現場校準最常調這個:相機鏡頭中軸相對水平往下的角度(度)
    CAM_PITCH_DEG = 30.0
    CAM_HEIGHT_MM = 1750.0            # 相機離地高度 (mm)
    CAM_VFOV_DEG  = 55.0             # OAK-D Lite 垂直視角 (約值)

    DROP_MM = 120.0                  # 低於地面多少 mm 算「落差像素」
    RISE_MM = 120.0                  # 高於地面多少 mm 算「隆起像素」
    RATIO_THRESH = 0.18              # 子圖異常像素比例超過此值 → 該半判定有異常

    DEPTH_SCAN_BAND_W = (0.40, 0.60)  # 取樣窗水平:中央 20% (只看正前方腳下,排除兩側牆壁/路邊/行人)
    DEPTH_SCAN_BAND_H = (0.35, 1.00)  # 取樣窗垂直:0.35(遠)~1.0(腳邊),上下兩半均分
    DEPTH_SCAN_MIN_VALID = 40         # 窗內有效像素少於此值 → 視為無資料 (窗收窄到中央20%,門檻同步調低避免漏報)
    DEPTH_MIN_MM = 200
    DEPTH_MAX_MM = 8000

    DEPTH_TEMPORAL_FRAMES = 5         # 軟體端多幀中位數 (洗單幀崩壞)
    DEPTH_CONFIRM_FRAMES = 12         # 連續 N 幀符合才觸發 (調高:減少瞬間誤觸,平地雜訊難連續 12 幀)
    DEPTH_WARN_COOLDOWN = 2.5         # 播報冷卻 (秒)
    DEPTH_CONFIDENCE_THRESH = 200     # OAK 立體匹配信心門檻 0-255 (越低越嚴格;只留高信心像素,治平地雜訊誤報)

    # ---- RANSAC 地面平面擬合 (取代固定俯角校正,自適應、抗相機晃動) ----
    # 平地上「每列中位深度 vs 列號」是平滑單調關係;RANSAC 擬合這條地面模型,
    # 每像素殘差 (實際深度 − 該列預期地面深度) 即落差/隆起信號:
    #   殘差 > +GROUND_DROP_MM → 比預期地面遠 → 往下掉 (落差)
    #   殘差 < −GROUND_RISE_MM → 比預期地面近 → 隆起 (上行台階)
    # 不需相機俯角/高度/內參,走路晃動也成立。擬合失敗時退回固定俯角法。
    GROUND_RANSAC_ENABLED = True
    GROUND_RANSAC_ITERS = 40          # 迭代次數 (40 足夠且快;對 N 列線性擬合很輕)
    GROUND_RANSAC_INLIER_MM = 80      # 列中位深度離地面模型多少 mm 內算 inlier
    GROUND_RANSAC_MIN_ROWS = 10       # 有效列數少於此 → 放棄 RANSAC,退固定俯角
    GROUND_DROP_MM = 180              # 殘差超過此值 (比地面遠) → 落差像素 (下行訊號強,門檻可高)
    GROUND_RISE_MM = 90               # 殘差低於負此值 (比地面近) → 隆起像素 (上行訊號弱,門檻放低才抓得到)
    # 自適應雜訊門檻:實際門檻 = max(固定門檻, GROUND_NOISE_K × 地面殘差標準差σ)。
    # 平地遠處雜訊大,σ 大 → 門檻自動抬高,雜訊不會被當落差 (治平地誤報下行)。
    GROUND_NOISE_K = 3.0
    # 上半 (遠處) 雜訊大不可靠,要求的異常比例額外乘此倍數 (>1 = 更嚴格,減遠處誤報)
    GROUND_FAR_STRICT = 1.6
    # 上下半各自的觸發比例門檻 (取代單一 RATIO_THRESH 用於 RANSAC 路徑)
    GROUND_DROP_RATIO = 0.30          # 落差:異常像素需佔該半多少比例 (調高治誤報)
    GROUND_RISE_RATIO = 0.22          # 隆起:比例放低 (上行訊號弱,否則漏報)

    # 三點區 (僅供 HUD 顯示距離參考,不參與判斷)
    DEPTH_ZONE_FAR   = (0.50, 0.65)
    DEPTH_ZONE_FRONT = (0.65, 0.80)
    DEPTH_ZONE_NEAR  = (0.80, 1.00)
    DEPTH_H_BAND     = (0.40, 0.60)

    # 樓梯雙來源融合
    STAIRS_FUSION_INTERVAL = 5.0
    STAIRS_YOLO_CLASSES = {"stairs"}
    STAIRS_NEAR_MM = 1500             # 第一階測距 < 此值 → 靠近 (high/warn);否則前方 (medium/guide)

    # ---- v19 第一階測距:地面斷點法 (方向無關) ----
    STAIRS_EDGE_DEV_MM = 150          # 偏離地面線門檻 (與 3σ 取大)。平地誤觸發→調高;階緣認太晚→調低
    STAIRS_EDGE_ROWS = 4              # 連續幾列偏離才認定階緣
    STAIRS_GROUND_MIN_ROWS = 10       # 平地帶最少列數,不足退回保守取近法

    # ============================================
    # Search mode
    # ============================================
    SEARCH_LOST_TIMEOUT = 5.0           # 找不到提示間隔
    SEARCH_GUIDE_INTERVAL = 5.0         # CENTER/REACH 引導間隔
    SEARCH_LOCK_PIXEL_OK = 60           # 抵達判定:像素偏差
    SEARCH_LOCK_PIXEL_TOL = 30          # 鎖定容忍抖動
    SEARCH_LOCK_Z_MM = 80               # 抵達判定:深度差
    SEARCH_LOCK_DURATION = 2.5          # 鎖定持續秒數
    SEARCH_TARGET_LOST_REGRESS = 1.5    # REACH 退回 CENTER 的時間
    SEARCH_HAND_PROMPT_INTERVAL = 5.0   # 提示「請伸手」間隔
    SEARCH_OWN_HAND_PROMPT_INTERVAL = 5.0  # 提示「他人的手」間隔
    SEARCH_AT_TARGET_HOLD_INTERVAL = 5.0   # 抵達後「保持手不動」提示
    SEARCH_REACH_DISTANCE_MM = 1000000      # 目標 < 70cm 才進 REACH (伸手範圍)
    SEARCH_TOO_FAR_INTERVAL = 5.0       # 「對準但太遠」播報冷卻
    SEARCH_MEMORY_TIMEOUT = 3.0         # 用記憶超過此秒數仍未抵達 → 提示重試

    # Search 冷啟動 (用 VLM 廣角預掃標籤先給粗略引導)
    SEARCH_PRESCAN_HINT_ENABLED = False # 已停用:搜尋改為 YOLO記憶→(miss)→VLM即時推理
    SEARCH_PRESCAN_HINT_COOLDOWN = 5.0  # 粗略引導重複播報間隔

    # 馬達方位「遲滯 + 中央死區」(治目標卡邊界時的左右擺盪)
    # 註:死區/遲滯過大會導致「目標偏一點也一律當正前 → 只震中間」,故收斂。
    SEARCH_ZONE_HYS_MARGIN = 0.015  # 要離開目前那格,ratio 需越過邊界這麼多才換
    SEARCH_ZONE_DEADZONE   = 0.04   # |ratio-0.5| 小於此 → 一律當正前(motor 3)

    # 垂直引導 (7 馬達為水平陣列,無法表上下 → 用語音提示抬頭/低頭)
    # 以目標框中心 cy 對畫面高度的比例判斷;偏上緣=頭太低需抬頭,偏下緣=頭太高需低頭。
    SEARCH_VERT_TOP_RATIO    = 0.30  # cy/H < 此 → 目標偏上 → 「請把頭抬高一點」
    SEARCH_VERT_BOTTOM_RATIO = 0.78  # cy/H > 此 → 目標偏下 → 「請把頭往下一點」
    SEARCH_VERT_INTERVAL     = 5.0   # 垂直提示冷卻 (秒)
    # 多目標追蹤:鎖定後若最接近的候選離上次框中心超過此比例(對畫面寬) → 視為已換目標
    SEARCH_TRACK_MAX_JUMP  = 0.35

    # ============================================
    # Traffic mode
    # ============================================
    TRAFFIC_LIGHT_INTERVAL = 1.3        # 紅燈等待提示間隔
    CROSSING_GUIDE_INTERVAL = 1.0       # 過馬路引導間隔
    CROSSING_ARRIVED_FRAMES = 15        # 連續 N 幀沒看到斑馬線 = 抵達
    CROSSING_MIN_DURATION = 5.0         # CROSSING 至少持續 N 秒才能判定抵達
    NO_LIGHT_TIMEOUT = 4.0              # 找不到號誌的間隔
    CROSSING_OFFSET_OK = 40             # 直走容忍偏移 (px)
    CROSSING_ANGLE_OK = 15             # 中線傾斜容忍角度 (度);超過視為使用者面向歪 → 引導轉頭

    # 斑馬線方向估計 (PCA 主軸) 與時序穩定化
    CROSSWALK_MIN_MASK_PX   = 500       # mask 前景像素少於此 → 視為沒抓到
    CROSSWALK_MIN_REGION_PX = 50        # 上/下取樣帶最少像素
    CROSSWALK_TOP_BAND      = 0.40      # 遠端取樣帶:y < 0.40H
    CROSSWALK_BOTTOM_BAND   = 0.66      # 近端取樣帶:y > 0.66H
    # (原 CROSSWALK_PCA_MIN_RATIO 已移除:PCA 主軸法實測不準,已從 crosswalk.py 拿掉)
    CROSSWALK_EMA_ALPHA     = 0.4       # offset/angle EMA 係數 (越小越穩但越鈍)
    CROSSWALK_ALMOST_CONFIRM = 3        # "即將到對岸" 需連續確認幀數
    CROSSWALK_MISS_RESET    = 5         # 連續沒抓到幾幀就重置平滑狀態

    # Hough 條紋斜率法 (crosswalk_hough.py;PCA angle 抓不準時的替代/增強)
    # 在 mask 範圍內對原始灰階抓條紋邊緣,取走道方向加權平均,比填滿區域 PCA 穩。
    CW_HOUGH_CANNY_LO       = 50        # Canny 低門檻
    CW_HOUGH_CANNY_HI       = 150       # Canny 高門檻
    CW_HOUGH_THRESH         = 30        # HoughLinesP 累積票數門檻
    CW_HOUGH_MIN_LEN        = 25        # 線段最短長度 (px);太短的雜訊不要
    CW_HOUGH_MAX_GAP        = 8         # 同線段最大間隙 (px)
    CW_HOUGH_MIN_LINES      = 4         # 至少抓到幾條才採信 (否則退 PCA)
    CW_HOUGH_MIN_CONSISTENCY = 0.85     # 方向一致性 (cos),太發散退 PCA

    # Tian et al. (2021) 斑馬線方向法參數 (crosswalk_tian.py)
    TIAN_WHITE_S_MAX     = 60      # HSV 白色萃取:飽和度上限
    TIAN_WHITE_V_MIN     = 150     # HSV 白色萃取:明度下限
    TIAN_CANNY_LO        = 50
    TIAN_CANNY_HI        = 150
    TIAN_MIN_LEN_RATIO   = 0.25    # 條件(a):線長 ≥ 畫面寬此比例
    TIAN_HOUGH_THRESH    = 50
    TIAN_HOUGH_MAX_GAP   = 10
    TIAN_THETA_MAX_DEG   = 10.0    # 條件(b):θ ∈ [0, 此值]
    TIAN_MIN_LINES       = 10      # 條件(c):通過線數門檻
    RED_WARN_INTERVAL = 2.5             # 過馬路時變紅燈警告間隔
    CROSSING_NO_MASK_HINT_INTERVAL = 2.0  # 「看不清楚斑馬線」間隔

    # 燈號穩定化 (連續 N 幀才確認狀態切換)
    LIGHT_CONFIRM_RED = 1               # 紅燈 1 幀即可 (保守安全)
    LIGHT_CONFIRM_GREEN = 5             # 綠燈 5 幀 (~0.17 秒, 避免誤判闖紅燈)
    LIGHT_CONFIRM_NONE = 10             # 無號誌 10 幀 (~0.33 秒, 避免遮擋誤觸發)

    # ============================================
    # VLM
    # ============================================
    VLM_MODEL_NAME = "qwen3-vl:2b"
    VLM_USE_CLOUD = True
    # 鎖小 context,避免 KV cache 撐爆 VRAM 掉 CPU。
    # 描述/OCR 任務 prompt+圖片 token 遠不到 4096,綽綽有餘。
    VLM_NUM_CTX = 4096
    # API key 一律從環境變數讀取,不在程式碼留明文。
    # 設定方式 (PowerShell): $env:GEMINI_API_KEY="你的key"
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_MODEL = "gemini-2.5-flash"       # 免費層可用,多模態 (gemini-3.5-flash 不存在)

    # VLM busy 卡死防護:超過此秒數沒收到 is_final 自動解鎖
    VLM_BUSY_TIMEOUT = 30.0

    # Ollama HTTP API 端點 (直接走 REST,base64 餵圖,標準 /api/chat vision 格式)
    OLLAMA_HOST = "http://127.0.0.1:11434"

    # YOLO 強制推論裝置 (0 = 第一張 NVIDIA GPU;"cpu" = 退回 CPU)
    # 雙顯卡 + spawn 多進程下 ultralytics 自動偵測會誤判掉到 CPU,故明確指定。
    YOLO_DEVICE = 0

    # ============================================
    # IMU (OAK-D Pro 內建 BNO086) — 即時俯角
    # rotation vector 只有 BNO086 支援,camera.py 會用 getConnectedIMU() 驗型號,
    # 型號不符/讀取失敗一律停用,所有吃俯角的模組退回 CAM_PITCH_DEG。
    # ============================================
    IMU_ENABLED          = True
    IMU_RATE_HZ          = 50        # rotation vector 取樣率
    IMU_EMA_ALPHA        = 0.25      # 俯角 EMA 平滑係數 (越小越穩越鈍)
    IMU_PITCH_OFFSET_DEG = 0.0       # ★ 機構夾角補償,須實測填入
    IMU_STALE_SEC        = 1.0       # 超過幾秒沒新資料就當沒有 IMU

    # ============================================
    # 自動身高估計 (由 stairs_fusion 的地面模型反推相機離地高度)
    # ★ 手動輸入 = 校驗基準:一旦使用者從 UI 設定過身高,自動估計只做
    #   「差距過大就警告」,不再靜默覆蓋。差距大通常代表 IMU_PITCH_OFFSET_DEG
    #   沒校準好,靜默覆蓋會把校準錯誤變成看不見的系統性偏差。
    # ============================================
    AUTO_HEIGHT_ENABLED     = True
    AUTO_HEIGHT_MIN_MM      = 1200.0
    AUTO_HEIGHT_MAX_MM      = 2100.0
    AUTO_HEIGHT_EMA_ALPHA   = 0.15   # 多次擬合的平滑係數
    AUTO_HEIGHT_MIN_SAMPLES = 5      # 至少累積幾次擬合才採信
    AUTO_HEIGHT_WARN_DIFF_MM = 150.0 # 自動 vs 手動差距超過此值 → 警告
    AUTO_HEIGHT_WARN_COOLDOWN = 60.0

    # ============================================
    # 樓梯 v20 多階量測
    # ============================================
    STAIRS_MAX_STEPS      = 6        # 硬上限
    STAIRS_RELIABLE_N     = 2        # 只有前 N 階標為可信 (相對式誤差會累積)
    STAIRS_DEAD_ROWS      = 12       # 連續幾列無有效深度就停止掃描
    STAIRS_STEP_MIN_MM    = 90       # 判定為「一階」的最小高度跳升
    STAIRS_STEP_ROWS      = 3        # 高度跳升需連續幾列維持
    STAIRS_STEP_H_MIN_MM  = 90       # 合理階高下限 (低於此視為雜訊)
    STAIRS_STEP_H_MAX_MM  = 300      # 合理階高上限

    # ============================================
    # IR 主動/被動深度動態切換 (ir_mode.py)
    # 戶外強光下 IR 點陣被陽光蓋過,主動投影反而讓深度更糟 → 切純被動;
    # 室內/夜間無紋理處被動立體失效 → 切主動。用無效像素比例當指標。
    # ============================================
    IR_MODE_ENABLED       = True
    IR_DOT_BRIGHTNESS_MA  = 800      # 點陣投影器亮度 (mA);depthai 2.15+ 的 mA 版 API
    IR_FLOOD_BRIGHTNESS_MA = 0       # 泛光燈 (夜視灰階用,預設關)
    IR_INVALID_HIGH       = 0.45     # 無效像素比例 > 此 → 需要主動投影
    IR_INVALID_LOW        = 0.20     # 無效像素比例 < 此 → 可以純被動 (遲滯下門檻)
    IR_CONFIRM_FRAMES     = 15       # 連續幾幀符合才切換
    IR_SWITCH_COOLDOWN    = 8.0      # 切換後冷卻 (秒),防在門檻附近抖動

    # ============================================
    # MQTT 斷線提醒
    # ============================================
    MQTT_OFFLINE_WARN_SEC      = 8.0    # 離線超過幾秒語音提醒
    MQTT_OFFLINE_WARN_COOLDOWN = 60.0

    # ============================================
    # HTTPS (語音輸入需要安全來源;自簽憑證方案)
    # ============================================
    HTTPS_ENABLED   = True
    HTTPS_PORT      = 5443
    HTTPS_CERT_FILE = r"certs/cert.pem"
    HTTPS_KEY_FILE  = r"certs/key.pem"

    # ============================================
    # 第三代:AprilTag / Zone Memory / 偽VLM / 觸覺信心編碼
    # 〔建議〕值為工程起始值,需實機校準
    # ============================================
    GEN3_ENABLED  = True
    GEN3_DATA_DIR = r"gen3_data"

    # ---- AprilTag / QR ----
    TAG_SIZE_MM         = 130.0      # 實體邊長 (建議 12-15cm)
    TAG_MIN_MARGIN      = 35.0       # L1 門檻 (僅 pupil_apriltags 有此值)
    TAG_REPEAT_WINDOW   = 1.5        # 無 margin 時連續確認的時間窗 (秒)
    TAG_FRESH_SEC       = 0.5
    TAG_DETECT_INTERVAL = 0.15
    TAG_QUAD_DECIMATE   = 1.5
    TAG_CAM_HFOV_DEG    = 95.0       # 無內參時的距離估計用
    TAG_QR_ENABLED      = True

    # ---- 信心階梯 L1-L5 ----
    L2_MAX_SEC           = 2.0
    L3_MAX_SEC           = 10.0
    L4_MAX_SEC           = 30.0
    SLAM_MIN_FEATURES    = 80
    SLAM_UPDATE_INTERVAL = 0.2
    SLAM_ORB_NFEATURES   = 500
    SLAM_DOWNSCALE       = 0.5
    ZONE_CHANGE_DIST_MM  = 4500.0    # 位移降級 (需 SLAM 位移訊號才生效)
    DOOR_SUSPECT_WINDOW  = 5.0
    ZONE_VISIT_COOLDOWN  = 60.0

    # ---- 偽VLM ----
    PSEUDO_VLM_MIN_TRUST     = 0.45  # 第一層:單筆偵測信心門檻 (環境無關)
    PSEUDO_VLM_WINDOW_FRAMES = 5     # 第一層:滑動視窗長度
    PSEUDO_VLM_MIN_HITS      = 3     # 第一層:視窗內最少命中幀數
    PSEUDO_VLM_SCORE_THRESH  = 15.0  # 總分啟動門檻
    PSEUDO_VLM_MARGIN        = 5.0   # 主導差距門檻
    PSEUDO_VLM_COUNT_DECAY   = 0.5   # 第 N 個同類物件 ×0.5^(N-1)
    PSEUDO_VLM_MAX_COUNT     = 4
    SCENE_VOTE_INTERVAL      = 1.0
    SCENE_HINT_TTL           = 20.0  # L4/L5 備用猜測有效期
    SCENE_HINT_SAY_INTERVAL  = 20.0
    GAP_MARGIN_THRESH        = 5.0   # 記憶落差偵測主導差距門檻
    GAP_CONFIRM_SCANS        = 3     # 連續確認次數

    # ---- 物件記憶 ----
    OBJ_DEDUP_MM          = 300.0
    OBJ_MAX_PER_CLASS     = 5
    OBJ_WRITE_MAX_LEVEL   = 2        # 只有 L1/L2 能寫入座標
    OBJ_HALFLIFE_DAYS     = 7.0
    OBJ_DECAY_WARN_THRESH = 0.40     # 低於此值改「可能已被移動」
    DECAY_TICK_INTERVAL   = 60.0

    # ---- 觸覺信心編碼 ----
    CONF_HAPTIC_ENABLED     = True
    CONF_HAPTIC_MOTOR       = 3      # 2×7 到位後改上排專屬馬達
    CONF_HAPTIC_SAFETY_HOLD = 1.0

    # ---- 隱私 ----
    PRIVACY_RETENTION_DAYS = 30

    # ============================================
    # 中文標籤
    # ============================================
    LABELS_ZH = {
        # 紅綠燈
        "Red_light": "紅燈", "Green_light": "綠燈",
        "crosswalk": "斑馬線",
        # daily 重訓三類
        "kettle": "水壺",
        "stairs": "樓梯",
        "cone":   "三角錐",
        # 既有
        "hole": "坑洞", "sharp": "尖銳物",
        "person": "人", "bicycle": "腳踏車",
        "car": "汽車", "motorcycle": "機車",
        "bus": "公車", "truck": "卡車",
        "chair": "椅子", "table": "桌子", "bench": "長椅",
        "backpack": "背包", "umbrella": "雨傘",
        "handbag": "手提包", "tie": "領帶", "suitcase": "行李箱",
        "fork": "叉子", "spoon": "湯匙", "bowl": "碗", "cup": "杯子",
        "bottle": "瓶子", "door": "門",
        "tv": "電視", "laptop": "筆電",
        "mouse": "滑鼠", "remote": "遙控器", "dining table": "餐桌",
        "toilet": "馬桶", "keyboard": "鍵盤", "cell phone": "手機",
        "microwave": "微波爐", "oven": "烤箱", "sink": "洗手台",
        "refrigerator": "冰箱", "book": "書", "clock": "時鐘",
        "scissors": "剪刀", "hair drier": "吹風機",
    }

    # 預掃別名表 (治召回):VLM 常用不同詞講同一物件。
    # 比對時除了 LABELS_ZH 的標準名,也吃這裡的別名 (僅供 prescan,不影響顯示)。
    # 別名一律 ≥2 字 (避免單字過度命中)。
    PRESCAN_ALIASES = {
        "kettle":      ["保溫瓶", "熱水瓶", "水瓶"],
        "cone":        ["交通錐", "路障", "錐桶", "雪糕筒"],
        "stairs":      ["階梯", "台階", "樓梯間"],
        "motorcycle":  ["摩托車", "重機"],
        "bicycle":     ["自行車", "單車"],
        "car":         ["車子", "轎車", "小客車"],
        "bus":         ["巴士", "公車站"],
        "truck":       ["貨車", "大卡車"],
        "person":      ["行人", "路人"],
        "chair":       ["座椅", "板凳"],
        "table":       ["茶几", "工作台"],
        "dining table":["餐桌椅"],
        "door":        ["大門", "門口", "玻璃門"],
        "refrigerator":["電冰箱"],
        "cell phone":  ["行動電話", "智慧型手機", "智慧手機"],
        "laptop":      ["筆記型電腦", "筆記本電腦"],
        "tv":          ["電視機", "螢幕"],
        "backpack":    ["後背包", "雙肩包"],
        "handbag":     ["提包", "包包"],
        "suitcase":    ["旅行箱", "拉桿箱"],
        "bottle":      ["寶特瓶", "罐子"],
        "cup":         ["馬克杯", "紙杯", "茶杯"],
        "bowl":        ["碗公"],
        "sink":        ["水槽", "流理台", "洗手槽"],
        "umbrella":    ["雨傘架", "陽傘"],
        "toilet":      ["馬桶蓋"],
    }


# ============================================
# 訊號常數
# ============================================
SIG_RETURN_IDLE = "__RETURN_IDLE__"