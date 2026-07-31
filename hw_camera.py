"""
影像擷取:OAK-D Pro/Lite + webcam fallback
==========================================
本檔除了 RGB/Depth 串流,另外負責兩件事:

① IMU 即時俯角(OAK-D Pro 內建 BNO086)
   樓梯物理模型 depth = k / sin(ray_pitch) 裡的 ray_pitch 依賴相機俯角。
   過去俯角是 config 的固定常數,走路時頭一晃就不準;有 IMU 後改吃即時值。
   ★ 只有 BNO086 支援 rotation vector(四元數),BNO085/沒有 IMU 的批次要
     優雅跳過,不能讓整條 pipeline 建不起來 —— 用 getConnectedIMU() 判型號。

② 相機內參
   AprilTag solvePnP(percept_apriltag)需要 fx/fy/cx/cy,沒有內參就只能用
   針孔近似估距離、拿不到位姿。開機讀一次存起來,供外部查詢。
"""
import math
import time
import threading
import cv2

from shared_config import CONFIG


class CameraManager:
    def __init__(self):
        self.use_oak = False
        self.rgb_frame = None
        self.depth_frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.cap = None
        self.device = None
        self.q_rgb = None
        self.q_depth = None
        self.q_still = None      # OCR 專用:OAK 全解析度靜態擷取
        self.q_control = None    # 送 setCaptureStill() 觸發用
        self.still_lock = threading.Lock()  # capture_still() 同時只能一人觸發
        # ---- IMU ----
        self.q_imu = None
        self.imu_ok = False              # IMU node 是否成功建立
        self.imu_model = None            # getConnectedIMU() 回報的型號
        self._pitch_deg = None           # EMA 平滑後的即時俯角
        self._pitch_ts = 0.0
        # ---- 內參 ----
        self.intrinsics = None           # (fx, fy, cx, cy),讀不到為 None

    def start(self):
        try:
            import depthai as dai
            available = dai.Device.getAllAvailableDevices()
            if len(available) == 0:
                raise RuntimeError("沒偵測到 OAK 裝置")
            print(f"📷 偵測到 {len(available)} 個 OAK 裝置")
            for d in available:
                print(f"   - {d.getMxId()} ({d.state})")
            self._start_oak()
            self.use_oak = True
            print("✅ 使用 OAK-D-Lite (RGB + Depth)")
        except Exception as e:
            print(f"⚠️ OAK 不可用 ({e}), fallback webcam")
            self._start_webcam()
            self.use_oak = False

    def _start_oak(self):
        import depthai as dai
        pipeline = dai.Pipeline()

        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(CONFIG.FRAME_W, CONFIG.FRAME_H)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(30)

        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setFps(30)
        mono_right.setFps(30)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.HIGH_DENSITY
        )
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setExtendedDisparity(False)
        stereo.initialConfig.setMedianFilter(
            dai.StereoDepthProperties.MedianFilter.KERNEL_5x5
        )
        # 註:之前加的整套後處理濾波 (confidence/spatial/temporal/speckle/threshold)
        # 已移除 — 疑似增加 OAK 端負載/USB 頻寬導致 X_LINK_ERROR 斷線。
        # 樓梯改純視覺判定後也不再依賴深度品質,故不需要那套濾波。


        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        x_rgb = pipeline.create(dai.node.XLinkOut)
        x_rgb.setStreamName("rgb")
        cam_rgb.preview.link(x_rgb.input)

        x_depth = pipeline.create(dai.node.XLinkOut)
        x_depth.setStreamName("depth")
        stereo.depth.link(x_depth.input)

        # ---- OCR 專用:1080p still 擷取 (不常態串流,不佔 USB 頻寬) ----
        # 平常 daily/search/traffic 共用的是上面 640x480 preview,給 YOLO/深度用就夠,
        # 但小字 (菜單、站牌) 在這個解析度下 VLM 讀不清楚。still 是「按需觸發」的
        # 單張擷取 (走 setCaptureStill control,不是常態 stream),平時不耗頻寬,
        # 只有呼叫 capture_still() 那一瞬間才觸發一次。
        # ★ 上限受 cam_rgb.setResolution() 綁死:still 不能超過目前感光元件模式
        #   (THE_1080_P → 1920x1080),不是感光元件真正的 13MP 硬體規格。
        #   要拿到真 13MP 得把整條 pipeline 的 setResolution 一起改,但那樣會動到
        #   preview/depth 共用的即時串流,USB 頻寬吃更重,容易重踩之前拔深度後處理
        #   濾波才解決的 X_LINK_ERROR;而且 DepthAI 解析度是建置時固定,不能中途切換,
        #   划不來。1920x1080 已比 640x480 的 YOLO 共用預覽多 4 倍面積,夠用。
        cam_rgb.setStillSize(1920, 1080)
        x_still = pipeline.create(dai.node.XLinkOut)
        x_still.setStreamName("still")
        cam_rgb.still.link(x_still.input)

        x_ctrl = pipeline.create(dai.node.XLinkIn)
        x_ctrl.setStreamName("control")
        x_ctrl.out.link(cam_rgb.inputControl)

        # ---- IMU(可選,失敗自動跳過)----
        imu_wanted = bool(getattr(CONFIG, "IMU_ENABLED", True))
        if imu_wanted:
            try:
                imu = pipeline.create(dai.node.IMU)
                imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR,
                                    int(getattr(CONFIG, "IMU_RATE_HZ", 50)))
                imu.setBatchReportThreshold(1)
                imu.setMaxBatchReports(10)
                x_imu = pipeline.create(dai.node.XLinkOut)
                x_imu.setStreamName("imu")
                imu.out.link(x_imu.input)
                self.imu_ok = True
            except Exception as e:
                print(f"⚠️ IMU node 建立失敗,改用固定俯角: {e}")
                self.imu_ok = False

        self.device = dai.Device(pipeline)

        # ★ 型號檢查必須在 Device 建立後:rotation vector 只有 BNO086 支援。
        #   建置階段無法預知型號,所以先建 node、開起來再驗證,型號不對就停用,
        #   讓所有吃俯角的模組退回 CONFIG.CAM_PITCH_DEG。
        if self.imu_ok:
            try:
                self.imu_model = str(self.device.getConnectedIMU())
                if "BNO086" not in self.imu_model.upper():
                    print(f"⚠️ IMU 型號為 {self.imu_model},非 BNO086,"
                          f"rotation vector 不保證支援 → 停用 IMU 俯角")
                    self.imu_ok = False
                else:
                    print(f"🧭 IMU: {self.imu_model}(rotation vector 已啟用)")
            except Exception as e:
                print(f"⚠️ getConnectedIMU() 失敗,停用 IMU 俯角: {e}")
                self.imu_ok = False
        if self.imu_ok:
            self.q_imu = self.device.getOutputQueue("imu", maxSize=10, blocking=False)
        self.q_rgb = self.device.getOutputQueue("rgb", maxSize=2, blocking=False)
        self.q_depth = self.device.getOutputQueue("depth", maxSize=2, blocking=False)
        self.q_still = self.device.getOutputQueue("still", maxSize=1, blocking=False)
        self.q_control = self.device.getInputQueue("control")

        try:
            calib = self.device.readCalibration()
            K = calib.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_A, CONFIG.FRAME_W, CONFIG.FRAME_H
            )
            self.intrinsics = (float(K[0][0]), float(K[1][1]),
                               float(K[0][2]), float(K[1][2]))
            print(f"📐 RGB 內參 fx={self.intrinsics[0]:.1f} fy={self.intrinsics[1]:.1f} "
                  f"cx={self.intrinsics[2]:.1f} cy={self.intrinsics[3]:.1f}")
        except Exception as e:
            print(f"⚠️ 讀 calibration 失敗 (AprilTag 將退回針孔近似估距): {e}")

        self.running = True
        self.thread = threading.Thread(target=self._oak_loop, daemon=True)
        self.thread.start()

    def _oak_loop(self):
        err_count = 0
        while self.running:
            try:
                in_rgb = self.q_rgb.tryGet()
                in_depth = self.q_depth.tryGet()
                err_count = 0          # 成功讀取 → 清除錯誤計數
            except Exception as e:
                # X_LINK_ERROR 等 USB 通訊斷線:不讓執行緒死掉,記錄後續嘗試。
                err_count += 1
                if err_count == 1 or err_count % 50 == 0:
                    print(f"⚠️ OAK 讀取錯誤 ({err_count} 次): {e}")
                with self.lock:
                    self.ret = False    # 通知主迴圈本幀無效 (會顯示等待畫面)
                time.sleep(0.1)
                continue

            if self.q_imu is not None:
                self._drain_imu()

            with self.lock:
                if in_rgb is not None:
                    self.rgb_frame = in_rgb.getCvFrame()
                    self.ret = True
                if in_depth is not None:
                    depth_data = in_depth.getFrame()
                    if depth_data.shape[:2] != (CONFIG.FRAME_H, CONFIG.FRAME_W):
                        depth_data = cv2.resize(
                            depth_data,
                            (CONFIG.FRAME_W, CONFIG.FRAME_H),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    self.depth_frame = depth_data

            time.sleep(0.005)

    # ============================================================
    # IMU:四元數 → 俯角
    # ============================================================
    def _drain_imu(self):
        """把 IMU queue 讀乾,只留最新一筆換算成俯角。失敗一律靜默停用。"""
        try:
            latest = None
            while True:
                pkt = self.q_imu.tryGet()
                if pkt is None:
                    break
                latest = pkt
            if latest is None:
                return
            rv = latest.packets[-1].rotationVector
            pitch = self._quat_to_pitch_deg(rv.real, rv.i, rv.j, rv.k)
            if pitch is None:
                return
            # 機構夾角補償:IMU 座標系與鏡頭光軸不一定重合,實測後填 config
            pitch += float(getattr(CONFIG, "IMU_PITCH_OFFSET_DEG", 0.0))
            a = float(getattr(CONFIG, "IMU_EMA_ALPHA", 0.25))
            with self.lock:
                self._pitch_deg = (pitch if self._pitch_deg is None
                                   else (1 - a) * self._pitch_deg + a * pitch)
                self._pitch_ts = time.time()
        except Exception as e:
            print(f"⚠️ IMU 讀取失敗,停用 IMU 俯角: {e}")
            self.q_imu = None
            self.imu_ok = False

    @staticmethod
    def _quat_to_pitch_deg(w, x, y, z):
        """
        四元數 → 俯角(度,鏡頭往下為正,與 CONFIG.CAM_PITCH_DEG 同號)。
        取 sin(pitch) = 2(wy - zx),夾在 [-1,1] 防浮點溢出導致 asin 爆掉。
        ★ 軸向與正負號需實機驗證:裝設方向不同,可能要改成 -asin 或換軸。
        """
        try:
            t = 2.0 * (w * y - z * x)
            t = max(-1.0, min(1.0, t))
            return math.degrees(math.asin(t))
        except Exception:
            return None

    def get_pitch(self):
        """
        即時俯角(度)。沒有 IMU、或資料過期 → 回 None,呼叫端自行退回
        CONFIG.CAM_PITCH_DEG。過期門檻預設 1 秒:IMU 掛掉時不能一直餵舊值,
        那比固定常數更危險(使用者以為是即時的)。
        """
        with self.lock:
            if self._pitch_deg is None:
                return None
            if time.time() - self._pitch_ts > float(getattr(CONFIG, "IMU_STALE_SEC", 1.0)):
                return None
            return float(self._pitch_deg)

    def get_intrinsics(self):
        """(fx, fy, cx, cy) 或 None。"""
        return self.intrinsics

    def _start_webcam(self):
        self.cap = cv2.VideoCapture(CONFIG.CAM_ID_HEAD)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.FRAME_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True
        self.thread = threading.Thread(target=self._webcam_loop, daemon=True)
        self.thread.start()

    def _webcam_loop(self):
        fail_count = 0
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    fail_count = 0
                    with self.lock:
                        self.ret = True
                        self.rgb_frame = frame
                        self.depth_frame = None
                else:
                    fail_count += 1
                    time.sleep(0.1)
                    if fail_count > 30:
                        print("⚠️ webcam 訊號遺失,重連")
                        self.cap.release()
                        time.sleep(0.5)
                        self.cap = cv2.VideoCapture(CONFIG.CAM_ID_HEAD)
                        fail_count = 0
            else:
                time.sleep(0.1)

    def capture_still(self, timeout=2.0):
        """
        OCR 專用:觸發 OAK 1920x1080 靜態擷取,回傳 BGR frame 或 None。
        不是 self.rgb_frame (那是 640x480 YOLO 共用預覽) —— 小字要這張才讀得清楚。
        webcam fallback (沒有 OAK) 時直接回 None,呼叫端應自行退回 read() 的預覽幀。
        阻塞呼叫,約 200ms~1s;只在使用者主動觸發 OCR 時呼叫一次,不進主迴圈。
        """
        if not self.use_oak or self.q_control is None or self.q_still is None:
            return None
        with self.still_lock:
            import depthai as dai
            try:
                # 清掉舊的殘留幀,避免拿到上一次的 still
                while self.q_still.tryGet() is not None:
                    pass
                ctrl = dai.CameraControl()
                ctrl.setCaptureStill(True)
                self.q_control.send(ctrl)

                deadline = time.time() + timeout
                while time.time() < deadline:
                    still = self.q_still.tryGet()
                    if still is not None:
                        return still.getCvFrame()
                    time.sleep(0.02)
                print("⚠️ OCR still 擷取逾時,退回預覽幀")
                return None
            except Exception as e:
                print(f"⚠️ OCR still 擷取失敗: {e}")
                return None

    def read(self):
        with self.lock:
            if self.rgb_frame is None:
                return False, None, None
            depth = self.depth_frame.copy() if self.depth_frame is not None else None
            return self.ret, self.rgb_frame.copy(), depth

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        if self.device:
            try: self.device.close()
            except Exception: pass