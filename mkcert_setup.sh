#!/usr/bin/env bash
# ================================================================
# mkcert_setup.sh — 產生本機 HTTPS 憑證(語音輸入必需)
# ================================================================
# 為什麼需要:瀏覽器的 SpeechRecognition API 只在「安全來源」下可用。
#   http://192.168.x.x  → 不算安全來源,語音輸入無法啟動
#   https://192.168.x.x → 可以,但需要一張被手機信任的憑證
# 自簽憑證 + 把根 CA 裝進手機,是區網環境下最省事的做法。
#
# ★ 安全性揭露(報告需寫):把 mkcert 的根 CA 裝進手機,等於信任這張 CA
#   簽出的**所有**憑證,不只本系統這一張。這是自簽方案的固有代價。
#   對外正式部署應改用正式網域 + Let's Encrypt。
#
# ★ 憑證綁 IP:Jetson 必須設固定 IP,否則 DHCP 換 IP 後憑證就失效。
# ================================================================
set -e

CERT_DIR="$(dirname "$0")/certs"
IP="${1:-}"

if [ -z "$IP" ]; then
    IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    echo "ℹ️ 未指定 IP,自動偵測為: $IP"
    echo "   (要指定請用: ./mkcert_setup.sh 192.168.1.50)"
fi

if ! command -v mkcert >/dev/null 2>&1; then
    echo "📦 安裝 mkcert..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y libnss3-tools wget
        ARCH=$(dpkg --print-architecture)
        wget -qO /tmp/mkcert "https://dl.filippo.io/mkcert/latest?for=linux/${ARCH}"
        chmod +x /tmp/mkcert && sudo mv /tmp/mkcert /usr/local/bin/mkcert
    else
        echo "❌ 找不到 apt-get,請自行安裝 mkcert 後重跑"
        exit 1
    fi
fi

mkdir -p "$CERT_DIR"
mkcert -install
mkcert -cert-file "$CERT_DIR/cert.pem" -key-file "$CERT_DIR/key.pem" \
       "$IP" localhost 127.0.0.1 ::1

echo ""
echo "✅ 憑證已產生於 $CERT_DIR"
echo "   重啟 main.py 後 https://$IP:5443 即可使用"
echo ""
echo "── 手機安裝根 CA(每支手機各做一次)──"
echo "根 CA 位置: $(mkcert -CAROOT)/rootCA.pem"
echo ""
echo "Android:"
echo "  1. 把 rootCA.pem 傳到手機(改副檔名為 .crt 較容易被辨識)"
echo "  2. 設定 → 安全性 → 加密與憑證 → 安裝憑證 → CA 憑證 → 選檔案"
echo "  3. Chrome 開 https://$IP:5443 應顯示鎖頭"
echo ""
echo "iOS:"
echo "  1. 用 Safari 或 AirDrop 把 rootCA.pem 傳到手機並安裝描述檔"
echo "  2. 設定 → 一般 → 關於本機 → 憑證信任設定 → 開啟完全信任"
echo "  3. Safari 開 https://$IP:5443"
