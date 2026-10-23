# 🚭 Smoke Guard AI — Phát Hiện Hút Thuốc Qua Camera & Video

Phát hiện hành vi hút thuốc lá / vape bằng YOLO11 tùy biến (EMA + LSKA, P3/P4/P5, imgsz 640).
2 chế độ: **Camera trực tiếp** (laptop hoặc điện thoại) và **Upload ảnh/video** có sẵn.

## Chạy nhanh nhất (Windows)

Chỉ cần **đúp chuột** vào file:
- **`Chay web.bat`** — chạy server + tự mở trình duyệt. Dùng cho laptop và upload từ điện thoại.
- **`Chay web (camera dien thoai).bat`** — chạy chế độ HTTPS, bắt buộc khi muốn dùng CAMERA điện thoại.

Cửa sổ đen hiện ra chính là server — thu nhỏ, đừng tắt. Muốn dừng thì đóng nó.

## Cài đặt lần đầu

```powershell
py -m pip install -r requirements.txt
```
(Nếu đã cài từ bản trước, chỉ cần thêm: `py -m pip install waitress cryptography`)

## Chạy bằng lệnh (thay cho .bat)

```powershell
py serve.py            # HTTP thường — laptop + upload điện thoại
py serve.py --https    # HTTPS tự ký — bắt buộc cho CAMERA điện thoại
```

Khi chạy, cửa sổ sẽ in sẵn địa chỉ cho laptop và điện thoại, ví dụ:
```
Trên laptop     : http://localhost:5000
Trên điện thoại : http://192.168.1.15:5000   (cùng WiFi)
```

## Dùng trên ĐIỆN THOẠI (cùng WiFi với laptop)

**Upload ảnh/video:** chạy `py serve.py` (HTTP thường), điện thoại mở `http://<IP-laptop>:5000`.

**Camera điện thoại:** chạy `py serve.py --https`, điện thoại mở `https://<IP-laptop>:5000`,
trình duyệt cảnh báo chứng chỉ → **Nâng cao → Tiếp tục truy cập** → bật camera, chọn cam trước/sau.

> IP laptop: mở terminal gõ `ipconfig`, lấy dòng **IPv4 Address** (mục Wi-Fi).
> Điện thoại không vào được: cho phép Python qua Windows Firewall (Private + Public), cổng 5000.

## Vì sao dùng serve.py thay vì app.py?

`serve.py` chạy bằng **waitress** — server ổn định, xử lý upload ảnh/video lớn từ điện thoại
không bị "mất kết nối" như server dev mặc định của Flask. Chế độ `--https` tạo chứng chỉ tự ký
thật (`cert.pem`/`key.pem`, tạo 1 lần) ổn định hơn chứng chỉ 'adhoc' tạm thời.

`app.py` vẫn chạy được trực tiếp (`py app.py`) để test nhanh trên laptop.

## Ghi chú

- Model load 1 lần — lần phát hiện đầu chậm vài giây (nạp lên GPU), sau đó nhanh.
- `app.py` import `custom_yolo_blocks` trước khi load model (bắt buộc, file .pt cần các class này).
- Upload nhiều file liên tiếp không cần F5. Có GPU (RTX 4050): camera ~10–20 FPS, video 47s ~30–60s.
- Ngưỡng tin cậy mặc định 0.5: hạ nếu bỏ sót, nâng nếu báo nhầm.
