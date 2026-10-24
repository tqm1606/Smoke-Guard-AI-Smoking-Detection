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
- Upload nhiều file liên tiếp không cần F5. Tốc độ phụ thuộc phần cứng và chưa được benchmark theo giao thức cố định.
- Ngưỡng tin cậy mặc định 0.5: hạ nếu bỏ sót, nâng nếu báo nhầm.


## Kết quả đánh giá và cách tái lập

`results.csv` hiện chỉ ghi **số liệu validation được lưu trong checkpoint**, không phải phép validation/test mới chạy trên máy này. Đọc `train_metrics` trong file `EMA_LSKA_SCRATCH_P3P4P5_640.pt` cho Precision 82.739%, Recall 73.564%, mAP@50 80.341% và mAP@50–95 54.382%. Bốn giá trị khớp epoch 89 trong `train_results`. Metadata ghi split `val`, ảnh 640 px, batch huấn luyện 16, Ultralytics 8.4.86; **không ghi số ảnh, số box nhãn, phần cứng và batch validation**. Checkpoint được tạo ngày 2026-07-03 (metadata không ghi múi giờ). SHA-256: `487c8d87d8e9dbae0ee61281b9a29869e658b68d3dafc306266589e36719f8f1`.

Bản ZIP ứng dụng và repository này không có tập validation/test được gán nhãn, `data.yaml` hay log đánh giá độc lập. Ảnh/video demo và confidence threshold 0.5 không thể dùng để tính Precision, Recall, mAP hoặc Accuracy. Vì vậy hiện **chưa có kết quả test độc lập** và không có FPS đo theo giao thức chuẩn.

### Chạy lại khi có dữ liệu gán nhãn

1. Đặt checkpoint vào `model/EMA_LSKA_SCRATCH_P3P4P5_640.pt` (không commit weights nếu chưa rõ quyền phân phối), cài dependencies bằng `python -m pip install -r requirements.txt`.
2. Chuẩn bị dataset YOLO detection với `data.yaml`, khóa `val` hoặc `test`, `names: {0: cigarette, 1: vape}`. Mỗi ảnh phải có tình trạng nhãn được xác minh: file `.txt` chứa box đúng định dạng, hoặc file rỗng cho ảnh âm tính đã kiểm tra. Ghi số ảnh/box của split và kiểm tra không trùng dữ liệu train. Không dùng ảnh demo không nhãn.
3. Chạy từ thư mục gốc repository, thay đường dẫn dataset và thiết bị cho phù hợp:

```powershell
python -c "import custom_yolo_blocks; from ultralytics import YOLO; m=YOLO('model/EMA_LSKA_SCRATCH_P3P4P5_640.pt'); r=m.val(data='/absolute/path/data.yaml', split='val', imgsz=640, batch=16, device='cpu', plots=False); print({'precision':r.box.mp, 'recall':r.box.mr, 'map50':r.box.map50, 'map50_95':r.box.map})"
```

Chỉ đổi `split='test'` nếu YAML có split test được gán nhãn. Import `custom_yolo_blocks` trước khi load weights để Ultralytics nạp đúng kiến trúc EMA/LSKA. Ghi phiên bản Ultralytics/PyTorch, hash checkpoint và dataset, ngày chạy, số ảnh/box, thiết bị, batch, resolution, confidence/IoU settings cùng bốn chỉ số vào **dòng mới** của `results.csv`; giữ dòng checkpoint lịch sử để phân biệt nguồn. Có thể thay `device='cpu'` bằng GPU thực tế. Chỉ báo FPS sau khi đo có warmup, số lần lặp, cùng phần cứng, batch và resolution.
