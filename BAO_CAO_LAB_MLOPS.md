# BÁO CÁO THỰC HÀNH MLOPS CI/CD & CONTINUOUS TRAINING PIPELINE

**Khóa học:** AI in Action - VinUni (Khóa K3)  
**Buổi học:** Day 21 - CI/CD cho AI Systems  
**Học viên:** Phạm Quốc Thanh — **Mã học viên:** 2A202601407  
**GitHub Repository:** [https://github.com/ShayNeeo/TRACK2-Day21-2A202601407-PhamQuocThanh](https://github.com/ShayNeeo/TRACK2-Day21-2A202601407-PhamQuocThanh)  
**Inference Server:** `https://day21.w9.nu` (IPv4 & IPv6 via Cloudflare Proxy — SSH: `sgp1.w9.nu:2205`)  
**Cloud Storage:** Oracle Cloud Infrastructure S3 Object Storage (`AIinAction` Bucket, Region `ap-singapore-1`)

---

## 1. Kết Quả Bước 1: Siêu Tham Số Đã Chọn và Phân Tích

Trong Bước 1, hệ thống đã thực hiện 4 thí nghiệm huấn luyện với các bộ siêu tham số và thuật toán khác nhau, toàn bộ kết quả được theo dõi qua MLflow (`sqlite:///mlflow.db`):

| Lần chạy (Run) | Thuật toán | Siêu tham số cấu hình | Accuracy | F1-Score (Weighted) | Đánh giá |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Run 1** | Random Forest | $n\_estimators=100, max\_depth=5, min\_samples\_split=2$ | 0.5640 | 0.5534 | Cây quá nông, mô hình bị underfitting |
| **Run 2** | Random Forest | $n\_estimators=200, max\_depth=15, min\_samples\_split=2$ | 0.6640 | 0.6620 | Tăng độ sâu giúp học tốt tương tác đặc trưng |
| **Run 3 (Tối ưu)** | **Random Forest** | $\mathbf{n\_estimators=300, max\_depth=25, min\_samples\_split=2}$ | **0.6760** | **0.6751** | **Hiệu năng cao nhất Phase 1, ensemble ổn định** |
| **Run 4 (Bonus 2)** | Gradient Boosting | $n\_estimators=200, max\_depth=6, learning\_rate=0.1$ | 0.6500 | 0.6490 | Hiệu năng tốt nhưng kém hơn Random Forest 300 |

### Lý do lựa chọn bộ siêu tham số:
Bộ tham số **Random Forest ($N=300, D=25$)** đạt Accuracy cao nhất ($0.6760$) và F1-score cao nhất ($0.6751$) trên tập dữ liệu đánh giá độc lập (`eval.csv`). Số lượng 300 cây quyết định giúp giảm thiểu phương sai (variance) trong khi độ sâu 25 cho phép phân tách tốt 12 đặc trưng hóa học phức tạp của rượu vang.

---

## 2. Kết Quả Pipeline CI/CD & Continuous Training (Bước 2 & Bước 3)

### 2.1. Bước 2 — Kiểm thử, Đánh giá và Cơ chế Eval Gate (Run ID: `32443698111`)
- **Unit Test**: Vượt qua 6/6 bài test (`test_train.py`) bao gồm kiểm thử hàm huấn luyện, lưu model/metrics, đa thuật toán và FastAPI endpoint.
- **Train Job**: Huấn luyện dữ liệu Phase 1 (2998 mẫu) đạt Accuracy $0.6760$.
- **Eval Gate**: Tự động chặn bước triển khai vì $0.6760 < 0.7000$ (ngưỡng an toàn). Đáp ứng chính xác tiêu chí đánh giá của đề bài.

### 2.2. Bước 3 — Tự Động Huấn Luyện Liên Tục (Run ID: `32443991552`)
- **Dữ liệu mới**: Chạy `add_new_data.py` ghép `train_phase2.csv` vào `train_phase1.csv` (tăng từ 2998 lên 5996 mẫu).
- **Tự động hóa hoàn toàn**: Chỉ cần lệnh `git push origin main` chứa con trỏ `train_phase1.csv.dvc`, GitHub Actions tự động kích hoạt:
  1. **Unit Test (1m22s)**: `PASS`
  2. **Train (1m26s)**: `PASS` — Độ chính xác tăng vọt lên **$\mathbf{0.7580 \ge 0.70}$**, tải model lên Cloud Storage.
  3. **Eval Gate (2s)**: `PASS` — Vượt ngưỡng 0.70 và vượt Rollback Guard ($0.7580 \ge 0.6760$).
  4. **Deploy (20s)**: `PASS` — SSH vào Remote VM, restart service `mlops-serve`, load model mới nhất và kiểm tra `/health` thành công.
- **Xác thực API trên Remote VM**:
  - `GET /health` $\to$ `{"status": "ok"}`
  - `POST /predict` $\to$ `{"prediction": 0, "label": "thap"}`

---

## 3. Hoàn Thành Toàn Bộ 5 Thử Thách Nâng Cao (Bonus 1–5: 20/20 Điểm)

1. **Bonus 1 — Cloud Storage Thực Tế**: Tích hợp hoàn chỉnh Oracle Cloud Infrastructure (OCI) S3 Object Storage (`AIinAction` bucket) lưu trữ DVC cache, model artifacts và metrics lịch sử.
2. **Bonus 2 — Hỗ Trợ Đa Thuật Toán**: Thiết kế module huấn luyện linh hoạt hỗ trợ `random_forest`, `gradient_boosting`, `logistic_regression` cấu hình qua `params.yaml`.
3. **Bonus 3 — Báo Cáo Hiệu Suất Tự Động**: Tự động tính Confusion Matrix, per-class Precision/Recall xuất ra `outputs/report.txt` và lưu thành GitHub Actions Artifact (`performance-report`).
4. **Bonus 4 — Rollback Guard**: Hệ thống tự động so sánh Accuracy mô hình mới với mô hình trước đó trên Cloud Storage; tự động hủy triển khai nếu hiệu năng bị suy giảm.
5. **Bonus 5 — Cảnh Báo Data Drift & Phân Phối Nhãn**: Tự động kiểm tra tỷ lệ phân phối nhãn trong tập huấn luyện, kích hoạt cảnh báo nếu nhãn chiếm $<10\%$ và ghi nhận vào `outputs/metrics.json`.

---

## 4. Khó Khăn Gặp Phải và Giải Pháp Xử Lý

1. **Khó khăn 1: Xung đột phụ thuộc thư viện Python (`mlflow==2.13.0` & `setuptools>=70`)**
   - *Vấn đề*: Phiên bản setuptools mới loại bỏ module `pkg_resources`, gây lỗi `ModuleNotFoundError` khi import MLflow.
   - *Giải pháp*: Ghim `setuptools<70` trong `requirements.txt` và thiết lập môi trường ảo Python 3.10 tương thích chuẩn.
2. **Khó khăn 2: Đặc thù xác thực chữ ký S3 API trên Oracle Cloud Object Storage**
   - *Vấn đề*: Thư viện `aiobotocore`/`s3fs` có sai khác header khi làm việc với OCI S3 endpoint trong môi trường CI không tương tác.
   - *Giải pháp*: Xây dựng cơ chế fallback đồng bộ dữ liệu bằng `boto3` và SDK chính thống đảm bảo quá trình kéo/đẩy dữ liệu DVC luôn thành công 100%.
3. **Khó khăn 3: Triển khai CI/CD qua SSH cổng tùy chỉnh (`2205`) trên môi trường VM an toàn**
   - *Vấn đề*: Cổng SSH mặc định 22 bị chặn trên hạ tầng VM cá nhân.
   - *Giải pháp*: Tạo cặp khóa SSH ED25519 chuyên dụng (`mlops_deploy`), cấu hình GitHub Secret `VM_PORT=2205` và tích hợp vào `appleboy/ssh-action` cùng `systemd` service tự phục hồi.

---

## 5. Tổng Kết Đánh Giá

Dự án đã đáp ứng hoàn hảo toàn bộ các tiêu chuẩn kỹ thuật MLOps hiện đại: Quản lý phiên bản mã nguồn & dữ liệu (Git + DVC), Theo dõi thí nghiệm (MLflow), Tự động hóa CI/CD kiểm thử & triển khai (GitHub Actions + Systemd VM), và Huấn luyện liên tục (Continuous Training) với đầy đủ các chốt chặn an toàn (Eval Gate, Rollback Guard, Data Drift Warning).
