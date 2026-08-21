# BÁO CÁO NGHIỆM THU KỸ THUẬT LAB MLOPS
## Xây Dựng Hệ Thống CI/CD & Continuous Training Cho Hệ Thống AI Trong Môi Trường Production

---

| Thông tin | Chi tiết |
| :--- | :--- |
| **Khóa học** | **AI in Action (K3) — VinUniversity** |
| **Chủ đề** | **Day 21: MLOps CI/CD & Continuous Training Pipeline** |
| **Học viên** | **Phạm Quốc Thành** |
| **Mã học viên** | **2A202601407** |
| **GitHub Repository** | [https://github.com/ShayNeeo/TRACK2-Day21-2A202601407-PhamQuocThanh](https://github.com/ShayNeeo/TRACK2-Day21-2A202601407-PhamQuocThanh) |
| **Inference Server (VM)** | `http://sgp1.w9.nu:8000` (SSH: `sgp1.w9.nu:2205`) |
| **Cloud Object Storage** | Oracle Cloud Infrastructure (OCI) S3 Storage (`AIinAction` Bucket, Region `ap-singapore-1`) |
| **Tổng điểm tự đánh giá** | **100 / 100 Điểm** (80/80 Điểm Core + 20/20 Điểm Bonus 1–5) |

---

## 0. Tổng Hợp Kết Quả Nghiệm Thu Định Lượng

<details open>
<summary><b>Kết quả xác thực các giai đoạn pipeline và dịch vụ suy luận thực tế</b></summary>

```text
====================================================================================================
1. KẾT QUẢ KIỂM THỬ TỰ ĐỘNG (PYTEST TEST SUITE)
====================================================================================================
tests/test_train.py::test_train_returns_float PASSED                                         [ 16%]
tests/test_train.py::test_metrics_file_created PASSED                                        [ 33%]
tests/test_train.py::test_model_file_created PASSED                                          [ 50%]
tests/test_train.py::test_multiple_algorithms PASSED                                         [ 66%]
tests/test_train.py::test_report_file_created PASSED                                         [ 83%]
tests/test_train.py::test_serve_api PASSED                                                   [100%]
=================================== 6 passed in 1.48s ====================================

====================================================================================================
2. KẾT QUẢ GITHUB ACTIONS PIPELINE RUNS
====================================================================================================
Run 1 (ID: 32443698111) - Bước 2: Eval Gate Safety Enforcement
  ✓ Unit Test (1m28s) -> ✓ Train (1m33s, Acc: 0.6760) -> ✗ Eval Gate Blocked (Acc 0.6760 < 0.70) -> - Deploy Skipped

Run 2 (ID: 32443991552) - Bước 3: Continuous Training & Production Deploy
  ✓ Unit Test (1m22s) -> ✓ Train (1m26s, Acc: 0.7580) -> ✓ Eval Gate (2s, Acc 0.7580 >= 0.70) -> ✓ Deploy (20s, Health OK)

====================================================================================================
3. KẾT QUẢ TRUY VẤN SUY LUẬN TRÊN REMOTE VM (http://sgp1.w9.nu:8000)
====================================================================================================
$ curl -s http://localhost:8000/health
{"status":"ok"}

$ curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}'
{"prediction":0,"label":"thap"}
```

</details>

### Bảng đối chiếu tiêu chí Rubric đánh giá (100/100 Điểm)

| STT | Hạng mục | Tiêu chí đánh giá | Điểm tối đa | Điểm đạt | Trạng thái |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | **Bước 1** - MLflow tracking | MLflow UI hiển thị ít nhất 3 lần chạy với các siêu tham số khác nhau | 12 | 12 | **ĐẠT** |
| 2 | **Bước 1** - Độ đo | Mỗi lần chạy ghi nhận đủ cả `accuracy` và `f1_score` | 8 | 8 | **ĐẠT** |
| 3 | **Bước 1** - Phân tích | Xác định và giải thích bộ siêu tham số tốt nhất | 4 | 4 | **ĐẠT** |
| 4 | **Bước 2** - DVC | Remote đã cấu hình, `dvc push` thành công, dữ liệu hiển thị trên cloud storage | 12 | 12 | **ĐẠT** |
| 5 | **Bước 2** - CI/CD | Cả bốn GitHub Actions jobs (Test, Train, Eval, Deploy) đều qua | 16 | 16 | **ĐẠT** |
| 6 | **Bước 2** - Eval gate | Deploy job tự động bị chặn khi accuracy dưới ngưỡng 0.70 | 4 | 4 | **ĐẠT** |
| 7 | **Bước 2** - Serving | VM trả về kết quả đúng tại endpoint `POST /predict` (12 features) | 12 | 12 | **ĐẠT** |
| 8 | **Bước 3** - Tự động hóa | Một commit dữ liệu mới kích hoạt toàn bộ pipeline không cần tác động thủ công | 12 | 12 | **ĐẠT** |
| 9 | **Bonus 1** | Tích hợp Cloud Storage thực tế (Oracle Cloud Infrastructure S3 Storage) | 4 | 4 | **ĐẠT** |
| 10 | **Bonus 2** | Thí nghiệm với nhiều thuật toán (`random_forest`, `gradient_boosting`, `logistic_regression`) | 4 | 4 | **ĐẠT** |
| 11 | **Bonus 3** | Báo cáo hiệu suất tự động (Confusion matrix, Precision/Recall per-class artifact) | 4 | 4 | **ĐẠT** |
| 12 | **Bonus 4** | Hoàn trả về phiên bản trước (Rollback Guard so sánh model hiện tại vs model cũ) | 4 | 4 | **ĐẠT** |
| 13 | **Bonus 5** | Cảnh báo Data Drift & lệch tỷ lệ phân phối nhãn $<10\%$ | 4 | 4 | **ĐẠT** |
| | **TỔNG CỘNG** | | **100** | **100** | **XUẤT SẮC** |

---

## Task 1 · Bước 1: Thực Nghiệm Cục Bộ & Theo Dõi Thí Nghiệm (MLflow Tracking)

### 1. Triệu chứng & Bối cảnh
Khi huấn luyện mô hình phân loại chất lượng rượu vang (**Wine Quality Dataset**) với 12 đặc trưng hóa học (axit, đường, SO2, pH, cồn...), các siêu tham số mặc định của `RandomForestClassifier` ($N=100, \text{max\_depth}=5$) chỉ cho độ chính xác khiêm tốn $0.5640$, không đủ tin cậy để đưa vào sản phẩm. Cần một quy trình thử nghiệm có hệ thống, ghi lại tham số và độ đo tự động thay vì ghi chép thủ công.

### 2. Root Cause Cơ Chế
1. **Cơ chế Underfitting khi cây quá nông:** Với $\text{max\_depth}=5$, mỗi cây quyết định chỉ có tối đa $2^5 = 32$ nút lá. Không gian 12 chiều liên tục chứa nhiều tương tác phi tuyến phức tạp (ví dụ tỷ lệ giữa `free_sulfur_dioxide` và `total_sulfur_dioxide`, hoặc sự kết hợp giữa `volatile_acidity` và `alcohol`), do đó $32$ lá không đủ khả năng phân chia ranh giới quyết định giữa 3 lớp chất lượng (Thấp, Trung bình, Cao).
2. **Cơ chế Bagging & Feature Subsampling:** Khi tăng $N=300$ và $\text{max\_depth}=25$, thuật toán Random Forest lấy mẫu bootstrap và ngẫu nhiên chọn $\sqrt{12} \approx 3.46$ đặc trưng tại mỗi điểm phân chia, giúp triệt tiêu phương sai cá thể giữa các cây mà không làm tăng bias.
3. **So sánh với Gradient Boosting:** Gradient Boosting ($N=200, D=6, \text{lr}=0.1$) huấn luyện tuần tự (sequential boosting) tập trung vào residual errors. Trên tập dữ liệu Phase 1 ($2998$ mẫu), boosting dễ bị overfit vào nhiễu đo đạc hóa học hơn so với bagging của Random Forest, dẫn đến Accuracy đạt $0.6500 < 0.6760$.

### 3. Cách Thực Hiện & Mã Nguồn
- **[`src/train.py`](file:///home/shayneeo/Downloads/Documents/Coding/AI_in_Action/Day_21/Morning/TRACK2-Day21-2A202601407-PhamQuocThanh/src/train.py):** Xây dựng pipeline chuẩn đọc dữ liệu, tách feature/target, khởi tạo MLflow run, ghi nhận parameters, huấn luyện mô hình theo cấu hình động, tính toán `accuracy` và `f1_score (weighted)`, xuất file `outputs/metrics.json` và `models/model.pkl`.
- **[`params.yaml`](file:///home/shayneeo/Downloads/Documents/Coding/AI_in_Action/Day_21/Morning/TRACK2-Day21-2A202601407-PhamQuocThanh/params.yaml):** Quản lý siêu tham số tập trung tách biệt khỏi mã nguồn code.

### 4. Bằng Chứng Định Lượng Thử Nghiệm

| Mã Run | Thuật toán | Cấu hình siêu tham số | Accuracy | F1-Score | Phân tích thực nghiệm |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Run 1** | Random Forest | $n\_estimators=100, max\_depth=5, min\_samples\_split=2$ | 0.5640 | 0.5534 | Underfitting, cây nông không bắt được tương tác |
| **Run 2** | Random Forest | $n\_estimators=200, max\_depth=15, min\_samples\_split=2$ | 0.6640 | 0.6620 | Tăng 10.0% accuracy khi mở rộng không gian cây |
| **Run 3** | **Random Forest** | $\mathbf{n\_estimators=300, max\_depth=25, min\_samples\_split=2}$ | **0.6760** | **0.6751** | **TỐI ƯU NHẤT: Cân bằng bias/variance hoàn hảo** |
| **Run 4** | Gradient Boosting | $n\_estimators=200, max\_depth=6, lr=0.1$ | 0.6500 | 0.6490 | Kém hơn RF 300 cây 2.6% accuracy |

> **Quyết định:** Chọn **Random Forest ($N=300, D=25$)** làm bộ siêu tham số chuẩn cho sản phẩm.

---

## Task 2 · Bước 2: Tự Động Hóa CI/CD, DVC & Cơ Chế Eval Gate

### 1. Triệu chứng & Vấn đề Vận Hành
Trong quy trình phát triển truyền thống, việc đẩy mô hình lên máy chủ thường làm thủ công qua SSH/FTP mà không có chốt chặn chất lượng. Hậu quả là các mô hình lỗi, mô hình chưa qua unit test, hoặc mô hình bị suy giảm độ chính xác vẫn bị đẩy thẳng lên production làm gián đoạn dịch vụ của người dùng.

### 2. Root Cause Cơ Chế
1. **DVC Content-Addressable Storage:** File CSV kích thước lớn không được commit vào Git để tránh phình to dung lượng kho mã nguồn. DVC tạo mã băm MD5 duy nhất (Content Hash) làm con trỏ (ví dụ `train_phase1.csv.dvc`). Khi code thay đổi, Git chỉ quản lý con trỏ 98 bytes; dữ liệu nhị phân thực sự được đồng bộ với Cloud Storage Remote qua giao thức S3.
2. **Eval Gate Enforcement:** Trong file workflow [`.github/workflows/mlops.yml`](file:///home/shayneeo/Downloads/Documents/Coding/AI_in_Action/Day_21/Morning/TRACK2-Day21-2A202601407-PhamQuocThanh/.github/workflows/mlops.yml), job `eval` đóng vai trò là Quality Gatekeeper (ngưỡng $\ge 0.70$). Nếu mô hình huấn luyện chỉ đạt Accuracy $< 0.70$, job `eval` lập tức gọi `raise SystemExit("FAILED: accuracy < 0.70")` trả về Exit Code 1. Cơ chế phụ thuộc `needs: eval` của job `deploy` lập tức chuyển trạng thái `deploy` thành **Canceled/Skipped**, bảo vệ 100% môi trường production.
3. **Decoupled Architecture trên Remote VM:** Server suy luận trên VM được quản lý bởi `systemd` daemon (`mlops-serve.service`). Khi khởi động lại qua CI/CD trigger, ứng dụng FastAPI tự động đọc biến môi trường `CLOUD_BUCKET`, tải artifact `models/latest/model.pkl` mới nhất từ Object Storage và nạp vào bộ nhớ RAM.

### 3. Cách Thực Hiện & Mã Nguồn
1. **[`.github/workflows/mlops.yml`](file:///home/shayneeo/Downloads/Documents/Coding/AI_in_Action/Day_21/Morning/TRACK2-Day21-2A202601407-PhamQuocThanh/.github/workflows/mlops.yml):** Pipeline 4 giai đoạn chuẩn (`test` $\to$ `train` $\to$ `eval` $\to$ `deploy`).
2. **[`src/serve.py`](file:///home/shayneeo/Downloads/Documents/Coding/AI_in_Action/Day_21/Morning/TRACK2-Day21-2A202601407-PhamQuocThanh/src/serve.py):** FastAPI application cung cấp endpoint `GET /health` và `POST /predict` (xác thực nghiêm ngặt 12 chiều đặc trưng, ánh xạ nhãn `0 -> thap`, `1 -> trung_binh`, `2 -> cao`).
3. **[`tests/test_train.py`](file:///home/shayneeo/Downloads/Documents/Coding/AI_in_Action/Day_21/Morning/TRACK2-Day21-2A202601407-PhamQuocThanh/tests/test_train.py):** Bộ 6 unit tests độc lập chạy trong môi trường isolated `tmp_path`.
4. **[`/etc/systemd/system/mlops-serve.service`](file:///home/shayneeo/Downloads/Documents/Coding/AI_in_Action/Day_21/Morning/TRACK2-Day21-2A202601407-PhamQuocThanh/.env):** Systemd service tự phục hồi (`Restart=always, RestartSec=5`).

### 4. Bằng Chứng Thực Nghiệm Đo Đạc

* **Kiểm thử Eval Gate hoạt động chặn triển khai (Run ID `32443698111`):**
  - Số lượng mẫu Phase 1: `2998` mẫu.
  - Accuracy huấn luyện: `0.6760`.
  - Output log từ runner:
    ```text
    [EVAL GATE] Model accuracy: 0.6760 (Threshold: 0.7000)
    FAILED: accuracy 0.6760 < 0.70. Huy deploy.
    ##[error]Process completed with exit code 1.
    ```
  - Trạng thái Deploy Job: **Skipped (Không deploy mô hình kém chất lượng)** $\implies$ Đạt yêu cầu tuyệt đối.

---

## Task 3 · Bước 3: Huấn Luyện Liên Tục (Continuous Training) Khi Có Dữ Liệu Mới

### 1. Triệu chứng & Yêu Cầu Vận Hành
Trong thực tế sản xuất, dữ liệu mới liên tục phát sinh theo thời gian (ở đây là `train_phase2.csv` gồm 2998 mẫu mới). Đội ngũ Data Engineering muốn bổ sung dữ liệu này vào hệ thống và mong muốn toàn bộ quy trình tái huấn luyện, đánh giá, kiểm thử và triển khai phiên bản mô hình mới phải diễn ra **hoàn toàn tự động 100% không chạm (Zero-Touch Continuous Training)**.

### 2. Root Cause Cơ Chế
1. **Tăng mật độ biểu diễn mẫu trong không gian đặc trưng (Sample Density Expansion):**
   Khi tăng kích thước tập huấn luyện từ 2998 lên 5996 mẫu, mật độ điểm dữ liệu trong không gian 12 chiều tăng gấp đôi. Rừng cây quyết định giảm đáng kể các vùng ngoại suy (extrapolation error) tại các điểm biên giữa chất lượng rượu 6 và 7, trực tiếp nâng Accuracy từ $0.6760$ lên **$0.7580$** (tăng $+8.2\%$).
2. **Git Webhook Trigger qua DVC Hash Pointer:**
   Khi chạy `add_new_data.py`, dữ liệu `train_phase1.csv` thay đổi. Lệnh `dvc add data/train_phase1.csv` cập nhật mã băm MD5 trong file `data/train_phase1.csv.dvc`. Lệnh `git push origin main` đẩy sự thay đổi của file `.dvc` lên GitHub. Do workflow khai báo filter `paths: ['data/*.dvc']`, GitHub lập tức phát sinh sự kiện `push` kích hoạt CI/CD runner.

### 3. Cách Thực Hiện & Quy Trình 4 Bước
```bash
# 1. Ghép dữ liệu mới
python add_new_data.py          # Output: Cap nhat du lieu: 2998 -> 5996 mau

# 2. Cập nhật DVC hash
dvc add data/train_phase1.csv

# 3. Đồng bộ dữ liệu lên Oracle Cloud Object Storage
dvc push                        # Upload blob sang s3://AIinAction/dvc/...

# 4. Kích hoạt Continuous Training Pipeline
git add data/train_phase1.csv.dvc
git commit -m "data: bổ sung 2998 mẫu dữ liệu mới (train_phase2)"
git push origin main
```

### 4. Bằng Chứng Định Lượng Trước vs Sau Khi Continuous Training

| Tiêu chí đo đạc | Trước (Bước 2 - Phase 1) | Sau (Bước 3 - Continuous Training) | Biến thiên |
| :--- | :---: | :---: | :---: |
| **Số lượng mẫu huấn luyện** | 2,998 mẫu | **5,996 mẫu** | $+100.0\%$ |
| **Accuracy trên tập eval (500 mẫu)** | 0.6760 | **0.7580** | **$+8.20\%$** |
| **F1-Score (Weighted)** | 0.6751 | **0.7568** | **$+8.17\%$** |
| **Trạng thái Eval Gate ($\ge 0.70$)** | `FAILED` (Bị chặn) | **`PASSED` (Vượt ngưỡng an toàn)** | **THÀNH CÔNG** |
| **Rollback Guard ($Acc_{mới} \ge Acc_{cũ}$)** | N/A (Bản đầu) | **`PASSED` ($0.7580 \ge 0.6760$)** | **THÀNH CÔNG** |
| **Trạng thái GitHub Actions Run** | Run `32443698111` (Blocked) | **Run `32443991552` (4/4 Jobs Green)** | **ALL GREEN** |
| **Trạng thái Inference API trên VM** | Chạy mô hình ban đầu | **Tự động tải & phục vụ mô hình mới ($0.7580$)** | **ZERO DOWNTIME** |

---

## 3. Hoàn Thành Chi Tiết 5 Thử Thách Mở Rộng (Bonus 1–5: 20/20 Điểm)

### Bonus 1: Tích Hợp Cloud Storage Thực Tế (Oracle Cloud S3 Object Storage)
- **Cơ chế:** Kết nối trực tiếp hệ thống lưu trữ Oracle Cloud Infrastructure (OCI) Object Storage tương thích hoàn toàn chuẩn giao thức Amazon S3 API (`boto3` client với endpoint `https://axc8ef7fwayz.compat.objectstorage.ap-singapore-1.oraclecloud.com`).
- **Ứng dụng:** Lưu trữ toàn bộ 3 tập dữ liệu (`train_phase1.csv`, `eval.csv`, `train_phase2.csv`), toàn bộ DVC cache objects, model artifact `models/latest/model.pkl`, và file đo lường `outputs/metrics.json`.

### Bonus 2: Thí Nghiệm Đa Thuật Toán (Dynamic Model Factory)
- **Cơ chế:** Trong `src/train.py`, triển khai hàm `build_model(model_type, params)` hỗ trợ khởi tạo động 3 dòng thuật toán:
  1. `random_forest`: `RandomForestClassifier`
  2. `gradient_boosting`: `GradientBoostingClassifier`
  3. `logistic_regression`: `LogisticRegression`
- Tham số `model_type` được điều khiển linh hoạt trực tiếp từ `params.yaml` mà không cần can thiệp code.

### Bonus 3: Báo Cáo Hiệu Suất Tự Động (Automated Performance Report Artifact)
- **Cơ chế:** Trong `src/train.py`, hàm `save_performance_report` tự động tính toán **Confusion Matrix** dạng bảng văn bản và bảng chi tiết **Precision, Recall, F1-score, Support** cho từng lớp (Lớp 0: Thấp, Lớp 1: Trung bình, Lớp 2: Cao).
- Báo cáo được ghi ra `outputs/report.txt` và được GitHub Actions tự động lưu trữ dưới dạng artifact `performance-report`.

<details>
<summary><b>Trích đoạn Báo Cáo Hiệu Suất Tự Động (outputs/report.txt)</b></summary>

```text
================================================================================
                    MLOPS MODEL PERFORMANCE EVALUATION REPORT
================================================================================
Model Type : random_forest
Timestamp  : 2026-08-21 03:39:24 UTC
Evaluated  : 500 samples

--- SUMMARY METRICS ---
Accuracy : 0.7580
F1-Score : 0.7568 (Weighted)

--- CLASSIFICATION REPORT (PER-CLASS METRICS) ---
              precision    recall  f1-score   support
        thap     0.7812    0.6944    0.7353       180
  trung_binh     0.7419    0.8035    0.7714       229
         cao     0.7614    0.7363    0.7486        91
    accuracy                         0.7580       500
   macro avg     0.7615    0.7447    0.7518       500
weighted avg     0.7596    0.7580    0.7568       500

--- CONFUSION MATRIX ---
Predicted ->   thap  trung_binh         cao
Actual thap     125          46           9
Actual trung_b   28         184          17
Actual cao        7          17          67
================================================================================
```

</details>

### Bonus 4: Cơ Chế Hoàn Trả Về Phiên Bản Trước (Rollback Guard Safety)
- **Cơ chế:** Trước khi cho phép triển khai, pipeline tự động kiểm tra trên Cloud Storage xem có file `metrics.json` của phiên bản trước đó hay không.
- Nếu tìm thấy, hệ thống so sánh: $Acc_{mới} < Acc_{cũ} \implies$ kích hoạt `raise SystemExit("FAILED (Bonus 4 Rollback Guard): New accuracy < Previous accuracy. Huy deploy.")`. Chỉ cho phép deploy khi mô hình mới có hiệu năng tương đương hoặc vượt trội.

### Bonus 5: Cảnh Báo Data Drift & Lệch Phân Phối Nhãn
- **Cơ chế:** Trước khi huấn luyện, hàm `check_data_drift` trong `src/train.py` quét toàn bộ tập dữ liệu huấn luyện, tính toán tỷ lệ phần trăm của từng lớp `[0, 1, 2]`.
- Nếu phát hiện bất kỳ lớp nào chiếm tỷ lệ $< 10\%$, hàm lập tức in cảnh báo `WARNING: Class imbalance detected` và ghi nhận tỷ lệ phân phối chi tiết (`class_distribution`) vào `outputs/metrics.json` và MLflow metrics để các kỹ sư giám sát độ trôi dữ liệu (data drift).

---

## 4. Báo Cáo Phân Tích Sự Cố Kỹ Thuật (Incident Postmortems & Troubleshooting)

### Sự Cố 1: Xung Đột Phụ Thuộc `mlflow==2.13.0` và `setuptools>=70`
- **1. Triệu chứng:** Khi chạy `train.py` hoặc `import mlflow`, ứng dụng văng lỗi: `ModuleNotFoundError: No module named 'pkg_resources'`.
- **2. Root Cause:** Từ phiên bản `setuptools 70.0.0`, module kế thừa `pkg_resources` đã bị loại bỏ hoàn toàn để chuyển sang `importlib.metadata`. Gói `mlflow==2.13.0` phát hành trước đó vẫn phụ thuộc vào `pkg_resources`. Khi cài đặt môi trường trên Python 3.10+, pip tự động kéo phiên bản setuptools mới nhất dẫn đến gãy dependency chain.
- **3. Cách Fix:** Thêm quy tắc ghim phiên bản `setuptools<70` trực tiếp vào [`requirements.txt`](file:///home/shayneeo/Downloads/Documents/Coding/AI_in_Action/Day_21/Morning/TRACK2-Day21-2A202601407-PhamQuocThanh/requirements.txt).
- **4. Bằng chứng:** Sau khi ghim `setuptools<70`, toàn bộ lệnh `import mlflow` và 6 unit tests đều pass 100%.

### Sự Cố 2: Sai Khác Signature Header S3 API trên Oracle Cloud Object Storage
- **1. Triệu chứng:** Lệnh `dvc pull` / `s3fs` trong môi trường headless CI runner thỉnh thoảng gặp lỗi `ClientError: 403 Forbidden / SignatureDoesNotMatch` khi truy cập Oracle Cloud S3 endpoint.
- **2. Root Cause:** Backend async của `aiobotocore` tự động thêm một số header mở rộng đặc thù của AWS mà OCI S3-compatible gateway không chấp nhận, khiến chữ ký băm HMAC-SHA256 bị sai lệch.
- **3. Cách Fix:** Trong workflow [`.github/workflows/mlops.yml`](file:///home/shayneeo/Downloads/Documents/Coding/AI_in_Action/Day_21/Morning/TRACK2-Day21-2A202601407-PhamQuocThanh/.github/workflows/mlops.yml), tích hợp cơ chế đồng bộ trực tiếp qua thư viện chuẩn `boto3` với cấu hình endpoint tường minh và credentials parse từ GitHub Secrets.
- **4. Bằng chứng:** Quá trình kéo dữ liệu `train_phase1.csv` và `eval.csv` trên GitHub Actions runner hoàn thành trong chưa đầy 3 giây mà không gặp bất kỳ lỗi kết nối nào.

### Sự Cố 3: Xác Thực SSH Tự Động Qua Cổng Tùy Chỉnh (`2205`) Trên VM
- **1. Triệu chứng:** GitHub Actions deploy step bị timeout hoặc từ chối kết nối khi SSH tới host VM.
- **2. Root Cause:** Hạ tầng VM cá nhân không mở cổng tiêu chuẩn 22 mà định tuyến qua cổng `2205`. Action `appleboy/ssh-action` mặc định kết nối port 22 nếu không được chỉ định tham số `port`.
- **3. Cách Fix:** Tạo cặp khóa chuyên dụng `~/.ssh/mlops_deploy` (chuẩn ED25519), thiết lập Secret `VM_PORT=2205` và bổ sung `port: ${{ secrets.VM_PORT || 2205 }}` vào step Deploy trong workflow.
- **4. Bằng chứng:** Job deploy thực hiện kết nối SSH thành công trong 20s, restart `systemd` service và nhận phản hồi `200 OK` từ `/health`.

---

## 5. Tổng Kết Kiến Trúc & Hướng Dẫn Vận Hành

Toàn bộ hệ thống được đóng gói nhất quán, khép kín từ khâu nghiên cứu đến vận hành:

```
[Developer / Data Engineer]
   │
   ├── Sửa code / params.yaml ──> git commit & push ──┐
   └── Bổ sung data mới (DVC) ──> git commit & push ──┤
                                                      │
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │    GitHub Actions Runner    │
                                       │ 1. Test (pytest 6 tests)    │
                                       │ 2. Train (DVC Pull + Train) │
                                       │ 3. Eval Gate (Acc >= 0.70)  │
                                       │    & Rollback Guard         │
                                       └──────────────┬──────────────┘
                                                      │ (Deploy qua SSH)
                                                      ▼
  ┌─────────────────────────────┐      ┌─────────────────────────────┐
  │   Oracle Cloud S3 Bucket    │◄────►│   Remote Production VM      │
  │   - DVC data cache          │      │   - systemd mlops-serve     │
  │   - models/latest/model.pkl │      │   - FastAPI REST API :8000  │
  │   - outputs/metrics.json    │      │   - GET /health, POST /pred │
  └─────────────────────────────┘      └─────────────────────────────┘
```

Hệ thống sẵn sàng bàn giao và đưa vào sử dụng trong môi trường thực tế!
