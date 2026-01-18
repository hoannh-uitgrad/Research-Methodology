
# Nghiên cứu và Đánh giá hiệu năng mô hình RT-DETR trong bài toán Phát hiện đối tượng thời gian thực

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-green)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Đề cương môn học: Phương pháp nghiên cứu khoa học (CS2205.SEP2025)** > **Học viên:** Nguyễn Huy Hoàn  
> **MSSV:** 250101022  
> **Trường:** Đại học Công nghệ Thông tin (UIT) - ĐHQG TP.HCM

---

## 📖 Giới thiệu (Introduction)

Dự án này tập trung nghiên cứu, tái hiện và đánh giá hiệu năng của mô hình **RT-DETR (Real-Time Detection Transformer)** - mô hình Transformer đầu tiên đạt được hiệu năng thời gian thực trong bài toán phát hiện đối tượng.

Mục tiêu chính là so sánh RT-DETR với các mô hình **YOLOv8** (State-of-the-Art hiện tại) để chứng minh hiệu quả của việc loại bỏ thuật toán hậu xử lý **NMS (Non-Maximum Suppression)**, từ đó giải quyết vấn đề độ trễ biến thiên và tối ưu hóa quy trình End-to-End.

---

## 🚀 Tính năng nổi bật (Key Features)

* **Real-time End-to-End Object Detection:** Không cần NMS, dự đoán trực tiếp tập hợp đối tượng.
* **Efficient Hybrid Encoder:** Kiến trúc lai kết hợp AIFI (Attention) và CCFF (CNN) giúp tối ưu hóa tốc độ và độ chính xác.
* **Uncertainty-minimal Query Selection:** Cơ chế chọn lọc truy vấn thông minh dựa trên độ không chắc chắn, cải thiện khả năng khởi tạo đặc trưng.
* **High Performance:** Đạt tốc độ và độ chính xác vượt trội so với YOLOv8 trên cùng điều kiện phần cứng.

---

## 🛠️ Kiến trúc mô hình (Model Architecture)

Mô hình RT-DETR bao gồm 3 thành phần chính:

1.  **Backbone:** Sử dụng ResNet/HGNet để trích xuất đặc trưng đa quy mô $\{S_3, S_4, S_5\}$.
2.  **Efficient Hybrid Encoder:**
    * **AIFI (Intra-scale interaction):** Sử dụng Self-Attention trên tầng $S_5$ để nắm bắt ngữ cảnh.
    * **CCFF (Cross-scale fusion):** Sử dụng CNN để hợp nhất các tầng đặc trưng $S_3, S_4, S_5$.
3.  **Transformer Decoder:** Thực hiện dự đoán One-to-one với các truy vấn đối tượng (Object Queries).

![RT-DETR Architecture](assets/architecture_overview.png)
*(Hình ảnh minh họa kiến trúc tổng quan - Figure 4)*

---

## 📊 Thực nghiệm & Kết quả (Experiments & Results)

### Thiết lập thực nghiệm (Setup)
* **Dataset:** MS COCO val2017.
* **Hardware:** NVIDIA Tesla T4 GPU.
* **Environment:** TensorRT FP16.
* **Baseline:** So sánh với YOLOv5, YOLOv8 (Scale L và X).

### Kết quả Benchmark (Benchmark Results)

| Model | Backbone | AP (%) | Latency (ms) | FPS | Params (M) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **YOLOv8-L** | CSP-Darknet | 52.9% | 14.1 | 71 | 43 |
| **RT-DETR-R50** | ResNet-50 | **53.1%** | **9.3** | **108** | 42 |
| | | | | | |
| **YOLOv8-X** | CSP-Darknet | 53.9% | 20.0 | 50 | 68 |
| **RT-DETR-R101** | ResNet-101 | **54.3%** | **13.5** | **74** | 76 |

> **Kết luận:** RT-DETR-R50 vượt trội hơn YOLOv8-L cả về độ chính xác (+0.2% AP) và tốc độ (+52% FPS) nhờ loại bỏ hoàn toàn độ trễ của NMS.

---

## 💻 Cài đặt & Sử dụng (Installation & Usage)

### 1. Yêu cầu (Prerequisites)
* Python 3.8+
* PyTorch 2.0+
* CUDA (khuyến nghị để training/inference GPU)

### 2. Cài đặt (Installation)
```bash
# Clone repository
git clone [https://github.com/hoannh-uitgrad/Research-Methodology](https://github.com/hoannh-uitgrad/Research-Methodology)
cd Research-Methodology

# Install dependencies
pip install -r requirements.txt
pip install ultralytics  # Hoặc cài đặt RT-DETR từ source chính thức

```

### 3. Huấn luyện (Training)

```bash
# Training RT-DETR trên tập COCO
yolo train model=rtdetr-l.pt data=coco8.yaml epochs=100 imgsz=640

```

### 4. Kiểm thử (Validation/Inference)

```bash
# Đánh giá mô hình trên tập validation
yolo val model=rtdetr-l.pt data=coco8.yaml

# Chạy dự đoán trên video/ảnh
yolo predict model=rtdetr-l.pt source='path/to/video.mp4' show=True

```

---

## 🎥 video báo cáo

https://youtu.be/qCSqRuyEheQ

---

## 📚 Tài liệu tham khảo (References)

1. Lv, W., et al. (2024). *DETRs Beat YOLOs on Real-time Object Detection*. CVPR 2024.
2. Jocher, G., et al. (2023). *Ultralytics YOLO*.
3. Carion, N., et al. (2020). *End-to-End Object Detection with Transformers*. ECCV 2020.
4. Zhu, X., et al. (2021). *Deformable DETR: Deformable Transformers for End-to-End Object Detection*. ICLR 2021.

---

## 📬 Liên hệ (Contact)

**Nguyễn Huy Hoàn**

* 📧 Email: hoannh.20@grad.uit.edu.vn
* 🏛️ University of Information Technology (UIT) - VNU-HCM
* 🔗 GitHub: [huyhoanFithcmus](https://github.com/huyhoanFithcmus)

---

*Dự án này là một phần của môn học Phương pháp nghiên cứu khoa học tại trường Đại học Công nghệ Thông tin (UIT).*

