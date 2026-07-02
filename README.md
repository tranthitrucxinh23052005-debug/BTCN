# 🧠 Rào Cản Tâm Lý Với AI (AI Psychological Barriers Dashboard)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]((https://btcndv.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"Không phải AI chưa đủ thông minh để làm việc, mà là con người có muốn giao việc cho nó hay không."** > 
> Bảng điều khiển (Dashboard) này phân tích các yếu tố dự báo sự kháng cự với tự động hóa AI ở người lao động, sử dụng bộ dữ liệu WORKBank (5.731 đánh giá × 104 nghề × 1.500 người lao động).

---

## 1. 🚀 Triển Khai (Deployment & Links)
* **Live Dashboard (Streamlit):** [Xem ứng dụng tại đây](https://btcndv.streamlit.app/)
* **Dataset gốc (SALT-NLP/WORKBank):** [HuggingFace](https://huggingface.co/datasets/SALT-NLP/WORKBank)
* **Mã nguồn chính:** `l8.py`

<img width="956" height="425" alt="image" src="https://github.com/user-attachments/assets/68c78c1f-9339-4e2f-8809-52b5a2574eca" />


---

## 2. 💡 Ý Tưởng Cốt Lõi (Core Idea)
Phần lớn các báo cáo hiện nay chỉ tập trung vào "nỗi sợ mất việc" hoặc góc nhìn kinh tế của AI. Dự án này khai thác một góc nhìn hoàn toàn khác: **Rào cản tâm lý và Bản sắc nghề nghiệp**. 

Biến trung tâm của dự án là `trust_gap` (Khoảng trống niềm tin):
`trust_gap = Năng lực AI (Chuyên gia đánh giá) - Mong muốn tự động hóa (Người lao động tự đánh giá)`

Khi `trust_gap` > 0, AI hoàn toàn có khả năng làm tốt tác vụ, nhưng người lao động lại từ chối giao việc. Dự án dùng thống kê suy luận để tìm ra **nguyên nhân thực sự** đằng sau sự từ chối này.

---

## 3. ⚙️ Luồng Xử Lý Dữ Liệu (Workflow)

Quá trình ETL và xây dựng mô hình được thực hiện hoàn toàn tự động qua các bước:

1. **Data Aggregation:** Gộp điểm năng lực AI (`expert_rated_technological_capability.csv`) theo `Task ID` (từ 2.057 dòng thô xuống 846 tác vụ duy nhất).
2. **Merging & Feature Engineering:** Gộp dữ liệu năng lực với bảng mong muốn (`domain_worker_desires.csv`) và nhân khẩu học (`domain_worker_metadata.csv`), tính toán các biến phái sinh như `trust_gap` và chuẩn hóa Z-score.
3. **Mô Hình Hồi Quy (OLS):** Chạy hồi quy tuyến tính bội bằng `numpy.linalg.lstsq` và `statsmodels` để kiểm soát các biến nhiễu (tuổi, lương, học vấn).
   $$\text{Automation Desire}_i = \alpha + \sum_{k} \beta_k \cdot z(X_k)_i + \varepsilon_i$$
4. **Phân Rã Phương Sai (HLM):** Tính toán hệ số tương quan nội bộ (ICC) để xem rào cản đến từ cấu trúc ngành nghề hay từ đặc điểm cá nhân.
5. **Interactive Visualization:** Hiển thị tương tác bằng Plotly và Streamlit, tích hợp Gemini API làm trợ lý ảo.

---

## 4. 📊 Insights & Kết Quả Phân Tích (Key Findings)

<img width="534" height="268" alt="image" src="https://github.com/user-attachments/assets/4346240e-c46e-4729-8148-58c476d55987" />
<img width="619" height="417" alt="image" src="https://github.com/user-attachments/assets/f994db09-5221-40dd-9701-48906595a21a" />

* **Nghịch lý của sự thích thú (Enjoyment Paradox):** Người càng thích công việc của mình thì càng kháng cự AI, bất kể năng lực AI tốt đến đâu (Tương quan âm mạnh nhất: r = -0.34, p < 0.001).
* **Đạo đức vs. Kinh tế:** Niềm tin "AI có thể chịu khổ" (Moral Patienthood) có sức mạnh dự báo cao hơn nhiều so với "Lo ngại mất việc" (Job Security).
* **Hiệu ứng Khuếch đại Kỹ năng (Matthew Effect):** Tác động của AI không đồng đều. Người có học vấn cao (Thạc sĩ/Tiến sĩ) khi dùng AI thường xuyên sẽ gỡ bỏ rào cản tâm lý cực kỳ nhanh so với nhóm yếu thế (Hệ số tương tác β > 0, p < 0.001).
* **Nghịch lý ngành Công Nghệ (SOC 15):** Những kỹ sư IT (Database Admin, Programmer) hiểu rõ năng lực AI nhất lại có mức `trust_gap` cao kỷ lục. Họ từ chối AI không phải vì sợ, mà để bảo vệ "quyền kiểm soát" và "cái tôi chuyên môn".

---

## 5. 🎯 Kết Luận (Conclusion)
Sự kháng cự với AI trong kỷ nguyên mới **không phải là một bài toán kinh tế, mà là một bài toán tâm lý học hành vi.** Người lao động, đặc biệt là nhóm chuyên gia trình độ cao, coi AI như một mối đe dọa đối với **bản dạng nghề nghiệp (Identity Threat)** của họ. Họ viện cớ "AI thiếu chuyên môn ngành" để giữ lại phần công việc mà họ yêu thích hoặc tự hào, tạo ra một rào cản ngầm ngay cả khi AI đã đạt mức hoàn thiện.

---

## 6. 🛠️ Khuyến Nghị Ứng Dụng (Recommendations)

Dựa trên dữ liệu định lượng, dự án đề xuất chiến lược ứng dụng AI cho doanh nghiệp:

1. **Tự động hóa phần việc "nhàm chán", giữ lại phần "yêu thích":** Đừng cố gắng tự động hóa toàn bộ quy trình. Hãy định vị AI để xử lý các tác vụ có Enjoyment Rating thấp.
2. **Kiến trúc "Calibrated AI Agent":** Đề xuất thiết kế Agent có 3 tầng:
   * Nhận thức bản sắc (Identity-Aware).
   * Hiệu chuẩn năng lực động (If tác vụ khó $\rightarrow$ Chuyển sang chế độ Co-Pilot/QA).
   * Giao diện minh bạch (Giải thích tư duy thay vì Hộp đen).
3. **Đào tạo có chủ đích:** Cần có lớp hướng dẫn sử dụng AI riêng cho nhóm học vấn trung bình/thấp, tránh để xảy ra bất bình đẳng do hiệu ứng "Khuếch đại kỹ năng".
<img width="395" height="394" alt="image" src="https://github.com/user-attachments/assets/6e51b2b7-c98f-4310-af4e-90b2b3855882" />


---

## 7. 🧠 Tư Duy Phân Tích (Data Strategist Mindset)
Dự án được xây dựng với tư duy tránh bẫy "So What?" và đi tìm các **Plot Twist (Cú lật ngược trực giác)**:
* Thay vì phân tích xem "người lương thấp có sợ AI không" (góc nhìn cũ), dự án đưa biến *thái độ đạo đức* vào mô hình đa biến.
* Thay vì khảo sát lao động phổ thông, dự án soi chiếu *Nhóm chuyên gia IT (SOC 15)* để bóc trần ngụy biện kỹ thuật của họ khi muốn bảo vệ cái tôi nghề nghiệp.
* Sử dụng Hạng tử tương tác (Interaction Terms) thay vì trung bình thô để chứng minh sự bất bình đẳng mới trong việc tiếp nhận công nghệ.

---

### Liên hệ & Hỗ trợ
Nếu có bất kỳ câu hỏi nào về mô hình dữ liệu hoặc cấu trúc source code, vui lòng mở **Issue** trên repository này hoặc trò chuyện trực tiếp với trợ lý Gemini được tích hợp sẵn trong Dashboard(API lấy từ Google API Studio)
