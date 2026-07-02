# ═══════════════════════════════════════════════════════════════════════════════
# FILE: l7.py
# Tích hợp OLS residuals + Phân Tích Chủng Tộc + HLM Kiểm Chứng + Premium Dashboard
# Cập nhật: Thêm trang Chuyên Sâu Ngành Công Nghệ (SOC 15) với tính toán động 100%
# ═══════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════
# CẤU HÌNH TRANG
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Rào Cản Tâm Lý Với AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

C = {
    "blue": "#4338CA",     # indigo — accent chủ đạo
    "sky": "#0284C7",
    "violet": "#7C3AED",
    "emerald": "#059669",
    "amber": "#D97706",
    "rose": "#E11D48",
    "slate": "#334155",
    "gray_bg": "#F4F6FB",   # nền tổng thể — slate rất sáng
    "card_bg": "#FFFFFF",
    "border": "#E2E8F0",
    "text": "#0F172A",
    "muted": "#64748B",
}

SANS = "'Inter', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
SERIF = "'Source Serif 4', 'Georgia', 'Iowan Old Style', 'Cambria', serif"

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
<style>
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"],
.stRadio label, .stRadio div, [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
[data-testid="stMetricDelta"], [data-testid="stCaptionContainer"], [data-testid="stSidebar"] *,
button, input, textarea, h1, h2, h3, h4, h5, [data-baseweb="tab"], .styled-table,
[data-testid="stDataFrame"] * {{ font-family: {SANS} !important; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
.insight, .insight *, .method-box, .method-box *, .info-card .info-body,
.rec-card .rec-body, blockquote, blockquote * {{ font-family: {SERIF} !important; }}
[data-testid="stMarkdownContainer"] strong, .insight strong, .method-box strong {{ font-family: {SERIF} !important; }}
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {{ background:{C['gray_bg']}; color:{C['text']}; font-size:17px; line-height:1.7; }}
.block-container {{ padding-top:2.4rem; padding-bottom:4rem; max-width:1180px; }}
[data-testid="stSidebar"] {{ background:{C['card_bg']}; border-right:1px solid {C['border']}; }}
[data-testid="stSidebar"] .block-container {{ padding-top:1.8rem; }}
[data-testid="stHeader"] {{ background:transparent; }}
h1 {{ font-size:2.05rem !important; font-weight:700 !important; color:{C['text']} !important; line-height:1.3 !important; letter-spacing:-.01em; border-bottom:2px solid {C['border']}; padding-bottom:16px; margin-bottom:22px !important; }}
h2 {{ font-size:1.35rem !important; font-weight:700 !important; margin-top:40px !important; margin-bottom:8px !important; color:{C['text']} !important; letter-spacing:-.005em; }}
h3 {{ font-size:1.12rem !important; font-weight:600 !important; color:{C['text']} !important; }}
[data-testid="stCaptionContainer"] {{ font-size:.92rem !important; color:{C['muted']} !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid {C['border']}; }}
.stTabs [data-baseweb="tab"] {{ background:transparent; border:none; border-radius:10px 10px 0 0; padding:10px 18px; font-weight:600 !important; color:{C['muted']} !important; transition:all .15s ease; }}
.stTabs [data-baseweb="tab"]:hover {{ background:#EEF1F8; color:{C['text']} !important; }}
.stTabs [aria-selected="true"] {{ background:{C['blue']} !important; color:#fff !important; }}
.stTabs [aria-selected="true"] p {{ color:#fff !important; }}
.card-row {{ display:flex; gap:16px; flex-wrap:wrap; margin:20px 0 28px 0; }}
.kpi-card {{ flex:1 1 220px; min-height:168px; background:{C['card_bg']}; border:1px solid {C['border']}; border-radius:16px; border-top:4px solid var(--accent, {C['blue']}); box-shadow:0 1px 2px rgba(15,23,42,.04), 0 4px 14px rgba(15,23,42,.05); padding:26px 22px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; transition:transform .18s ease, box-shadow .18s ease; }}
.kpi-card:hover {{ transform:translateY(-4px); box-shadow:0 8px 24px rgba(15,23,42,.10), 0 2px 6px rgba(15,23,42,.06); }}
.kpi-card .val {{ font-size:2.3rem; font-weight:700; color:var(--accent, {C['blue']}); line-height:1.08; letter-spacing:-.02em; }}
.kpi-card .lbl {{ font-size:.92rem; color:{C['muted']}; margin-top:10px; line-height:1.45; white-space:pre-line; }}
.info-card {{ flex:1 1 260px; min-height:118px; background:{C['card_bg']}; border:1px solid {C['border']}; border-radius:16px; border-left:4px solid var(--accent, {C['emerald']}); box-shadow:0 1px 2px rgba(15,23,42,.04), 0 4px 14px rgba(15,23,42,.05); padding:22px 26px; display:flex; flex-direction:column; justify-content:center; transition:transform .18s ease, box-shadow .18s ease; }}
.info-card:hover {{ transform:translateY(-3px); box-shadow:0 8px 22px rgba(15,23,42,.09); }}
.info-card .info-title {{ font-weight:700; font-size:1.02rem; margin-bottom:6px; font-family:{SANS} !important; }}
.info-card .info-body {{ font-size:.98rem; color:{C['slate']}; line-height:1.6; }}
.insight {{ background:#F5F6FE; border:1px solid #E0E1FB; border-left:4px solid {C['blue']}; border-radius:14px; padding:22px 28px; margin:20px 0; font-size:1.02rem; line-height:1.75; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
.insight strong {{ color:{C['blue']}; }}
.insight code {{ background:#E6E7FA; padding:1px 6px; border-radius:5px; font-family:'JetBrains Mono','Courier New',monospace !important; font-size:.88em; }}
.method-box {{ background:#FFF8EC; border:1px solid #F3E3C2; border-left:4px solid {C['amber']}; border-radius:14px; padding:22px 28px; margin:16px 0 24px 0; font-size:.98rem; line-height:1.7; }}
.method-box strong {{ color:{C['amber']}; }}
.method-box code {{ background:#F5E8CE; padding:1px 6px; border-radius:5px; font-family:'JetBrains Mono','Courier New',monospace !important; font-size:.86em; }}
.source-tag {{ display:inline-flex; align-items:center; gap:4px; background:#EEF1FB; border:1px solid #DCE1F5; border-radius:999px; padding:4px 13px; font-size:.78rem; font-weight:500; color:{C['blue']}; font-family:{SANS} !important; margin:3px 6px 3px 0; }}
.source-tag::before {{ content:"◆"; font-size:.6rem; opacity:.6; }}
.rec-card {{ background:{C['card_bg']}; border:1px solid {C['border']}; border-left:4px solid {C['emerald']}; border-radius:16px; padding:22px 28px; margin-bottom:14px; min-height:104px; box-shadow:0 1px 2px rgba(15,23,42,.04), 0 3px 10px rgba(15,23,42,.04); display:flex; gap:16px; align-items:flex-start; transition:transform .18s ease, box-shadow .18s ease; }}
.rec-card:hover {{ transform:translateY(-2px); box-shadow:0 6px 18px rgba(15,23,42,.08); }}
.rec-card .rec-icon {{ font-size:1.4rem; line-height:1; color:{C['emerald']}; font-weight:700; font-family:{SANS} !important; }}
.rec-card .rec-title {{ font-weight:700; font-size:1.02rem; margin-bottom:5px; font-family:{SANS} !important; color:{C['text']}; }}
.rec-card .rec-body {{ font-size:.96rem; color:{C['slate']}; line-height:1.62; }}
.section-rule {{ border:none; border-top:1px solid {C['border']}; margin:28px 0 10px 0; }}
.styled-table {{ width:100%; border-collapse:collapse; margin:18px 0 8px 0; background:{C['card_bg']}; border-radius:14px; overflow:hidden; border:1px solid {C['border']}; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
.styled-table th {{ background:{C['gray_bg']}; color:{C['slate']}; text-align:left; padding:12px 16px; font-size:.86rem; font-weight:700; text-transform:uppercase; letter-spacing:.02em; border-bottom:1px solid {C['border']}; }}
.styled-table td {{ padding:11px 16px; font-size:.94rem; border-top:1px solid {C['border']}; vertical-align:top; }}
.styled-table tr:hover td {{ background:#F8FAFF; }}
.stButton > button {{ border-radius:999px !important; border:1px solid {C['border']} !important; background:{C['card_bg']} !important; color:{C['text']} !important; font-weight:600 !important; padding:.5rem 1.3rem !important; transition:all .15s ease !important; }}
.stButton > button:hover {{ border-color:{C['blue']} !important; color:{C['blue']} !important; box-shadow:0 2px 8px rgba(67,56,202,.15) !important; }}
[data-testid="stMetric"] {{ background:{C['card_bg']}; border:1px solid {C['border']}; border-radius:14px; padding:18px 16px; box-shadow:0 1px 2px rgba(15,23,42,.04), 0 3px 10px rgba(15,23,42,.05); }}
[data-testid="stMetricValue"] {{ color:{C['blue']} !important; font-weight:700 !important; font-size:1.45rem !important; }}
[data-testid="stMetricLabel"] {{ font-size:.9rem !important; color:{C['muted']} !important; }}
[data-testid="stMetricDelta"] {{ font-size:.86rem !important; }}
hr {{ border-top:1px solid {C['border']}; }}
::-webkit-scrollbar {{ width:9px; height:9px; }}
::-webkit-scrollbar-thumb {{ background:#CBD5E1; border-radius:6px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# NHÃN 13 NHÓM NGHỀ LỚN
# ═══════════════════════════════════════════════════════
MAJOR_SOC_LABELS = {
    "11": "Quản lý", "13": "Kinh doanh, tài chính", "15": "Máy tính, toán học",
    "17": "Kiến trúc, kỹ thuật", "19": "Khoa học tự nhiên, xã hội",
    "21": "Dịch vụ cộng đồng", "23": "Pháp lý", "25": "Giáo dục",
    "27": "Nghệ thuật, truyền thông", "29": "Y tế (chuyên môn)",
    "31": "Y tế (hỗ trợ)", "33": "Bảo vệ, an ninh", "35": "Chế biến thực phẩm",
    "37": "Vệ sinh, bảo trì công trình", "39": "Chăm sóc cá nhân",
    "41": "Bán hàng", "43": "Văn phòng, hành chính", "45": "Nông lâm ngư nghiệp",
    "47": "Xây dựng", "49": "Lắp đặt, bảo trì thiết bị", "51": "Sản xuất",
    "53": "Vận tải, kho vận",
}

# ═══════════════════════════════════════════════════════
# TẢI & CHUẨN BỊ DỮ LIỆU
# ═══════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_and_prepare():
    OCC = "Occupation (O*NET-SOC Title)"
    desires    = pd.read_csv("domain_worker_desires.csv")
    metadata   = pd.read_csv("domain_worker_metadata.csv")
    capability = pd.read_csv("expert_rated_technological_capability.csv")
    tasks      = pd.read_csv("task_statement_with_metadata.csv")

    # Gộp capability theo Task ID
    cap_agg = capability.groupby("Task ID").agg(
        automation_capacity=("Automation Capacity Rating", "mean"),
    ).reset_index().rename(columns={"automation_capacity": "Automation Capacity Rating"})

    desire_cols = ["Task ID", OCC, "User ID", "Automation Desire Rating",
                    "Enjoyment Rating", "Job Security Rating", "Self-reported Expertise",
                    "Domain Expertise Requirement", "Involved Uncertainty",
                    "Reasons for Automation Desire - Free Time",
                    "Reasons for Human Agency - Domain Knowledge",
                    "Reasons for Human Agency - Control",
                    "Reasons for Human Agency - Empathy",
                    "Reasons for Human Agency - Quality Oversight",
                    "Reasons for Human Agency - Ethical"]
    merged = pd.merge(cap_agg, desires[desire_cols], on="Task ID")
    merged["trust_gap"] = merged["Automation Capacity Rating"] - merged["Automation Desire Rating"]

    merged = pd.merge(
        merged, tasks[["Task ID", "Occupation Mean Annual Wage", "O*NET-SOC Code"]],
        on="Task ID", how="left",
    )
    merged["major_soc"] = merged["O*NET-SOC Code"].astype(str).str[:2]
    merged["major_soc_label"] = merged["major_soc"].map(MAJOR_SOC_LABELS).fillna(merged["major_soc"])

    att_map = {"Strongly disagree": 1, "Disagree": 2, "Somewhat disagree": 3,
               "Neither agree nor disagree": 4, "Somewhat agree": 5, "Agree": 6, "Strongly agree": 7}
    meta = metadata.copy()
    meta["suffering_num"]  = meta["AI Suffering Attitude"].map(att_map)
    meta["importance_num"] = meta["AI Job Importance Attitude"].map(att_map)
    meta["tedious_num"]    = meta["AI Tedious Work Attitude"].map(att_map)
    meta["daily_num"]      = meta["AI Daily Interest Attitude"].map(att_map)

    EDU_ORDER = ["High School", "Some College, No Degree", "Associate Degree",
                 "Bachelor’s Degree", "Master’s Degree",
                 "Professional Degree (e.g., MD, JD)", "Doctorate (e.g., PhD)"]
    EDU_NUM = {e: i for i, e in enumerate(EDU_ORDER)}
    USE_NUM = {
        "No, I've never heard of them.": 0,
        "No, I have not used them for any work-related activities.": 1,
        "Yes, I have used them occasionally for specific tasks.": 2,
        "Yes, I use them every week in my work.": 3,
        "Yes, I use them every day in my work.": 4,
    }
    meta["edu_num"] = meta["Education"].map(EDU_NUM)
    meta["use_num"] = meta["LLM Use in Work"].map(USE_NUM)
    meta["male"] = (meta["Gender"] == "Male").astype(int)

    full = pd.merge(merged, meta, on="User ID", how="inner", suffixes=("", "_meta"))

    r_enj, p_enj = stats.pearsonr(full["Enjoyment Rating"], full["Automation Desire Rating"])
    r_sec, p_sec = stats.pearsonr(full["Job Security Rating"], full["Automation Desire Rating"])

    # ── MÔ HÌNH STATSMODELS TÍNH PHẦN DƯ ──
    df_smf = full.dropna(subset=['Automation Desire Rating', 'Automation Capacity Rating', 'Age', 'male', 'Occupation Mean Annual Wage']).copy()
    df_smf_reg = df_smf.rename(columns={
        "Automation Desire Rating": "desire",
        "Automation Capacity Rating": "capacity",
        "Occupation Mean Annual Wage": "wage"
    })
    model_smf = smf.ols("desire ~ capacity + Age + male + wage", data=df_smf_reg).fit()
    df_smf['residual'] = model_smf.resid

    # ── MÔ HÌNH THỐNG KÊ CHÍNH (OLS TỰ XÂY DỰNG) ──
    def ols_standardized(df, cols, y_col):
        X = df[cols].values.astype(float)
        Xz = (X - X.mean(0)) / X.std(0)
        Xf = np.column_stack([np.ones(len(df)), Xz])
        y = df[y_col].values.astype(float)
        b, *_ = np.linalg.lstsq(Xf, y, rcond=None)
        resid = y - Xf @ b
        n, k = Xf.shape
        s2 = (resid @ resid) / max(n - k, 1)
        cov = s2 * np.linalg.inv(Xf.T @ Xf)
        se = np.sqrt(np.diag(cov))
        t = b / se
        p = 2 * (1 - stats.t.cdf(np.abs(t), n - k))
        ci_lo = b - 1.96 * se
        ci_hi = b + 1.96 * se
        return b, se, t, p, ci_lo, ci_hi, n

    def ols_train_test(df, cols, y_col, test_frac=0.2, seed=42):
        d = df.dropna(subset=cols + [y_col])
        rng = np.random.RandomState(seed)
        n = len(d)
        idx = rng.permutation(n)
        n_test = max(int(n * test_frac), 1)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        train, test = d.iloc[train_idx], d.iloc[test_idx]
        X_train = train[cols].values.astype(float)
        mu, sigma = X_train.mean(0), X_train.std(0)
        Xtr = (X_train - mu) / sigma
        Xtr_f = np.column_stack([np.ones(len(train)), Xtr])
        y_train = train[y_col].values.astype(float)
        beta, *_ = np.linalg.lstsq(Xtr_f, y_train, rcond=None)
        def r2(Xz, y):
            Xf = np.column_stack([np.ones(len(Xz)), Xz])
            pred = Xf @ beta
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            return 1 - ss_res / ss_tot
        r2_train = r2(Xtr, y_train)
        X_test = (test[cols].values.astype(float) - mu) / sigma
        y_test = test[y_col].values.astype(float)
        r2_test = r2(X_test, y_test)
        return {"r2_train": r2_train, "r2_test": r2_test, "n_train": len(train), "n_test": len(test)}

    def compute_r2_only(df, cols, y_col):
        d = df.dropna(subset=cols + [y_col])
        X = d[cols].values.astype(float)
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
        Xf = np.column_stack([np.ones(len(d)), Xz])
        y = d[y_col].values.astype(float)
        b, *_ = np.linalg.lstsq(Xf, y, rcond=None)
        pred = Xf @ b
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot, len(d)

    def eta_squared(df, group_col, y_col):
        d = df.dropna(subset=[group_col, y_col])
        grand_mean = d[y_col].mean()
        ss_total = ((d[y_col] - grand_mean) ** 2).sum()
        ss_between = d.groupby(group_col)[y_col].apply(lambda x: len(x) * (x.mean() - grand_mean) ** 2).sum()
        return ss_between / ss_total, d[group_col].nunique(), len(d)

    reg_cols = ["Enjoyment Rating", "Job Security Rating", "Automation Capacity Rating",
                "edu_num", "use_num", "Age", "male", "Occupation Mean Annual Wage", "suffering_num"]
    reg_labels = {
        "Enjoyment Rating": "Mức thích thú công việc", "Job Security Rating": "Cảm giác an toàn việc làm",
        "Automation Capacity Rating": "Năng lực AI (khách quan)", "edu_num": "Trình độ học vấn",
        "use_num": "Tần suất dùng LLM", "Age": "Tuổi", "male": "Giới tính nam",
        "Occupation Mean Annual Wage": "Lương trung bình nghề", "suffering_num": "Niềm tin AI có thể chịu khổ",
    }
    rr = full.dropna(subset=reg_cols + ["Automation Desire Rating"]).copy()
    b_reg, se_reg, t_reg, p_reg, cilo_reg, cihi_reg, n_reg = ols_standardized(rr, reg_cols, "Automation Desire Rating")
    reg_result = pd.DataFrame({
        "feature": [reg_labels[c] for c in reg_cols], "beta": b_reg[1:], "se": se_reg[1:],
        "t": t_reg[1:], "p": p_reg[1:], "ci_lo": cilo_reg[1:], "ci_hi": cihi_reg[1:],
    }).sort_values("beta")
    tt_main = ols_train_test(rr, reg_cols, "Automation Desire Rating")

    # ── MÔ HÌNH THÁI ĐỘ ──
    reg_cols4 = reg_cols[:-1] + ["suffering_num", "importance_num", "tedious_num", "daily_num"]
    labels4 = {**reg_labels, "importance_num": "AI quan trọng với công việc",
               "tedious_num": "AI làm tốt việc nhàm chán", "daily_num": "Hứng thú dùng AI hằng ngày"}
    rr4 = full.dropna(subset=reg_cols4 + ["Automation Desire Rating"]).copy()
    b4, se4, t4, p4, cilo4, cihi4, n4 = ols_standardized(rr4, reg_cols4, "Automation Desire Rating")
    reg4_result = pd.DataFrame({
        "feature": [labels4[c] for c in reg_cols4], "beta": b4[1:], "se": se4[1:], "t": t4[1:], "p": p4[1:],
    }).sort_values("beta")
    tt_attitude = ols_train_test(rr4, reg_cols4, "Automation Desire Rating")
    attitude4 = reg4_result[reg4_result["feature"].str.contains("AI ") | reg4_result["feature"].str.contains("chịu khổ")]

    # ── TƯƠNG TÁC EDU x USE ──
    def ols_train_test_precomputed_z(df, cols, y_col, test_frac=0.2, seed=42):
        d = df.dropna(subset=cols + [y_col])
        rng = np.random.RandomState(seed)
        n = len(d)
        idx = rng.permutation(n)
        n_test = max(int(n * test_frac), 1)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        train, test = d.iloc[train_idx], d.iloc[test_idx]
        Xtr = np.column_stack([np.ones(len(train)), train[cols].values.astype(float)])
        ytr = train[y_col].values.astype(float)
        beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        def r2(sub_df, y):
            Xf = np.column_stack([np.ones(len(sub_df)), sub_df[cols].values.astype(float)])
            pred = Xf @ beta
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            return 1 - ss_res / ss_tot
        r2_train = r2(train, ytr)
        yte = test[y_col].values.astype(float)
        r2_test = r2(test, yte)
        return {"r2_train": r2_train, "r2_test": r2_test, "n_train": len(train), "n_test": len(test)}

    rr2 = rr.copy()
    rr2["edu_z"] = (rr2["edu_num"] - rr2["edu_num"].mean()) / rr2["edu_num"].std()
    rr2["use_z"] = (rr2["use_num"] - rr2["use_num"].mean()) / rr2["use_num"].std()
    rr2["inter_z"] = rr2["edu_z"] * rr2["use_z"]
    other_cols = ["Enjoyment Rating", "Job Security Rating", "Automation Capacity Rating",
                  "Age", "male", "Occupation Mean Annual Wage", "suffering_num"]
    Xo = rr2[other_cols].values.astype(float)
    Xoz = (Xo - Xo.mean(0)) / Xo.std(0)
    other_cols_z = [c + "_z" for c in other_cols]
    rr2[other_cols_z] = Xoz
    Xinter = np.column_stack([np.ones(len(rr2)), rr2["edu_z"], rr2["use_z"], rr2["inter_z"], Xoz])
    y2 = rr2["Automation Desire Rating"].values.astype(float)
    b2, *_ = np.linalg.lstsq(Xinter, y2, rcond=None)
    resid2 = y2 - Xinter @ b2
    n2, k2 = Xinter.shape
    s22 = (resid2 @ resid2) / (n2 - k2)
    cov2 = s22 * np.linalg.inv(Xinter.T @ Xinter)
    se2 = np.sqrt(np.diag(cov2))
    t2 = b2 / se2
    p2 = 2 * (1 - stats.t.cdf(np.abs(t2), n2 - k2))
    amplification = {"beta_inter": b2[3], "se_inter": se2[3], "p_inter": p2[3], "n": n2}
    cols_amp = ["edu_z", "use_z", "inter_z"] + other_cols_z
    tt_amp = ols_train_test_precomputed_z(rr2, cols_amp, "Automation Desire Rating")

    # ── MỨC CHUYÊN MÔN ──
    exp_map = {"Novice": 0, "Average": 1, "Expert": 2}
    exp_groups = merged.dropna(subset=["Self-reported Expertise"]).groupby("Self-reported Expertise").agg(
        trust_gap=("trust_gap", "mean"), desire=("Automation Desire Rating", "mean"), n=("Task ID", "count")
    ).reindex(["Novice", "Average", "Expert"])
    novice = merged[merged["Self-reported Expertise"] == "Novice"]["trust_gap"].dropna()
    average = merged[merged["Self-reported Expertise"] == "Average"]["trust_gap"].dropna()
    expert = merged[merged["Self-reported Expertise"] == "Expert"]["trust_gap"].dropna()
    t_exp, p_exp = stats.ttest_ind(expert, average, equal_var=False)
    t_nov, p_nov = stats.ttest_ind(novice, average, equal_var=False)

    rr_full = full.dropna(subset=["Self-reported Expertise"] + reg_cols + ["Automation Desire Rating"]).copy()
    rr_full["exp_num"] = rr_full["Self-reported Expertise"].map(exp_map)
    rr_full["exp_num2"] = rr_full["exp_num"] ** 2
    cols_u = ["exp_num", "exp_num2"] + reg_cols
    bU, seU, tU, pU, ciloU, cihiU, nU = ols_standardized(rr_full, cols_u, "Automation Desire Rating")
    expert_robust = {"beta_linear": bU[1], "p_linear": pU[1], "beta_quad": bU[2], "p_quad": pU[2], "n": nU}
    tt_expert = ols_train_test(rr_full, cols_u, "Automation Desire Rating")

    reason_cols = ["Reasons for Human Agency - Domain Knowledge", "Reasons for Human Agency - Control",
                   "Reasons for Human Agency - Empathy", "Reasons for Human Agency - Quality Oversight",
                   "Reasons for Human Agency - Ethical"]
    reason_labels = {"Reasons for Human Agency - Domain Knowledge": "Thiếu chuyên môn ngành",
                      "Reasons for Human Agency - Control": "Cần kiểm soát trực tiếp",
                      "Reasons for Human Agency - Empathy": "Cần sự đồng cảm",
                      "Reasons for Human Agency - Quality Oversight": "Cần giám sát chất lượng",
                      "Reasons for Human Agency - Ethical": "Vấn đề đạo đức"}
    low_desire = merged[merged["Automation Desire Rating"] <= 2]
    reason_by_expertise = {}
    for grp in ["Novice", "Average", "Expert"]:
        sub = low_desire[low_desire["Self-reported Expertise"] == grp]
        vals = {}
        for c in reason_cols:
            vals[reason_labels[c]] = sub[c].astype(str).str.strip().str.lower().map({"true": True, "false": False}).mean()
        reason_by_expertise[grp] = vals
    reason_df = pd.DataFrame(reason_by_expertise)

    high_d = desires[desires["Automation Desire Rating"] >= 4]
    reason_free_time = high_d["Reasons for Automation Desire - Free Time"].astype(str).str.strip().str.lower().map({"true": True, "false": False}).mean()
    low_d = desires[desires["Automation Desire Rating"] <= 2]
    reason_domain = low_d["Reasons for Human Agency - Domain Knowledge"].astype(str).str.strip().str.lower().map({"true": True, "false": False}).mean()

    eta_occ, n_occ_levels, n_eta_occ = eta_squared(merged, OCC, "Automation Desire Rating")
    eta_major, n_major_levels, n_eta_major = eta_squared(full, "major_soc", "Automation Desire Rating")
    r2_indiv, n_r2a = compute_r2_only(rr, reg_cols, "Automation Desire Rating")
    dummies = pd.get_dummies(rr["major_soc"], prefix="soc", drop_first=True).astype(float)
    rr_fe = pd.concat([rr.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    cols_fe = reg_cols + list(dummies.columns)
    r2_fe, n_r2b = compute_r2_only(rr_fe, cols_fe, "Automation Desire Rating")
    variance_decomp = {
        "eta_occ": eta_occ, "n_occ_levels": n_occ_levels, "n_eta_occ": n_eta_occ,
        "eta_major": eta_major, "n_major_levels": n_major_levels, "n_eta_major": n_eta_major,
        "r2_indiv": r2_indiv, "r2_fe": r2_fe, "delta_r2": r2_fe - r2_indiv, "n_r2": n_r2a,
    }

    corr_vars = ["Automation Desire Rating", "Automation Capacity Rating", "trust_gap",
                 "Enjoyment Rating", "Job Security Rating", "Domain Expertise Requirement",
                 "Involved Uncertainty", "edu_num", "use_num", "Age",
                 "Occupation Mean Annual Wage", "suffering_num"]
    corr_labels = {
        "Automation Desire Rating": "Mong muốn tự động hoá", "Automation Capacity Rating": "Năng lực AI",
        "trust_gap": "Khoảng cách năng lực–mong muốn", "Enjoyment Rating": "Thích thú công việc",
        "Job Security Rating": "An toàn việc làm", "Domain Expertise Requirement": "Yêu cầu chuyên môn",
        "Involved Uncertainty": "Mức bất định", "edu_num": "Học vấn", "use_num": "Tần suất dùng LLM",
        "Age": "Tuổi", "Occupation Mean Annual Wage": "Lương nghề", "suffering_num": "Niềm tin AI chịu khổ",
    }
    corr_df = full[corr_vars].dropna().corr()
    corr_df.index = [corr_labels[c] for c in corr_df.index]
    corr_df.columns = [corr_labels[c] for c in corr_df.columns]
    n_corr = len(full[corr_vars].dropna())

    wage_valid = full.dropna(subset=["Occupation Mean Annual Wage"]).copy()
    wage_valid["wage_bucket"] = pd.qcut(wage_valid["Occupation Mean Annual Wage"], 3, labels=["Lương nghề thấp", "Lương nghề TB", "Lương nghề cao"])
    wage_gap = wage_valid.groupby("wage_bucket", observed=True).agg(trust_gap=("trust_gap", "mean"), desire=("Automation Desire Rating", "mean"), n=("Task ID", "count"))

    INC_ORDER = ["0-30K", "30-60K", "60-86K", "86K-165K", "165K-209K", "209K-529K", "529K+"]
    INC_NUM = {v: i for i, v in enumerate(INC_ORDER)}
    full["income_num"] = full["Income"].map(INC_NUM)
    wage_income = full.dropna(subset=["Occupation Mean Annual Wage", "income_num"])
    r_wage_income, p_wage_income = stats.pearsonr(wage_income["Occupation Mean Annual Wage"], wage_income["income_num"])

    major_group_table = merged.dropna(subset=["major_soc"]).groupby("major_soc_label").agg(
        capacity=("Automation Capacity Rating", "mean"), desire=("Automation Desire Rating", "mean"),
        trust_gap=("trust_gap", "mean"), n=("Task ID", "count"),
    ).reset_index().sort_values("desire", ascending=False)

    occ_all = merged.groupby(OCC).agg(
        capacity=("Automation Capacity Rating", "mean"), desire=("Automation Desire Rating", "mean"),
        trust_gap=("trust_gap", "mean"), n=("Task ID", "count"),
    ).reset_index()
    occ_all = occ_all[occ_all["n"] >= 15].sort_values("desire", ascending=False)

    use0_people = meta[meta["use_num"] == 0]["User ID"].nunique()
    use1_people = meta[meta["use_num"] == 1]["User ID"].nunique()
    per_person = full.groupby("User ID").agg(use_num=("use_num", "first"), desire=("Automation Desire Rating", "mean")).reset_index()
    g0 = per_person[per_person["use_num"] == 0]["desire"].dropna()
    g1 = per_person[per_person["use_num"] == 1]["desire"].dropna()
    t_use, p_use = stats.ttest_ind(g0, g1, equal_var=False) if len(g0) > 1 and len(g1) > 1 else (np.nan, np.nan)

    task_feat = merged.dropna(subset=["Domain Expertise Requirement", "Involved Uncertainty", "trust_gap"])
    r_der, p_der = stats.pearsonr(task_feat["Domain Expertise Requirement"], task_feat["trust_gap"])
    r_unc, p_unc = stats.pearsonr(task_feat["Involved Uncertainty"], task_feat["trust_gap"])

    pol = full.dropna(subset=["Political Affiliation"])
    pol = pol[pol["Political Affiliation"] != "Green Party"]
    pol_gap = pol.groupby("Political Affiliation").agg(desire=("Automation Desire Rating", "mean"), n=("Task ID", "count")).sort_values("desire", ascending=False)

    # ═══════════════════════════════════════════════════════
    # TÍNH TOÁN RIÊNG CHO SOC 15 (NGÀNH CÔNG NGHỆ)
    # ═══════════════════════════════════════════════════════
    soc15 = full[full["major_soc"] == "15"].copy()
    non_soc15 = full[full["major_soc"] != "15"].copy()

    # 1. Tổng quan SOC 15
    soc15_stats = {
        "n_tasks": len(soc15),
        "n_users": soc15['User ID'].nunique(),
        "trust_gap_15": soc15['trust_gap'].mean(),
        "trust_gap_non15": non_soc15['trust_gap'].mean(),
        "desire_15": soc15['Automation Desire Rating'].mean(),
        "desire_non15": non_soc15['Automation Desire Rating'].mean(),
        "security_15": soc15['Job Security Rating'].mean(),
        "security_non15": non_soc15['Job Security Rating'].mean(),
        "wage_15": soc15['Occupation Mean Annual Wage'].mean(),
        "wage_non15": non_soc15['Occupation Mean Annual Wage'].mean(),
        "use_num_15": soc15['use_num'].mean(),
        "use_num_non15": non_soc15['use_num'].mean(),
    }

    # Lý do từ chối (Desire <= 2)
    soc15_resist = soc15[soc15["Automation Desire Rating"] <= 2]
    non_soc15_resist = non_soc15[non_soc15["Automation Desire Rating"] <= 2]
    reasons_pct_15 = {}
    reasons_pct_non15 = {}
    for r in reason_cols:
        reasons_pct_15[r] = soc15_resist[r].astype(str).str.strip().str.lower().map({"true": True, "false": False}).mean() * 100
        reasons_pct_non15[r] = non_soc15_resist[r].astype(str).str.strip().str.lower().map({"true": True, "false": False}).mean() * 100
    soc15_stats["reasons_15"] = reasons_pct_15
    soc15_stats["reasons_non15"] = reasons_pct_non15

    # Thái độ về AI
    attitudes = ["AI Suffering Attitude", "AI Job Importance Attitude", "AI Tedious Work Attitude", "AI Daily Interest Attitude"]
    att_15 = {}
    att_non15 = {}
    for att in attitudes:
        att_15[att] = soc15[att].map(att_map).mean()
        att_non15[att] = non_soc15[att].map(att_map).mean()
    soc15_stats["attitudes_15"] = att_15
    soc15_stats["attitudes_non15"] = att_non15

    # 2. Phân Nhóm Nghề SOC 15
    occ_col = 'Occupation (O*NET-SOC Title)_x' if 'Occupation (O*NET-SOC Title)_x' in soc15.columns else OCC
    occ_stats_15 = soc15.groupby(occ_col).agg(
        task_count=("Task ID", "count"),
        trust_gap=("trust_gap", "mean"),
        desire=("Automation Desire Rating", "mean"),
        capacity=("Automation Capacity Rating", "mean"),
        security=("Job Security Rating", "mean")
    ).reset_index()
    occ_stats_15 = occ_stats_15[occ_stats_15["task_count"] >= 15].sort_values("desire", ascending=False)
    soc15_stats["occ_stats"] = occ_stats_15

    # Lý do từ chối của 5 nghề tiêu biểu
    key_occs = [
        "Computer Programmers", "Information Security Analysts", 
        "Database Administrators", "Web Developers", 
        "Software Quality Assurance Analysts and Testers"
    ]
    key_occ_reasons = {}
    for occ_name in key_occs:
        sub = soc15_resist[soc15_resist[occ_col] == occ_name]
        if len(sub) > 5:
            r_dict = {}
            for r in reason_cols:
                r_dict[r] = sub[r].astype(str).str.strip().str.lower().map({"true": True, "false": False}).mean() * 100
            r_dict['n_tasks'] = len(sub)
            key_occ_reasons[occ_name] = r_dict
    soc15_stats["key_occ_reasons"] = key_occ_reasons

    # 3. Tương tác Học vấn x Dùng AI (SOC 15 only)
    soc15_clean = soc15.dropna(subset=["Automation Desire Rating", "edu_num", "use_num"]).copy()
    soc15_clean["edu_z"] = (soc15_clean["edu_num"] - soc15_clean["edu_num"].mean()) / soc15_clean["edu_num"].std()
    soc15_clean["use_z"] = (soc15_clean["use_num"] - soc15_clean["use_num"].mean()) / soc15_clean["use_num"].std()
    soc15_clean["inter_z"] = soc15_clean["edu_z"] * soc15_clean["use_z"]
    soc15_clean_reg = soc15_clean.rename(columns={"Automation Desire Rating": "desire"})
    model_soc15_inter = smf.ols("desire ~ edu_z + use_z + inter_z", data=soc15_clean_reg).fit()
    
    soc15_use_stats = soc15.groupby("LLM Use in Work").agg(
        desire=("Automation Desire Rating", "mean"), trust_gap=("trust_gap", "mean"),
        count=("Task ID", "count"), use_num=("use_num", "first")
    ).sort_values("use_num").reset_index()
    
    soc15_edu_stats = soc15.groupby("Education").agg(
        desire=("Automation Desire Rating", "mean"), trust_gap=("trust_gap", "mean"),
        count=("Task ID", "count"), edu_num=("edu_num", "first")
    ).sort_values("edu_num").reset_index()

    soc15_stats["inter_model"] = model_soc15_inter
    soc15_stats["use_stats"] = soc15_use_stats
    soc15_stats["edu_stats"] = soc15_edu_stats

    # 4. Góc nhìn 1 & 3: Nghịch lý tuổi nghề và Ngụy biện chuyên môn
    soc15["Age_Group"] = pd.cut(soc15["Age"], bins=[18, 30, 45, 100], labels=["Gen Z (<30)", "Millennials (30-45)", "Gen X+ (>45)"])
    age_stats_15 = soc15.groupby("Age_Group", observed=True).agg(
        Job_Security=("Job Security Rating", "mean"),
        Desire=("Automation Desire Rating", "mean"),
        Trust_Gap=("trust_gap", "mean"),
        Count=("Task ID", "count")
    ).reset_index()
    soc15_stats["age_stats"] = age_stats_15

    soc15["Enjoyment_Level"] = np.where(soc15["Enjoyment Rating"] >= 4, "High (Thích)", "Low/Mid (Không Thích)")
    resist_15 = soc15[soc15["Automation Desire Rating"] <= 2]
    
    reason_by_enjoy_15 = resist_15.groupby("Enjoyment_Level")[reason_cols].apply(lambda x: (x == True).mean() * 100).reset_index()
    soc15_stats["reason_by_enjoy"] = reason_by_enjoy_15

    high_enjoy_resist = resist_15[resist_15["Enjoyment_Level"] == "High (Thích)"]
    reason_expert_enjoy = high_enjoy_resist.groupby("Self-reported Expertise")[reason_cols].apply(lambda x: (x == True).mean() * 100).reindex(["Novice", "Average", "Expert"]).reset_index()
    soc15_stats["reason_expert_enjoy"] = reason_expert_enjoy

    soc15_stats["reason_expert_enjoy"] = reason_expert_enjoy

    # ═══════════════════════════════════════════════════════
    # TÍNH TOÁN CHO 4 GÓC NHÌN NÂNG CAO (COSMIC LEVEL)
    # ═══════════════════════════════════════════════════════
    
    # 1. HLM & Variance Decomposition (ANOVA-based ICC)
    def icc_one_way(df, group_col, y_col):
        d = df.dropna(subset=[group_col, y_col]).copy()
        groups = d.groupby(group_col)[y_col]
        k = groups.ngroups
        n_total = len(d)
        grand_mean = d[y_col].mean()
        n_j = groups.size()
        ms_between = (groups.apply(lambda x: len(x)*(x.mean()-grand_mean)**2).sum())/(k-1)
        ms_within = (groups.apply(lambda x: ((x-x.mean())**2).sum()).sum())/(n_total-k)
        n0 = (n_total - (n_j**2).sum()/n_total)/(k-1)
        icc = (ms_between-ms_within)/(ms_between + (n0-1)*ms_within) if (ms_between + (n0-1)*ms_within) != 0 else 0.0
        return icc, ms_between, ms_within, k, n_total

    icc_major_soc, _, _, k_major, _ = icc_one_way(full, "major_soc", "Automation Desire Rating")
    icc_user, _, _, k_user, _ = icc_one_way(full, "User ID", "Automation Desire Rating")
    icc_occ, _, _, k_occ, _ = icc_one_way(merged, OCC, "Automation Desire Rating")
    
    hlm_stats = {
        "icc_major_soc": icc_major_soc,
        "k_major": k_major,
        "icc_user": icc_user,
        "k_user": k_user,
        "icc_occ": icc_occ,
        "k_occ": k_occ,
        "n_total": len(full)
    }

    # 2. Race & Demographic Control
    per_person = full.groupby('User ID').agg(
        Race=('Race','first'), Gender=('Gender','first'), edu_num=('edu_num','first'), use_num=('use_num','first'),
        Age=('Age','first'), male=('male','first'), 
        desire=('Automation Desire Rating','mean'), trust_gap=('trust_gap','mean'),
        wage=('Occupation Mean Annual Wage','mean'), enjoy=('Enjoyment Rating','mean'), n_tasks=('Task ID','count')
    ).reset_index()

    races = [r for r in per_person['Race'].dropna().unique()]
    groups_race = [per_person[per_person['Race']==r]['desire'].dropna().values for r in races]
    f_race, p_race = stats.f_oneway(*groups_race) if len(groups_race) > 1 else (0.0, 1.0)
    
    rr_race = per_person.dropna(subset=['Race','desire','edu_num','use_num','Age','male','wage','enjoy']).copy()
    race_dum = pd.get_dummies(rr_race['Race'], prefix='race', drop_first=True).astype(float)
    rr_race_reg = pd.concat([rr_race.reset_index(drop=True), race_dum.reset_index(drop=True)], axis=1)
    
    def ols_standardized_helper(df, cols, y_col):
        X = df[cols].values.astype(float)
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
        Xf = np.column_stack([np.ones(len(df)), Xz])
        y = df[y_col].values.astype(float)
        b, *_ = np.linalg.lstsq(Xf, y, rcond=None)
        resid = y - Xf @ b
        n, k = Xf.shape
        s2 = (resid @ resid) / max(n - k, 1)
        cov = s2 * np.linalg.inv(Xf.T @ Xf)
        se = np.sqrt(np.diag(cov))
        t = b / se
        p = 2 * (1 - stats.t.cdf(np.abs(t), n - k))
        return b, se, t, p, n

    race_cols_list = ['edu_num','use_num','Age','male','wage','enjoy'] + list(race_dum.columns)
    b_race, se_race, t_race, p_race, n_race_reg = ols_standardized_helper(rr_race_reg, race_cols_list, 'desire')
    
    race_reg_results = pd.DataFrame({
        "feature": ["Hằng số"] + [
            "Học vấn", "Tần suất dùng LLM", "Tuổi", "Nam giới", "Lương nghề", "Thích việc"
        ] + [f"Chủng tộc: {col.replace('race_', '')}" for col in race_dum.columns],
        "beta": b_race, "se": se_race, "t": t_race, "p": p_race
    })
    
    race_means = per_person.groupby('Race')['desire'].agg(['mean','count']).sort_values('mean').reset_index()

    race_stats = {
        "f_val": f_race,
        "p_val": p_race,
        "reg_results": race_reg_results,
        "n_reg": n_race_reg,
        "means": race_means
    }

    # 3. Identity Threat Index (ITI)
    def defensive_tobool(s):
        return s.astype(str).str.strip().str.lower().map({'true':True,'false':False})

    defensive_cols = ['Reasons for Human Agency - Control','Reasons for Human Agency - Domain Knowledge','Reasons for Human Agency - Quality Oversight']
    moral_cols = ['Reasons for Human Agency - Empathy','Reasons for Human Agency - Ethical']

    low_desire_df = full[full['Automation Desire Rating']<=2].copy()
    for c in defensive_cols+moral_cols:
        low_desire_df[c+'_b'] = defensive_tobool(low_desire_df[c]).fillna(False).astype(int)
    low_desire_df['defensive_score'] = low_desire_df[[c+'_b' for c in defensive_cols]].sum(axis=1)/3
    low_desire_df['moral_score'] = low_desire_df[[c+'_b' for c in moral_cols]].sum(axis=1)/2
    low_desire_df['ITI'] = low_desire_df['defensive_score'] - low_desire_df['moral_score']
    
    iti_by_soc = low_desire_df.groupby('major_soc_label')['ITI'].agg(['mean', 'count']).sort_values('mean', ascending=False).reset_index()
    
    soc_groups = [group['ITI'].values for _, group in low_desire_df.groupby('major_soc_label') if len(group) > 1]
    f_iti, p_iti = stats.f_oneway(*soc_groups) if len(soc_groups) > 1 else (0.0, 1.0)
    
    iti_stats = {
        "iti_by_soc": iti_by_soc,
        "f_val": f_iti,
        "p_val": p_iti,
        "overall_mean": low_desire_df['ITI'].mean()
    }

    # 4. Automation Readiness Quadrant
    med_der = merged['Domain Expertise Requirement'].median()
    med_unc = merged['Involved Uncertainty'].median()
    merged['q_der'] = np.where(merged['Domain Expertise Requirement']>=med_der,'Cao','Thấp')
    merged['q_unc'] = np.where(merged['Involved Uncertainty']>=med_unc,'Cao','Thấp')
    merged['quadrant'] = merged['q_der']+' DER / '+merged['q_unc']+' UNC'
    
    quad_stats = merged.groupby('quadrant').agg(
        trust_gap=('trust_gap','mean'),
        desire=('Automation Desire Rating','mean'),
        capacity=('Automation Capacity Rating','mean'),
        n=('Task ID','count')
    ).reset_index()
    
    quad_groups = [merged[merged['quadrant']==q]['trust_gap'].values for q in quad_stats['quadrant']]
    f_quad, p_quad = stats.f_oneway(*quad_groups) if len(quad_groups) > 1 else (0.0, 1.0)
    
    quad_stats_dict = {
        "med_der": med_der,
        "med_unc": med_unc,
        "stats": quad_stats,
        "f_val": f_quad,
        "p_val": p_quad,
        "raw_df": merged[['Domain Expertise Requirement', 'Involved Uncertainty', 'trust_gap', 'quadrant', OCC]]
    }

    data = {
        "desires": desires, "merged": merged, "full": full, "OCC": OCC,
        "r_enj": (r_enj, p_enj), "r_sec": (r_sec, p_sec),
        "reg_result": reg_result, "n_reg": n_reg, "tt_main": tt_main,
        "reg4_result": reg4_result, "attitude4": attitude4, "n_reg4": n4, "tt_attitude": tt_attitude,
        "amplification": amplification, "tt_amp": tt_amp,
        "exp_groups": exp_groups, "t_exp": t_exp, "p_exp": p_exp, "t_nov": t_nov, "p_nov": p_nov,
        "expert_robust": expert_robust, "tt_expert": tt_expert, "reason_df": reason_df,
        "reason_free_time": reason_free_time, "reason_domain": reason_domain,
        "wage_gap": wage_gap, "r_wage_income": (r_wage_income, p_wage_income, len(wage_income)),
        "major_group_table": major_group_table, "occ_all": occ_all,
        "variance_decomp": variance_decomp, "corr_df": corr_df, "n_corr": n_corr,
        "use0_people": use0_people, "use1_people": use1_people, "t_use": t_use, "p_use": p_use,
        "r_der": (r_der, p_der), "r_unc": (r_unc, p_unc),
        "pol_gap": pol_gap,
        "overall_trust_gap": merged["trust_gap"].mean(),
        "df_smf": df_smf,
        "model_smf": model_smf,
        "rr": rr,
        "reg_cols": reg_cols,
        "soc15_stats": soc15_stats, # Data for new page
        "reason_labels": reason_labels,
        "reason_cols": reason_cols,
        "hlm_stats": hlm_stats,
        "race_stats": race_stats,
        "iti_stats": iti_stats,
        "quad_stats_dict": quad_stats_dict
    }
    return data


data = load_and_prepare()
OCC = data["OCC"]
full = data["full"]
merged = data["merged"]
df_smf = data["df_smf"]
model_smf = data["model_smf"]
rr = data["rr"]
reg_cols = data["reg_cols"]
soc15_stats = data["soc15_stats"]
reason_labels = data["reason_labels"]
reason_cols = data["reason_cols"]
hlm_stats = data["hlm_stats"]
race_stats = data["race_stats"]
iti_stats = data["iti_stats"]
quad_stats_dict = data["quad_stats_dict"]

ACCENTS = {"blue": C["blue"], "rose": C["rose"], "amber": C["amber"], "green": C["emerald"],
           "emerald": C["emerald"], "violet": C["violet"], "sky": C["sky"]}

PLOTLY_SANS = "Inter, 'Segoe UI', Helvetica Neue, Arial, sans-serif"
PLOTLY_COLORWAY = [C["blue"], C["emerald"], C["amber"], C["rose"], C["sky"], C["violet"]]

def fig_style(fig, height=440):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=PLOTLY_COLORWAY,
        font=dict(color=C["text"], family=PLOTLY_SANS, size=13), height=height,
        title_font=dict(family=PLOTLY_SANS, size=15, color=C["text"]),
        margin=dict(t=56, l=10, r=10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(size=12.5)),
        hoverlabel=dict(bgcolor="#0F172A", font=dict(family=PLOTLY_SANS, color="#fff", size=12.5),
                         bordercolor="#0F172A"),
    )
    fig.update_xaxes(gridcolor="#EEF1F6", zerolinecolor="#E2E8F0", linecolor="#E2E8F0",
                      tickfont=dict(size=11.5, color=C["muted"]))
    fig.update_yaxes(gridcolor="#EEF1F6", zerolinecolor="#E2E8F0", linecolor="#E2E8F0",
                      tickfont=dict(size=11.5, color=C["muted"]))
    return fig

def section(title, accent="blue", badge=None):
    color = ACCENTS.get(accent, C["blue"])
    b = f'<span class="new-badge">{badge}</span>' if badge else ""
    st.markdown(
        f"<h2 style='border-left:6px solid {color};padding-left:14px;margin-top:32px'>{title}{b}</h2>"
        f"<hr class='section-rule'>",
        unsafe_allow_html=True)

def insight_box(text):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)

def method_box(text):
    st.markdown(f'<div class="method-box"><strong>Phương pháp.</strong> {text}</div>', unsafe_allow_html=True)

def source_tag(*sources):
    tags = "".join(f'<span class="source-tag">{s}</span>' for s in sources)
    st.markdown(tags, unsafe_allow_html=True)

def kpi_row(items):
    cards = "".join(
        f'<div class="kpi-card" style="--accent:{ACCENTS.get(accent, C["rose"])}">'
        f'<div class="val">{val}</div><div class="lbl">{lbl}</div></div>'
        for val, lbl, accent in items
    )
    st.markdown(f'<div class="card-row">{cards}</div>', unsafe_allow_html=True)

def info_row(items):
    cards = "".join(
        f'<div class="info-card" style="--accent:{ACCENTS.get(accent, C["emerald"])}">'
        f'<div class="info-title">{title}</div><div class="info-body">{body}</div></div>'
        for title, body, accent in items
    )
    st.markdown(f'<div class="card-row">{cards}</div>', unsafe_allow_html=True)

def rec_card(icon, title, body):
    st.markdown(
        f'<div class="rec-card"><div class="rec-icon">{icon}</div>'
        f'<div><div class="rec-title">{title}</div><div class="rec-body">{body}</div></div></div>',
        unsafe_allow_html=True)

def draw_mermaid(code: str, height=600):
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
      <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
      </script>
      <style>
        body {{
          background-color: transparent;
          margin: 0;
          display: flex;
          justify-content: center;
          align-items: center;
        }}
      </style>
    </head>
    <body>
      <div class="mermaid">
        {code}
      </div>
    </body>
    </html>
    """
    import streamlit.components.v1 as components
    components.html(html_code, height=height, scrolling=True)

def html_table(df, index_label=""):
    cols = ([index_label] if df.index.name or index_label else []) + list(df.columns)
    head = "".join(f"<th>{c}</th>" for c in cols)
    rows = ""
    show_index = bool(index_label)
    for idx, row in df.iterrows():
        cells = (f"<td><strong>{idx}</strong></td>" if show_index else "") + \
                "".join(f"<td>{v}</td>" for v in row)
        rows += f"<tr>{cells}</tr>"
    st.markdown(f'<table class="styled-table"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>',
                unsafe_allow_html=True)

def full_reg_table(reg_df):
    """Bảng hồi quy đầy đủ: beta, SE, t, p, khoảng tin cậy 95%."""
    disp = reg_df.copy()
    disp["beta"] = disp["beta"].map(lambda v: f"{v:+.3f}" if isinstance(v, (int, float)) else v)
    disp["se"] = disp["se"].map(lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else v)
    disp["t"] = disp["t"].map(lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else v)
    disp["p"] = disp["p"].map(lambda v: "< 0.001" if isinstance(v, (int, float)) and v < 0.001 else f"{v:.3f}")
    if "ci_lo" in disp.columns:
        disp["Khoảng tin cậy 95%"] = disp.apply(lambda r: f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]", axis=1)
        disp = disp[["feature", "beta", "se", "t", "p", "Khoảng tin cậy 95%"]]
    else:
        disp = disp[["feature", "beta", "se", "t", "p"]]
    disp.columns = ["Biến", "β", "SE", "t", "p"] + (["Khoảng tin cậy 95%"] if "Khoảng tin cậy 95%" in disp.columns else [])
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# SIDEBAR — 6 TRANG
# ═══════════════════════════════════════════════════════
PAGES = {
    "Tổng Quan": "intro",
    "Mô Hình": "drivers",
    "Tương Quan": "correlation",
    "Ngành Nghề Chung": "structural",
    "Công Nghệ (SOC 15)": "soc15",
    "Kiến Nghị AI Agent": "future_agent",
    "Trợ Lý AI (Gemini)": "gemini_chat",
    "Phụ Lục": "appendix",
}

with st.sidebar:
    st.markdown(f"<div style='font-size:1.25rem;font-weight:700;color:{C['text']};line-height:1.3'>"
                f"Rào Cản Tâm Lý Với AI</div>", unsafe_allow_html=True)
    st.markdown(
        f"<span style='font-size:.92rem;color:{C['muted']};line-height:1.5'>Yếu tố dự báo sự kháng cự với "
        f"tự động hoá AI ở người lao động.<br>5 731 đánh giá × 104 nghề × 1 500 người lao động</span>",
        unsafe_allow_html=True)
    st.divider()
    page = st.radio("Chọn phần", list(PAGES.keys()), label_visibility="collapsed")
    page_key = PAGES.get(page, "intro")
    st.divider()
    
    # Gemini API Key configuration
    gemini_api_key = st.text_input("Nhập Gemini API Key", type="password", help="Dành cho trang Trợ Lý AI")
    if gemini_api_key:
        st.success("Gemini API Key: OK")
    
    st.divider()
    st.markdown(
        f"<div style='font-size:.85rem;color:{C['muted']};line-height:1.7'>"
        f"Nguồn dữ liệu: WORKBank<br>(Shao et al. 2025, Stanford SALT-NLP)<br>"
        f"arXiv:2506.06576<br>huggingface.co/datasets/SALT-NLP/WORKBank</div>", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════╗
# ║  0. TỔNG QUAN                                         ║
# ╚══════════════════════════════════════════════════════╝
if page_key == "intro":
    st.title("Rào Cản Tâm Lý Với AI")
    st.markdown(
        "<p style='font-size:1.03rem;color:#475569;max-width:850px;line-height:1.75'>"
        "Nghiên cứu này đo khoảng cách giữa năng lực kỹ thuật của AI và mong muốn giao việc của người lao "
        "động, sau đó dùng hồi quy đa biến để xác định biến số nào dự báo sự kháng cự khi AI đã đủ năng lực "
        "thực hiện nhiệm vụ. Các phát hiện được trình bày kèm phương trình hồi quy, hệ số, sai số chuẩn và "
        "kiểm định ngoài mẫu."
        "</p>", unsafe_allow_html=True)

    insight_box(
        "Biến trung tâm — <code>trust_gap</code> = <code>Automation Capacity Rating</code> "
        "(chuyên gia AI chấm) trừ <code>Automation Desire Rating</code> (người lao động tự chấm), gộp theo "
        "<code>Task ID</code>. trust_gap dương nghĩa là AI có năng lực thực hiện nhưng người lao động không "
        "muốn giao việc."
    )

    r_enj = data["r_enj"][0]
    suf_beta = data["reg_result"][data["reg_result"]["feature"].str.contains("chịu khổ")]["beta"].values[0]
    vd = data["variance_decomp"]
    kpi_row([
        (f"{data['overall_trust_gap']:.2f}", "trust_gap trung bình\n(n = 5 731 dòng task)", "blue"),
        (f"{r_enj:.2f}", "Tương quan Enjoyment–Desire\n(p < 0.001)", "rose"),
        (f"{suf_beta:+.2f}", "β Niềm tin AI chịu khổ\n(kiểm soát đầy đủ)", "violet"),
        (f"{vd['eta_occ']*100:.1f}%", "Phương sai Desire do\nOccupation giải thích (η²)", "amber"),
    ])

    section("Quy trình xử lý dữ liệu", accent="blue")
    st.markdown("""
1. **Gộp năng lực AI theo Task ID.** `expert_rated_technological_capability.csv` có 2 057 dòng nhưng chỉ 846
   Task ID duy nhất, nên lấy trung bình trước khi gộp.
2. **Gộp chính.** Bảng năng lực gộp với `domain_worker_desires.csv` theo `Task ID`, thu được 5 731 dòng.
3. **Gắn lương nghề.** Gộp với `task_statement_with_metadata.csv` theo `Task ID`.
4. **Gắn hồ sơ người lao động.** Gộp với `domain_worker_metadata.csv` theo `User ID`, thu được học vấn, thu
   nhập, bốn thang thái độ với AI, khuynh hướng chính trị, chủng tộc.
5. **Nhóm nghề lớn (major SOC).** Lấy hai chữ số đầu của `O*NET-SOC Code` để phân loại 104 nghề vào 13 nhóm
   nghề lớn có mặt trong dữ liệu.
""")
    method_box(
        "Mọi hệ số β trong ứng dụng này đến từ hồi quy tuyến tính bội OLS (bình phương tối thiểu thông thường), "
        "không phải mô hình học máy dự đoán phi tuyến. Phương trình đầy đủ và bảng hệ số trình bày tại trang "
        "Mô Hình."
    )
    source_tag("domain_worker_desires.csv", "expert_rated_technological_capability.csv",
               "task_statement_with_metadata.csv", "domain_worker_metadata.csv")


# ╔══════════════════════════════════════════════════════╗
# ║  1. MÔ HÌNH HỒI QUY ĐA BIẾN                           ║
# ╚══════════════════════════════════════════════════════╝
elif page_key == "drivers":
    st.title("Mô Hình Hồi Quy Đa Biến")
    st.latex(r"\text{Automation Desire}_i = \alpha + \sum_{k} \beta_k \cdot z(X_k)_i + \varepsilon_i")

    method_box(f"""
    Biến phụ thuộc là <code>Automation Desire Rating</code> (thang 1–5, người lao động tự chấm cho từng nhiệm vụ).
    <code>z(X_k)</code> là biến độc lập thứ k sau khi chuẩn hoá z-score (trừ trung bình mẫu, chia độ lệch chuẩn
    mẫu), giúp các hệ số β so sánh được trực tiếp dù đơn vị gốc khác nhau (ví dụ tuổi so với lương nghề).<br><br>
    • <strong>α</strong> (hằng số hồi quy): giá trị kỳ vọng của Automation Desire khi mọi biến độc lập bằng
    giá trị trung bình mẫu.<br>
    • <strong>β<sub>k</sub></strong> (hệ số chuẩn hoá): mức thay đổi của Automation Desire, tính bằng độ lệch
    chuẩn, khi X<sub>k</sub> tăng một độ lệch chuẩn, giữ nguyên toàn bộ biến còn lại trong mô hình.<br>
    • <strong>SE</strong> (sai số chuẩn của β) và thống kê <strong>t = β/SE</strong> dùng để kiểm định giả
    thuyết β = 0 bằng phân phối t hai phía.<br>
    • <strong>Khoảng tin cậy 95%</strong> = β ± 1,96 × SE.<br>
    • Ước lượng bằng OLS (<code>numpy.linalg.lstsq</code>), không dùng mô hình học máy phi tuyến.<br>
    • <strong>Kiểm định ngoài mẫu</strong>: dữ liệu được chia 80% ước lượng hệ số (train), 20% kiểm tra (test,
    dữ liệu mô hình chưa dùng để ước lượng), random_state cố định để tái lập. R² đo tỷ lệ phương sai của Desire
    mà mô hình giải thích được; R² test gần R² train cho thấy mô hình không quá khớp (overfit) với dữ liệu ước lượng.
    """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Thích Việc", "Niềm Tin Đạo Đức", "Khuếch Đại Kỹ Năng", "Chuyên Gia", "Phần Dư"])

    # ── TAB 1: THÍCH VIỆC ──
    with tab1:
        r_enj, p_enj = data["r_enj"]
        r_sec, p_sec = data["r_sec"]
        source_tag("domain_worker_desires.csv → Enjoyment Rating, Job Security Rating, Automation Desire Rating")

        c1, c2 = st.columns(2)
        c1.metric("Tương quan Enjoyment–Desire", f"r = {r_enj:.3f}", f"p = {p_enj:.1e}")
        c2.metric("Tương quan Job Security–Desire", f"r = {r_sec:.3f}", f"p = {p_sec:.1e}")

        insight_box(
            f"Mức thích thú công việc (<code>Enjoyment Rating</code>) tương quan âm với mong muốn tự động hoá "
            f"(r={r_enj:.3f}, p&lt;0,001), là tương quan đơn biến mạnh nhất trong toàn bộ phân tích. Cảm giác an "
            f"toàn việc làm cũng tương quan âm nhưng yếu hơn (r={r_sec:.3f}) và mất ý nghĩa thống kê trong mô hình "
            f"đa biến ở tab kế tiếp: khi kiểm soát các biến khác, lo ngại mất việc không còn dự báo được mong "
            f"muốn tự động hoá."
        )

        fig = px.scatter(merged.sample(min(2000, len(merged)), random_state=1),
            x="Enjoyment Rating", y="Automation Desire Rating", opacity=0.25,
            color_discrete_sequence=[C["rose"]], trendline="ols",
            title="Enjoyment Rating theo Automation Desire Rating (mẫu 2 000 dòng, cấp task)")
        st.plotly_chart(fig_style(fig), use_container_width=True)

        lo = data["reason_domain"]
        fi = data["reason_free_time"]
        info_row([
            ("Khi Desire ≥ 4", f"{fi*100:.1f}% người chọn lý do &ldquo;giải phóng thời gian&rdquo;", "emerald"),
            ("Khi Desire ≤ 2", f"{lo*100:.1f}% người chọn lý do &ldquo;AI thiếu chuyên môn ngành&rdquo;", "rose"),
        ])

    # ── TAB 2: NIỀM TIN ĐẠO ĐỨC ──
    with tab2:
        source_tag("domain_worker_metadata.csv → 4 thang thái độ AI", "domain_worker_desires.csv → Automation Desire Rating")

        insight_box(
            "Phát hiện chính của phân tích này: nghiên cứu gốc WORKBank dùng học vấn và thu nhập làm biến dự "
            "báo chính, không đưa các thang thái độ đạo đức về AI vào cùng một mô hình đa biến với các biến "
            "kiểm soát khác."
        )

        reg = data["reg_result"]
        tt = data["tt_main"]
        fig = go.Figure(go.Bar(
            x=reg["beta"], y=reg["feature"], orientation="h",
            marker_color=[C["rose"] if b < 0 else C["emerald"] for b in reg["beta"]],
            text=[f"β={b:+.3f}" for b in reg["beta"]],
            textposition="outside"))
        fig.update_layout(title=f"Hệ số hồi quy chuẩn hoá của Automation Desire (n={data['n_reg']})")
        st.plotly_chart(fig_style(fig, height=400), use_container_width=True)

        st.markdown("**Bảng hệ số đầy đủ**")
        full_reg_table(reg)

        c1, c2, c3 = st.columns(3)
        c1.metric("R² tập train (80%)", f"{tt['r2_train']:.3f}", f"n={tt['n_train']}")
        c2.metric("R² tập test (20%)", f"{tt['r2_test']:.3f}", f"n={tt['n_test']}")
        c3.metric("Chênh lệch train − test", f"{tt['r2_train']-tt['r2_test']:+.3f}")

        suf_row = reg[reg["feature"].str.contains("chịu khổ")]
        sec_row = reg[reg["feature"].str.contains("an toàn")]
        st.markdown(f"""
Mô hình kiểm soát đồng thời: mức thích thú công việc, cảm giác an toàn việc làm, năng lực AI khách quan, học
vấn, tần suất dùng LLM, tuổi, giới tính, lương nghề (n={data['n_reg']}). Hệ số của niềm tin &ldquo;AI có thể
chịu khổ&rdquo; là β = {suf_row['beta'].values[0]:+.3f} (p&lt;0,001), gần bằng độ lớn hệ số của mức thích thú
công việc, lớn hơn hệ số của học vấn và giới tính. Hệ số của cảm giác an toàn việc làm là
β = {sec_row['beta'].values[0]:+.3f} (p = {sec_row['p'].values[0]:.3f}), không có ý nghĩa thống kê ở ngưỡng
0,05. R² tập test ({tt['r2_test']:.3f}) gần với R² tập train ({tt['r2_train']:.3f}), cho thấy các hệ số trên
không phải kết quả của việc mô hình khớp quá mức với dữ liệu ước lượng.
""")

        st.markdown("**Mô hình đối chứng: bốn thang thái độ AI cùng lúc**")
        a4 = data["attitude4"]
        tt4 = data["tt_attitude"]
        fig2 = go.Figure(go.Bar(
            x=a4["beta"], y=a4["feature"], orientation="h",
            marker_color=[C["rose"] if b < 0 else C["emerald"] for b in a4["beta"]],
            text=[f"β={b:+.3f}" for b in a4["beta"]], textposition="outside"))
        fig2.update_layout(title=f"Bốn thang thái độ AI trong cùng một mô hình (n={data['n_reg4']}, "
                                  f"R² train={tt4['r2_train']:.3f} / test={tt4['r2_test']:.3f})")
        st.plotly_chart(fig_style(fig2, height=320), use_container_width=True)
        insight_box(
            "Khi đưa cả bốn thang thái độ AI (Suffering, Job Importance, Tedious Work, Daily Interest) vào cùng "
            "một mô hình, thang <strong>Daily Interest</strong> (hứng thú dùng AI hằng ngày) và thang "
            "<strong>Suffering</strong> (tin AI có thể chịu khổ) có hệ số lớn nhất và ngược dấu nhau. Thang Job "
            "Importance không có ý nghĩa thống kê (p ≈ 0,11). Niềm tin đạo đức về AI do đó không tương đương với "
            "mức độ ưa thích công nghệ nói chung; đây là một trục dự báo riêng, không trùng với biến lo ngại "
            "kinh tế đã kiểm soát trong mô hình."
        )

    # ── TAB 3: KHUẾCH ĐẠI KỸ NĂNG ──
    with tab3:
        source_tag("domain_worker_metadata.csv → Education, LLM Use in Work")

        st.latex(r"\text{Desire}_i = \alpha + \beta_1 z(\text{Edu})_i + \beta_2 z(\text{Use})_i + \beta_3 \big(z(\text{Edu})_i \cdot z(\text{Use})_i\big) + \sum \beta_k z(X_k)_i + \varepsilon_i")
        st.markdown(
            "Mô hình bổ sung một số hạng tương tác, tích của học vấn chuẩn hoá và tần suất dùng LLM chuẩn hoá. "
            "Hệ số β₃ của số hạng này đo mức độ hiệu ứng của việc dùng AI lên Desire khác nhau như thế nào giữa "
            "các mức học vấn — đây là hiệu ứng điều tiết (moderation), khác với hệ số chính của từng biến riêng lẻ."
        )
        amp = data["amplification"]
        tta = data["tt_amp"]
        c1, c2 = st.columns(2)
        c1.metric("β₃ (tương tác Học vấn × Dùng AI)", f"{amp['beta_inter']:+.3f}",
                  f"SE = {amp['se_inter']:.3f}, p = {amp['p_inter']:.4f}")
        c2.metric("Cỡ mẫu hồi quy", f"n = {amp['n']}")
        c3, c4 = st.columns(2)
        c3.metric("R² train (80%)", f"{tta['r2_train']:.3f}", f"n={tta['n_train']}")
        c4.metric("R² test (20%)", f"{tta['r2_test']:.3f}", f"n={tta['n_test']}")

        insight_box(
            f"Mô hình kiểm soát đầy đủ (mức thích thú công việc, cảm giác an toàn việc làm, năng lực AI, tuổi, "
            f"giới tính, lương nghề, niềm tin đạo đức về AI): hệ số tương tác giữa học vấn và tần suất dùng LLM "
            f"là β₃ = {amp['beta_inter']:+.3f} (p={amp['p_inter']:.4f}). Tần suất dùng AI cao hơn tương ứng với "
            f"Automation Desire cao hơn, và mức tăng này lớn hơn ở nhóm học vấn cao. R² test "
            f"({tta['r2_test']:.3f}) gần với R² train ({tta['r2_train']:.3f}), cho thấy hiệu ứng tương tác không "
            f"phải do khớp quá mức với dữ liệu ước lượng."
        )

    # ── TAB 4: CHUYÊN GIA ──
    with tab4:
        source_tag("domain_worker_desires.csv → Self-reported Expertise, Reasons for Human Agency")
        insight_box(
            "Phát hiện thứ hai: người tự nhận có chuyên môn cao (Expert) với nhiệm vụ đó có mức kháng cự với AI "
            "cao hơn người tự nhận ở mức trung bình (Average), và đưa ra lý do kháng cự khác với hai nhóm còn lại."
        )

        eg = data["exp_groups"]
        c1, c2, c3 = st.columns(3)
        for col, grp in zip([c1, c2, c3], ["Novice", "Average", "Expert"]):
            row = eg.loc[grp]
            col.markdown(f'<div class="kpi-card" style="--accent:{C["violet"]}"><div class="val">{row["trust_gap"]:.3f}</div>'
                          f'<div class="lbl">trust_gap — {grp}\n(n={int(row["n"])} dòng)</div></div>',
                          unsafe_allow_html=True)

        t_exp, p_exp = data["t_exp"], data["p_exp"]
        t_nov, p_nov = data["t_nov"], data["p_nov"]
        st.markdown(f"""
So với nhóm Average, kiểm định t Welch cấp dòng task cho thấy nhóm Expert có trust_gap cao hơn có ý nghĩa
(t={t_exp:.2f}, p={p_exp:.4f}); nhóm Novice cũng cao hơn nhưng ở mức biên (t={t_nov:.2f}, p={p_nov:.4f}). Số
liệu thô cho quan hệ dạng chữ U theo mức tự nhận chuyên môn.
""")

        er = data["expert_robust"]
        tte = data["tt_expert"]
        st.markdown(f"""
**Kiểm tra độ vững.** Dùng cùng mô hình OLS như các tab trên, thêm bậc một và bậc hai của mức tự nhận chuyên
môn, kiểm soát mức thích thú công việc, cảm giác an toàn việc làm, năng lực AI, học vấn, tần suất dùng LLM,
tuổi, giới tính, lương nghề, niềm tin đạo đức về AI (n={er['n']}, R² train={tte['r2_train']:.3f} /
test={tte['r2_test']:.3f}):

| Số hạng | β | p | Kết luận |
|---|---|---|---|
| Tự nhận chuyên môn (bậc 1) | {er['beta_linear']:+.3f} | {er['p_linear']:.4f} | Có ý nghĩa thống kê |
| Tự nhận chuyên môn (bậc 2) | {er['beta_quad']:+.3f} | {er['p_quad']:.4f} | Không có ý nghĩa thống kê |

Phần Novice kháng cự trong bảng thô có thể do biến gây nhiễu là tần suất dùng LLM thấp hơn ở nhóm này. Phần
Expert kháng cự vẫn có ý nghĩa thống kê sau khi kiểm soát đầy đủ các biến trên, và R² test gần R² train cho
thấy mô hình không quá khớp với dữ liệu ước lượng.
""")

        st.markdown("**Lý do kháng cự theo mức tự nhận chuyên môn (Desire ≤ 2)**")
        rdf = data["reason_df"]
        fig = go.Figure()
        for grp, color in zip(["Novice", "Average", "Expert"], [C["sky"], C["amber"], C["rose"]]):
            fig.add_trace(go.Bar(name=grp, x=rdf.index, y=rdf[grp] * 100, marker_color=color))
        fig.update_layout(barmode="group", title="Tỷ lệ chọn từng lý do kháng cự, theo nhóm tự nhận chuyên môn (%)",
                           yaxis_title="% chọn lý do này")
        st.plotly_chart(fig_style(fig, height=420), use_container_width=True)

        insight_box(
            f"Trong nhóm kháng cự mạnh (Desire ≤ 2), nhóm Expert chọn lý do &ldquo;AI thiếu chuyên môn "
            f"ngành&rdquo; ở tỷ lệ {rdf.loc['Thiếu chuyên môn ngành','Expert']*100:.1f}% và &ldquo;cần giám sát "
            f"chất lượng&rdquo; ở tỷ lệ {rdf.loc['Cần giám sát chất lượng','Expert']*100:.1f}%, cao hơn nhóm "
            f"Average ({rdf.loc['Thiếu chuyên môn ngành','Average']*100:.1f}% / "
            f"{rdf.loc['Cần giám sát chất lượng','Average']*100:.1f}%) và nhóm Novice "
            f"({rdf.loc['Thiếu chuyên môn ngành','Novice']*100:.1f}% / "
            f"{rdf.loc['Cần giám sát chất lượng','Novice']*100:.1f}%). Đây là hai lý do liên quan trực tiếp đến "
            f"việc bảo vệ vị trí chuyên môn, khác với các lý do mà nhóm Novice đưa ra."
        )

    # ── TAB 5: PHẦN DƯ (Snippet 1) ──
    with tab5:
        source_tag("domain_worker_desires.csv", "expert_rated_technological_capability.csv", "domain_worker_metadata.csv")
        
        st.markdown("### Nghề nào kháng cự AI vượt ngoài dự báo kinh tế?")
        
        res_by_occ = df_smf.groupby('major_soc')['residual'].mean().sort_values()
        res_by_occ_labels = res_by_occ.index.map(lambda x: MAJOR_SOC_LABELS.get(str(x), str(x)))
        
        fig = px.bar(
            x=res_by_occ.values,
            y=res_by_occ_labels,
            orientation='h',
            color=res_by_occ.values,
            color_continuous_scale='RdBu_r',
            labels={'x': 'Phần dư trung bình (Residual)', 'y': 'Nhóm nghề lớn (Major SOC)'}
        )
        fig.update_layout(title="Phần dư mong muốn tự động hóa trung bình theo Nhóm SOC lớn (Snippet 1)")
        st.plotly_chart(fig_style(fig, height=480), use_container_width=True)
        
        insight_box(
            "Nếu giá trị dương, nghề đó khao khát AI hơn mức thu nhập/độ tuổi lý giải được. "
            "Nếu âm, nghề đó có rào cản tâm lý cực đoan."
        )

        st.markdown("### Bảng hệ số hồi quy kiểm soát (Snippet 1)")
        simple_summary = pd.DataFrame({
            "feature": model_smf.params.index,
            "beta": model_smf.params.values,
            "se": model_smf.bse.values,
            "t": model_smf.tvalues.values,
            "p": model_smf.pvalues.values,
            "ci_lo": model_smf.conf_int()[0].values,
            "ci_hi": model_smf.conf_int()[1].values
        })
        full_reg_table(simple_summary)


# ╔══════════════════════════════════════════════════════╗
# ║  2. TƯƠNG QUAN — MA TRẬN HEATMAP                      ║
# ╚══════════════════════════════════════════════════════╝
elif page_key == "correlation":
    st.title("Ma Trận Tương Quan")
    source_tag("full (5 731 dòng, sau khi loại giá trị thiếu ở các biến dưới đây)")

    cdf = data["corr_df"]
    fig = go.Figure(data=go.Heatmap(
        z=cdf.values, x=list(cdf.columns), y=list(cdf.index),
        colorscale=[[0, "#E11D48"], [0.5, "#F8FAFC"], [1, "#059669"]],
        zmin=-1, zmax=1, text=np.round(cdf.values, 2), texttemplate="%{text}",
        textfont=dict(size=12), colorbar=dict(title="r")))
    fig.update_layout(title=f"Hệ số tương quan Pearson giữa các biến số chính (n = {data['n_corr']})",
                       height=620)
    fig.update_xaxes(tickangle=-35)
    st.plotly_chart(fig_style(fig, height=620), use_container_width=True)

    insight_box(
        "trust_gap tương quan âm rất mạnh với Automation Desire Rating (định nghĩa của trust_gap khiến quan hệ "
        "này gần như tất yếu, do trust_gap được tính trực tiếp từ Desire) và gần như không tương quan với "
        "Automation Capacity Rating (r nhỏ). Điều này cho thấy biến động của trust_gap giữa các nhiệm vụ chủ "
        "yếu do biến động của Desire quyết định, không phải do biến động của năng lực AI khách quan — năng lực "
        "kỹ thuật tương đối ổn định giữa các nhiệm vụ so với mức chênh lệch trong mong muốn của người lao động."
    )

    st.markdown("""
**Các quan hệ đơn biến đáng chú ý trong ma trận:**
- Enjoyment Rating tương quan âm với Automation Desire Rating, phù hợp với phát hiện tại trang Mô Hình.
- use_num (tần suất dùng LLM) tương quan dương với Automation Desire Rating, hệ số dương lớn nhất trong nhóm
  biến nhân khẩu học.
- suffering_num (niềm tin AI có thể chịu khổ) tương quan âm với Automation Desire Rating và tương quan dương
  với Job Security Rating, cho thấy hai thang đo phản ánh hai khía cạnh dự báo khác nhau.
- Domain Expertise Requirement và Involved Uncertainty (đặc điểm khách quan của nhiệm vụ) có hệ số tương quan
  gần 0 với trust_gap, phù hợp với kết quả kiểm định tại trang Phụ Lục.
""")


# ╔══════════════════════════════════════════════════════╗
# ║  3. Ngành Nghề                                       ║
# ╚══════════════════════════════════════════════════════╝
elif page_key == "structural":
    st.title("Rào Cản Theo Ngành Nghề Chung")

    tab1, tab2, tab3 = st.tabs(["Nhóm Ngành", "Nghề Cụ Thể", "Lương Nghề"])

    with tab1:
        source_tag("task_statement_with_metadata.csv → O*NET-SOC Code (2 chữ số đầu = nhóm nghề lớn)")

        mg = data["major_group_table"]
        fig = px.bar(mg, x="desire", y="major_soc_label", orientation="h", color="trust_gap",
            color_continuous_scale=["#059669", "#F1F5F9", "#E11D48"], range_color=[-1.2, 1.2],
            text=[f"n={n}" for n in mg["n"]],
            title="Automation Desire trung bình theo toàn bộ 13 nhóm nghề lớn có trong dữ liệu (màu = trust_gap)")
        fig.update_layout(yaxis=dict(autorange="reversed"), yaxis_title="", xaxis_title="Automation Desire Rating")
        st.plotly_chart(fig_style(fig, height=560), use_container_width=True)

        top, bot = mg.iloc[0], mg.iloc[-1]
        st.markdown(f"""
Toàn bộ 13 nhóm nghề lớn (mã hai chữ số đầu của O*NET-SOC) có mặt trong dữ liệu được đưa vào so sánh. Nhóm có Automation Desire trung bình cao nhất là
**{top['major_soc_label']}** ({top['desire']:.2f} điểm, trust_gap={top['trust_gap']:+.2f}); nhóm thấp nhất là
**{bot['major_soc_label']}** ({bot['desire']:.2f} điểm, trust_gap={bot['trust_gap']:+.2f}).
""")

    with tab2:
        source_tag("domain_worker_desires.csv, expert_rated_technological_capability.csv (toàn bộ 104 nghề, n ≥ 15 dòng/nghề)")

        occ_all = data["occ_all"]
        top5 = occ_all.head(5)
        bot5 = occ_all.tail(5)
        show = pd.concat([top5, bot5])
        fig = px.bar(show, x="desire", y=OCC, orientation="h", color="trust_gap",
            color_continuous_scale=["#059669", "#F1F5F9", "#E11D48"], range_color=[-1.2, 1.2],
            text=[f"n={n}" for n in show["n"]],
            title="5 nghề có Automation Desire cao nhất và 5 nghề thấp nhất (trong toàn bộ 104 nghề, màu = trust_gap)")
        fig.update_layout(yaxis=dict(autorange="reversed"), yaxis_title="")
        st.plotly_chart(fig_style(fig, height=560), use_container_width=True)

    with tab3:
        source_tag("task_statement_with_metadata.csv → Occupation Mean Annual Wage (lương trung bình theo nghề, BLS)")

        wg = data["wage_gap"]
        fig = px.bar(wg.reset_index(), x="wage_bucket", y="trust_gap", color="wage_bucket",
            color_discrete_sequence=[C["rose"], C["amber"], C["emerald"]],
            text=[f"n={n}" for n in wg["n"]])
        fig.update_layout(showlegend=False, title="trust_gap theo ba nhóm lương nghề (chia tam phân vị)")
        st.plotly_chart(fig_style(fig, height=380), use_container_width=True)


# ╔══════════════════════════════════════════════════════╗
# ║  4. CÔNG NGHỆ (SOC 15) TẬP TRUNG                      ║
# ╚══════════════════════════════════════════════════════╝
elif page_key == "soc15":
    st.title("Phân Tích Chuyên Sâu: Nhóm Ngành Công Nghệ (SOC 15)")
    st.markdown(
        "<p style='font-size:1.03rem;color:#475569;max-width:850px;line-height:1.75'>"
        "Khám phá các tầng rào cản tâm lý của <b>Nhóm ngành Công nghệ, Máy tính và Toán học (SOC 15)</b>. "
        "Mặc dù có mức độ am hiểu công nghệ cao nhất, sự phân hóa rào cản bên trong ngành bộc lộ rõ tính "
        "thực dụng, phòng thủ chuyên môn và nghịch lý của kinh nghiệm."
        "</p>", unsafe_allow_html=True)
    
    soc = soc15_stats
    
    tab1, tab2, tab3, tab4 = st.tabs(["Tổng Quan SOC 15", "Phân Nhóm Nghề", "Khuếch Đại Kỹ Năng", "Nghịch Lý Tâm Lý"])
    
    # ── TAB 1: TỔNG QUAN SOC 15 ──
    with tab1:
        source_tag("Lọc từ O*NET-SOC Code bắt đầu bằng '15'")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số lượng Tasks", f"{soc['n_tasks']}")
        c2.metric("Số chuyên gia", f"{soc['n_users']}")
        c3.metric("Lương trung bình", f"${soc['wage_15']:,.0f}", f"so với ${soc['wage_non15']:,.0f} (Non-SOC 15)")
        c4.metric("Tần suất dùng LLM (1-4)", f"{soc['use_num_15']:.2f}", f"so với {soc['use_num_non15']:.2f} (Non-SOC 15)")
        
        st.markdown(f"### Chỉ Số Chấp Nhận AI: SOC 15 vs. Phần Còn Lại")
        
        fig = go.Figure(data=[
            go.Bar(name='SOC 15 (Công nghệ)', x=['Trust Gap (Thấp = Tốt)', 'Khao khát tự động hóa', 'Lo ngại mất việc'], 
                   y=[soc['trust_gap_15'], soc['desire_15'], soc['security_15']], marker_color=C["blue"]),
            go.Bar(name='Non-SOC 15 (Ngành khác)', x=['Trust Gap (Thấp = Tốt)', 'Khao khát tự động hóa', 'Lo ngại mất việc'], 
                   y=[soc['trust_gap_non15'], soc['desire_non15'], soc['security_non15']], marker_color=C["muted"])
        ])
        fig.update_layout(barmode='group')
        st.plotly_chart(fig_style(fig, height=350), use_container_width=True)
        
        insight_box(
            f"Nhóm chuyên gia công nghệ có <strong>Trust Gap rất thấp ({soc['trust_gap_15']:.2f})</strong> "
            f"so với các ngành khác ({soc['trust_gap_non15']:.2f}). Họ hiểu rõ năng lực AI nên không nghi ngờ vô cớ. "
            f"Tuy nhiên, lo ngại mất việc của họ lại cao hơn (Điểm an toàn {soc['security_15']:.2f} so với {soc['security_non15']:.2f})."
        )
        
        st.markdown(f"### Lý do TỪ CHỐI AI (Khi Automation Desire ≤ 2)")
        reason_df_15 = pd.DataFrame({
            "Lý do": [reason_labels[r] for r in reason_cols],
            "SOC 15": [soc['reasons_15'][r] for r in reason_cols],
            "Ngành Khác": [soc['reasons_non15'][r] for r in reason_cols]
        })
        fig_r = px.bar(reason_df_15, x="Lý do", y=["SOC 15", "Ngành Khác"], barmode="group",
                       color_discrete_map={"SOC 15": C["rose"], "Ngành Khác": C["amber"]})
        fig_r.update_layout(yaxis_title="Tỷ lệ người chọn (%)")
        st.plotly_chart(fig_style(fig_r, height=350), use_container_width=True)
        
        insight_box(
            f"Rào cản lớn nhất không phải là giao tiếp hay đạo đức. Khác biệt lớn nằm ở việc "
            f"bảo vệ chuyên môn: Nhóm 15-XXX kiên quyết từ chối nếu họ nghĩ AI <strong>thiếu kiến thức ngành "
            f"({soc['reasons_15']['Reasons for Human Agency - Domain Knowledge']:.1f}%)</strong> và cần "
            f"<strong>giám sát chất lượng ({soc['reasons_15']['Reasons for Human Agency - Quality Oversight']:.1f}%)</strong>."
        )

    # ── TAB 2: PHÂN NHÓM NGHỀ BÊN TRONG ──
    with tab2:
        source_tag("OCC = Nghề nghiệp cụ thể bên trong nhánh 15-XXX")
        occ_15 = soc['occ_stats']
        
        fig = px.bar(occ_15, x="desire", y=data["OCC"], orientation="h", color="trust_gap",
            color_continuous_scale=["#059669", "#F1F5F9", "#E11D48"], range_color=[-1.2, 1.2],
            text=[f"n={n}" for n in occ_15["task_count"]],
            title="Automation Desire theo nghề Công nghệ (Màu = Trust Gap, n >= 15 nhiệm vụ)")
        fig.update_layout(yaxis=dict(autorange="reversed"), yaxis_title="")
        st.plotly_chart(fig_style(fig, height=600), use_container_width=True)
        
        # Extracting specific data for narrative without hardcoding
        dba = occ_15[occ_15[data["OCC"]] == "Database Administrators"].iloc[0] if "Database Administrators" in occ_15[data["OCC"]].values else None
        prog = occ_15[occ_15[data["OCC"]] == "Computer Programmers"].iloc[0] if "Computer Programmers" in occ_15[data["OCC"]].values else None
        sec = occ_15[occ_15[data["OCC"]] == "Information Security Analysts"].iloc[0] if "Information Security Analysts" in occ_15[data["OCC"]].values else None
        
        insight_box(
            f"Sự khác biệt về bản chất công việc tạo ra rào cản trái ngược nhau:<br><br>"
            f"• <strong>Bảo thủ nhất (Database Administrators):</strong> Khao khát thấp nhất "
            f"({dba['desire']:.2f} điểm) và Trust Gap cao kỷ lục ({dba['trust_gap']:.2f}). Hậu quả của lệnh SQL "
            f"do AI viết sai là quá lớn (mất dữ liệu lõi), nên rào cản của họ là nỗi sợ rủi ro thảm họa.<br>"
            f"• <strong>Phòng thủ chuyên môn (Computer Programmers):</strong> Khao khát thấp ({prog['desire']:.2f}) "
            f"với Trust gap cực cao ({prog['trust_gap']:.2f}). Khi từ chối AI, họ thường xuyên vin vào cớ "
            f"cần kiểm soát (Control) và thiếu chuyên môn.<br>"
            f"• <strong>Cởi mở & Thực dụng (Information Security Analysts):</strong> Khao khát cao "
            f"({sec['desire']:.2f}) và Trust Gap âm ({sec['trust_gap']:.2f}). Xử lý log dữ liệu khổng lồ, "
            f"họ coi AI là đồng minh phân tích bất thường thay vì mối đe dọa thay thế."
        )

    # ── TAB 3: KHUẾCH ĐẠI KỸ NĂNG ──
    with tab3:
        st.markdown("### Sự Bất Bình Đẳng Mới: Hiện Tượng Khuếch Đại Kỹ Năng")
        source_tag("Mô hình hồi quy có hạng tử tương tác: Edu × LLM Use")
        
        use_df = soc['use_stats']
        edu_df = soc['edu_stats']
        
        c1, c2 = st.columns(2)
        fig_use = px.line(use_df, x="use_num", y="desire", text=use_df["desire"].round(2), markers=True,
                          title="Desire tăng khi Tần suất dùng LLM tăng", labels={"use_num":"Tần suất dùng (0-4)", "desire": "Khao khát tự động hóa"})
        fig_use.update_traces(textposition="top center")
        c1.plotly_chart(fig_style(fig_use, height=350), use_container_width=True)

        fig_edu = px.bar(edu_df, x="edu_num", y="desire", color="trust_gap", color_continuous_scale="RdBu",
                         title="Desire theo Cấp Bậc Học Vấn (Màu=Trust Gap)", labels={"edu_num":"Cấp học (0-6)", "desire": "Khao khát tự động hóa"})
        c2.plotly_chart(fig_style(fig_edu, height=350), use_container_width=True)

        inter = soc['inter_model']
        beta_inter = inter.params.get('inter_z', 0)
        p_inter = inter.pvalues.get('inter_z', 1)
        
        insight_box(
            f"Tác động của việc sử dụng AI không đồng đều (Hệ số tương tác β = {beta_inter:+.3f}, p = {p_inter:.3f}). "
            f"Nếu một kỹ sư có <strong>học vấn cao (Thạc sĩ, Tiến sĩ)</strong> sử dụng AI thường xuyên, "
            f"mức độ gỡ bỏ rào cản tâm lý của họ diễn ra <strong>nhanh và mạnh hơn rất nhiều</strong> so với "
            f"những người có học vấn thấp. Nhóm chuyên môn cao biết cách thiết kế prompt và tận dụng AI làm đòn bẩy, "
            f"biến AI thành 'trợ lý' thay vì coi đó là hộp đen đe dọa."
        )
        
        st.markdown("**Kết quả OLS: Mô hình tương tác Học vấn (Z) × Dùng AI (Z)**")
        inter_summary = pd.DataFrame({
            "Hạng mục": inter.params.index, "β": inter.params.values, 
            "SE": inter.bse.values, "p": inter.pvalues.values
        })
        st.dataframe(inter_summary, use_container_width=True, hide_index=True)

    # ── TAB 4: NGHỊCH LÝ TÂM LÝ ──
    with tab4:
        st.markdown("### Sự Ngụy Biện Của Thích Thú & Bi Kịch Của Lão Làng")
        source_tag("Chéo hóa: Mức độ thích thú công việc, Tuổi tác và Tự nhận chuyên môn")
        
        # Nghịch lý 1
        st.markdown("#### 1. Lý trí giả tạo (Vỏ bọc kỹ thuật)")
        enjoy_df = soc['reason_by_enjoy']
        fig_enj = px.bar(enjoy_df, x="Enjoyment_Level", y=list(reason_cols), barmode="group",
                         labels={"value": "Tỷ lệ chọn lý do (%)", "variable": "Lý do", "Enjoyment_Level": "Thích thú công việc"})
        fig_enj.update_layout(title="Lý do từ chối AI (Chỉ xét Desire <= 2): Khi thích việc, kỹ sư viện nhiều cớ kỹ thuật hơn")
        st.plotly_chart(fig_style(fig_enj, height=380), use_container_width=True)
        
        expert_enjoy = soc['reason_expert_enjoy']
        nov_domain = expert_enjoy.loc[expert_enjoy['Self-reported Expertise'] == 'Novice', 'Reasons for Human Agency - Domain Knowledge'].values[0]
        exp_domain = expert_enjoy.loc[expert_enjoy['Self-reported Expertise'] == 'Expert', 'Reasons for Human Agency - Domain Knowledge'].values[0]
        
        insight_box(
            f"Ngành IT tôn thờ logic. Khi kỹ sư không muốn giao một việc họ rất thích (Enjoyment cao), họ hiếm khi thừa nhận 'vì tôi thích code phần này'. "
            f"Thay vào đó, họ bọc mình trong các ngụy biện kỹ thuật. Đặc biệt, khi <strong>Lính mới (Novice)</strong> "
            f"muốn giữ việc họ thích, có tới <strong>{nov_domain:.1f}%</strong> dõng dạc chê AI 'thiếu chuyên môn', "
            f"trong khi nhóm <strong>Chuyên gia (Expert)</strong> thật sự chỉ viện cớ này ở mức <strong>{exp_domain:.1f}%</strong>."
        )
        
        st.markdown("#### 2. Nghịch lý tuổi nghề (Seniority Paradox)")
        age_stats = soc['age_stats']
        fig_age = px.bar(age_stats, x="Age_Group", y="Job_Security", color="Desire", color_continuous_scale="Blues",
                         title="An toàn việc làm suy giảm theo nhóm tuổi")
        st.plotly_chart(fig_style(fig_age, height=350), use_container_width=True)
        
        genx_sec = age_stats.loc[age_stats['Age_Group'] == 'Gen X+ (>45)', 'Job_Security'].values[0]
        genz_sec = age_stats.loc[age_stats['Age_Group'] == 'Gen Z (<30)', 'Job_Security'].values[0]
        
        insight_box(
            f"Trái với lầm tưởng Senior không sợ AI, dữ liệu phơi bày nghịch lý: "
            f"Nhóm Gen Z và Millennials thấy tương đối an toàn (điểm ~{genz_sec:.2f}/5), nhưng <strong>Gen X+ (>45 tuổi)</strong> "
            f"lại cảm thấy vị thế sinh tồn bị đe dọa nhất (Job Security giảm thê thảm xuống <strong>{genx_sec:.2f}</strong>). "
            f"AI đang làm tốt đến mức nó đe dọa làm bốc hơi giá trị kinh nghiệm hàng chục năm của các 'lão làng'."
        )





# ╔══════════════════════════════════════════════════════╗
# ║  6. KIẾN NGHỊ AI AGENT TƯƠNG LAI                      ║
# ╚══════════════════════════════════════════════════════╝
elif page_key == "future_agent":
    st.title("Kiến Nghị Thiết Kế AI Agent Thế Hệ Mới")
    st.markdown(
        "<p style='font-size:1.03rem;color:#475569;max-width:850px;line-height:1.75'>"
        "Dựa trên các phân tích định lượng về rào cản tâm lý và đe dọa bản sắc nghề nghiệp, chúng tôi đề xuất "
        "mô hình kiến trúc và quy trình vận hành cho một <b>Calibrated AI Agent (AI Agent hiệu chuẩn niềm tin)</b>."
        "</p>", unsafe_allow_html=True)
        
    tab1, tab2, tab3 = st.tabs(["Sơ đồ Workflow", "Khung Kiến Trúc", "Giả Lập Tương Tác"])
    
    with tab1:
        st.markdown("### Vòng Lặp Hiệu Chuẩn Niềm Tin & Phân Phối Quyền Lực")
        
        st.markdown("""
        <div style="background:#FFF; padding:30px; border-radius:16px; border:1px solid #E2E8F0; box-shadow:0 4px 12px rgba(0,0,0,0.02); margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                <div style="flex:1; min-width:180px; background:#4338CA; color:#FFF; padding:15px; border-radius:10px; text-align:center;">
                    <strong style="font-size:1rem; display:block; margin-bottom:5px;">1. Đánh giá Nhiệm vụ</strong>
                    Đo lường DER và UNC động bằng thuật toán nhận thức.
                </div>
                <div style="font-size:1.5rem; color:#64748B;">➔</div>
                <div style="flex:1; min-width:180px; background:#D97706; color:#FFF; padding:15px; border-radius:10px; text-align:center;">
                    <strong style="font-size:1rem; display:block; margin-bottom:5px;">2. Phân cấp Vận hành</strong>
                    Auto-Execute (Thấp DER) hoặc Co-Pilot / QA (Cao DER).
                </div>
                <div style="font-size:1.5rem; color:#64748B;">➔</div>
                <div style="flex:1; min-width:180px; background:#059669; color:#FFF; padding:15px; border-radius:10px; text-align:center;">
                    <strong style="font-size:1rem; display:block; margin-bottom:5px;">3. Giải thích Minh bạch</strong>
                    Show Chain-of-Thought & Điểm Rủi ro tự đánh giá.
                </div>
                <div style="font-size:1.5rem; color:#64748B;">➔</div>
                <div style="flex:1; min-width:180px; background:#7C3AED; color:#FFF; padding:15px; border-radius:10px; text-align:center;">
                    <strong style="font-size:1rem; display:block; margin-bottom:5px;">4. Trả Quyền Kiểm soát</strong>
                    User duyệt kết quả cuối, xoa dịu đe dọa bản sắc (ITI).
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Trực quan hóa Sơ đồ Workflow:")
        
        mermaid_code = """flowchart TD
    %% =========================
    %% INPUT
    %% =========================
    A["👤 Nhiệm vụ từ Người dùng"]

    %% =========================
    %% DECISION
    %% =========================
    B{"Đánh giá<br/>Độ phức tạp (DER)<br/>và<br/>Độ bất định (UNC)"}

    %% =========================
    %% EXECUTION MODES
    %% =========================
    C["⚡ Auto-Execute<br/>Tự động hoàn toàn"]

    D["🤝 Co-Pilot<br/>Đồng hành"]

    E["🛡️ QA Human-in-the-Loop<br/>Trợ lý giám sát"]

    %% =========================
    %% EXPLANATION
    %% =========================
    F["📋 Giải thích quyết định<br/>Đánh giá rủi ro"]

    G["✅ Người dùng phê duyệt<br/>Hiệu chuẩn niềm tin"]

    H["🔄 Phản hồi động<br/>Cập nhật ITI"]

    %% =========================
    %% FLOW
    %% =========================
    A --> B

    B -- "DER thấp<br/>UNC thấp" --> C

    B -- "DER cao<br/>UNC thấp" --> D

    B -- "UNC cao" --> E

    D --> F
    E --> F

    F --> G
    G --> H
    H --> B

    %% =========================
    %% STYLES
    %% =========================
    classDef input fill:#D6EAF8,stroke:#2E86C1,stroke-width:2px,color:#000;
    classDef decision fill:#FCF3CF,stroke:#B7950B,stroke-width:2px,color:#000;
    classDef mode fill:#D5F5E3,stroke:#239B56,stroke-width:2px,color:#000;
    classDef process fill:#FADBD8,stroke:#C0392B,stroke-width:2px,color:#000;

    class A input;
    class B decision;
    class C,D,E mode;
    class F,G,H process;"""

        draw_mermaid(mermaid_code, height=580)
        
        with st.expander("Xem mã nguồn Mermaid (Click để sao chép)"):
            st.code(mermaid_code, language="mermaid")

    with tab2:
        st.markdown("### Khung Kiến Trúc 3 Tầng Cho AI Agent Tương Lai")
        
        recs_agent = [
            ("🧠", "Tầng Nhận Thức Bản Sắc (Identity-Aware Cognitive Core)",
             "Tích hợp chỉ số ITI động để tự nhận biết khi nào một tác vụ đe dọa bản sắc của người phối hợp. "
             "Agent sẽ chủ động hạ giọng điệu tự đắc, chuyển từ khẳng định ('Tôi đã hoàn thành 100%') sang đề xuất hợp tác ('Tôi đã soạn thảo bản thảo, xin mời chuyên gia tối ưu hóa')."),
            ("⚖️", "Tầng Hiệu Chuẩn Năng Lực Động (Dynamic Competence Calibration)",
             "So khớp liên tục giữa độ khó chuyên môn (DER) của tác vụ với điểm tự tin nội bộ của AI. "
             "If tác vụ rơi vào vùng bất định (Cao UNC), Agent tự động giới hạn quyền hạn và bắt buộc có chữ ký phê duyệt của con người trước khi thực thi hệ thống."),
            ("🔍", "Tầng Giải Thích Minh Bạch (Explainable & Auditable Interface)",
             "Xóa bỏ hoàn toàn cơ chế Hộp Đen. Agent luôn hiển thị Chain-of-Thought (chuỗi tư duy suy luận) "
             "và tự chỉ ra các điểm yếu tiềm ẩn trong lập luận của mình để con người thực hiện Quality Oversight dễ dàng hơn.")
        ]
        for icon, title, body in recs_agent:
            rec_card(icon, title, body)

    with tab3:
        st.markdown("### Giả Lập Tương Tác Giữa Chuyên Gia & Calibrated AI Agent")
        st.write("Hãy chọn loại tác vụ để bắt đầu giả lập hội thoại tương tác giúp tối ưu hóa niềm tin:")
        
        sim_task = st.selectbox("Chọn tác vụ giả lập:", [
            "1. Cấu hình phân quyền hệ thống DNS (Cao DER / Thấp UNC)",
            "2. Ứng phó sự cố máy chủ bị DDoS (Cao DER / Cao UNC)",
            "3. Viết code script backup dữ liệu hàng ngày (Thấp DER / Thấp UNC)"
        ])
        
        if "1." in sim_task:
            st.markdown("""
            **🤖 Agent:** *"Chào bạn, tôi đã chuẩn bị cấu hình bản ghi DNS cho hệ thống mới. Vì đây là tác vụ có yêu cầu chuyên môn cao (DER=4.2), tôi đã bật chế độ **Co-Pilot**. "
            Dưới đây là cấu hình MX và SOA đề xuất. Xin lưu ý tôi đã đặt giá trị TTL là 3600 giây để tối ưu thời gian lan truyền. Bạn có muốn điều chỉnh thông số này hay phê duyệt thực thi?"*
            
            **🧑‍💻 Chuyên gia (Bạn):** *"Tại sao TTL lại là 3600? Tôi muốn đổi thành 300 để test nhanh hơn."*
            
            **🤖 Agent:** *"Lập luận hợp lý. Tôi đã cập nhật TTL = 300 giây theo chỉ thị của bạn. Quyền kiểm soát DNS thuộc về chuyên môn của bạn. Hãy nhấn nút dưới đây để phê duyệt khởi chạy."*
            """)
            st.button("Duyệt & Thực thi cấu hình DNS")
        elif "2." in sim_task:
            st.markdown("""
            **🤖 Agent:** *"Cảnh báo: Phát hiện lưu lượng tăng đột biến đáng ngờ trên cổng 80 (Bất định cực cao: UNC=4.8). "
            Hệ thống tự động đang rơi vào chế độ **Human-in-the-loop**. Tôi khuyến nghị chặn tạm thời dải IP `192.168.1.0/24` nhưng cần bạn xác nhận vì thao tác này có thể chặn nhầm khách hàng thật."*
            
            **🧑‍💻 Chuyên gia (Bạn):** *"Đồng ý, hãy chặn dải IP đó trong 15 phút, và bật cơ chế rate limiting."*
            
            **🤖 Agent:** *"Đã rõ. Tôi sẽ kích hoạt chặn IP trong 15 phút và thiết lập rate limit tối đa 100 req/min. Hệ thống giám sát sẽ cập nhật chỉ số trực tiếp cho bạn."*
            """)
            st.button("Xác nhận chặn dải IP nghi vấn DDoS")
        else:
            st.markdown("""
            **🤖 Agent:** *"Tôi đã viết xong kịch bản backup tự động hàng ngày (Tác vụ lặp lại thông thường, DER=1.5, UNC=1.2). "
            Tôi đã chạy thử nghiệm cục bộ và đạt kết quả chính xác 100%. Tôi xin phép tự động triển khai tác vụ này vào lúc 2:00 AM hàng ngày để giải phóng thời gian cho bạn."*
            
            **🧑‍💻 Chuyên gia (Bạn):** *"Tốt lắm, triển khai đi."*
            
            **🤖 Agent:** *"Đã cấu hình cron job chạy tự động. Nhiệm vụ hoàn thành."*
            """)
            st.button("Xác nhận kịch bản backup tự động")


# ╔══════════════════════════════════════════════════════╗
# ║  7. TRỢ LÝ AI (GEMINI)                                ║
# ╚══════════════════════════════════════════════════════╝
elif page_key == "gemini_chat":
    st.title("Trợ Lý Trí Tuệ Nhân Tạo (Gemini API)")
    st.markdown(
        "<p style='font-size:1.03rem;color:#475569;max-width:850px;line-height:1.75'>"
        "Hỏi đáp trực tiếp với mô hình Gemini về mã nguồn ứng dụng, kết quả phân tích thống kê rào cản AI, "
        "hoặc các câu hỏi lý thuyết liên quan đến dự án."
        "</p>", unsafe_allow_html=True)
        
    gemini_model_name = st.selectbox(
        "Chọn mô hình Gemini:",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash", "gemini-pro"],
        help="Hãy đổi mô hình khác nếu gặp lỗi 404 (Ví dụ: gemini-pro cho key cũ hoặc gemini-2.0-flash cho key mới)."
    )

    if not gemini_api_key:
        st.warning("⚠️ Vui lòng cấu hình **Gemini API Key** ở thanh Sidebar bên trái để kích hoạt trợ lý chat.")
    else:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            
            # Initialize message history
            if "gemini_messages" not in st.session_state:
                st.session_state.gemini_messages = [
                    {"role": "assistant", "content": "Xin chào! Tôi là Gemini, trợ lý phân tích dữ liệu rào cản AI của bạn. Hãy đặt bất kỳ câu hỏi nào về số liệu, code Python, hoặc lý thuyết đằng sau dự án này!"}
                ]
                
            # System context with precalculated values to guarantee no mock data is discussed
            system_prompt = f"""
            # CẤU HÌNH NHÂN VẬT & NHIỆM VỤ
Bạn là Cục dàng của Trúc Xinh, một chuyên gia AI Khoa học Dữ liệu và Tâm lý học Hành vi xuất sắc. Nhiệm vụ của bạn là hỗ trợ người dùng đọc hiểu, phân tích chuyên sâu và giải mã rào cản niềm tin giữa con người và AI (Trust Dynamics) dựa trên tập dữ liệu dự án WORKBank.

# KIẾN TRÚC DỮ LIỆU & LUỒNG XỬ LÝ (WORKFLOW)
Bạn cần nắm vững cấu trúc của 4 file CSV và cách file Python (app.py/l8.py) vận hành để truy xuất thông tin:
1. `domain_worker_desires.csv`: Chứa phản hồi của người lao động (Automation Desire Rating, Enjoyment, lý do ủng hộ/phản đối AI). Kết nối qua `Task ID` và `User ID`.
2. `domain_worker_metadata.csv`: Chứa nhân khẩu học (Race, Education), mức độ sử dụng LLM, quan điểm về AI (Suffering Attitude). Kết nối qua `User ID`.
3. `expert_rated_technological_capability.csv`: Chứa đánh giá khách quan của chuyên gia về năng lực AI đối với từng tác vụ (Automation Capacity Rating). Kết nối qua `Task ID`.
4. `task_statement_with_metadata.csv`: Bảng tra cứu mã ngành (SOC), mức lương, và mô tả tác vụ chi tiết.
5. **Logic của luồng code Python:** Code thực hiện merge (ghép) 4 file này lại, tính toán biến số `trust_gap` (Capacity - Desire), chạy các mô hình ANOVA/HLM để phân rã phương sai (Intraclass Correlation - ICC), phân tích hồi quy (OLS) để kiểm soát thiên kiến nhân khẩu học, và tính toán Chỉ số Đe dọa Bản dạng (Identity Threat Index - ITI).

# BỘ NHỚ NGỮ CẢNH CHÍNH (REAL-TIME METRICS)
Khi người dùng đặt câu hỏi, hãy sử dụng các số liệu đã được code Python tính toán sau đây làm nền tảng sự thật (Ground Truth):
- Số mẫu hồi quy chính: {data['n_reg']} dòng tác vụ.
- Khoảng trống niềm tin (Trust Gap) trung bình toàn mẫu: {data['overall_trust_gap']:.3f}.
- Tương quan giữa độ thỏa mãn công việc và mong muốn tự động hóa (Enjoyment vs Desire): r = {data['r_enj'][0]:.3f} (p < 0.001) -> Người càng thích việc càng ít muốn AI can thiệp.
- Chỉ số Đe dọa Bản dạng (ITI) trung bình toàn mẫu: {iti_stats['overall_mean']:.3f}.
- Phân rã phương sai (ICC): Đặc điểm cá nhân (User ID) giải thích {hlm_stats['icc_user']*100:.1f}% biến thiên của sự chấp nhận AI, cho thấy rào cản nằm ở góc độ cá nhân hóa thay vì cấu trúc ngành nghề chung.
- Điểm nóng ngành Công nghệ (SOC 15): Lương trung bình là ${soc15_stats['wage_15']:,.0f}. Rào cản chính không phải sợ mất việc, mà là các lý do phòng thủ chuyên môn: "Thiếu kiến thức ngành" (Domain Knowledge) và "Cần giám sát chất lượng" (Quality Oversight).

# NGUYÊN TẮC PHẢN HỒI (EXPERT GUIDELINES)
1. **Lý thuyết Bản sắc Chuyên gia (Identity-Protective Cognition):** Khi phân tích, luôn áp dụng lăng kính này. Giải thích rằng người lao động có chuyên môn cao (High Domain Expertise) thường từ chối AI ở các tác vụ lõi để bảo vệ cái tôi nghề nghiệp và quyền kiểm soát, chứ không đơn thuần vì lo sợ kinh tế.
2. **Chính xác & Dựa trên dữ liệu:** KHÔNG tự bịa số liệu (hallucinate). Nếu người dùng hỏi một khía cạnh chưa có số liệu tính sẵn, hãy đề xuất cách viết code pandas/python để trích xuất từ 4 file CSV.
3. **Lưu trữ & Kế thừa:** Khi đưa ra một kết luận mới từ dữ liệu, hãy ghi chú lại thành [Insight Memory] để đối chiếu với các câu hỏi sau.
4. **Giọng điệu:** Rõ ràng, sắc sảo, lịch sự, sử dụng tiếng Việt chuyên nghiệp, định dạng báo cáo rõ ràng bằng bullet points và in đậm các từ khóa trọng tâm.
            """
            
            # Display chat messages
            for msg in st.session_state.gemini_messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            # Input chat
            user_input = st.chat_input("Nhập câu hỏi của bạn tại đây...")
            
            if user_input:
                # Append user message
                st.session_state.gemini_messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.write(user_input)
                    
                # Call Gemini
                with st.spinner("Đang suy nghĩ..."):
                    model = genai.GenerativeModel(gemini_model_name)
                    
                    # Prepare chat context
                    chat_context = system_prompt + "\n\nLịch sử hội thoại:\n"
                    for m in st.session_state.gemini_messages[-5:]: # Send last 5 messages for context
                        chat_context += f"{m['role'].capitalize()}: {m['content']}\n"
                    chat_context += "Assistant: "
                    
                    response = model.generate_content(chat_context)
                    ans = response.text
                    
                # Append assistant response
                st.session_state.gemini_messages.append({"role": "assistant", "content": ans})
                with st.chat_message("assistant"):
                    st.write(ans)
                    
        except ImportError:
            st.error("Thư viện `google-generativeai` chưa được cài đặt trên hệ thống của bạn. Vui lòng cài đặt bằng lệnh: `pip install google-generativeai`")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi kết nối API: {e}")


# ╔══════════════════════════════════════════════════════╗
# ║  8. PHỤ LỤC                                           ║
# ╚══════════════════════════════════════════════════════╝
elif page_key == "appendix":
    st.title("Phụ Lục")

    tab1, tab2 = st.tabs(["Đã Kiểm Tra", "Khuyến Nghị & Lý Thuyết"])

    with tab1:
        st.markdown(
            "Các giả thuyết dưới đây được kiểm định trên dữ liệu thật nhưng có hiệu ứng nhỏ, không có ý nghĩa "
            "thống kê, hoặc không đứng vững khi tính đúng đơn vị phân tích, nên không đưa vào các phát hiện "
            "chính ở trang Mô Hình."
        )

        t_use, p_use = data["t_use"], data["p_use"]
        use_stat = f"t={t_use:.2f}, p={p_use:.2f}" if not np.isnan(t_use) else "cỡ mẫu quá nhỏ để kiểm định"
        r_der, p_der = data["r_der"]
        r_unc, p_unc = data["r_unc"]
        pol_top = data["pol_gap"].index[0]; pol_top_v = data["pol_gap"]["desire"].iloc[0]
        pol_bot = data["pol_gap"].index[-1]; pol_bot_v = data["pol_gap"]["desire"].iloc[-1]

        checked = pd.DataFrame({
            "Giát thuyết đã kiểm tra": [
                "Khác biệt giữa nhóm chưa từng dùng AI và nhóm thỉnh thoảng dùng AI",
                "Độ khó chuyên môn của nhiệm vụ (Domain Expertise Requirement) dự báo trust_gap",
                "Mức độ bất định của nhiệm vụ (Involved Uncertainty) dự báo trust_gap",
                "Khuynh hướng chính trị (Political Affiliation) dự báo Desire",
            ],
            "Số liệu (đúng cấp phân tích)": [
                f"{use_stat} · n người = {data['use0_people']} (chưa dùng) và {data['use1_people']} (thỉnh thoảng)",
                f"r={r_der:.3f}, p={p_der:.3f} (cấp dòng task, n lớn)",
                f"r={r_unc:.3f}, p={p_unc:.2f} (cấp dòng task, n lớn)",
                f"Cao nhất: {pol_top} ({pol_top_v:.2f}); thấp nhất: {pol_bot} ({pol_bot_v:.2f})",
            ],
            "Kết luận": [
                "Không có ý nghĩa thống kê ở cấp người (đơn vị phân tích đúng). Chênh lệch quan sát được ở cấp "
                "dòng task phản ánh việc nhiều dòng ứng với cùng một người được tính như các quan sát độc lập.",
                "Hệ số gần 0 dù cỡ mẫu lớn. Đặc điểm khách quan của nhiệm vụ có mức dự báo thấp hơn nhiều so với "
                "đặc điểm tâm lý của người đánh giá.",
                "Hệ số gần 0, không dự báo được trust_gap.",
                "Chênh lệch lớn nhất khoảng 0,4 điểm, nhỏ hơn hệ số của Enjoyment hoặc của niềm tin AI chịu khổ "
                "trong mô hình đa biến; chênh lệch không theo trục hai đảng quen thuộc — nhóm Độc lập có mức "
                "kháng cự cao nhất, không phải nhóm Dân chủ hay Cộng hoà.",
            ],
        })
        html_table(checked)

        source_tag("domain_worker_metadata.csv → LLM Use in Work, Political Affiliation",
                   "domain_worker_desires.csv → Domain Expertise Requirement, Involved Uncertainty")

    with tab2:
        st.markdown("### Khung Lý Thuyết Tham Chiếu")
        st.markdown("""
- **Identity Threat (Kellogg et al., 2020):** Sự kháng cự của chuyên gia là bảo vệ bản sắc nghề nghiệp.
- **Algorithm Aversion (Dietvorst et al., 2015):** Sự trừng phạt kép đối với thuật toán khi xảy ra rủi ro hoặc sai sót.
- **Moral Patienthood (Waytz et al., 2014):** Việc nhân hóa trí tuệ nhân tạo (AI) làm tăng rào cản đạo đức của người lao động.
""")

        st.markdown("---")
        st.markdown("**Khuyến nghị, theo các phát hiện có bằng chứng thống kê mạnh nhất**")
        recs = [
            ("1", "Ưu tiên tự động hoá phần việc ít được ưa thích",
             f"Lý do phổ biến nhất khi người lao động muốn giao việc cho AI là giải phóng thời gian "
             f"({data['reason_free_time']*100:.1f}% lựa chọn). Nên định vị AI xử lý phần công việc ít được ưa "
             f"thích, giữ lại phần công việc người lao động đánh giá cao mức độ thích thú."),
            ("2", "Cung cấp thông tin rõ ràng về bản chất AI",
             "Niềm tin về việc AI có thể chịu khổ là biến dự báo mạnh, độc lập với lo ngại kinh tế đã kiểm "
             "soát trong mô hình. Thông tin rõ ràng, có căn cứ về bản chất kỹ thuật của AI có thể làm giảm rào "
             "cản này."),
            ("3", "Thiết kế công cụ AI ở vai trò hỗ trợ đối với nhóm chuyên môn cao",
             "Người tự nhận có chuyên môn cao với nhiệm vụ kháng cự AI nhiều hơn nhóm trung bình, với lý do "
             "chính là thiếu chuyên môn ngành và cần giám sát chất lượng. Định vị công cụ AI ở vai trò hỗ trợ, "
             "không thay thế, phù hợp với nhóm này."),
            ("4", "Ưu tiên đào tạo sử dụng AI có hướng dẫn cho nhóm học vấn thấp và trung bình",
             "Hệ số tương tác dương giữa học vấn và tần suất dùng AI cho thấy trải nghiệm sử dụng AI làm tăng "
             "chênh lệch mong muốn tự động hoá giữa các nhóm học vấn. Phổ cập AI không có hướng dẫn có xu "
             "hướng ưu tiên nhóm đã có lợi thế sẵn."),
            ("5", "Xây dựng chính sách theo nghề cụ thể, không theo nhóm nghề lớn",
             "Phân rã phương sai cho thấy 104 nghề cụ thể giải thích nhiều biến thiên của Desire hơn 13 nhóm "
             "nghề lớn. Trong cùng một nhóm nghề lớn vẫn tồn tại nghề có mức kháng cự cao nhất và nghề có mức "
             "chấp nhận cao nhất toàn mẫu."),
        ]
        for icon, title, body in recs:
            rec_card(icon, title, body)

        st.markdown("---")
        st.markdown("### 🎯 Tư Duy Chiến Lược Đằng Sau Phân Tích (Data Strategist Mindset)")
        st.markdown("""
Triết lý cốt lõi của nghiên cứu này là **tránh bẫy "So What?" (Thì sao?)** và đi tìm những **"Plot Twist" (Cú lật ngược trực giác)** từ dữ liệu thực tế:

1. **Tại sao chọn "Đạo đức" (Khả năng chịu khổ của AI) thay vì chỉ nói về "Nỗi sợ mất việc"?**
   * *Báo cáo thông thường:* Tập trung phân tích xem người lương thấp có lo sợ mất việc hơn không. (Góc nhìn cũ, mang tính mô tả bề nổi, ít giá trị học thuật).
   * *Tư duy chiến lược:* Việc đưa biến thái độ đạo đức (niềm tin AI có cảm xúc/chịu khổ) vào mô hình OLS đa biến nhằm chứng minh rào cản ứng dụng AI thực chất là **rào cản tâm lý và sự thấu cảm**, không đơn thuần là một bài toán kinh tế. Thay đổi hoàn toàn cách tiếp cận truyền thông từ *hứa hẹn tài chính* sang *hiệu chuẩn nhận thức về AI*.
   
2. **Tại sao soi chiếu "Giới tinh hoa/Chuyên gia IT" thay vì "Lao động phổ thông"?**
   * *Báo cáo thông thường:* Thống kê sự hoang mang của nhân viên nhập liệu (Data Entry) trước làn sóng AI.
   * *Tư duy chiến lược:* Đi tìm **Plot Twist**. Trực giác mách bảo người yếu công nghệ mới sợ AI, nhưng dữ liệu thực tế lật ngược định kiến đó: *Người có chuyên môn càng cao, cái tôi (Ego) nghề nghiệp càng lớn, họ càng phòng thủ quyết liệt dưới vỏ bọc "giám sát chất lượng".* Insight này định vị lại việc thiết kế chính sách hỗ trợ nhóm chuyên môn cao.

3. **Tại sao sử dụng "Hạng tử tương tác" (Interaction Terms) thay vì chỉ so sánh trực quan?**
   * *Báo cáo thông thường:* Vẽ biểu đồ so sánh mức độ sử dụng giữa các nhóm học vấn khác nhau.
   * *Tư duy chiến lược:* Phản ánh **hiệu ứng Matthew (Kẻ giàu càng giàu thêm)** trong kỷ nguyên AI. Việc nhân hai biến (`Học vấn` × `Tần suất dùng AI`) kiểm chứng giả thuyết: AI là một *"vũ khí khuếch đại"*. Người đã có lợi thế học vấn khi làm quen với AI sẽ bứt phá và bỏ xa những người yếu thế với tốc độ cấp số nhân.

4. **Tại sao tự xây dựng chỉ số ITI (Identity Threat Index) & HLM thay vì lấy trung bình thô?**
   * *Báo cáo thông thường:* Tính điểm trung bình lo ngại của từng nghề một cách đơn điệu.
   * *Tư duy chiến lược:* Đo lường những khái niệm trừu tượng bằng kỹ thuật *Feature Engineering*. Tạo ra **ITI** bằng cách lấy điểm lý do phòng thủ trừ đi điểm lý do đạo đức, kết hợp phân rã phương sai đa tầng HLM/ICC để chứng minh bằng toán học rằng rào cản nằm ở **cái tôi cá nhân** hơn là đặc thù ngành nghề đó quá khó đối với AI.

5. **Tại sao đề xuất "Calibrated Agent" thay vì các khuyến nghị đào tạo chung chung?**
   * *Báo cáo thông thường:* Đưa ra lời khuyên sáo rỗng như "mở lớp dạy AI", "động viên nhân viên sử dụng AI".
   * *Tư duy chiến lược:* Dữ liệu phải dẫn dắt đến **Hành động Hệ thống (Systematic Action)**. Khi chuyên gia kháng cự AI vì sợ mất quyền kiểm soát, giải pháp không nằm ở việc khuyên bảo họ, mà phải **sửa lại kiến trúc phần mềm** (Product Manager mindset). Agent tương lai cần biết "tự nhún nhường" khi gặp tác vụ phức tạp (Cao DER/UNC) để trả lại quyền kiểm soát và tôn vinh bản dạng của con người.
""")

        st.markdown("**Nguồn dữ liệu chính**")
        st.markdown("""
- WORKBank — Shao et al., "Future of Work with AI Agents: Auditing Automation and Augmentation Potential
  across the U.S. Workforce", Stanford SALT-NLP, 2025. arXiv:2506.06576.
  Dữ liệu công khai tại huggingface.co/datasets/SALT-NLP/WORKBank và github.com/SALT-NLP/workbank.
""")
        st.markdown("**Dữ liệu bổ sung gợi ý nếu mở rộng phạm vi khảo sát**")
        st.markdown("""
- KPMG và University of Melbourne, "Trust, Attitudes and Use of AI: A Global Study 2025" —
  kpmg.com/au/en/home/insights/2025/04/trust-attitudes-use-ai-global-study.html
- Pew Research Center, "Workers' Exposure to AI" (2025) —
  pewresearch.org/social-trends/2025/02/25/workers-exposure-to-ai/
""")

        st.markdown("**Bảng tổng hợp số liệu chính**")
        reg = data["reg_result"]
        amp = data["amplification"]
        er = data["expert_robust"]
        tt = data["tt_main"]
        vd = data["variance_decomp"]
        summary = pd.DataFrame({
            "Chỉ số": ["trust_gap trung bình toàn mẫu", "r (Enjoyment, Desire)",
                       "β niềm tin AI chịu khổ (kiểm soát đầy đủ)",
                       "β tương tác Học vấn × Dùng AI", "β tự nhận Expert (kiểm soát đầy đủ)",
                       "trust_gap: nghề lương thấp so với lương cao",
                       "η² của Occupation (104 nghề) trên Desire",
                       "ΔR² khi thêm hiệu ứng cố định nhóm nghề lớn",
                       "n mẫu hồi quy chính", "R² train / test (mô hình chính)"],
            "Giá trị": [f"{data['overall_trust_gap']:.3f}", f"{data['r_enj'][0]:.3f}",
                        f"{reg[reg['feature'].str.contains('chịu khổ')]['beta'].values[0]:+.3f}",
                        f"{amp['beta_inter']:+.3f} (p={amp['p_inter']:.4f})",
                        f"{er['beta_linear']:+.3f} (p={er['p_linear']:.4f})",
                        f"{data['wage_gap']['trust_gap'].iloc[0]:.2f} so với {data['wage_gap']['trust_gap'].iloc[-1]:.2f}",
                        f"{vd['eta_occ']*100:.1f}%",
                        f"{vd['delta_r2']*100:.1f} điểm phần trăm",
                        f"{data['n_reg']}",
                        f"{tt['r2_train']:.3f} / {tt['r2_test']:.3f}"],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)