# -*- coding: utf-8 -*-
# ============================================================
# CHƯƠNG TRÌNH THU THẬP VÀ TỔNG HỢP THÔNG TIN KINH TẾ VĨ MÔ
# CỦA VIỆT NAM TRÊN THỊ TRƯỜNG TÀI CHÍNH (VERSION 9 - STABLE)
# ============================================================

import os
import io
import math
import time
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# (Luôn bật) Gemini cho "AI phân tích và tư vấn"
GEMINI_OK = False
try:
    import google.generativeai as genai
    GEMINI_OK = True
except Exception:
    GEMINI_OK = False

# ---------------------------
# Tham số & Từ điển hiển thị
# ---------------------------
WB_API_BASE = "https://api.worldbank.org/v2"
UNDP_API_BASE = "https://hdr.undp.org/sites/default/files/api/"
HDI_PSEUDO_CODE = "UNDP.HDI"

# Danh mục mặc định (code, EN name, unit)
DEFAULT_INDICATORS = [
    ("NY.GDP.MKTP.KD.ZG", "GDP growth (annual %)", "%"),
    ("FP.CPI.TOTL.ZG", "Inflation, CPI (annual %)", "%"),
    ("SL.UEM.TOTL.ZS", "Unemployment (% labor force)", "%"),
    ("NE.EXP.GNFS.ZS", "Exports of goods & services (% of GDP)", "%"),
    ("NE.IMP.GNFS.ZS", "Imports of goods & services (% of GDP)", "%"),
    ("GC.DOD.TOTL.GD.ZS", "Central government debt (% of GDP)", "%"),
    ("BX.KLT.DINV.WD.GD.ZS", "FDI, net inflows (% of GDP)", "%"),
    ("SP.POP.TOTL", "Population (total)", "persons"),
    ("NY.GDP.PCAP.CD", "GDP per capita (current US$)", "USD"),
]

# Chỉ số mở rộng (SBV/IMF/GSO)
EXTENDED_INDICATORS = [
    ("FR.INR.LEND", "Lãi suất cho vay (%)", "%", "WB proxy (IMF/GSO)"),
    ("FR.INR.DPST", "Lãi suất tiền gửi (%)", "%", "WB proxy (IMF/GSO)"),
    ("PA.NUS.FCRF", "Tỷ giá chính thức (LCU/USD)", "LCU/USD", "WB proxy (SBV)"),
    ("SBV.POLICY.RATE", "Lãi suất điều hành (SBV) (%)", "%", "SBV (placeholder)"),
]

VN_NAME_MAP = {
    "NY.GDP.MKTP.KD.ZG": ("Tăng trưởng GDP (năm)", "%"),
    "FP.CPI.TOTL.ZG": ("Lạm phát CPI (năm)", "%"),
    "SL.UEM.TOTL.ZS": ("Tỷ lệ thất nghiệp", "%"),
    "NE.EXP.GNFS.ZS": ("Xuất khẩu hàng hóa & dịch vụ", "% GDP"),
    "NE.IMP.GNFS.ZS": ("Nhập khẩu hàng hóa & dịch vụ", "% GDP"),
    "GC.DOD.TOTL.GD.ZS": ("Nợ chính phủ", "% GDP"),
    "BX.KLT.DINV.WD.GD.ZS": ("FDI, dòng vốn ròng", "% GDP"),
    "SP.POP.TOTL": ("Dân số", "người"),
    "NY.GDP.PCAP.CD": ("GDP bình quân đầu người", "USD"),
    HDI_PSEUDO_CODE: ("Chỉ số phát triển con người (HDI)", ""),
    "FR.INR.LEND": ("Lãi suất cho vay", "%"),
    "FR.INR.DPST": ("Lãi suất tiền gửi", "%"),
    "PA.NUS.FCRF": ("Tỷ giá chính thức (LCU/USD)", "LCU/USD"),
    "SBV.POLICY.RATE": ("Lãi suất điều hành (SBV)", "%"),
}

AGRIBANK_RGB = "rgb(174,28,63)"   # #AE1C3F

def get_vn_label_with_unit(code: str) -> str:
    vn_name, vn_unit = VN_NAME_MAP.get(code, (None, None))
    if vn_name is None:
        en_unit, en_name = "", code
        for c, name, unit in DEFAULT_INDICATORS + [(x[0], x[1], x[2]) for x in EXTENDED_INDICATORS]:
            if c == code:
                en_name = name
                en_unit = unit
                break
        return f"{en_name} ({en_unit})" if en_unit else en_name
    return f"{vn_name} ({vn_unit})" if vn_unit else vn_name

def is_percent_unit(code: str) -> bool:
    _, unit = VN_NAME_MAP.get(code, (None, None))
    if unit is None:
        for c, _, u in DEFAULT_INDICATORS + [(x[0], x[1], x[2]) for x in EXTENDED_INDICATORS]:
            if c == code:
                unit = u
                break
    unit = (unit or "").lower()
    return "%" in unit

# ---------------------------
# Tối ưu: cache & song song
# ---------------------------
@st.cache_data(show_spinner=False, ttl=60*60)
def list_wb_countries():
    url = f"{WB_API_BASE}/country?format=json&per_page=400"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for item in data[1]:
            if item.get("region", {}).get("id") != "Aggregates":
                rows.append({"id": item["id"], "name": item["name"]})
        df = pd.DataFrame(rows).sort_values("name")
        return df
    except Exception:
        # Fallback nếu API lỗi
        return pd.DataFrame([{"id": "VNM", "name": "Vietnam"}])

@st.cache_data(show_spinner=False, ttl=60*60)
def _fetch_wb_indicator(country_code: str, indicator_code: str, start_year: int, end_year: int) -> pd.DataFrame:
    url = f"{WB_API_BASE}/country/{country_code}/indicator/{indicator_code}?date={start_year}:{end_year}&format=json&per_page=12000"
    max_retries = 3
    for i in range(max_retries):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            js = r.json()
            if not isinstance(js, list) or len(js) < 2 or js[1] is None:
                return pd.DataFrame(columns=["Year", indicator_code])
            rows = []
            for rec in js[1]:
                y = rec.get("date")
                val = rec.get("value")
                try:
                    y = int(y)
                except:
                    continue
                rows.append({"Year": y, indicator_code: val})
            return pd.DataFrame(rows).sort_values("Year")
        except requests.exceptions.RequestException:
            if i == max_retries - 1:
                return pd.DataFrame(columns=["Year", indicator_code])
            time.sleep(2)
    return pd.DataFrame(columns=["Year", indicator_code])

@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_wb_indicators_parallel(country_code: str, indicator_codes: list, start_year: int, end_year: int) -> dict:
    out = {}
    with ThreadPoolExecutor(max_workers=min(8, len(indicator_codes) or 1)) as ex:
        futures = {ex.submit(_fetch_wb_indicator, country_code, code, start_year, end_year): code for code in indicator_codes}
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                out[code] = fut.result()
            except Exception:
                out[code] = pd.DataFrame(columns=["Year", code])
    return out

@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_undp_hdi(country_iso3: str, start_year: int, end_year: int) -> pd.DataFrame:
    try:
        url = f"{UNDP_API_BASE}v1/indicators/137506?countries={country_iso3}&years={start_year}-{end_year}"
        r = requests.get(url, timeout=40)
        r.raise_for_status()
        js = r.json()
        data = js.get("data", [])
        rows = [{"Year": int(it.get("year")), HDI_PSEUDO_CODE: it.get("value")} for it in data if it.get("year") is not None]
        return pd.DataFrame(rows).sort_values("Year")
    except Exception:
        return pd.DataFrame(columns=["Year", HDI_PSEUDO_CODE])

def fetch_extended_indicator(country_code: str, code: str, start_year: int, end_year: int) -> pd.DataFrame:
    if code == "SBV.POLICY.RATE":
        return pd.DataFrame(columns=["Year", code])
    return _fetch_wb_indicator(country_code, code, start_year, end_year)

def merge_wide(dfs: list) -> pd.DataFrame:
    out = None
    for d in dfs:
        if d is None or d.empty:
            continue
        out = d if out is None else pd.merge(out, d, on="Year", how="outer")
    return (out.sort_values("Year").reset_index(drop=True)) if out is not None else pd.DataFrame(columns=["Year"])

def impute_missing(df: pd.DataFrame, method: str):
    if df.empty:
        return df, {}
    df2 = df.copy()
    report = {}
    numeric_cols = [c for c in df2.columns if c != "Year"]
    if method == "Giữ nguyên (N/A)":
        pass
    elif method == "Forward/Backward fill":
        df2[numeric_cols] = df2[numeric_cols].ffill().bfill()
    elif method == "Điền trung bình theo cột":
        for c in numeric_cols:
            df2[c] = df2[c].fillna(df2[c].mean(skipna=True))
    elif method == "Điền median theo cột":
        for c in numeric_cols:
            df2[c] = df2[c].fillna(df2[c].median(skipna=True))
    for c in numeric_cols:
        report[c] = int(df2[c].isna().sum())
    return df2, report

def _format_number_vn(val, decimals_auto=True, force_decimals=None):
    import pandas as _pd
    if _pd.isna(val):
        return ""
    try:
        v = float(val)
    except Exception:
        return str(val)
    if force_decimals is not None:
        d = force_decimals
    elif decimals_auto:
        av = abs(v)
        if av >= 1000:
            d = 0
        elif av >= 1:
            d = 2
        else:
            d = 3
    else:
        d = 2
    s = f"{v:,.{d}f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s

def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    stats = []
    numeric_cols = [c for c in df.columns if c != "Year"]
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            stats.append({"Chỉ tiêu": get_vn_label_with_unit(col), "Giá trị TB (Mean)": np.nan,
                          "Độ lệch chuẩn (Std)": np.nan, "Nhỏ nhất (Min)": np.nan, "Năm Min": None,
                          "Lớn nhất (Max)": np.nan, "Năm Max": None, "Trung vị (Median)": np.nan,
                          "Q1": np.nan, "Q3": np.nan, "Hệ số biến thiên (CV%)": np.nan})
            continue
        mean, std = s.mean(), s.std(ddof=1)
        min_val, max_val = s.min(), s.max()
        min_year = df.loc[df[col].idxmin(), "Year"] if not s.empty else None
        max_year = df.loc[df[col].idxmax(), "Year"] if not s.empty else None
        median, q1, q3 = s.median(), s.quantile(0.25), s.quantile(0.75)
        cv = (std/mean*100.0) if mean and not math.isclose(mean, 0.0, abs_tol=1e-12) else np.nan
        stats.append({"Chỉ tiêu": get_vn_label_with_unit(col), "Giá trị TB (Mean)": mean, "Độ lệch chuẩn (Std)": std,
                      "Nhỏ nhất (Min)": min_val, "Năm Min": int(min_year) if pd.notna(min_year) else None,
                      "Lớn nhất (Max)": max_val, "Năm Max": int(max_year) if pd.notna(max_year) else None,
                      "Trung vị (Median)": median, "Q1": q1, "Q3": q3, "Hệ số biến thiên (CV%)": cv})
    return pd.DataFrame(stats)

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c != "Year"]
    return df[cols].corr(method="pearson") if len(cols) >= 2 else pd.DataFrame()

def add_trendline(df: pd.DataFrame, x: str, y: str):
    sub = df[[x, y]].dropna()
    if len(sub) < 2:
        return None
    a, b = np.polyfit(sub[x], sub[y], deg=1)
    x_line = np.linspace(sub[x].min(), sub[x].max(), 100)
    y_line = a * x_line + b
    return x_line, y_line, a, b

def to_excel_bytes(df_data: pd.DataFrame, df_stats: pd.DataFrame, corr
