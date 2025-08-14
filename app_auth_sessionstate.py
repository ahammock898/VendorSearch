
# app_auth_sessionstate.py — Vendor Finder with session_state-based auth handling
# Uses streamlit_authenticator's session keys to avoid unpack/None issues.

import io
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import yaml as _yaml

st.set_page_config(page_title="Vendor Finder (Freq → Recent → Price)", layout="wide")

WEIGHTS = dict(freq=5.0, recency=3.0, price=1.0)

COLUMN_MAP = {
    "product_id": ["Product ID","Eclipse Product ID","Item Number","Item ID","Product","SKU"],
    "vendor_name": ["Vendor Name","Vendor","Supplier Name"],
    "vendor_id": ["Vendor ID","VendorID","Vendor Id","Supplier ID"],
    "description": ["Description","Item Description","Product Description"],
    "unit_price": ["Unit Price","Order Price (Unit)","Unit Cost","UnitPrice","Price","Cost"],
    "order_date": ["Order Date","PO Date","Date","Invoice Date","OrderDate"],
    "priority": ["Priority"],
    "keywords": ["Keywords"]
}

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for c in candidates:
        for col in df.columns:
            if c.lower() in col.lower():
                return col
    return None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {std: pick_col(df, alts) for std, alts in COLUMN_MAP.items()}
    rename = {mapping[k]: k for k in mapping if mapping[k] is not None}
    out = df.rename(columns=rename).copy()
    if "order_date" in out: out["order_date"] = pd.to_datetime(out["order_date"], errors="coerce")
    if "unit_price" in out: out["unit_price"] = pd.to_numeric(out["unit_price"], errors="coerce")
    if "product_id" in out: out["product_id"] = out["product_id"].astype(str).str.strip()
    if "vendor_name" in out: out["vendor_name"] = out["vendor_name"].astype(str).str.strip()
    if "keywords" in out: out["keywords"] = out["keywords"].fillna("").astype(str)
    return out

def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        ws = writer.sheets[sheet_name]
        nrows, ncols = df.shape
        try:
            ws.add_table(0, 0, max(nrows,1), max(ncols-1,0), {
                "name": "Results",
                "columns": [{"header": c} for c in df.columns],
                "style": "Table Style Light 15"
            })
        except Exception:
            pass
        for idx, col in enumerate(df.columns):
            series = df[col].astype(str)
            max_len = max(series.map(len).max(), len(col))
            width = min(80, max(10, int(max_len * 1.1)))
            ws.set_column(idx, idx, width)
    return output.getvalue()

def normalize_series(s: pd.Series) -> pd.Series:
    if s.notna().any():
        smin, smax = s.min(), s.max()
        rng = (smax - smin) if (smax - smin) not in [0, np.nan] else 1.0
        return (s - smin) / rng
    return pd.Series(0.0, index=s.index)

# -------- Auth --------
try:
    import os
    here = os.path.dirname(__file__)
    secrets_path = os.path.join(here, "secrets_auth.yaml")
    with open(secrets_path, "r") as f:
        config = _yaml.safe_load(f)
except FileNotFoundError:
    st.error("Missing credentials. Place 'secrets_auth.yaml' in the same folder as this app.")
    st.stop()

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

# Render login and rely on session_state for status
authenticator.login(location="sidebar", fields={"Form name": "Login", "Username": "Username", "Password": "Password", "Login": "Login"}, key="auth_form")

auth_status = st.session_state.get("authentication_status", None)
username = st.session_state.get("username", None)
name = st.session_state.get("name", None)

if auth_status is False:
    st.sidebar.error("Username or password is incorrect.")
    st.stop()
elif auth_status is None:
    st.sidebar.info("Please log in.")
    st.stop()

st.sidebar.success(f"✅ Logged in as {name} ({username})")
st.sidebar.markdown("---")
authenticator.logout("Logout", "sidebar")

ROLE = config["credentials"]["usernames"].get(username, {}).get("role", "user")
st.sidebar.caption(f"Role: `{ROLE}`")

# -------- App UI --------
st.title("Vendor Finder")
st.caption("Ranking = Frequency → Most Recent PO → Cheapest Price.")

if ROLE == "admin":
    st.subheader("Data Upload (Admin)")
    uploads = st.file_uploader("Upload one or more Excel files (.xlsx)", type=["xlsx"], accept_multiple_files=True, key="uploads_key")
    dfs = []
    if uploads:
        for up in uploads:
            try:
                df_in = pd.read_excel(up)
                dfs.append(df_in)
            except Exception as e:
                st.warning(f"Failed to read {up.name}: {e}")
    if not dfs:
        st.info("👉 Upload vendor files to proceed. The page stays active after login.")
        st.stop()
else:
    st.warning("This account cannot upload. Please login as admin to upload data.", icon="⚠️")
    st.stop()

raw = pd.concat(dfs, ignore_index=True)
df = normalize_columns(raw)

needed = ["product_id","vendor_name","unit_price","order_date"]
missing = [c for c in needed if c not in df.columns]
if missing:
    st.error(f"Missing required columns after normalization: {missing}")
    st.stop()

with st.expander("Preview (first 25 rows after normalization)"):
    st.dataframe(df.head(25), use_container_width=True)

# Aggregate per product/vendor
agg = (
    df.groupby(["product_id","vendor_name"], dropna=False)
      .agg(
          last_order_date=("order_date","max"),
          min_price=("unit_price","min"),
          avg_price=("unit_price","mean"),
          orders=("unit_price","size"),
          vendor_id=("vendor_id","first"),
          sample_description=("description","first"),
          keywords=("keywords", lambda s: ", ".join(dict.fromkeys([x for v in s.fillna('').astype(str).tolist() for x in v.split(', ') if x])))
      )
      .reset_index()
)

# Scoring
agg["freq_score"] = normalize_series(agg["orders"])
if agg["last_order_date"].notna().any():
    max_d = agg["last_order_date"].max()
    min_d = agg["last_order_date"].min()
    span_days = max((max_d - min_d).days, 1)
    agg["recency_score"] = agg["last_order_date"].apply(lambda d: 1.0 - max(0, (max_d - d).days)/span_days if pd.notna(d) else 0.0)
else:
    agg["recency_score"] = 0.0

if agg["avg_price"].notna().any():
    pmin = agg["avg_price"].min()
    pmax = agg["avg_price"].max()
    prange = max(pmax - pmin, 1e-9)
    agg["price_score"] = 1.0 - (agg["avg_price"] - pmin) / prange
else:
    agg["price_score"] = 0.0

agg["combined_score"] = (
    WEIGHTS["freq"]   * agg["freq_score"] +
    WEIGHTS["recency"]* agg["recency_score"] +
    WEIGHTS["price"]  * agg["price_score"]
)

agg = agg.sort_values(["product_id","combined_score","orders","last_order_date","avg_price"],
                      ascending=[True, False, False, False, True])
agg["rank"] = agg.groupby("product_id").cumcount() + 1

# Search
st.subheader("Search")
c1, c2, c3 = st.columns([1,1,2])
q_product = c1.text_input("Product ID (exact match or contains)")
q_vendor  = c2.text_input("Vendor Name (contains)")
q_keywords = c3.text_input("Keywords (comma separated; matches any)")

def filter_results(agg: pd.DataFrame, pid: str, vname: str, keywords: str) -> pd.DataFrame:
    view = agg.copy()
    if pid:
        pid = pid.strip()
        view = view[view["product_id"].str.contains(pid, case=False, na=False)]
    if vname:
        vname = vname.strip()
        view = view[view["vendor_name"].str.contains(vname, case=False, na=False)]
    if keywords:
        toks = [t.strip().lower() for t in keywords.split(",") if t.strip()]
        if toks:
            mask = view["keywords"].fillna("").str.lower().apply(lambda s: any(tok in s for tok in toks))
            view = view[mask]
    view = view.sort_values(["product_id","combined_score","orders","last_order_date","avg_price"],
                            ascending=[True, False, False, False, True])
    view["rank"] = view.groupby("product_id").cumcount() + 1
    return view

results = filter_results(agg, q_product, q_vendor, q_keywords)

if results.empty and (q_product or q_vendor or q_keywords):
    st.info("No matches found. Try broadening your search or removing a filter.")
elif results.empty:
    st.caption("Enter a Product ID, Vendor name, and/or Keywords to get ranked vendors.")
else:
    st.subheader("Ranked Vendors (Freq → Recent → Cheapest)")
    show_cols = ["product_id","vendor_name","rank","orders","last_order_date","avg_price","min_price","vendor_id","sample_description","keywords","freq_score","recency_score","price_score","combined_score"]
    st.dataframe(results[show_cols], use_container_width=True)

# Export
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "⬇️ Download All Aggregated Vendors (Excel)",
        data=to_excel_bytes(agg, "vendors_all"),
        file_name="vendors_all.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
with c2:
    if not results.empty:
        st.download_button(
            "⬇️ Download Current Search Results (Excel)",
            data=to_excel_bytes(results, "search_results"),
            file_name="vendor_search_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.button("⬇️ Download Current Search Results (Excel)", disabled=True)
