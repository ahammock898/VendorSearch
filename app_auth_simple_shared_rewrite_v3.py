
# app_auth_simple_shared_rewrite_v3.py
# Change: make vendor_id optional (no failures if missing). Dynamic aggregation keys.
# Admin can Search/Publish/Audit. Duplicate-header-safe normalization, FRP ranking.

import streamlit as st

# Call page_config once per session, before any other st.* call.
if not st.session_state.get("_page_configured", False):
    st.set_page_config(
        page_title="Vendor Finder — Shared Dataset (FRP)",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.session_state["_page_configured"] = True

# Place this at the very top of your layout after page_config
logo_path = "Logo.png"
st.image(logo_path, width=200)  # adjust width as needed


# --- USERS (top of file) ---
USERS = {
    "Admin": {"password": "Admin123!", "role": "admin"},
    "User1": {"password": "User123!", "role": "user"},
}

# ------------ AUTH: inline login panel + safe logout gate ------------

# Ensure session auth container exists
auth = st.session_state.get("auth")
if not isinstance(auth, dict):
    auth = {"status": None, "user": None}
    st.session_state.auth = auth

username = auth.get("user")


def _find_user(u: str):
    """Return user record by exact key, then case-insensitive fallback."""
    if u in USERS:  # exact match first
        return u, USERS[u]
    # case-insensitive fallback
    lower_map = {k.lower(): k for k in USERS.keys()}
    k = lower_map.get(u.lower())
    return (k, USERS[k]) if k else (None, None)

# If NOT logged in: render login panel and stop
if not username:
    with st.sidebar.form("login_form", clear_on_submit=False):
        st.subheader("Log in")
        u_raw = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        u = (u_raw or "").strip()
        p = (p or "").strip()
        key, user_rec = _find_user(u)
        if user_rec and p == str(user_rec.get("password", "")):
            st.session_state.auth = {"status": True, "user": key}
            st.rerun()  # IMPORTANT: use st.rerun(), not experimental
        else:
            st.sidebar.error("Invalid username or password")
    st.stop()

# If we get here, we are logged in
role = USERS.get(username, {}).get("role")

# ---- Logout (inline — not a callback) ----
if st.sidebar.button("Logout", key="logout_btn"):
    st.session_state.clear()  # wipe everything, including _page_configured
    st.rerun()                # immediately re-render; login panel shows

# ------------ END AUTH GATE ------------
import io
import os
from datetime import datetime
import numpy as np
import pandas as pd

# --------------------------- App config ---------------------------

HERE = os.path.dirname(__file__)
RAW_PATH_DEFAULT = os.path.join(HERE, "shared_dataset.parquet")
AGG_PATH_DEFAULT = os.path.join(HERE, "shared_dataset_agg.parquet")

WEIGHTS = dict(freq=5.0, recency=3.0, price=1.0)

# Column mapping used for rename (expanded vendor_id aliases)
COLUMN_MAP = {
    "product_id": ["Product ID","Eclipse Product ID","Item Number","Item ID","Product","SKU","Part Number","Part #"],
    "vendor_name": ["Vendor Name","Vendor","Supplier Name","Vendor Name ","Vendor_Name"],
    "vendor_id": ["Vendor ID","VendorID","Vendor Id","Supplier ID","SupplierID","Vendor Number","Vendor #","Vendor Code"],
    "description": ["Description","Item Description","Product Description","Desc","Item_Description"],
    "unit_price": ["Unit Price","Order Price (Unit)","Unit Cost","UnitPrice","Price","Cost"],
    "order_date": ["Order Date","PO Date","Date","Invoice Date","OrderDate","Last Order Date","Ordered Date"],
    "priority": ["Priority","Rank","Priority Level"],
    "keywords": ["Keywords","Key Words","Tags"]
}

# --------------------------- Helpers ---------------------------
def _dedupe_cols(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ~frame.columns.duplicated(keep="first")]

def _first_series(frame: pd.DataFrame, col: str):
    if col not in frame.columns:
        return None
    obj = frame[col]
    if isinstance(obj, pd.DataFrame):
        for i in range(obj.shape[1]):
            s = obj.iloc[:, i]
            if s.notna().any():
                return s
        return obj.iloc[:, 0]
    return obj

def pick_col(df: pd.DataFrame, candidates: list) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for c in candidates:
        for col in df.columns:
            try:
                if c.lower() in str(col).lower():
                    return col
            except Exception:
                continue
    return None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = _dedupe_cols(df)
    mapping = {std: pick_col(df, alts) for std, alts in COLUMN_MAP.items()}
    rename = {src: std for std, src in mapping.items() if src is not None}
    out = df.rename(columns=rename).copy()

    s = _first_series(out, "order_date")
    if s is not None:
        out["order_date"] = pd.to_datetime(s, errors="coerce")

    s = _first_series(out, "unit_price")
    if s is not None:
        if s.dtype == object:
            s = s.replace({r"[,\$]": ""}, regex=True)
        out["unit_price"] = pd.to_numeric(s, errors="coerce")

    s = _first_series(out, "product_id")
    if s is not None:
        out["product_id"] = s.astype(str).str.strip()

    s = _first_series(out, "vendor_name")
    if s is not None:
        out["vendor_name"] = s.astype(str).str.strip()

    s = _first_series(out, "keywords")
    if s is not None:
        out["keywords"] = s.fillna("").astype(str)

    out = _dedupe_cols(out)
    return out

def normalize_series(s: pd.Series) -> pd.Series:
    if s.notna().any():
        smin = s.min()
        smax = s.max()
        rng = (smax - smin) if pd.notna(smax) and pd.notna(smin) and (smax != smin) else 1.0
        return (s - smin) / rng
    return pd.Series(0.0, index=s.index)

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    g = _dedupe_cols(g)

    # Force keys to be 1-D strings
    for c in ["product_id", "vendor_name"]:
        if c in g.columns:
            s = _first_series(g, c)
            if s is not None:
                g[c] = s.astype(str).str.strip()

    # Build aggregation dict dynamically so missing columns don't error
    agg_dict = {
        "last_order_date": ("order_date", "max"),
        "min_price": ("unit_price", "min"),
        "avg_price": ("unit_price", "mean"),
        "orders": ("unit_price", "size"),
    }
    if "vendor_id" in g.columns:
        agg_dict["vendor_id"] = ("vendor_id", "first")
    if "description" in g.columns:
        agg_dict["sample_description"] = ("description", "first")
    if "keywords" in g.columns:
        agg_dict["keywords"] = ("keywords", lambda s: ", ".join(
            dict.fromkeys(
                x for v in s.fillna("").astype(str).tolist()
                for x in v.split(", ") if x
            )
        ))

    grouped = g.groupby(["product_id","vendor_name"], dropna=False).agg(**agg_dict).reset_index()

    # Ensure optional columns exist even if absent (so downstream columns exist)
    for opt in ["vendor_id","sample_description","keywords"]:
        if opt not in grouped.columns:
            grouped[opt] = ""

    return grouped

def score_rank(agg: pd.DataFrame) -> pd.DataFrame:
    agg = agg.copy()
    agg["freq_score"] = normalize_series(agg["orders"])

    if agg["last_order_date"].notna().any():
        max_d = agg["last_order_date"].max()
        min_d = agg["last_order_date"].min()
        span_days = max((max_d - min_d).days, 1)
        agg["recency_score"] = agg["last_order_date"].apply(
            lambda d: 1.0 - max(0, (max_d - d).days) / span_days if pd.notna(d) else 0.0
        )
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
        WEIGHTS["freq"]   * agg["freq_score"]
        + WEIGHTS["recency"] * agg["recency_score"]
        + WEIGHTS["price"] * agg["price_score"]
    )

    agg.sort_values(
        ["product_id", "combined_score", "orders", "last_order_date", "avg_price"],
        ascending=[True, False, False, False, True],
        inplace=True
    )
    agg["rank"] = agg.groupby("product_id").cumcount() + 1
    return agg

def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        ws = writer.sheets[sheet_name]
        nrows, ncols = df.shape
        try:
            ws.add_table(0, 0, max(nrows, 1), max(ncols - 1, 0), {
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

def append_only_newer_filtered(existing: pd.DataFrame, df_new: pd.DataFrame, scope: str) -> pd.DataFrame:
    df_new = df_new.copy()
    existing = existing.copy()

    for c in ["product_id", "vendor_name"]:
        if c in df_new.columns:
            df_new[c] = df_new[c].astype(str)
        if c in existing.columns:
            existing[c] = existing[c].astype(str)

    if scope == "None":
        return df_new

    if scope == "Per product + vendor" and {"product_id","vendor_name","order_date"}.issubset(existing.columns) and {"product_id","vendor_name","order_date"}.issubset(df_new.columns):
        max_per_key = existing.groupby(["product_id", "vendor_name"], dropna=False)["order_date"].max()
        lookup = df_new[["product_id", "vendor_name"]].apply(tuple, axis=1)
        existing_max_series = lookup.map(max_per_key)
        df_new["existing_max_date"] = existing_max_series
        out = df_new[
            (df_new["existing_max_date"].isna()) |
            (pd.to_datetime(df_new["order_date"], errors="coerce") > pd.to_datetime(df_new["existing_max_date"], errors="coerce"))
        ].drop(columns=["existing_max_date"])
        return out

    if scope == "Per product" and {"product_id","order_date"}.issubset(existing.columns) and {"product_id","order_date"}.issubset(df_new.columns):
        max_per_pid = existing.groupby(["product_id"], dropna=False)["order_date"].max()
        existing_max_series = df_new["product_id"].map(max_per_pid)
        df_new["existing_max_date"] = existing_max_series
        out = df_new[
            (df_new["existing_max_date"].isna()) |
            (pd.to_datetime(df_new["order_date"], errors="coerce") > pd.to_datetime(df_new["existing_max_date"], errors="coerce"))
        ].drop(columns=["existing_max_date"])
        return out

    return df_new

# --------------------------- Sidebar dataset paths ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Shared Dataset")
raw_path = st.sidebar.text_input("Raw dataset path", value=RAW_PATH_DEFAULT)
agg_path = st.sidebar.text_input("Aggregated dataset path", value=AGG_PATH_DEFAULT)

# --------------------------- Search component (shared) ---------------------------
def render_search(agg_path: str, raw_path: str):
    st.title("Vendor Finder — Search")
    st.caption("Ranking = Frequency → Recency → Price")

    # ---------------- Load data (AGG -> RAW) ----------------
    agg = None
    loaded_from = None

    if os.path.exists(agg_path):
        try:
            agg = pd.read_parquet(agg_path)
            loaded_from = "AGG"
        except Exception as e:
            st.warning(f"Couldn't read AGG dataset; falling back to RAW. Error: {e}")

    if agg is None:
        if not os.path.exists(raw_path):
            st.error(f"Couldn't find shared dataset: {raw_path}")
            st.stop()
        try:
            df_raw = pd.read_parquet(raw_path)
            df_raw = normalize_columns(df_raw)
            df_raw = _dedupe_cols(df_raw)
            agg = score_rank(aggregate(df_raw))
            loaded_from = "RAW"
        except Exception as e:
            st.error(f"Failed to build rankings from RAW: {e}")
            st.stop()

    st.caption(f"Loaded from: **{loaded_from}**")

    # ---------------- Search inputs ----------------
    st.subheader("Search")
    c1, c2, c3 = st.columns([1, 1, 2])
    q_product = c1.text_input("Product ID (exact match or contains)")
    q_vendor = c2.text_input("Vendor Name (contains)")
    q_keywords = c3.text_input("Keywords (comma separated; matches any)")

    # ---------------- Filtering helper ----------------
    def filter_results(agg_df: pd.DataFrame, pid: str, vname: str, keywords: str) -> pd.DataFrame:
        view = agg_df.copy()
        if pid:
            pid = pid.strip()
            view = view[view["product_id"].astype(str).str.contains(pid, case=False, na=False)]
        if vname:
            vname = vname.strip()
            view = view[view["vendor_name"].astype(str).str.contains(vname, case=False, na=False)]
        if keywords:
            toks = [t.strip().lower() for t in keywords.split(",") if t.strip()]
            if toks and "keywords" in view.columns:
                mask = view["keywords"].fillna("").astype(str).str.lower().apply(
                    lambda s: any(tok in s for tok in toks)
                )
                view = view[mask]

        # Sort + rerank
        view = view.sort_values(
            ["product_id", "combined_score", "orders", "last_order_date", "avg_price"],
            ascending=[True, False, False, False, True],
        )
        view["rank"] = view.groupby("product_id").cumcount() + 1
        return view

    # ---------------- Run filter ----------------
    results = filter_results(agg, q_product, q_vendor, q_keywords)

    # ---------------- Display ----------------
    if results.empty and (q_product or q_vendor or q_keywords):
        st.info("No matches found. Try broadening your search or removing a filter.")
        return
    elif results.empty:
        st.caption("Enter a Product ID, Vendor name, and/or Keywords to get ranked vendors.")
        return

    # Format/rename/hide columns as requested
    # 1) Remove commas from vendor_id (display-only)
    if "vendor_id" in results.columns:
        results["vendor_id"] = results["vendor_id"].astype(str).str.replace(",", "", regex=False)

    # 2) Build display frame and rename columns
    #    Hide: orders, avg_price (but keep them in 'results' for ranking logic)
    pretty_map = {
        "product_id": "Product ID",
        "vendor_name": "Vendor",
        "rank": "Rank",
        "last_order_date": "Last Order Date",
        "min_price": "Min Price",
        "vendor_id": "Vendor ID",
        "sample_description": "Description",
        "keywords": "Keywords",
    }

    # Optional: add a visual marker for top rank
    results["Top"] = results["rank"].eq(1).map({True: "⭐ Top", False: ""})

    show_cols = [
        "Top",
        "product_id",
        "vendor_name",
        "rank",
        "last_order_date",
        "min_price",
        "vendor_id",
        "sample_description",
        "keywords",
    ]
    # only keep columns that actually exist
    show_cols = [c for c in show_cols if c in results.columns]

    # Prepare a view for display
    view_df = results[show_cols].rename(columns=pretty_map).copy()

    # Column formatting for nicer display
    if "Last Order Date" in view_df.columns:
        # Convert to date-only string for display
        view_df["Last Order Date"] = pd.to_datetime(view_df["Last Order Date"], errors="coerce").dt.date.astype(str)

    if "Min Price" in view_df.columns:
        # Format as currency-like string (no currency sign to keep it simple)
        view_df["Min Price"] = pd.to_numeric(view_df["Min Price"], errors="coerce").map(
            lambda x: f"{x:,.2f}" if pd.notna(x) else ""
        )

    # Show results
    st.subheader("Ranked Vendors (Freq → Recent → Cheapest)")
    st.caption("Rows with ⭐ are the top-ranked vendor for that product.")
    st.dataframe(view_df, use_container_width=True)

# --------------------------- Admin Views ---------------------------
if role == "admin":
    view = st.sidebar.radio("Admin View", ["Search", "Publish", "Audit"], index=0, horizontal=True)
    if view == "Search":
        render_search(agg_path, raw_path)

    elif view == "Publish":
        st.title("Vendor Finder — Admin Publish")
        st.caption("Upload, normalize, and publish a shared dataset.")

        uploads = st.file_uploader("Upload one or more Excel files (.xlsx)", type=["xlsx"], accept_multiple_files=True)
        dfs = []
        if uploads:
            for up in uploads:
                try:
                    df_in = pd.read_excel(up)
                    dfs.append(df_in)
                except Exception as e:
                    st.warning(f"Failed to read {up.name}: {e}")

        if dfs:
            raw = pd.concat(dfs, ignore_index=True)
            df = normalize_columns(raw)

            needed = ["product_id","vendor_name","unit_price","order_date"]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                st.error(f"Missing required columns after normalization: {missing}")
                st.stop()

            with st.expander("Preview (first 25 rows)"):
                st.dataframe(df.head(25), use_container_width=True)

            agg_preview = score_rank(aggregate(df))

            st.subheader("Publish Options")
            append_mode = st.checkbox("Append to existing RAW dataset", value=False)
            append_scope = st.selectbox(
                "Append only newer — scope",
                ["None", "Per product", "Per product + vendor"],
                index=2,
                help="When appending, only include rows with order_date newer than existing max for the selected scope."
            )
            publish_aggregated = st.checkbox("Also publish AGGREGATED dataset for instant user queries", value=True)

            c1, c2, c3 = st.columns([1,1,2])
            with c1:
                st.download_button(
                    "⬇️ Download Aggregated Preview (Excel)",
                    data=to_excel_bytes(agg_preview, "vendors_all"),
                    file_name="vendors_all_preview.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with c2:
                if st.button("📢 Publish Now"):
                    try:
                        if append_mode and os.path.exists(raw_path):
                            try:
                                existing = pd.read_parquet(raw_path)
                                existing = normalize_columns(existing)
                                existing = _dedupe_cols(existing)
                                to_append = append_only_newer_filtered(existing, df, append_scope)
                                combined = pd.concat([existing, to_append], ignore_index=True)
                                dedupe_cols = [c for c in ["product_id","vendor_name","order_date","unit_price","vendor_id"] if c in combined.columns]
                                if dedupe_cols:
                                    combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
                                combined = _dedupe_cols(combined)
                                if os.path.dirname(raw_path):
                                    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                                combined.to_parquet(raw_path, index=False)
                                st.success(f"Appended to RAW dataset: {raw_path} (rows: {len(combined)}) — added {len(to_append)} new rows.")
                            except Exception as e:
                                st.error(f"Failed to append; falling back to overwrite: {e}")
                                if os.path.dirname(raw_path):
                                    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                                df.to_parquet(raw_path, index=False)
                                st.warning(f"Overwrote RAW dataset: {raw_path}")
                        else:
                            if os.path.dirname(raw_path):
                                os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                            df.to_parquet(raw_path, index=False)
                            st.success(f"Published RAW dataset: {raw_path} (rows: {len(df)})")

                        if publish_aggregated:
                            if os.path.dirname(agg_path):
                                os.makedirs(os.path.dirname(agg_path), exist_ok=True)
                            df_current = pd.read_parquet(raw_path)
                            df_current = normalize_columns(df_current)
                            df_current = _dedupe_cols(df_current)
                            agg_full = score_rank(aggregate(df_current))
                            agg_full.to_parquet(agg_path, index=False)
                            st.success(f"Published AGG dataset: {agg_path} (rows: {len(agg_full)})")
                    except Exception as e:
                        st.error(f"Failed to publish: {e}")
            with c3:
                st.write("RAW = normalized rows (appendable).")
                st.write("AGG = precomputed rankings for fast user queries.")
                st.write("'Append only newer' uses order_date by scope.")
        else:
            st.info("Upload vendor files or a consolidated export to continue.")

    else:  # Audit
        st.title("Vendor Finder — Admin Audit")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**RAW Dataset**")
            if os.path.exists(raw_path):
                info = os.stat(raw_path)
                st.write(f"Path: `{raw_path}`")
                st.write(f"Size: {info.st_size:,} bytes")
                st.write(f"Modified: {datetime.fromtimestamp(info.st_mtime)}")
                try:
                    df_raw = pd.read_parquet(raw_path)
                    st.write(f"Rows: {len(df_raw):,}")
                    if "order_date" in df_raw.columns:
                        od = pd.to_datetime(df_raw["order_date"], errors="coerce")
                        st.write(f"Date range: {od.min()} → {od.max()}")
                    if "vendor_name" in df_raw.columns:
                        st.write("Top Vendors (by rows):")
                        st.dataframe(df_raw["vendor_name"].astype(str).value_counts().head(10).rename_axis("vendor_name").reset_index(name="rows"), use_container_width=True)
                    if "product_id" in df_raw.columns:
                        st.write("Top Products (by rows):")
                        st.dataframe(df_raw["product_id"].astype(str).value_counts().head(10).rename_axis("product_id").reset_index(name="rows"), use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to read RAW parquet: {e}")
            else:
                st.info("RAW dataset not found.")
        with cols[1]:
            st.markdown("**AGG Dataset**")
            if os.path.exists(agg_path):
                info = os.stat(agg_path)
                st.write(f"Path: `{agg_path}`")
                st.write(f"Size: {info.st_size:,} bytes")
                st.write(f"Modified: {datetime.fromtimestamp(info.st_mtime)}")
                try:
                    df_agg = pd.read_parquet(agg_path)
                    st.write(f"Rows: {len(df_agg):,}")
                    if "last_order_date" in df_agg.columns:
                        lod = pd.to_datetime(df_agg["last_order_date"], errors="coerce")
                        st.write(f"Last order dates: {lod.min()} → {lod.max()}")
                    st.write("Sample:")
                    st.dataframe(df_agg.head(20), use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to read AGG parquet: {e}")
            else:
                st.info("AGG dataset not found.")

# --------------------------- User (search-only) ---------------------------
else:
    render_search(agg_path, raw_path)
