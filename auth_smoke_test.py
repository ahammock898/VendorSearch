
# auth_smoke_test.py — Minimal login test to confirm authenticator works
import streamlit as st
import streamlit_authenticator as stauth
import yaml as _yaml

st.set_page_config(page_title="Auth Smoke Test")

st.write("Auth Smoke Test — ensure secrets_auth.yaml is in the same folder as this file.")

with open("secrets_auth.yaml", "r") as f:
    config = _yaml.safe_load(f)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

authenticator.login(location="sidebar", fields={"Form name": "Login"}, key="auth_form")
auth_status = st.session_state.get("authentication_status", None)
name = st.session_state.get("name", None)
username = st.session_state.get("username", None)

st.write(f"authentication_status={auth_status!r}, name={name!r}, username={username!r}")
if auth_status:
    st.success(f"Logged in as {name} ({username})")
else:
    st.info("Not logged in yet.")
