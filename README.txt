Vendor Finder (Frequency → Recency → Price)
==========================================

Files:
  - app.py
  - secrets_auth.yaml
  - requirements.txt

Quick Start (Windows / PowerShell):
  python -m venv .venv
  . .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
  streamlit run app.py

Logins:
  - admin / Admin123!  (Admin: upload, export)
  - user1 / User123!   (User: search-only)
