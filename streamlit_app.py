import streamlit as st
import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import google.generativeai as genai

BASE = "https://v3.football.api-sports.io"
WINDOW_HOURS = 3

st.set_page_config(page_title="Football Signal Engine", page_icon="⚽", layout="wide")

# ---------- Helper Functions ----------
def api_get(key, path, params=None):
    r = requests.get(f"{BASE}{path}", headers={"x-apisports-key": key}, params=params or {}, timeout=30)
    r.raise_for_status()
    remaining = r.headers.get("x-ratelimit-requests-remaining")
    return r.json(), remaining

def calculate_poisson_probs(home_xg, away_xg, max_goals=6):
    """Generates outcome probabilities based on Poisson distribution."""
    matrix = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            matrix[h, a] = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
            
    p_home = float(np.sum(np.tril(matrix, -1)))
    p_draw = float(np.sum(np.diag(matrix)))
    p_away = float(np.sum(np.triu(matrix, 1)))
    
    p_o25 = float(np.sum([matrix[h, a] for h in range(max_goals) for a in range(max_goals) if h + a > 2.5]))
    p_u25 = 1.0 - p_o25
    
    return {
        "Home Win": p_home,
        "Draw": p_draw,
        "Away Win": p_away,
        "1X": p_home + p_draw,
        "X2": p_away + p_draw,
        "Over 2.5 Goals": p_o25,
        "Under 2.5 Goals": p_u25,
    }

def evaluate_with_gemini(gemini_key, match_data, picks):
    """Sends calculated statistics and top candidate options to Gemini for review."""
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = f"""
    You are an expert sports quantitative analyst. Review the statistical picks for the following football match:
    Match: {match_data['home']} vs {match_data['away']}
    League: {match_data['league']}
    
    Calculated Poisson Probabilities & Market Odds:
    {picks}
    
    Task:
    Provide a concise evaluation (3-4 bullet points maximum) explaining the strongest statistical signal, market edge, and any data risk (e.g. unconfirmed lineups, volatile odds). Do NOT invent any statistics.
    """
    response = model.generate_content(prompt)
    return response.text

# ---------- UI Layout ----------
st.title("⚽ Football Signal Engine")
st.caption("Stage 2: Poisson Model, Odds Edge Scoring & Gemini Analysis Integration")

col_api, col_gem, col_tz = st.columns([3, 3, 2])
with col_api:
    api_key = st.text_input("API-Football Key", type="password")
with col_gem:
    gemini_key = st.text_input("Gemini API Key", type="password")
with col_tz:
    tz_name = st.text_input("Time Zone", value="Africa/Lagos")

go = st.button("Refresh & Calculate Signals", type="primary", disabled=not (api_key and gemini_key))

if not (api_key and gemini_key):
    st.info("Enter both your API-Football Key and Gemini API Key to proceed.")
    st.stop()

if go:
    st.success("Keys accepted! Processing match data and generating Poisson matrices...")
