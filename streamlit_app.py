import streamlit as st
import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from google import genai

BASE = "https://v3.football.api-sports.io"
WINDOW_HOURS = 12

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
        "Over 2.5 Goals": p_o25,
        "Under 2.5 Goals": p_u25,
    }

def evaluate_with_gemini(gemini_key, match_data, probs):
    """Sends calculated statistics and outcome probabilities to Gemini for review."""
    client = genai.Client(api_key=gemini_key)
    
    prompt = f"""
    You are an expert sports quantitative analyst. Review the statistical model output for the following football match:
    Match: {match_data['home']} vs {match_data['away']}
    League: {match_data['league']}
    Match Time: {match_data['time']}
    
    Model Calculated Outcome Probabilities (Poisson):
    {probs}
    
    Task:
    Provide a concise evaluation (3 key bullet points max) explaining:
    1. The strongest statistical signal based on expected goals.
    2. Tactical context or market risk to consider.
    3. Final quantitative verdict.
    Do NOT invent any statistics.
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# ---------- UI Layout ----------
st.title("⚽ Football Signal Engine")
st.caption("Live Match Fetching, Poisson Probability Modeling & Gemini AI Synthesis")

col_api, col_gem, col_tz = st.columns([3, 3, 2])
with col_api:
    api_key = st.text_input("API-Football Key", type="password")
with col_gem:
    gemini_key = st.text_input("Gemini API Key", type="password")
with col_tz:
    tz_name = st.text_input("Time Zone", value="Africa/Lagos")

go = st.button("Refresh & Calculate Signals", type="primary", disabled=not (api_key and gemini_key))

if go:
    try:
        user_tz = ZoneInfo(tz_name)
    except Exception:
        st.error(f"Invalid timezone string '{tz_name}'. Defaulting to UTC.")
        user_tz = ZoneInfo("UTC")

    now = datetime.now(user_tz)
    date_str = now.strftime("%Y-%m-%d")

    st.info(f"Fetching upcoming fixtures for {date_str}...")

    fixtures = []
    try:
        res, remaining = api_get(api_key, "/fixtures", {"date": date_str})
        st.sidebar.metric("API Requests Remaining Today", remaining)
        fixtures = res.get("response", [])
    except Exception as e:
        st.error(f"Error fetching data from API-Football: {e}")

    upcoming_matches = []
    for f in fixtures:
        f_date = datetime.fromisoformat(f["fixture"]["date"]).astimezone(user_tz)
        if now <= f_date <= now + timedelta(hours=WINDOW_HOURS):
            upcoming_matches.append({
                "id": f["fixture"]["id"],
                "league": f["league"]["name"],
                "home": f["teams"]["home"]["name"],
                "away": f["teams"]["away"]["name"],
                "time": f_date.strftime("%H:%M %Z")
            })

    if not upcoming_matches:
        st.warning("No live match fixtures returned for this window. Generating sample statistical analysis:")
        sample_matches = [
            {"home": "Arsenal", "away": "Chelsea", "league": "Premier League", "time": "17:30 WAT", "home_xg": 1.85, "away_xg": 1.10},
            {"home": "Real Madrid", "away": "Barcelona", "league": "La Liga", "time": "20:00 WAT", "home_xg": 1.95, "away_xg": 1.65}
        ]
        
        for match in sample_matches:
            st.subheader(f"⚽ {match['home']} vs {match['away']} — {match['league']} ({match['time']})")
            probs = calculate_poisson_probs(home_xg=match["home_xg"], away_xg=match["away_xg"])
            
            cols = st.columns(len(probs))
            for col, (k, v) in zip(cols, probs.items()):
                col.metric(k, f"{v*100:.1f}%")
                
            with st.spinner("Generating Gemini AI Analysis..."):
                ai_eval = evaluate_with_gemini(gemini_key, match, probs)
                st.markdown("### 🤖 Gemini AI Signal Evaluation")
                st.markdown(ai_eval)
                st.divider()
    else:
        st.success(f"Found {len(upcoming_matches)} live match(es) starting in the next {WINDOW_HOURS} hours.")
        for match in upcoming_matches:
            st.subheader(f"⚽ {match['home']} vs {match['away']} — {match['league']} ({match['time']})")
            
            home_xg, away_xg = 1.65, 1.20
            probs = calculate_poisson_probs(home_xg, away_xg)
            
            cols = st.columns(len(probs))
            for col, (k, v) in zip(cols, probs.items()):
                col.metric(k, f"{v*100:.1f}%")
                
            with st.spinner("Generating Gemini AI Analysis..."):
                ai_eval = evaluate_with_gemini(gemini_key, match, probs)
                st.markdown("### 🤖 Gemini AI Signal Evaluation")
                st.markdown(ai_eval)
                st.divider()
