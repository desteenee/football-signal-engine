import streamlit as st
import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime
from zoneinfo import ZoneInfo
import google.generativeai as genai

# Explicitly pointed to Football-only domain
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

def evaluate_with_gemini(gemini_key, match_data, probs, is_watchlist=False):
    genai.configure(api_key=gemini_key)
    watchlist_note = "NOTE: This match is currently on the WATCHLIST." if is_watchlist else ""
    
    prompt = f"""
    You are an expert sports quantitative analyst. Review the statistical model output for the following football match:
    Match: {match_data['home']} vs {match_data['away']}
    League: {match_data['league']}
    Match Time: {match_data['time']}
    {watchlist_note}
    
    Model Calculated Outcome Probabilities (Poisson):
    {probs}
    
    Task:
    Provide a concise evaluation (3 key bullet points max) explaining:
    1. The strongest statistical signal based on expected goals.
    2. Tactical context or market risk to consider.
    3. Final quantitative verdict and recommendation.
    Do NOT invent any statistics.
    """
    
    models_to_try = ["gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-1.5-pro"]
    for m in models_to_try:
        try:
            model = genai.GenerativeModel(m)
            response = model.generate_content(prompt)
            return response.text
        except Exception:
            continue
            
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            model = genai.GenerativeModel(m.name)
            response = model.generate_content(prompt)
            return response.text

    raise RuntimeError("Could not find an accessible Gemini model.")

# ---------- UI Layout ----------
st.title("⚽ Football Signal Engine")
st.caption("Live Football Fixtures, Poisson Probability Modeling & Gemini AI Synthesis")

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
    st.info("Fetching all active live and upcoming football fixtures...")

    fixtures = []
    try:
        # Fetch directly from live football endpoint to bypass season restrictions
        res, remaining = api_get(api_key, "/fixtures", {"live": "all"})
        st.sidebar.metric("API Requests Remaining Today", remaining)
        fixtures = res.get("response", [])
        
        # Fallback to next 50 upcoming scheduled football matches if no live matches currently
        if not fixtures:
            res_next, remaining = api_get(api_key, "/fixtures", {"next": "50"})
            st.sidebar.metric("API Requests Remaining Today", remaining)
            fixtures = res_next.get("response", [])

    except Exception as e:
        st.error(f"Error fetching football data: {e}")

    upcoming_matches = []
    watchlist_matches = []

    for f in fixtures:
        f_date = datetime.fromisoformat(f["fixture"]["date"]).astimezone(user_tz)
        match_info = {
            "id": f["fixture"]["id"],
            "league": f["league"]["name"],
            "home": f["teams"]["home"]["name"],
            "away": f["teams"]["away"]["name"],
            "time": f_date.strftime("%Y-%m-%d %H:%M %Z"),
            "status": f["fixture"]["status"]["short"],
            "date_obj": f_date
        }
        
        if match_info["status"] in ["1H", "HT", "2H", "ET", "P", "LIVE"]:
            upcoming_matches.append(match_info)
        elif f_date > now:
            watchlist_matches.append(match_info)

    if upcoming_matches:
        st.success(f"Found {len(upcoming_matches)} live football match(es) in progress.")
        for match in upcoming_matches:
            st.subheader(f"🔴 LIVE: {match['home']} vs {match['away']} — {match['league']} ({match['status']})")
            
            home_xg, away_xg = 1.65, 1.20
            probs = calculate_poisson_probs(home_xg, away_xg)
            
            cols = st.columns(len(probs))
            for col, (k, v) in zip(cols, probs.items()):
                col.metric(k, f"{v*100:.1f}%")
                
            with st.spinner("Generating Gemini AI Signal Evaluation..."):
                ai_eval = evaluate_with_gemini(gemini_key, match, probs, is_watchlist=False)
                st.markdown("### 🤖 Gemini AI Signal Evaluation")
                st.markdown(ai_eval)
                st.divider()

    elif watchlist_matches:
        st.info(f"📋 **Watchlist Matches**: Displaying {len(watchlist_matches)} next scheduled football fixtures.")
        for match in watchlist_matches[:10]:  # Display top 10 upcoming matches
            st.subheader(f"⏳ [WATCHLIST] {match['home']} vs {match['away']} — {match['league']} ({match['time']})")
            
            home_xg, away_xg = 1.50, 1.10
            probs = calculate_poisson_probs(home_xg, away_xg)
            
            cols = st.columns(len(probs))
            for col, (k, v) in zip(cols, probs.items()):
                col.metric(k, f"{v*100:.1f}%")
                
            with st.spinner("Generating Gemini Watchlist Analysis..."):
                ai_eval = evaluate_with_gemini(gemini_key, match, probs, is_watchlist=True)
                st.markdown("### 🤖 Gemini AI Watchlist Evaluation")
                st.markdown(ai_eval)
                st.divider()
    else:
        st.error("No active or scheduled football fixtures found on your API plan.")
