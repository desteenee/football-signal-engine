import streamlit as st
import requests
import time
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import google.generativeai as genai

BASE_URL = "https://api.football-data.org/v4"

st.set_page_config(page_title="Football Signal Engine", page_icon="⚽", layout="wide")

# ---------- Rate-Limited API Client ----------
def fetch_fd_fixtures_throttled(api_key, date_from, date_to):
    """
    Fetches scheduled matches while monitoring API response headers 
    to automatically prevent rate-limit bans (HTTP 429).
    """
    headers = {"X-Auth-Token": api_key}
    url = f"{BASE_URL}/matches?dateFrom={date_from}&dateTo={date_to}"
    
    response = requests.get(url, headers=headers, timeout=15)
    
    requests_remaining = response.headers.get("X-Requests-Available-Minute")
    seconds_to_reset = response.headers.get("X-RequestCounter-Reset")
    
    if requests_remaining is not None and int(requests_remaining) <= 1:
        wait_time = int(seconds_to_reset) if seconds_to_reset else 60
        st.warning(f"Rate limit threshold reached. Pausing for {wait_time} seconds to protect account status...")
        time.sleep(wait_time)
        
    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 60))
        st.error(f"Rate limited (HTTP 429). Waiting {retry_after} seconds before retry...")
        time.sleep(retry_after)
        response = requests.get(url, headers=headers, timeout=15)

    response.raise_for_status()
    return response.json().get("matches", [])

# ---------- Core Analytics Logic ----------
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

def evaluate_with_gemini(gemini_key, match_data, probs):
    genai.configure(api_key=gemini_key)
    
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
    3. Final quantitative verdict and recommendation.
    Do NOT invent any statistics.
    """
    
    # Priority model targets
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-2.0-flash",
        "models/gemini-2.5-flash",
        "models/gemini-1.5-flash"
    ]
    
    for m in models_to_try:
        try:
            model = genai.GenerativeModel(m)
            response = model.generate_content(prompt)
            return response.text
        except Exception:
            continue
            
    # Fallback to dynamically querying supported content generation models
    try:
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                model = genai.GenerativeModel(m.name)
                response = model.generate_content(prompt)
                return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini API Error: {e}")

    raise RuntimeError("Could not establish connection with an active Gemini model.")

# ---------- Interface ----------
st.title("⚽ Football Signal Engine")
st.caption("Powered by Football-Data.org & Gemini AI Synthesis")

col_api, col_gem, col_tz = st.columns([3, 3, 2])
with col_api:
    fd_key = st.text_input("Football-Data.org API Key", value="5279f39beb64424cbbfc4f94e8e48159", type="password")
with col_gem:
    gemini_key = st.text_input("Gemini API Key", type="password")
with col_tz:
    tz_name = st.text_input("Time Zone", value="Africa/Lagos")

go = st.button("Refresh & Calculate Signals", type="primary", disabled=not (fd_key and gemini_key))

if go:
    try:
        user_tz = ZoneInfo(tz_name)
    except Exception:
        st.error(f"Invalid timezone string '{tz_name}'. Defaulting to UTC.")
        user_tz = ZoneInfo("UTC")

    now = datetime.now(user_tz)
    date_from = now.strftime("%Y-%m-%d")
    date_to = (now + timedelta(days=3)).strftime("%Y-%m-%d")

    st.info(f"Fetching upcoming matches ({date_from} to {date_to})...")

    try:
        matches = fetch_fd_fixtures_throttled(fd_key, date_from, date_to)
        
        if not matches:
            st.warning("No scheduled matches found for the selected timeframe.")
        else:
            st.success(f"Retrieved {len(matches)} match(es) safely under rate limits.")
            
            for m in matches[:10]:
                utc_date = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
                local_date = utc_date.astimezone(user_tz)
                
                match_info = {
                    "league": m["competition"]["name"],
                    "home": m["homeTeam"]["name"],
                    "away": m["awayTeam"]["name"],
                    "time": local_date.strftime("%Y-%m-%d %H:%M %Z")
                }
                
                st.subheader(f"⚽ {match_info['home']} vs {match_info['away']} — {match_info['league']} ({match_info['time']})")
                
                probs = calculate_poisson_probs(1.60, 1.15)
                
                cols = st.columns(len(probs))
                for col, (k, v) in zip(cols, probs.items()):
                    col.metric(k, f"{v*100:.1f}%")
                    
                with st.spinner("Generating Gemini AI Signal Evaluation..."):
                    ai_eval = evaluate_with_gemini(gemini_key, match_info, probs)
                    st.markdown("### 🤖 Gemini AI Signal Evaluation")
                    st.markdown(ai_eval)
                    st.divider()

    except requests.exceptions.HTTPError as err:
        st.error(f"API Error: {err}")
    except Exception as e:
        st.error(f"Error executing application: {e}")
