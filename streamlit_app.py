import streamlit as st
import requests
import time
import numpy as np
from scipy.stats import poisson
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import google.generativeai as genai

BASE_URL = "https://api.football-data.org/v4"

st.set_page_config(page_title="Football Signal Engine", page_icon="⚽", layout="wide")

# ---------- Rate-Limited API Client ----------
def fetch_fd_fixtures_throttled(api_key, date_from, date_to):
    headers = {"X-Auth-Token": api_key}
    url = f"{BASE_URL}/matches?dateFrom={date_from}&dateTo={date_to}"
    
    response = requests.get(url, headers=headers, timeout=15)
    
    requests_remaining = response.headers.get("X-Requests-Available-Minute")
    seconds_to_reset = response.headers.get("X-RequestCounter-Reset")
    
    if requests_remaining is not None and int(requests_remaining) <= 1:
        wait_time = int(seconds_to_reset) if seconds_to_reset else 60
        st.warning(f"Rate limit threshold reached. Pausing for {wait_time} seconds...")
        time.sleep(wait_time)
        
    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 60))
        st.error(f"Rate limited (HTTP 429). Waiting {retry_after} seconds...")
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
    Status: {match_data['status']}
    
    Model Calculated Outcome Probabilities (Poisson):
    {probs}
    
    Task:
    Provide a concise evaluation (3 key bullet points max) explaining:
    1. The strongest statistical signal based on expected goals.
    2. Tactical context or market risk to consider.
    3. Final quantitative verdict and recommendation.
    Do NOT invent any statistics.
    """
    
    models_to_try = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash"
    ]
    
    for m in models_to_try:
        try:
            model = genai.GenerativeModel(m)
            response = model.generate_content(prompt)
            return response.text
        except Exception:
            continue
            
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

col_mode, col_slide = st.columns([2, 4])
with col_mode:
    include_in_play = st.checkbox("Include Live / In-Play Matches", value=True)
with col_slide:
    hours_ahead = st.slider("Lookahead Window (Hours from now):", min_value=1, max_value=24, value=12, step=1)

if fd_key and gemini_key:
    try:
        user_tz = ZoneInfo(tz_name)
    except Exception:
        user_tz = ZoneInfo("UTC")

    now_utc = datetime.now(timezone.utc)
    date_from = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")
    date_to = (now_utc + timedelta(days=2)).strftime("%Y-%m-%d")

    try:
        all_matches = fetch_fd_fixtures_throttled(fd_key, date_from, date_to)
        
        cutoff_time = now_utc + timedelta(hours=hours_ahead)
        match_options = {}

        for m in all_matches:
            utc_date = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            status = m.get("status", "")
            
            # Show live matches OR matches kicking off within the specified lookahead window
            is_live = include_in_play and status in ["IN_PLAY", "PAUSED", "HALFTIME"]
            is_upcoming = (now_utc - timedelta(hours=3)) <= utc_date <= cutoff_time and status not in ["FINISHED", "AWARDED"]

            if is_live or is_upcoming:
                local_date = utc_date.astimezone(user_tz)
                status_tag = f"🔴 {status}" if is_live else local_date.strftime("%b %d, %H:%M %Z")
                label = f"[{status_tag}] {m['homeTeam']['name']} vs {m['awayTeam']['name']} ({m['competition']['name']})"
                
                match_options[label] = {
                    "league": m["competition"]["name"],
                    "home": m["homeTeam"]["name"],
                    "away": m["awayTeam"]["name"],
                    "time": local_date.strftime("%Y-%m-%d %H:%M %Z"),
                    "status": status
                }

        if not match_options:
            st.warning(f"No covered matches found in the API response. Football-Data free tier only includes top 12 leagues. If matches are happening in lower/regional leagues right now, they won't appear.")
        else:
            selected_label = st.selectbox("Select Match to Analyze:", list(match_options.keys()))
            
            if st.button("Analyze Selected Match", type="primary"):
                selected_match = match_options[selected_label]
                
                st.subheader(f"⚽ {selected_match['home']} vs {selected_match['away']}")
                st.caption(f"{selected_match['league']} | {selected_match['time']} | Status: {selected_match['status']}")
                
                probs = calculate_poisson_probs(1.60, 1.15)
                
                cols = st.columns(len(probs))
                for col, (k, v) in zip(cols, probs.items()):
                    col.metric(k, f"{v*100:.1f}%")
                    
                st.divider()
                
                with st.spinner("Generating AI Analysis..."):
                    ai_eval = evaluate_with_gemini(gemini_key, selected_match, probs)
                    st.markdown("### 🤖 Gemini AI Signal Evaluation")
                    st.markdown(ai_eval)

    except requests.exceptions.HTTPError as err:
        st.error(f"API Error: {err}")
    except Exception as e:
        st.error(f"Error executing application: {e}")
else:
    st.info("Please enter your API keys above to load matches.")
