import streamlit as st
import pickle
import pandas as pd
import base64
import random
import os

st.set_page_config(page_title="IPL Win Predictor", layout="wide")

# ---------------- BACKGROUND ----------------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_path = os.path.join(os.path.dirname(__file__), "stad.jpg")

if os.path.exists(bg_path):
    bin_str = get_base64(bg_path)
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.85)),
                              url("data:image/jpg;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        h1, h2, h3, p, span, label {{
            color: white !important;
        }}
        div[data-testid="stBlock"] {{
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
        }}
        </style>
    """, unsafe_allow_html=True)

# ---------------- LOAD MODEL (FIXED) ----------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "pipe.pkl")

    if not os.path.exists(model_path):
        st.error("❌ pipe.pkl not found in repo")
        st.stop()

    with open(model_path, "rb") as f:
        return pickle.load(f)

pipe = load_model()

# ---------------- LOGOS ----------------
logos = {
    'Sunrisers Hyderabad': 'https://upload.wikimedia.org/wikipedia/en/8/81/Sunrisers_Hyderabad.svg',
    'Mumbai Indians': 'https://upload.wikimedia.org/wikipedia/en/c/cd/Mumbai_Indians_Logo.svg',
    'Royal Challengers Bangalore': 'https://upload.wikimedia.org/wikipedia/en/2/2b/Royal_Challengers_Bangalore_2020_design_logo.svg',
    'Kolkata Knight Riders': 'https://upload.wikimedia.org/wikipedia/en/4/4c/Kolkata_Knight_Riders_Logo.svg',
    'Kings XI Punjab': 'https://upload.wikimedia.org/wikipedia/en/d/d4/Punjab_Kings_Logo.svg',
    'Chennai Super Kings': 'https://upload.wikimedia.org/wikipedia/en/2/2b/Chennai_Super_Kings_Logo.svg',
    'Rajasthan Royals': 'https://upload.wikimedia.org/wikipedia/en/6/60/Rajasthan_Royals_Logo.svg',
    'Delhi Capitals': 'https://upload.wikimedia.org/wikipedia/en/f/f5/Delhi_Capitals_Logo.svg'
}

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'>🏏 IPL Win Predictor</h1>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
teams = list(logos.keys())
cities = ['Hyderabad','Bangalore','Mumbai','Kolkata','Delhi','Chennai','Jaipur','Pune']

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Batting Team', teams)
with col2:
    bowling_team = st.selectbox('Bowling Team', teams)

selected_city = st.selectbox('City', cities)
target = st.number_input('Target', value=150)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', value=50)
with col4:
    overs = st.number_input('Overs', value=10.0)
with col5:
    wickets = st.number_input('Wickets', value=3)

predict = st.button("🚀 Predict")

# ---------------- PREDICTION ----------------
if predict:

    if overs == 0:
        st.error("Overs cannot be 0")
        st.stop()

    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets_left = 10 - wickets

    if score >= target or wickets >= 10:
        st.error("Match already finished!")
    else:
        crr = score / overs
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team':[batting_team],
            'bowling_team':[bowling_team],
            'city':[selected_city],
            'runs_left':[runs_left],
            'balls_left':[balls_left],
            'wickets':[wickets_left],
            'total_runs_x':[target],
            'crr':[crr],
            'rrr':[rrr]
        })

        result = pipe.predict_proba(input_df)
        win = result[0][1]

        st.markdown("## 🏆 Result")

        col1, col2 = st.columns(2)

        with col1:
            st.image(logos[batting_team], width=120)
            st.markdown(f"### {batting_team}")
            st.metric("Win Chance", f"{round(win*100)}%")

        with col2:
            st.image(logos[bowling_team], width=120)
            st.markdown(f"### {bowling_team}")
            st.metric("Win Chance", f"{round((1-win)*100)}%")

        st.progress(int(win * 100))

        # ---------------- INSIGHTS ----------------
        st.markdown("### Match Insights")

        if rrr > 12:
            st.warning("High pressure chase!")
        elif rrr > 8:
            st.info("Balanced game")
        else:
            st.success("Comfortable")

        if wickets_left <= 3:
            st.error("⚠️ Few wickets left!")

        # ---------------- SIMULATION ----------------
        def simulate(runs_left, balls_left, wickets_left):
            while balls_left > 0 and runs_left > 0 and wickets_left > 0:
                outcome = random.choice([0,1,2,4,6,'W'])
                if outcome == 'W':
                    wickets_left -= 1
                else:
                    runs_left -= outcome
                balls_left -= 1
            return runs_left <= 0

        wins = sum(simulate(runs_left, balls_left, wickets_left) for _ in range(100))

        st.markdown(f"### Simulation Win Chance: {wins}%")

        # ---------------- WHAT IF ----------------
        extra = st.slider("Next over runs?", 0, 30, 10)
        st.write(f"Projected Score: {score + extra}")

        # ---------------- DOWNLOAD ----------------
        report = f"""
Win Probability: {round(win*100)}%
CRR: {crr}
RRR: {rrr}
Simulation: {wins}%
"""
        st.download_button("Download Report", report)

        # ---------------- CELEBRATION ----------------
        if win > 0.7:
            st.balloons()
            st.success("Dominating position!")