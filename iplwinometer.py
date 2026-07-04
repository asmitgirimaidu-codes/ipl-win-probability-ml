import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random

# ==========================================
# 1. CONFIG & PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="IPL Win-O-Meter",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for that "Premium" look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
    }
    h1 {
        color: #ff4b4b;
        font-family: 'Arial Black', sans-serif;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4c4c52;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA GENERATION (Synthetic Data)
# ==========================================
@st.cache_data
def generate_mock_data():
    """
    Since we can't load external CSV files easily in a demo script,
    we generate realistic IPL data on the fly to train our model.
    """
    teams = ['Mumbai Indians', 'Chennai Super Kings', 'RCB', 'KKR', 'SRH', 'DC', 'RR', 'PBKS', 'GT', 'LSG']
    venues = ['Wankhede', 'Chepauk', 'Eden Gardens', 'MCG', ' Arun Jaitley Stadium']
    
    data = []
    # Generate 500 matches worth of data
    for _ in range(500):
        team_a = random.choice(teams)
        team_b = random.choice([t for t in teams if t != team_a])
        venue = random.choice(venues)
        
        # Random score simulation
        runs_a = random.randint(120, 240)
        wickets_a = random.randint(2, 10)
        overs_a = random.randint(5, 20)
        
        # Logic: Winner is determined loosely by score (this is for training)
        # In a real scenario, you use historical match results CSV
        winner = 1 if runs_a > 170 else 0 
        
        data.append([team_a, team_b, venue, runs_a, wickets_a, overs_a, winner])
        
    df = pd.DataFrame(data, columns=['Batting_Team', 'Bowling_Team', 'Venue', 'Runs', 'Wickets', 'Overs', 'Win'])
    return df

# ==========================================
# 3. MACHINE LEARNING MODEL
# ==========================================
def train_ml_model(df):
    # Encode categorical variables
    le_team = LabelEncoder()
    le_venue = LabelEncoder()
    
    df['Batting_Team_Enc'] = le_team.fit_transform(df['Batting_Team'])
    df['Bowling_Team_Enc'] = le_team.transform(df['Bowling_Team'])
    df['Venue_Enc'] = le_venue.fit_transform(df['Venue'])
    
    # Features and Target
    X = df[['Batting_Team_Enc', 'Bowling_Team_Enc', 'Venue_Enc', 'Runs', 'Wickets', 'Overs']]
    y = df['Win']
    
    # Train Logistic Regression
    model = LogisticRegression()
    model.fit(X, y)
    
    return model, le_team, le_venue

# ==========================================
# 4. SIDEBAR & INPUTS
# ==========================================
st.sidebar.title("🏏 Match Settings")

teams = ['Mumbai Indians', 'Chennai Super Kings', 'RCB', 'KKR', 'SRH', 'DC', 'RR', 'PBKS', 'GT', 'LSG']
venues = ['Wankhede (Mumbai)', 'Chepauk (Chennai)', 'Eden Gardens (Kolkata)', 'MCG (Melbourne)']

batting_team = st.sidebar.selectbox("Select Batting Team", teams)
bowling_team = st.sidebar.selectbox("Select Bowling Team", [t for t in teams if t != batting_team])
venue = st.sidebar.selectbox("Select Venue", venues)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Live Score Input")

current_score = st.sidebar.number_input("Current Total Runs", min_value=0, value=120, step=1)
wickets_down = st.sidebar.number_input("Wickets Lost", min_value=0, max_value=10, value=4)
overs_completed = st.sidebar.slider("Overs Completed", 0.0, 20.0, 15.0, 0.5)

# Calculate Projected Score for visualization
run_rate = current_score / overs_completed if overs_completed > 0 else 0
projected_final = run_rate * 20

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
# Load Model
df = generate_mock_data()
model, le_team, le_venue = train_ml_model(df)

# Encode Inputs for Prediction
try:
    bt_enc = le_team.transform([batting_team])[0]
    bw_enc = le_team.transform([bowling_team])[0]
    v_enc = le_venue.transform([venue])[0]
except ValueError:
    # Fallback if combination is unseen (common in small mock data)
    bt_enc = 5
    bw_enc = 2
    v_enc = 0

# Create Input Array
input_features = np.array([[bt_enc, bw_enc, v_enc, current_score, wickets_down, overs_completed]])

# Get Probability
prob_win = model.predict_proba(input_features)[0][1] * 100
prob_loss = 100 - prob_win

# Adjust probability slightly based on Current Run Rate (The "Human Factor" logic)
if run_rate > 9.0: prob_win += 10
if wickets_down > 7: prob_win -= 15

# Clamp values
prob_win = max(0, min(100, prob_win))

# ==========================================
# 6. MAIN DASHBOARD UI
# ==========================================

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🏏 IPL Win-O-Meter")
    st.caption(f"Live Analysis: {batting_team} vs {bowling_team} at {venue}")
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/en/7/7b/Indian_Premier_League_Cricket_2013.png", width=100)

st.markdown("---")

# KPI Metrics
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(label="Current Score", value=f"{current_score}/{wickets_down}", delta=f"CRR: {run_rate:.2f}")

with kpi2:
    st.metric(label="Projected Score (20 Overs)", value=f"{int(projected_final)}")

with kpi3:
    st.metric(label="Win Probability", value=f"{prob_win:.1f}%", delta_color="normal")

# Visualization: The Gauge
fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prob_win,
    title = {'text': "Chance of Winning", 'font': {'size': 24, 'color': 'white'}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
        'bar': {'color': "#ff4b4b"},
        'bgcolor': "#262730",
        'borderwidth': 2,
        'bordercolor': "white",
        'steps': [
            {'range': [0, 40], 'color': "#590d0d"},
            {'range': [40, 60], 'color': "#594a0d"},
            {'range': [60, 100], 'color': "#0d5918"}
        ],
    }
))

st.plotly_chart(fig_gauge, use_container_width=True)

# ==========================================
# 7. STRATEGIC CHARTS
# ==========================================
c1, c2 = st.columns(2)

with c1:
    # Worm Chart Simulation
    over_labels = [f"Over {i}" for i in range(1, int(overs_completed)+1)]
    team_runs = np.cumsum(np.random.randint(5, 15, int(overs_completed)))
    opp_runs = np.cumsum(np.random.randint(4, 14, int(overs_completed)))
    
    chart_df = pd.DataFrame({
        'Overs': over_labels,
        batting_team: team_runs,
        bowling_team: opp_runs
    })
    
    fig_line = px.line(chart_df, x='Overs', y=[batting_team, bowling_team], 
                      title="Run Rate Comparison (Worm)")
    fig_line.update_layout(paper_bgcolor="transparent", plot_bgcolor="rgba(0,0,0,0.5)", font_color="white")
    st.plotly_chart(fig_line, use_container_width=True)

with c2:
    # Win Probability by Over
    over_range = list(range(1, 21))
    win_prob_sim = [max(0, min(100, 50 + (i/20)*15 + random.randint(-5,5))) for i in over_range]
    
    df_prob = pd.DataFrame({'Over': over_range, 'Win Chance': win_prob_sim})
    fig_bar = px.area(df_prob, x='Over', y='Win Chance', title="Historical Win Probability Curve")
    fig_bar.update_layout(paper_bgcolor="transparent", plot_bgcolor="rgba(0,0,0,0.5)", font_color="white")
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.write("💻 Built with Python & Streamlit | 🤖 Model: Logistic Regression")