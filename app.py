import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import requests
from scipy.stats import norm

# --- Configuration & Setup ---
st.set_page_config(page_title="Election Forecaster 2026", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS ---
st.markdown("""
<style>
    .metric-card { background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .metric-label { font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 5px; }
    .metric-value { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .dem-Text { color: #60a5fa; }
    .rep-Text { color: #f87171; }
    h1 { background: linear-gradient(to right, #60a5fa, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
</style>
""", unsafe_allow_html=True)

st.title("Election Forecaster")
st.markdown("*Statistical Model: Presidential Approval vs. House Popular Vote*")

# --- Data Loading (Hardcoded for Stability) ---
@st.cache_data
def load_data():
    data = [
        {"Year": 1992, "President": "Clinton", "Party": "D", "Approval": np.nan, "Rep_House_Vote": 45.10, "Dem_House_Vote": 50.10, "Is_Election": 1},
        {"Year": 1993, "President": "Clinton", "Party": "D", "Approval": 48.90, "Rep_House_Vote": np.nan, "Dem_House_Vote": np.nan, "Is_Election": 0},
        {"Year": 1994, "President": "Clinton", "Party": "D", "Approval": 45.90, "Rep_House_Vote": 51.50, "Dem_House_Vote": 44.70, "Is_Election": 1},
        {"Year": 1995, "President": "Clinton", "Party": "D", "Approval": 46.80, "Rep_House_Vote": np.nan, "Dem_House_Vote": np.nan, "Is_Election": 0},
        {"Year": 1996, "President": "Clinton", "Party": "D", "Approval": 55.70, "Rep_House_Vote": 48.15, "Dem_House_Vote": 48.22, "Is_Election": 1},
        {"Year": 1997, "President": "Clinton", "Party": "D", "Approval": 57.00, "Rep_House_Vote": np.nan, "Dem_House_Vote": np.nan, "Is_Election": 0},
        {"Year": 1998, "President": "Clinton", "Party": "D", "Approval": 63.80, "Rep_House_Vote": 48.40, "Dem_House_Vote": 47.30, "Is_Election": 1},
        {"Year": 1999, "President": "Clinton", "Party": "D", "Approval": 59.50, "Rep_House_Vote": np.nan, "Dem_House_Vote": np.nan, "Is_Election": 0},
        {"Year": 2000, "President": "Clinton", "Party": "D", "Approval": 60.70, "Rep_House_Vote": 47.60, "Dem_House_Vote": 47.10, "Is_Election": 1},
        {"Year": 2001, "President": "GWBush", "Party": "R", "Approval": 67.10, "Rep_House_Vote": np.nan, "Dem_House_Vote": np.nan, "Is_Election": 0},
        {"Year": 2002, "President": "GWBush", "Party": "R", "Approval": 71.40, "Rep_House_Vote": 50.00, "Dem_House_Vote": 45.20, "Is_Election": 1},
        {"Year": 2003, "President": "GWBush", "Party": "R", "Approval": 59.60, "Rep_House_Vote": np.nan, "Dem_House_Vote": np.nan, "Is_Election": 0},
        {"Year": 2004, "President": "GWBush", "Party": "R", "Approval": 50.10, "Rep_House_Vote": 49.40, "Dem_House_Vote": 46.80, "Is_Election": 1},
        {"Year": 2005, "President": "GWBush", "Party": "R", "Approval": 46.10, "Rep_House_Vote": np.nan, "Dem_House_Vote": np.nan, "Is_Election": 0},
        {"Year": 2006, "President": "GWBush", "Party": "R", "Approval": 39.80, "Rep_House_Vote": 44.30, "Dem_House_Vote": 52.30, "Is_Election": 1},
        {"Year": 2007, "President": "GWBush", "Party": "R", "Approval": 33.30, "Rep_House_Vote": np.nan, "Dem_House_Vote": np.nan, "Is_Election": 0},
        {"Year": 2008, "President": "GWBush", "Party": "R", "Approval": 26.90, "Rep_House_Vote": 42.60, "Dem_House_Vote": 53.20, "Is_Election": 1},
        {"Year": 2009, "President": "Obama", "Party": "D", "Approval": 57.20, "Rep_House_Vote": np.nan, "Dem_House_Vote": np.nan, "Is_Election": 0},
        {"Year": 2010, "President": "Obama", "Party": "D", "Approval": 46.70, "Rep_House_Vote": 51.70, "Dem_House_Vote": 44.90, "Is_Election": 1},
        {"Year": 2011, "President": "Obama", "Party": "D", "Approval": 44.40, "Rep_House_Vote": np.nan, "Dem_House_Vote": np.nan, "Is_Election": 0},
        {"Year": 2012, "President": "Obama", "Party": "D", "Approval": 47.90, "Rep_House_Vote": 47.70, "Dem_House_Vote": 48.80, "Is_Election": 1},
        {"Year": 2013, "President": "Obama", "Party": "D", "Approval": 45.90, "Rep_House_Vote": np.nan, "Dem_House_Vote": np.nan, "Is_Election": 0},
        {"Year": 2014, "President": "Obama", "Party": "D", "Approval": 42.60, "Rep_House_Vote": 51.20, "Dem_House_Vote": 45.50, "Is_Election": 1},
        {"Year": 2015, "President": "Obama", "Party": "D", "Approval": 46.50, "Rep_House_Vote": np.nan, "Dem_House_Vote": np.nan, "Is_Election": 0},
        {"Year": 2016, "President": "Obama", "Party": "D", "Approval": 51.00, "Rep_House_Vote": 48.30, "Dem_House_Vote": 47.30, "Is_Election": 1},
        {"Year": 2017, "President": "Trump", "Party": "R", "Approval": 38.40, "Rep_House_Vote": np.nan, "Dem_House_Vote": np.nan, "Is_Election": 0},
        {"Year": 2018, "President": "Trump", "Party": "R", "Approval": 40.40, "Rep_House_Vote": 44.80, "Dem_House_Vote": 53.40, "Is_Election": 1},
        {"Year": 2019, "President": "Trump", "Party": "R", "Approval": 42.20, "Rep_House_Vote": np.nan, "Dem_House_Vote": np.nan, "Is_Election": 0},
        {"Year": 2020, "President": "Trump", "Party": "R", "Approval": 42.80, "Rep_House_Vote": 47.20, "Dem_House_Vote": 50.30, "Is_Election": 1},
        {"Year": 2021, "President": "Biden", "Party": "D", "Approval": 48.90, "Rep_House_Vote": np.nan, "Dem_House_Vote": np.nan, "Is_Election": 0},
        {"Year": 2022, "President": "Biden", "Party": "D", "Approval": 41.00, "Rep_House_Vote": 50.00, "Dem_House_Vote": 47.30, "Is_Election": 1},
        {"Year": 2023, "President": "Biden", "Party": "D", "Approval": 39.80, "Rep_House_Vote": np.nan, "Dem_House_Vote": np.nan, "Is_Election": 0},
        {"Year": 2024, "President": "Biden", "Party": "D", "Approval": 39.10, "Rep_House_Vote": 49.80, "Dem_House_Vote": 47.20, "Is_Election": 1},
        {"Year": 2025, "President": "Trump", "Party": "R", "Approval": 41.00, "Rep_House_Vote": np.nan, "Dem_House_Vote": np.nan, "Is_Election": 0}
    ]
    return pd.DataFrame(data)

df = load_data()

# --- Sidebar ---
st.sidebar.header("Forecast Settings")
input_party = st.sidebar.radio("President's Party", ["Democrat", "Republican"], index=1)
input_cycle = st.sidebar.radio("Election Cycle", ["Midterm (e.g. '26)", "Presidential (e.g. '28)"], index=0)

input_approval = st.sidebar.slider("Approval (%)", 20.0, 80.0, 36.0, 0.1)

st.sidebar.markdown("---")
low_thresh = st.sidebar.slider("Low Approval Threshold (Penalty)", 35, 55, 45)
high_thresh = st.sidebar.slider("High Approval Threshold (Boost)", 50, 70, 55)

confidence_level = st.sidebar.slider("Confidence Interval (%)", 50, 99, 90)

# --- Preprocessing & Feature Engineering ---
def prepare_features(row, low_t, high_t):
    approval = row['Approval']
    if pd.isna(approval): 
        return pd.Series([np.nan]*4, index=['Dem_Anchor', 'Low_Impact', 'High_Impact', 'Cycle_Impact'])
    
    # 1. Anchor
    dem_anchor = approval if row['Party'] == 'D' else (100 - approval - 4)
    
    # 2. Low Penalty
    penalty = max(0, low_t - approval)
    low_impact = -1 * penalty if row['Party'] == 'D' else 1 * penalty
    
    # 3. High Boost
    boost = max(0, approval - high_t)
    high_impact = 1 * boost if row['Party'] == 'D' else -1 * boost
    
    # 4. Cycle Effect
    # Midterm = Year % 4 != 0
    is_midterm = (row['Year'] % 4 != 0)
    cycle_impact = 0
    if row['Party'] == 'D':
        cycle_impact = -1 if is_midterm else 1
    else:
        cycle_impact = 1 if is_midterm else -1
        
    return pd.Series([dem_anchor, low_impact, high_impact, cycle_impact], 
                     index=['Dem_Anchor', 'Low_Impact', 'High_Impact', 'Cycle_Impact'])

# Apply dynamic features based on current slider values
features_df = df.apply(lambda row: prepare_features(row, low_thresh, high_thresh), axis=1)
df = pd.concat([df, features_df], axis=1)

# --- Train Model ---
train_df = df[df['Is_Election'] == 1].dropna(subset=['Dem_House_Vote', 'Approval']).copy()

X = sm.add_constant(train_df[['Dem_Anchor', 'Low_Impact', 'High_Impact', 'Cycle_Impact']])
y = train_df['Dem_House_Vote']

model = sm.OLS(y, X).fit()
rse = np.sqrt(model.scale)

# --- Predict Current Scenario ---
if input_party == "Democrat":
    curr_anchor = input_approval
    party_sign = 1 # Used for D perspective logic later
else:
    curr_anchor = 100 - input_approval - 4
    party_sign = -1

# Current Features
curr_penalty = max(0, low_thresh - input_approval)
# If D is prez, penalty hurts D (-1). If R is prez, penalty helps D (+1)
curr_low_impact = -1 * curr_penalty if input_party == "Democrat" else 1 * curr_penalty

curr_boost = max(0, input_approval - high_thresh)
curr_high_impact = 1 * curr_boost if input_party == "Democrat" else -1 * curr_boost

is_curr_midterm = "Midterm" in input_cycle
curr_cycle_impact = 0
if input_party == "Democrat":
    curr_cycle_impact = -1 if is_curr_midterm else 1
else:
    curr_cycle_impact = 1 if is_curr_midterm else -1

exog = pd.DataFrame({'const': [1.0], 
                     'Dem_Anchor': [curr_anchor], 
                     'Low_Impact': [curr_low_impact],
                     'High_Impact': [curr_high_impact],
                     'Cycle_Impact': [curr_cycle_impact]})

pred_dem_mean = model.predict(exog)[0]
pred_rep_mean = 100 - 4 - pred_dem_mean

# Simulation
sims = np.random.normal(pred_dem_mean, rse, 10000)
lb_dem = np.percentile(sims, (100 - confidence_level) / 2)
ub_dem = np.percentile(sims, 100 - (100 - confidence_level) / 2)
lb_rep = 100 - 4 - ub_dem
ub_rep = 100 - 4 - lb_dem

# --- UI Layout ---
st.subheader(f"2026 House Forecast (Projected)")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<div class="metric-card" style="border-color:#3b82f6;"><div class="metric-label dem-Text">Democrats</div><div class="metric-value dem-Text">{pred_dem_mean:.1f}%</div><div style="font-size:0.8rem;color:#94a3b8;">{confidence_level}% Range: {lb_dem:.1f}% — {ub_dem:.1f}%</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card" style="border-color:#ef4444;"><div class="metric-label rep-Text">Republicans</div><div class="metric-value rep-Text">{pred_rep_mean:.1f}%</div><div style="font-size:0.8rem;color:#94a3b8;">{confidence_level}% Range: {lb_rep:.1f}% — {ub_rep:.1f}%</div></div>', unsafe_allow_html=True)

st.markdown("### Forecasting Uncertainty (Simulation)")
sim_tab1, sim_tab2 = st.tabs(["Democratic Distribution", "Republican Distribution"])

def plot_simulation(sim_data, title, color_bar, color_line, mean, lb, ub):
    fig, ax = plt.subplots(figsize=(10, 3), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    
    # Histogram
    ax.hist(sim_data, bins=50, color=color_bar, alpha=0.7, density=True)
    
    # Lines
    ax.axvline(mean, color='red', linewidth=2, label=f'Mean: {mean:.1f}%')
    ax.axvline(lb, color=color_line, linestyle='--', linewidth=2, label=f'Lower: {lb:.1f}%')
    ax.axvline(ub, color=color_line, linestyle='--', linewidth=2, label=f'Upper: {ub:.1f}%')
    
    ax.set_xlabel("Vote Share (%)", color='#94a3b8')
    ax.get_yaxis().set_visible(False)
    ax.tick_params(colors='#94a3b8')
    ax.spines['bottom'].set_color('#334155'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
    ax.legend(facecolor='#1e293b', labelcolor='#f8fafc', edgecolor='#334155', frameon=False, loc='upper right')
    return fig

with sim_tab1:
    st.pyplot(plot_simulation(sims, "Democrats", '#3b82f6', '#10b981', pred_dem_mean, lb_dem, ub_dem))

with sim_tab2:
    # Rep sims = 100 - 4 - dem_sims
    sims_rep = 100 - 4 - sims
    st.pyplot(plot_simulation(sims_rep, "Republicans", '#ef4444', '#10b981', pred_rep_mean, lb_rep, ub_rep))


st.markdown("### Historical Validation & Forecast")

# --- Backtesting for Chart ---
hist_df = train_df.copy()
hist_df['Pred_Dem'] = model.predict(X)
hist_df['Pred_Rep'] = 100 - 4 - hist_df['Pred_Dem']

z_score = norm.ppf(1 - (1 - confidence_level/100)/2)
margin = z_score * rse

hist_df['Dem_Lower'] = hist_df['Pred_Dem'] - margin
hist_df['Dem_Upper'] = hist_df['Pred_Dem'] + margin
hist_df['Rep_Lower'] = hist_df['Pred_Rep'] - margin
hist_df['Rep_Upper'] = hist_df['Pred_Rep'] + margin

tab1, tab2 = st.tabs(["Democratic Forecast", "Republican Forecast"])

def plot_chart(party, hist_actual, hist_pred, hist_lower, hist_upper, curr_mean, curr_lower, curr_upper, c_main, c_light):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    
    # History Interval
    ax.fill_between(hist_df['Year'], hist_df[hist_lower], hist_df[hist_upper], color=c_light, alpha=0.15, label=f'{confidence_level}% Conf. Interval')
    # History Lines
    ax.plot(hist_df['Year'], hist_df[hist_actual], marker='o', label='Actual', color=c_main)
    # NOTE: Fix for KeyError, using exact column names 'Pred_Dem' or 'Pred_Rep'
    ax.plot(hist_df['Year'], hist_df[hist_pred], marker='x', linestyle='--', label='Predicted', color=c_light, alpha=0.7)
    
    # 2026 Forecast
    ax.errorbar([2026], [curr_mean], yerr=[[curr_mean - curr_lower], [curr_upper - curr_mean]], fmt='*', markersize=15, color='#10b981', ecolor='#10b981', capsize=5, label='2026 Forecast', elinewidth=2)
    ax.text(2026, curr_upper + 0.5, f"{curr_mean:.1f}%", color='#10b981', ha='center', fontweight='bold')
    
    ax.set_ylabel("Vote Share (%)", color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, color='#334155', alpha=0.3)
    ax.spines['bottom'].set_color('#334155'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
    ax.legend(facecolor='#1e293b', labelcolor='#f8fafc', edgecolor='#334155', loc='upper left', ncol=4, frameon=False)
    return fig

with tab1:
    # Fix: Pass 'Pred_Dem' instead of 'Predicted_Dem_Vote'
    st.pyplot(plot_chart("Dem", 'Dem_House_Vote', 'Pred_Dem', 'Dem_Lower', 'Dem_Upper', pred_dem_mean, lb_dem, ub_dem, '#2563eb', '#93c5fd'))

with tab2:
    # Fix: Pass 'Pred_Rep' instead of 'Predicted_Rep_Vote'
    st.pyplot(plot_chart("Rep", 'Rep_House_Vote', 'Pred_Rep', 'Rep_Lower', 'Rep_Upper', pred_rep_mean, lb_rep, ub_rep, '#ef4444', '#fca5a5'))

with st.expander("Model Statistics"):
    st.write(model.summary())
