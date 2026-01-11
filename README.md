# Election Forecaster: Presidential Approval & House Vote Model

A statistical model and interactive dashboard that predicts U.S. House popular vote outcomes based on presidential approval ratings.

## 🚀 Live Demo

**[Click here to run the model](index.html)** 
*(If viewing on GitHub, enable GitHub Pages for this repository to see the live app)*

## 📊 Features

- **Interactive Forecast**: Input presidential approval ratings to see projected House vote shares.
- **Dynamic Modeling**:
    - **Approval Anchors**: Adjusts baselines based on party.
    - **Penalty/Boost Logic**: Applies penalties for low approval (<45%) and boosts for high approval (>55%).
    - **Election Cycle Effect**: Accounts for the historical "Midterm Penalty" vs. "Presidential Boost".
- **Advanced Visualization**:
    - **Dual Forecasts**: Separate projections for Democratic and Republican vote shares.
    - **Uncertainty Quantification**: 5-95% confidence intervals derived from Monte Carlo simulations.
    - **Historical Validation**: Compare the model's backtested predictions against actual election results (1992-2024).

## 🛠️ Tech Stack

- **Frontend**: Single-file HTML5 application.
- **Logic**: Vanilla JavaScript for statistical modeling and regression.
- **Libraries**:
    - [Plotly.js](https://plotly.com/javascript/) for interactive charting.
    - [Math.js](https://mathjs.org/) for matrix operations and OLS regression.

## 📂 Files

- `index.html`: The complete, standalone application. **(Main Entry Point)**
- `app.py`: (Optional) A Python/Streamlit prototype of the same model.
- `data/`: Historical election data used for training.

## 🔮 2026 Forecast Context

The app is pre-configured to load a "2026 Midterm" scenario by default, using recent 2025 approval data (e.g., ~36%) to demonstrate the compound effects of low approval and the midterm penalty.
