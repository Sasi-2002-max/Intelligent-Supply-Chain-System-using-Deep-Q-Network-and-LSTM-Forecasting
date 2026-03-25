import streamlit as st
import pandas as pd
import torch

# Import your modules
from forecasting.forecast import predict_next_day
from rl.environment import SupplyChainEnv
from rl.dqn_agent import Agent

st.set_page_config(page_title="Supply Chain AI", layout="wide")
st.title("📦 Supply Chain AI Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("Upload Sales CSV", type=["csv"])

if uploaded_file is not None:

    # -----------------------------
    # Load CSV in chunks safely
    # -----------------------------
    # -----------------------------
    # Load CSV in chunks safely
    # -----------------------------
    chunks = pd.read_csv(uploaded_file, chunksize=100000)
    df_list = []

    for chunk in chunks:
        # Normalize column names
        chunk.columns = (
            chunk.columns
            .str.strip()  # remove spaces
            .str.replace(' ', '_')  # replace spaces with underscores
            .str.lower()  # lowercase
        )

        # Check required columns
        if 'date' not in chunk.columns or 'sales' not in chunk.columns:
            st.error(f"Uploaded CSV must contain 'date' and 'sales'. Found: {list(chunk.columns)}")
            st.stop()

        # Keep only relevant columns
        df_list.append(chunk[['date', 'sales']])

    # Concatenate all chunks
    df = pd.concat(df_list)

    # Convert date to datetime & drop invalid rows
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date'])

    # Aggregate daily sales
    daily_sales = df.groupby('date')['sales'].sum().reset_index()

    # Limit last 1000 days for LSTM
    daily_sales = daily_sales.tail(1000)

    st.subheader("📊 Data Preview")
    st.dataframe(daily_sales.head())

    # -----------------------------
    # Forecast Section
    # -----------------------------
    st.subheader("📈 Demand Forecast")
    if st.button("Predict Tomorrow Demand"):
        try:
            # Use aggregated daily_sales
            demand, rmse, mae = predict_next_day(daily_sales)
            st.success(f"Predicted Demand for Tomorrow: {demand:.2f}")

            st.write("### 📊 Model Performance")
            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"MAE: {mae:.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

    # -----------------------------
    # RL Inventory Optimization
    # -----------------------------
    st.subheader("🤖 Inventory Optimization")
    if st.button("Run Optimization"):
        try:
            # Use daily_sales for RL environment
            env = SupplyChainEnv(daily_sales)
            agent = Agent()

            state = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0)

            action = agent.act(state)
            action_idx = agent.action_space.index(action)
            next_state, reward, done = env.step(action)

            order, warehouse, route = action

            st.write("### Results")
            st.write(f"📦 Order Quantity: {order}")
            st.write(f"🏬 Warehouse: {warehouse}")
            st.write(f"🚚 Route: {route}")
            st.write(f"Reward: {reward}")
            st.write(f"Next State: {next_state}")

        except Exception as e:
            st.error(f"Error in optimization: {e}")

else:
    st.info("Please upload a dataset to proceed.")
