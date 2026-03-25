import pandas as pd
from rl.environment import SupplyChainEnv
from rl.dqn_agent import Agent
from simulation.simulator import run_simulation

# Load and preprocess superstore data
data = pd.read_csv(r"C:\Users\rsasi\PycharmProjects\supply_chain_ai\data\superstore.csv",
                   encoding='ISO-8859-1')
data.columns = data.columns.str.strip()
data['Order_Date'] = data['Order_Date'].astype(str).str.replace('/', '-', regex=True)
data['Order_Date'] = pd.to_datetime(data['Order_Date'], dayfirst=True, errors='coerce')
data = data.dropna(subset=['Order_Date'])
data = data.sort_values('Order_Date')

daily_sales = data.groupby('Order_Date')['Sales'].sum().reset_index()
daily_sales = daily_sales.rename(columns={'Order_Date':'date', 'Sales':'sales'})

# Create environment with real demand
env = SupplyChainEnv(daily_sales)

agent = Agent()
agent.memory = []
episodes = 50

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    for _ in range(50):
        action = agent.act(state)
        next_state, reward, _ = env.step(action)

        agent.remember(state, action, reward, next_state)
        agent.train()

        state = next_state
        total_reward += reward

    print(f"Episode {ep}, Reward: {total_reward}")

# Run advanced simulation
run_simulation(agent)