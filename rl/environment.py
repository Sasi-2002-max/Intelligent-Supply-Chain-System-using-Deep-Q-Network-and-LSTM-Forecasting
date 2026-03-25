import numpy as np
import pandas as pd
from forecasting.forecast import predict_next_day

data = pd.read_csv(
    r"C:\Users\rsasi\PycharmProjects\supply_chain_ai\data\superstore.csv",
    encoding='ISO-8859-1'
)

# Strip column names
data.columns = data.columns.str.strip()

# Replace slashes with dashes for consistent formatting
data['Order_Date'] = data['Order_Date'].astype(str).str.replace('/', '-', regex=True)

# Parse dates with dayfirst
data['Order_Date'] = pd.to_datetime(data['Order_Date'], dayfirst=True, errors='coerce')

# Check for rows that couldn't be parsed
print(data[data['Order_Date'].isna()])

# Drop only the truly invalid rows
data = data.dropna(subset=['Order_Date'])

# Sort by date
data = data.sort_values('Order_Date')

print(data.head())

daily_sales = data.groupby('Order_Date')['Sales'].sum().reset_index()
daily_sales = daily_sales.rename(columns={'Order_Date':'date', 'Sales':'sales'})


import numpy as np

class SupplyChainEnv:
    def __init__(self, daily_sales):
        self.max_inventory = 100
        self.daily_sales = daily_sales['sales'].values
        self.n_days = len(self.daily_sales)

        # NEW: warehouses & routes
        self.num_warehouses = 2
        self.num_routes = 3

        # lead time for each route
        self.route_lead_time = [1, 2, 3]

        # route cost
        self.route_cost = [5, 3, 2]

        self.reset()

    def reset(self):
        self.inventory = 50
        self.current_day = 0
        self.warehouse = np.random.randint(0, self.num_warehouses)
        self.route = np.random.randint(0, self.num_routes)

        # Use last 10 days to predict next demand
        start_idx = max(0, self.current_day - 10)
        history = self.daily_sales[start_idx:self.current_day]

        # If not enough history, fallback to actual demand
        if len(history) < 10:
            self.demand = self.daily_sales[self.current_day]
        else:
            self.demand = predict_next_day(history)

        return np.array([
            self.inventory,
            self.demand,
            self.warehouse,
            self.route,
            self.route_lead_time[self.route],
            self.route_cost[self.route]
        ], dtype=float)

    def step(self, action):
        """
        Action = [order_qty, warehouse_choice, route_choice]
        """
        order_qty, warehouse, route = action

        self.inventory += order_qty
        self.warehouse = warehouse
        self.route = route

        # simulate delay effect
        lead_time = self.route_lead_time[route]

        # demand fulfillment
        sales = min(self.inventory, self.demand)
        self.inventory -= sales

        # cost calculations
        holding_cost = self.inventory * 1
        ordering_cost = order_qty * 2
        transport_cost = self.route_cost[route] * order_qty

        reward = sales * 10 - holding_cost - ordering_cost - transport_cost

        # move to next day
        self.current_day += 1
        done = self.current_day >= self.n_days

        if not done:
            start_idx = max(0, self.current_day - 10)
            history = self.daily_sales[start_idx:self.current_day]

            if len(history) < 10:
                self.demand = self.daily_sales[self.current_day]
            else:
                self.demand = predict_next_day(history)

        next_state = np.array([
            self.inventory,
            self.demand,
            self.warehouse,
            self.route,
            lead_time,
            self.route_cost[route]
        ], dtype=float)

        return next_state, reward, done

    # … your SupplyChainEnv class above

    # ======= Test the environment =======
if __name__ == "__main__":
        env = SupplyChainEnv(daily_sales)
        state = env.reset()
        print("Initial state:", state)

        next_state, reward, done = env.step((10, 1, 2)) # order 10 units
        print("Next state:", next_state, "Reward:", reward, "Done:", done)