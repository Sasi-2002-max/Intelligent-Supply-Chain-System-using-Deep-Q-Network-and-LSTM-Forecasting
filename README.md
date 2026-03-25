  <img width="1846" height="881" alt="Screenshot (674)" src="https://github.com/user-attachments/assets/75ebba8f-cd0b-4eed-ab01-f9069973586a" />

📝 Project Overview

Supply Chain AI is a Python-based project that combines time series forecasting and reinforcement learning to optimize inventory and supply chain operations. It allows users to:

* Forecast daily demand using an LSTM model.
* Simulate inventory decisions using an RL agent considering multiple warehouses, routes, lead times, and transportation costs.
* Visualize results interactively via a Streamlit dashboard.

This project demonstrates a real-world application of AI in retail and logistics operations.

🔹 Key Features
1. Demand Forecasting (LSTM)
    * Predicts the next day’s sales based on historical daily data.
    * Handles large datasets efficiently using chunked CSV loading.
    * Displays model performance: RMSE and MAE.
    * Scales and normalizes sales data for robust predictions.

2. Reinforcement Learning (Inventory Optimization)
   * Agent decides:
       1. Order quantity
       2. Warehouse selection
       3. Shipping route
   * Considers:
       1. Lead time (time to deliver stock via a route)
       2. Route cost (transport cost per unit)
       3. Inventory levels and predicted demand
       4. Reward function balances profit from sales with holding, ordering, and transport costs.

4. Streamlit Dashboard
   * Upload your sales CSV and visualize data.
   * Run demand forecasts and inventory optimization interactively.
   * Shows detailed results including:
        1. Predicted demand
        2. Agent’s order, warehouse, and route choices
        3. Reward and next state
    

📂 Dataset Requirements

The system expects a CSV file with at least the following columns:
| Column  | Type     | Description                 |
| ------- | -------- | --------------------------- |
| `date`  | datetime | Order or sales date         |
| `sales` | numeric  | Sales quantity for that day |

Optional columns for multivariate LSTM:
price, promo, weekday, month


🖥️ Installation
1. Clone the repository:
   git clone https://github.com/your-username/supply-chain-ai.git
   cd supply-chain-ai

2. Create a virtual environment and install dependencies:
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt

3. Run the Streamlit app:
   streamlit run app.py


📊 Usage
1. Open the dashboard in your browser (Streamlit URL will appear after running the app).
2. Upload your sales CSV file.
3. Demand Forecast: Click “Predict Tomorrow Demand”
   * See predicted sales, RMSE, and MAE.
     <img width="652" height="400" alt="Screenshot (676)" src="https://github.com/user-attachments/assets/508015ae-9e8f-4bec-9cd3-541740d7ecc8" />

4. Inventory Optimization: Click “Run Optimization”
   * See RL agent’s order, warehouse, route, reward, and next state.
     <img width="949" height="507" alt="Screenshot (675)" src="https://github.com/user-attachments/assets/4617e950-931e-447b-8416-d92b20b18909" />

