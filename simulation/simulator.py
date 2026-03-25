import simpy
import random

def supply_chain(env, agent):
    inventory = 50

    while True:
        demand = random.randint(10, 40)

        state = [inventory, demand]
        action = agent.act(state)

        inventory += action
        sales = min(inventory, demand)
        inventory -= sales

        reward = sales * 10 - action * 2 - inventory

        next_state = [inventory, random.randint(10,40)]

        agent.remember(state, action, reward, next_state)
        agent.train()

        yield env.timeout(1)

def run_simulation(agent):
    env = simpy.Environment()
    env.process(supply_chain(env, agent))
    env.run(until=100)