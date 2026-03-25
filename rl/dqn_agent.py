import torch
import torch.nn as nn
import numpy as np
import random


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        action_size = 24  # (4 order options * 2 warehouses * 3 routes)
        self.fc = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
        )

    def forward(self, x):
        return self.fc(x)


class Agent:
    def __init__(self):
        self.model = DQN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = []
        self.gamma = 0.9

        self.action_space = []

        for order in range(0, 20, 5):
            for wh in range(2):
                for route in range(3):
                    self.action_space.append((order, wh, route))

    def act(self, state):
        if random.random() < 0.1:
            return random.choice(self.action_space)

        state = torch.FloatTensor(state[:6]).reshape(1, -1)
        q_values = self.model(state)

        action_idx = torch.argmax(q_values).item()
        return self.action_space[action_idx]

    def remember(self, s, a, r, ns):
        self.memory.append((s, a, r, ns))

    def train(self):
        if len(self.memory) < 32:
            return

        batch = random.sample(self.memory, 32)

        for s, a, r, ns in batch:
            s = torch.FloatTensor(s[:6]).reshape(1, -1)  # ✅ force 6 features
            ns = torch.FloatTensor(ns[:6]).reshape(1, -1)  # ✅ force 6 features



            target = r + self.gamma * torch.max(self.model(ns)).item()
            action_idx = self.action_space.index(a)
            output = self.model(s)[0][action_idx]

            loss = self.criterion(output, torch.tensor(target, dtype=torch.float))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()