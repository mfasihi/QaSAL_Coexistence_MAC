import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import QNetwork, MultiHeadQNetwork


class DQNAgent:
    def __init__(self,
                 state_size,
                 pc1_action_size,
                 pc3_action_size,
                 ap_action_size,
                 gamma=0.99,
                 augmented_reward=False,
                 multi_objective=False,
                 alpha=0.5,
                 constraints=None):
        self.state_size = state_size
        self.pc1_action_size = pc1_action_size
        self.pc3_action_size = pc3_action_size
        self.ap_action_size = ap_action_size
        self.action_size = pc1_action_size * pc3_action_size * ap_action_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.epsilon = 1.0
        self.buffer_size = int(1e6)
        self.batch_size = 16
        self.gamma = gamma
        self.learning_rate = 0.00001

        self.augmented_reward = augmented_reward
        self.constraints = constraints
        self.multi_objective = multi_objective
        self.alpha = alpha

        self.memory = deque(maxlen=self.buffer_size)
        if self.multi_objective:
            self.network = MultiHeadQNetwork(self.state_size, self.action_size).to(self.device)
            self.target_network = MultiHeadQNetwork(self.state_size, self.action_size).to(self.device)
        else:
            self.network = QNetwork(self.state_size, self.action_size).to(self.device)
            self.target_network = QNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

        self.update_target_network()

        self.loss_history = []  # To store loss values
        self.q_value_history = []  # To store cumulative Q-values

    def remember(self, state, action, reward, violation, next_state, done):
        action_index = (action[0] * self.pc3_action_size * self.ap_action_size) + \
                       (action[1] * self.ap_action_size) + \
                       action[2]
        self.memory.append((state, action_index, reward, violation, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return (
                random.randint(0, self.pc1_action_size - 1),
                random.randint(0, self.pc3_action_size - 1),
                random.randint(0, self.ap_action_size - 1),
            )

        state = state if isinstance(state, torch.Tensor) else torch.FloatTensor(state).to(self.device)
        state = state.unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.multi_objective:
                q_delay, q_fair = self.network(state)  # two heads
                combined_q = self.alpha * q_delay + (1 - self.alpha) * q_fair
                action_idx = combined_q.argmax().item()
            else:
                q_values = self.network(state)
                action_idx = q_values.argmax().item()

            action_pc1 = action_idx // (self.pc3_action_size * self.ap_action_size)
            action_pc3 = (action_idx % (self.pc3_action_size * self.ap_action_size)) // self.ap_action_size
            action_ap = action_idx % self.ap_action_size

        return action_pc1, action_pc3, action_ap

    def replay(self, lambda_values=None):
        if len(self.memory) < self.batch_size:
            return

        experiences = random.sample(self.memory, self.batch_size)
        states, actions, rewards, violations, next_states, dones = zip(*experiences)

        states = torch.tensor(np.vstack(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        violations = torch.tensor(violations, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.vstack(next_states), dtype=torch.float32).to(self.device)

        if self.multi_objective:
            q_current_delay, q_current_fair = self.network(states)
            current_q_delay = q_current_delay.gather(1, actions)
            current_q_fair = q_current_fair.gather(1, actions)

            q_next_delay, q_next_fair = self.target_network(next_states)
            max_next_q_delay = q_next_delay.max(1, keepdim=True)[0]
            max_next_q_fair = q_next_fair.max(1, keepdim=True)[0]
        else:
            current_q_values = self.network(states).gather(1, actions)

            # next_actions = self.network(next_states).argmax(1).unsqueeze(1).to(self.device)
            # next_q_values = self.target_network(next_states).gather(1, next_actions)
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]

        with torch.no_grad():
            target_q_values = rewards

            if self.augmented_reward:
                lambda_delay = lambda_values["smoothed_delay_pc1"]
                lambda_tensor = torch.tensor([lambda_delay], dtype=torch.float32).to(self.device)
                constraint_penalty = lambda_tensor * torch.clamp(violations, min=0.0)
                target_q_values -= constraint_penalty

            if self.multi_objective:
                target_delay = rewards + self.gamma * max_next_q_delay
                target_fair = rewards + self.gamma * max_next_q_fair
            else:
                target_q_values = target_q_values + self.gamma * next_q_values
                target_q_values = torch.clamp(target_q_values, min=-100.0, max=100.0)  # Clip targets

        # loss = nn.MSELoss()
        loss_fn = nn.SmoothL1Loss()
        if self.multi_objective:
            loss_delay = loss_fn(current_q_delay, target_delay.clamp(-100.0, 100.0))
            loss_fair = loss_fn(current_q_fair, target_fair.clamp(-100.0, 100.0))
            loss = loss_delay + loss_fair  # combined loss
        else:
            loss = loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.loss_history.append(loss.item())

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        self.network.load_state_dict(torch.load(filename, map_location=self.device))

