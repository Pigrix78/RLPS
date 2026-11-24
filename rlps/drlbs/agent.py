import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    def __init__(self, bands: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(bands, 2 * bands),
            nn.ReLU(),
            nn.Linear(2 * bands, 2 * bands),
            nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(2 * bands, 2 * bands), nn.ReLU(), nn.Linear(2 * bands, 1))
        self.advantage = nn.Sequential(
            nn.Linear(2 * bands, 2 * bands),
            nn.ReLU(),
            nn.Linear(2 * bands, bands),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -0.1, 0.1)
                nn.init.zeros_(module.bias)

    def forward(self, state):
        features = self.features(state)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DuelingDoubleDQN:
    def __init__(
        self,
        bands: int,
        candidate_mask: np.ndarray,
        learning_rate: float,
        gamma: float,
        memory_size: int,
        batch_size: int,
        target_update: int,
        device: torch.device,
    ):
        self.bands = bands
        self.candidate_mask = np.asarray(candidate_mask, dtype=bool)
        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device
        self.policy = DuelingQNetwork(bands).to(device)
        self.target = DuelingQNetwork(bands).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        self.memory = np.zeros((memory_size, 2 * bands + 3), dtype=np.float32)
        self.memory_count = 0
        self.updates = 0

    def _available_numpy(self, state):
        return (np.asarray(state) < 0.5) & self.candidate_mask

    def _masked(self, values, states):
        candidate = torch.as_tensor(self.candidate_mask, device=self.device).view(1, -1)
        available = (states < 0.5) & candidate
        return values.masked_fill(~available, -1.0e9), available

    def q_values(self, state, masked=True):
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
            values = self.policy(tensor)
            if masked:
                values, _ = self._masked(values, tensor)
        return values.squeeze(0).cpu().numpy()

    def greedy_action(self, state):
        if not self._available_numpy(state).any():
            raise RuntimeError("No candidate band remains available.")
        return int(np.argmax(self.q_values(state)))

    def random_action(self, state):
        available = np.flatnonzero(self._available_numpy(state))
        if available.size == 0:
            raise RuntimeError("No candidate band remains available.")
        return int(np.random.choice(available))

    def store(self, state, action, reward, next_state, done):
        transition = np.hstack((state, [action, reward], next_state, [float(done)]))
        self.memory[self.memory_count % self.memory_size] = transition
        self.memory_count += 1

    def learn(self):
        if self.memory_count < self.batch_size:
            return None
        if self.updates % self.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())
        self.updates += 1

        upper = min(self.memory_count, self.memory_size)
        batch = self.memory[np.random.choice(upper, self.batch_size, replace=False)]
        states = torch.as_tensor(batch[:, : self.bands], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch[:, self.bands : self.bands + 1], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch[:, self.bands + 1 : self.bands + 2], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            batch[:, self.bands + 2 : 2 * self.bands + 2], dtype=torch.float32, device=self.device
        )
        done = torch.as_tensor(batch[:, -1:], dtype=torch.float32, device=self.device)

        current_q = self.policy(states).gather(1, actions)
        with torch.no_grad():
            policy_next, available = self._masked(self.policy(next_states), next_states)
            next_actions = policy_next.argmax(dim=1, keepdim=True)
            target_next = self.target(next_states).gather(1, next_actions)
            has_next = available.any(dim=1, keepdim=True).float()
            target_q = rewards + self.gamma * target_next * (1.0 - done) * has_next

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.optimizer.step()
        return float(loss.detach().cpu())
