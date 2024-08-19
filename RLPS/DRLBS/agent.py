import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchsummary import summary

class DQN(nn.Module):
    def __init__(self, band):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(band, 2 * band)
        self.fc2 = nn.Linear(2 * band, 2 * band)
        self.fc3 = nn.Linear(2 * band, band)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-0.1, 0.1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Dueling(nn.Module):
    def __init__(self, band):
        super(Dueling, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(band, 2 * band),
            nn.ReLU(),
            nn.Linear(2 * band, 2 * band),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(2 * band, 2 * band),
            nn.ReLU(),
            nn.Linear(2 * band, 1),
            # nn.ReLU()
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(2 * band, 2 * band),
            nn.ReLU(),
            nn.Linear(2 * band, band),
            # nn.ReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):                                         
                m.weight.data.uniform_(-0.1, 0.1)
                # m.weight.data.normal_(0, 0.02)
    
    def forward(self, state):
        features = self.feature_layer(state)                             
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_value = values + (advantages - advantages.mean())
        return torch.sigmoid(q_value)

class HsiAgent(object):
    def __init__(self, band, learning_rate, gamma, memory_size, target_replace_iter, batch_size):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.band = band
        self.gamma = gamma
        self.memory_size = memory_size
        self.target_replace_iter = target_replace_iter
        self.batch_size = batch_size
        self.memory_index = 0
        self.learn_counter = 0
        self.policy_net = Dueling(band).to(self.device)
        self.target_net = Dueling(band).to(self.device)
        # summary(self.policy_net, (1, band))
        self.memory = np.zeros((memory_size, band * 2 + 2))
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        # self.loss = nn.MSELoss()
        self.loss = nn.SmoothL1Loss()
   
    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)
        pred = self.policy_net.forward(state)
        pred = (1 - state) * pred
        if self.device == 'cpu':
            action = torch.max(pred, 1)[1].detach().numpy()[0]
        else:
            action = torch.max(pred, 1)[1].detach().cpu().numpy()[0]
        return action
    
    def store_transition(self, state, action, reward, state_):
        transition = np.hstack((state, [action, reward], state_)) 
        index = self.memory_index % self.memory_size
        self.memory[index] = transition
        self.memory_index += 1
    
    def learn(self):
        if self.learn_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_counter += 1

        if self.memory_index < self.memory_size:
            sample_index = np.random.choice(self.memory_index, self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, self.batch_size)
        b_memory = self.memory[sample_index]
        b_state = torch.FloatTensor(b_memory[:, :self.band]).to(self.device)
        b_action = torch.LongTensor(b_memory[:, self.band:self.band + 1].astype(int)).to(self.device)
        b_reward = torch.FloatTensor(b_memory[:, self.band + 1:self.band + 2]).to(self.device)
        b_state_ = torch.FloatTensor(b_memory[:, -self.band:]).to(self.device)
        # b_reward = F.normalize(b_reward)

        q_eval = self.policy_net(b_state).gather(1, b_action)
        # q_next = self.target_net(b_state_).max(1)[0].view(-1, 1)
        q_next_idx = self.policy_net(b_state_).argmax(1).unsqueeze(1)                                             
        q_next = self.target_net(b_state_).detach().gather(1, q_next_idx)
        q_target = b_reward + self.gamma * q_next

        loss = self.loss(q_eval, q_target)
        q = q_target.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()  

        return q, loss

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
    
    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))