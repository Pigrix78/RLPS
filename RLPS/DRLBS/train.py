import re
from util import *
from agent import HsiAgent
import numpy as np
import random
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import train_test_split
import torch
# from tensorboardX import SummaryWriter   
# writer = SummaryWriter(path + '/log')

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def rlTrain(n_select_feature, learning_rate=learning_rate, memory_index=memory_size, batch_size=batch_size, gamma=gamma):
    bs = batch_size
    dqn = HsiAgent(band, learning_rate, gamma, memory_size, target_replace_iter, bs)
    # dqn.load_model(model_path)

    epsilon = 1
    epsilon_decay = 0.995
    num_episodes = 1000

    path = './' + flag + '/' + str(n_select_feature)
    model_path = path + '/model.pth'

    if not os.path.exists(path):
        os.makedirs(path)

    pbar = tqdm(range(num_episodes))
    for episode in pbar:
        state = np.zeros((band, ))
        pbar.set_description("%s" % (epsilon))
        
        idx = []
        for i in range(n_select_feature):
            if random.random() > epsilon:
                action = dqn.choose_action(state)
            else:
                action = np.random.choice(np.where(state <= 0)[0])     

            if i > 0:
                d = np.mean(D[idx, :][:, idx])
                e = MIE[action]

            state_ = state
            state_[action] = 1
            idx.append(action)

            d_ = np.mean(D[idx, :][:, idx])
            e_ = np.mean(MIE[idx])

            if i == 0:
                reward = MIE[action]
            else:
                reward = (d_ - d) * (e_ - e)

            dqn.store_transition(state, action, reward, state_)
            
            state = state_
            
        if dqn.memory_index > batch_size:
            q_target, loss = dqn.learn()     

            epsilon = epsilon * epsilon_decay
            
            # writer.add_scalar('loss', loss, episode)
            # writer.add_scalar('q_target', q_target, episode)

    dqn.save_model(model_path)

    # writer.close()

    feature_idx = []
    state = np.zeros((band, ))

    for t in range(n_select_feature):
        action = dqn.choose_action(state)
        feature_idx.append(action)
        state[action] = 1
        print("{}/{}".format(t, n_select_feature))

    # feature_idx = np.sort(feature_idx)

    print(feature_idx)
    sio.savemat(path + '/feature_idx.mat', {'feature_idx': feature_idx})
    
    return