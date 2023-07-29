#!/usr/bin/env python
# coding: utf-8

# In[1]:


# AI for Autonomous Vehicles - Build a Self-Driving Car

# Importing the libraries
import gym
import pygame
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q-Learning

class Dqn(object):
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(capacity = 100000)
        self.optimizer = optim.Adam(params = self.model.parameters())
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state))*100)
        action = probs.multinomial(len(probs))
        return action.data[0,0]
    
    def learn(self, batch_states, batch_actions, batch_rewards, batch_next_states):
        batch_outputs = self.model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
        batch_next_outputs = self.model(batch_next_states).detach().max(1)[0]
        batch_targets = batch_rewards + self.gamma * batch_next_outputs
        td_loss = F.smooth_l1_loss(batch_outputs, batch_targets)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
    
    def update(self, new_state, new_reward):
        new_state = torch.Tensor(new_state).float().unsqueeze(0)
        self.memory.push((self.last_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]), new_state))
        new_action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_states, batch_actions, batch_rewards, batch_next_states = self.memory.sample(100)
            self.learn(batch_states, batch_actions, batch_rewards, batch_next_states)
        self.last_state = new_state
        self.last_action = new_action
        self.last_reward = new_reward
        return new_action.item()
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")



# In[2]:


args= {'gamma': 0.9,
      'batch_size': 32,
      'input_size': 2,
      'n_action': 3}


# In[3]:


env=gym.make('MountainCar-v0', render_mode='human')


# In[4]:


print('State space: ', env.observation_space)
print(env.observation_space.low)
print(env.observation_space.high)


# In[5]:


print('State space: ', env.action_space)


# In[6]:


state, _=env.reset()
print(state)
reward = 0


# In[7]:


brain = Dqn(2,3,0.9)


# In[8]:


action = brain.update(state, reward)
print(action)


# In[9]:


class GameRunner:
    def __init__(self,model,env,render = False):
        self.env = env
        self.model = model
        self.render = render
        self.steps = 0
        self.list_steps = []
        self.reward_store = []
        self.max_x_store = []
    def run(self):
        state, _ = self.env.reset()
        reward = 0
        action = self.model.update(state, reward)
        tot_reward = 0
        max_x = -100
        self.list_steps = []
        while True:
            if self.render:
                self.env.render()
            self.steps +=1
            next_state, reward, done, info, _ = env.step(action)
            if next_state[0] >=-0.25:
                reward +=1
            elif next_state[0] >= 0.1:
                reward +=1
            elif next_state[0] >= 0.25:
                reward +=1
            elif next_state[0] >= 0.5:
                reward +=200
            
            action = self.model.update(next_state, reward)
            
            if next_state[0] > max_x:
                max_x = next_state[0]
                
            if max_x > 0.5:
                print("You Win")
                
            if done or self.steps > 1000:
                next_state = None
                
                
            state = next_state
            tot_reward += reward
            
            
            if done or self.steps > 200:
                self.reward_store.append(tot_reward)
                self.max_x_store.append(max_x)
                self.list_steps.append(self.steps)
                print("Step {}, Total reward: {}, Max: ".format(self.steps, tot_reward, max_x))
                if self.steps < 180:
                    self.model.save()
                self.steps = 0
                break


# In[ ]:


steps=[]
env = gym.make('MountainCar-v0',render_mode='human')
gr = GameRunner(model = brain, env=env, render=False)
for i in range(20000):
    gr.run()


# In[ ]:


plt.plot(gr.max_x_store)


# In[ ]:


plt.plot(gr.list_steps)


# In[ ]:


brain.load()
env = gym.make('MountainCar-v0', render_model='human')
gr = GamerRunner(model=brain, env = env, render=False)
for i in range(10):
    gr.run


# In[ ]:


gr.run()


# In[ ]:




