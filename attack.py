
import gym
import torch
import numpy as np
from torchvision import transforms

ENV_NAME = "CartPole-v1"

LEARNING_RATE = 0.001 
GAMMA = 0.99 
BATCH_SIZE = 32 
EPISODES = 1000

ATTACK_METHOD = "FGSM"
EPSILON = 0.01 
ALPHA = 0.001
ITERATIONS = 10 

env = gym.make(ENV_NAME)
n_actions = env.action_space.n
state_shape = env.observation_space.shape

class PolicyNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)


model = PolicyNetwork(state_shape[0], n_actions)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.load_state_dict(torch.load('policy_net.pth'))

def choose_action(state):

  state = torch.tensor(state, dtype=torch.float32).view(1, -1)
  probs = model(state)
  action = torch.multinomial(probs, 1).item()
  
  return action

def generate_adversarial_example(state):

  state = torch.tensor(state, dtype=torch.float32).view(1, -1)
  adv_state = torch.autograd.Variable(state, requires_grad=True)
 
  if ATTACK_METHOD == "FGSM":
   
    loss = torch.nn.functional.cross_entropy(model(state), model(adv_state))
   
    grad = torch.autograd.grad(loss, adv_state)[0]

    adv_state = adv_state + EPSILON * torch.sign(grad)
    
    adv_state = torch.clamp(adv_state, 0, 1)
  elif ATTACK_METHOD == "MI-FGSM":
    
    momentum = torch.zeros_like(adv_state)
   
    for i in range(ITERATIONS):
      
      loss = torch.nn.functional.cross_entropy(model(state), model(adv_state))
      
      grad = torch.autograd.grad(loss, adv_state)[0]
      
      momentum = GAMMA * momentum + grad / torch.mean(torch.abs(grad))
     
      adv_state = adv_state + EPSILON * torch.sign(momentum)
     
      adv_state = torch.clamp(adv_state, 0, 1)
  elif ATTACK_METHOD == "NI-FGSM":
      
      noise = torch.zeros_like(adv_state)
      
      for i in range(ITERATIONS):
        
        loss = torch.nn.functional.cross_entropy(model(state), model(adv_state))
       
        grad = torch.autograd.grad(loss, adv_state)[0]
       
        noise = GAMMA * noise + grad / torch.mean(torch.abs(grad))
        
        adv_state = adv_state + EPSILON * torch.sign(noise)
      
def clip(x):
    return torch.clamp(x, 0, 1)


def generate_adversarial_example(state):
   
    state = torch.tensor(state, dtype=torch.float32, requires_grad=True)
   
    action = choose_action(state)
    
    log_prob = model(state)[action]
    
    loss = -log_prob
    
    loss.backward()
   
    grad = state.grad.data
   
    adv_state = state + EPSILON * grad.sign()
    
    adv_state = clip(adv_state)
    
    return adv_state

def test_model(adv=False):
   
    total_reward = 0
  
    success_count = 0
    
    for i in range(100):
       
        state, _ = env.reset()
        
        reward = 0
      
        while True:
            
            if adv:
                state = generate_adversarial_example(state)
           
            action = choose_action(state)
            
            next_state, reward, done, _, _ = env.step(action)
           
            total_reward += reward
           
            state = next_state
          
            if done:
                break
     
        if total_reward >= 195:
          
            success_count += 1
 
    avg_reward = total_reward / 100
    
    success_rate = success_count / 100
    
    return avg_reward, success_rate


epsilons = [0.01, 0.02, 0.03, 0.04, 0.05]

for epsilon in epsilons:

  print(f"Epsilon: {epsilon}")
  
  EPSILON = epsilon

  normal_reward, normal_rate = test_model(adv=False)

  print(f"Normal Reward: {normal_reward}, Normal Rate: {normal_rate}")

  adv_reward, adv_rate = test_model(adv=True)
 
  print(f"Adversarial Reward: {adv_reward}, Adversarial Rate: {adv_rate}")

  print()
