import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import random


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EPSILONS = [0, .05, .1, .15, .2, .25, .3]  
PRETRAINED_MODEL = "checkpoint/lenet_mnist_model.pth"  
BATCH_SIZE = 1  
NUM_WORKERS = 4  
LEARNING_RATE = 0.01  
NUM_EPOCHS = 100  
TARGET = 7  
C = 0.1  


transform = transforms.Compose([transforms.ToTensor()])
test_dataset = torchvision.datasets.MNIST(root='./datasets', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = LeNet().to(device)
model.load_state_dict(torch.load(PRETRAINED_MODEL, map_location='cpu'))


model.eval()


def fgsm_attack(image, epsilon, data_grad):
  
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
  
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image


def mi_fgsm_attack(image, epsilon, data_grad, decay_factor, g):
  
    sign_data_grad = data_grad.sign()
    g = decay_factor * g + sign_data_grad
    perturbed_image = image + epsilon * g
  
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image, g

# 定义 NI-FGSM 攻击函数
def ni_fgsm_attack(image, epsilon, data_grad, decay_factor, g):
    
    sign_data_grad = data_grad.sign()
    g = decay_factor * g + sign_data_grad / torch.norm(sign_data_grad, p=1)
    perturbed_image = image + epsilon * g
   
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
  
    return perturbed_image, g

# 定义 JSMA 攻击函数
def jsma_attack(image, target, model, theta, gamma):
   
    image = torch.tensor(image, dtype=torch.float32, requires_grad=True)
    
    adv_jsma = image.clone()

    n, c, h, w = image.shape
   
    pixels = [(i, j) for i in range(h) for j in range(w)]
    
    random.shuffle(pixels)
    
    target_prob = model(adv_jsma)[0, target]
    
    stop = False
   
    for i, j in pixels:
     
        if target_prob > gamma or torch.norm(adv_jsma - image) > theta or stop:
            break
       
        for k in range(c):
            
            original_value = adv_jsma[0, k, i, j].item()
            
            adv_jsma[0, k, i, j] = original_value + theta
         
            adv_jsma.requires_grad = True
            out = model(adv_jsma)
            target_grad = torch.autograd.grad(out[0, target], adv_jsma, retain_graph=True)[0][0, k, i, j].item()
            other_grad = torch.autograd.grad(torch.sum(out[0, :target] + out[0, target + 1:]), adv_jsma)[0][0, k, i, j].item()
         
            adv_jsma[0, k, i, j] = original_value
           
            saliency = target_grad * other_grad
           
            if saliency > 0:
              
                adv_jsma[0, k, i, j] = torch.clamp(original_value + theta, 0, 1)
            
                target_prob = model(adv_jsma)[0, target]
              
                if target_prob == 1:
                    stop = True
             
                break
 
    return adv_jsma


def simple_black_box_attack(image, epsilon):
    perturbed_image = image.clone()
    num_pixels = image.shape[-1]
    pixels_to_modify = np.random.choice(num_pixels, int(epsilon * num_pixels), replace=False)
    
    for pixel in pixels_to_modify:
        perturbed_image[0, 0, pixel // 28, pixel % 28] = torch.rand(1)
    
    return torch.clamp(perturbed_image, 0, 1)


for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    data.requires_grad = True

   
    output = model(data)
    loss = F.nll_loss(output, target)
    model.zero_grad()
    loss.backward()

  
    perturbed_data = fgsm_attack(data, 0.1, data.grad.data)


    g = torch.zeros_like(data.grad.data)
    perturbed_data_mi, g = mi_fgsm_attack(data, 0.1, data.grad.data, 0.9, g)

    
    g = torch.zeros_like(data.grad.data)
    perturbed_data_ni, g = ni_fgsm_attack(data, 0.1, data.grad.data, 0.9, g)

   
    perturbed_data_jsma = jsma_attack(data, TARGET, model, 0.1, 0.8)


    perturbed_data_simple_black_box = simple_black_box_attack(data, 0.1)
