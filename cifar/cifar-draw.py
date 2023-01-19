#pip install git+https://github.com/facebookresearch/EGG.git

import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils import data

import torch.distributions
import matplotlib.pyplot as plt

import egg.core as core
from utils import *
from draw import *

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

opts = core.init(params=['--random_seed=7', 
                         '--n_epochs=70',
                         '--lr=1e-4',
                         '--vocab_size=64',
                         '--batch_size=1024'])

class Sample(data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset.data = (torch.tensor(self.dataset.data).permute(0,3,1,2).type(torch.float32)) / 255
        #self.dataset.data = self.dataset.data[:,0:1]
        self.subsets = { name: dataset.data[torch.tensor(dataset.targets) == label]
                     for name, label in dataset.class_to_idx.items()}
        print(dataset.class_to_idx.items())

    def __len__(self):
        return min(map(len, self.subsets.values()))

    def __getitem__(self, idx):
        i = torch.randint(len(self), (len(self.subsets), ))
        y = torch.stack([self.subsets[name][id]
                                  for name, id in zip(self.subsets, i)]) \
                 .type(torch.FloatTensor)
        
        label = idx % len(self.subsets)
        return y[label], label, y

kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader = data.DataLoader(
    Sample(torchvision.datasets.CIFAR10('./data', train=True, download=True)),
    batch_size=opts.batch_size, **kwargs)

validation_loader = data.DataLoader(
    Sample(torchvision.datasets.CIFAR10('./data', train=False, download=False)),
    batch_size=opts.batch_size, **kwargs)

###################################### GAME ######################################

class Eyes(nn.Sequential):
    def __init__(self, ndim=128):
        super().__init__()
        self.ndim = ndim

        self.add_module("conv", Conv([3,8,64]))
        self.add_module("fc", Linear(64*64, ndim))

class Sender(nn.Module):
    def __init__(self, eyes, draw, ndim=100):
        super().__init__()
        self.eyes = eyes
        self.ndim = ndim
        
        self.fc = Linear(eyes.ndim, 128, 4 * 2 * 3)
        self.draw = draw

    def forward(self, x, _=None):
        z = self.eyes(x)
        z = self.fc(z)
#         z = z + torch.rand(z.shape).to(torch.device("cuda:0")) * 1e-1
#         return self.draw(z[:, 0:4])+ self.draw(z[:, 4:8])+ self.draw(z[:, 8:12])+ self.draw(z[:, 12:16])
        a1 = torch.max(torch.stack([self.draw(z[:, 0:4]), self.draw(z[:, 4:8])], dim=-1), dim=-1).values
        a2 = torch.max(torch.stack([self.draw(z[:, 8:12]), self.draw(z[:, 12:16])], dim=-1), dim=-1).values
        a3 = torch.max(torch.stack([self.draw(z[:, 16:20]), self.draw(z[:, 20:24])], dim=-1), dim=-1).values
        return torch.cat((a1,a2,a3),dim = 1)

class Receiver(nn.Module):
    def __init__(self, eyes, ndim=100):
        super().__init__()
        self.eyes = eyes
        self.ndim = ndim
        
        self.news = Eyes(opts.vocab_size)
    
    def forward(self, x, y, _=None):
        #x = x + torch.rand(x.shape).to(torch.device("cuda:0")) * 1e-1
        z = self.news(x)[:, None, :]
        zs = self.eyes(y.view(-1, 3, 32, 32)).view(y.shape[0], y.shape[1], -1)
        return -torch.mean((zs - z)**2, dim=-1)

# PRETRAIN

device = torch.device("cuda:0")
eyes = Eyes(opts.vocab_size)
eyes.to(device)

def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
    cel = F.cross_entropy(receiver_output, labels)
    acc = (labels == receiver_output.argmax(dim=1)).float().mean()
    return cel, { "acc": acc.unsqueeze(0) }


vae = VAE(4)
vae.to(device)
vae.load_state_dict(torch.load("draw_cifar.pth"))
draw = vae.d
for param in draw.parameters():
    param.require_grad = False

sender = Sender(eyes, draw, opts.vocab_size)
receiver = Receiver(eyes, opts.vocab_size)

game = core.SymbolGameGS(sender, receiver, loss)
optimizer = core.build_optimizer(game.parameters())

trainer = core.Trainer(game=game, optimizer=optimizer, 
                       train_data=train_loader,
                       validation_data=validation_loader,
                       callbacks=[
                           core.ConsoleLogger(as_json=True, print_train_loss=True),
                       ])

trainer.train(n_epochs=opts.n_epochs)

# 让我康康
''''''
_, _, images = next(iter(validation_loader))

plt.clf()
with torch.no_grad():
    generated = sender(images[:10].view(-1, 3, 32, 32).to(device)).view(10, 10, 3, 32, 32)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 8))
    for i in range(0, 100):
        fig.add_subplot(10, 10, i+1)
        image = generated[i//10, i%10]
        plt.xticks([])  # 去掉x轴
        plt.yticks([])  # 去掉y轴
        plt.axis('off')  # 去掉坐标轴
        t = torch.max(torch.stack([image[0],image[1],image[2]], dim=-1), dim=-1).values
        plt.imshow(t.cpu())
    plt.savefig("./results/cifar-draw.jpg")

def draw_clustering(X_tsne, name):
    c = [i%10 for i in range(100)]    
    plt.figure(figsize=(6, 5))
    plt.clf()
    fs = 14
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=c, s=15)
#     plt.xlabel('axis 1', fontsize = fs)
#     plt.ylabel('axis 2', fontsize = fs)
    plt.tick_params(labelsize=fs)
#     plt.title(f"T-SNE of {name}", fontsize = fs)
    plt.grid(True)
    plt.savefig(f'./t-SNE/cifar.{name}.png')
    # plt.legend()
    plt.show()
    
with torch.no_grad():
    X_gen = sender(images[:10].view(-1, 3, 32, 32).to(device)).view(100, -1).cpu().detach().numpy()
X_tsne = TSNE(n_components=2, random_state=42, method = 'exact').fit_transform(X_gen)
draw_clustering(X_tsne, "Message from Draw")

X_origin = images[:10].view(-1, 3*1024).cpu()
X_origin_tsne = TSNE(n_components=2,random_state=33).fit_transform(X_origin)
draw_clustering(X_origin_tsne, 'Original Images')
