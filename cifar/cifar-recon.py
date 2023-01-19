#pip install git+https://github.com/facebookresearch/EGG.git

import torch
import torchvision
import torchvision.transforms as transforms

from torch import nn
from torch.nn import functional as F
from torch.utils import data

import torch.distributions
import matplotlib.pyplot as plt

import egg.core as core
import numpy as np

import cv2
import sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

opts = core.init(params=['--random_seed=7', 
                         '--n_epochs=50',
                         '--lr=1e-3',
                         '--vocab_size=64',
                         '--batch_size=1024'])
opts.recon_error = True if sys.argv[1].lower() == 'true' else False

class Sample(data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset.data = (torch.tensor(self.dataset.data).permute(0,3,1,2).type(torch.float32)) / 255
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

class Conv(nn.Sequential):
    def __init__(self, c):
        super().__init__()
        for x, (i, o) in enumerate(zip(c[:-1], c[1:])):
            self.add_module(f"{x}", nn.Conv2d(i, o, kernel_size=3, stride=2, padding=1))
            self.add_module(f"{x}bn", nn.BatchNorm2d(o))
            self.add_module(f"{x}ac", nn.LeakyReLU())
        self.add_module(f"flatten", nn.Flatten())

class Linear(nn.Sequential):
    def __init__(self, *d):
        super().__init__()
        for x, (i, o) in enumerate(zip(d[:-1], d[1:])):
            self.add_module(f"{x}", nn.Linear(i, o))
            self.add_module(f"{x}bn", nn.BatchNorm1d(o))
            self.add_module(f"{x}ac", nn.LeakyReLU())

###################################### GAME ######################################

class Eyes(nn.Sequential):
    def __init__(self, ndim=128):
        super().__init__()
        self.ndim = ndim

        self.add_module("conv", Conv([3,8,64]))
        self.add_module("fc", Linear(64*64, ndim))

class Draw(nn.Module):
    def __init__(self, ndim=128):
        super().__init__()
        self.ndim = ndim
        
        self.fc = Linear(ndim, 64*64)
        self.conv1 = nn.ConvTranspose2d(64, 8, 4,2,1)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv1_ac = nn.LeakyReLU()
        self.conv2 = nn.ConvTranspose2d(8, 3, 4,2,1)
        self.conv2_bn = nn.BatchNorm2d(3)
        self.conv2_ac = nn.LeakyReLU()

    def forward(self, z):
        x = self.fc(z).reshape(-1, 64, 8, 8)
        x = self.conv1_ac(self.conv1_bn(self.conv1(x)))
        x = self.conv2_bn(self.conv2(x))
        return x

class Sender(nn.Module):
    def __init__(self, eyes, ndim=100):
        super().__init__()
        self.eyes = eyes
        self.ndim = ndim

        self.draw = Draw(ndim)

    def forward(self, x, _=None):
        z = self.eyes(x)
        z = self.draw(z)
        return z

class Receiver(nn.Module):
    def __init__(self, eyes, ndim=100):
        super().__init__()
        self.eyes = eyes
        self.ndim = ndim
        
        self.news = Eyes(opts.vocab_size)
    
    def forward(self, x, y, _=None):
        x = x + torch.rand(x.shape).to(torch.device("cuda:0")) * 1e-1
        z = self.news(x)[:, None, :]
        zs = self.eyes(y.view(-1, 3, 32, 32)).view(y.shape[0], y.shape[1], -1)
        return -torch.mean((zs - z)**2, dim=-1)
    
def loss(sender_input, message, _receiver_input, receiver_output, labels, _aux_input):
    cel = F.cross_entropy(receiver_output, labels)
    if opts.recon_error:
        cel += 1000* ((message - sender_input)**2).mean()
    acc = (labels == receiver_output.argmax(dim=1)).float().mean()
    return cel, { "acc": acc.unsqueeze(0) }

# PRETRAIN

device = torch.device("cuda:0")
eyes = Eyes(opts.vocab_size)
eyes.to(device)


sender = Sender(eyes, opts.vocab_size)
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

_, _, images = next(iter(validation_loader))
with torch.no_grad():
    generated = sender(images[:10].view(-1, 3, 32, 32).to(device)).view(10, 10, 3, 32, 32)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 8))
    for i in range(0, 100):
        fig.add_subplot(10, 10, i+1)
        image = generated[i//10, i%10]
        t = image
        #t = image[0]*0.299 + image[1]*0.587 + image[2]*0.114
        t[0] = (t[0] - t[0].min())/(t[0].max() - t[0].min())
        t[1] = (t[1] - t[1].min())/(t[1].max() - t[1].min())
        t[2] = (t[2] - t[2].min())/(t[2].max() - t[2].min())
        t = t.permute(1,2,0)
        plt.xticks([])  # 去掉x轴
        plt.yticks([])  # 去掉y轴
        plt.axis('off')  # 去掉坐标轴
        plt.imshow(t.cpu())
    if opts.recon_error:
        name = 'with'
    else:
        name = 'without'
    plt.savefig("./results/" + name + "_recon.jpg")
    
def draw_clustering(X_tsne, name):
    c = np.arange(10).repeat(10).reshape(10, 10).transpose().reshape(100)
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
    plt.show()
    
X_gen2 = sender(images[:10].view(-1, 3, 32, 32).to(device)).view(-1, 3*1024).cpu().detach().numpy()
X_gen2_tsne = TSNE(n_components=2,random_state=33).fit_transform(X_gen2)
draw_clustering(X_gen2_tsne, 'Message_' + name + '_recon')

X_origin = images[:10].view(-1, 3*1024).cpu()
X_origin_tsne = TSNE(n_components=2,random_state=33).fit_transform(X_origin)
draw_clustering(X_origin_tsne, 'Original Images')
