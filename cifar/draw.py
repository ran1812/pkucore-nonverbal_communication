import torch
import torch.nn as nn
import torch.nn.functional as F

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

class E(nn.Module):
    
    def __init__(self, nz):
        super().__init__()

#         self.conv1 = nn.Conv2d(1, 16, 3,2,1)
#         self.conv2 = nn.Conv2d(16, 1, 3,2,1)
#         self.fc = Linear(49, 128, nz)
        self.fc = Linear(1024, 512)
        self.fc2 = nn.Linear(512, nz)

    def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         z = self.fc(x.flatten(1))
#         return z
        return self.fc2(self.fc(x.view(-1, 1024)))

class D(nn.Module):
    
    def __init__(self, nz):
        super().__init__()

#         self.fc = Linear(nz, 128, 49)
#         self.conv1 = nn.ConvTranspose2d(1, 16, 4,2,1)
#         self.conv2 = nn.ConvTranspose2d(16, 1, 4,2,1)
        self.fc = Linear(nz, 512)
        self.fc2 = nn.Linear(512, 1024)

    def forward(self, z):
#         x = self.fc(z).reshape(-1, 1, 7, 7)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         return F.sigmoid(x)
        return torch.sigmoid(self.fc2(self.fc(z)).view(-1, 1, 32, 32))

class VAE(nn.Sequential):
    
    def __init__(self, nz):
        super().__init__()
        self.e = E(nz)
        self.d = D(nz)

import argparse
import matplotlib.pyplot as plt
from torch.utils import data
import torchvision
parser = argparse.ArgumentParser()
parser.add_argument("--nz", type=int, default=4)
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--batch", type=int, default=2048)
parser.add_argument("--load", type=bool, default=True)
args = parser.parse_args()

device = torch.device("cuda:0")

model = VAE(args.nz)
model.to(device)

if args.load:
    def draw(imgs):
        fig = plt.figure()
        for i, img in enumerate(imgs):
            fig.add_subplot(len(imgs), 1, i+1)
            plt.imshow(img)
        plt.savefig("draw.jpg")
    
    def gen(n):
        with torch.no_grad():
            x = model.d(torch.rand((n, 4)).to(device))
            draw(x.cpu().squeeze())
            
    class Sample(data.Dataset):
        def __init__(self, dataset):
            super().__init__()
            self.dataset = dataset
            self.subsets = { name: dataset.data[dataset.targets == label]
                         for name, label in dataset.class_to_idx.items()}

        def __len__(self):
            return min(map(len, self.subsets.values()))

        def __getitem__(self, idx):
            i = torch.randint(len(self), (len(self.subsets), ))
            y = torch.stack([self.subsets[name][id]
                                      for name, id in zip(self.subsets, i)]) \
                     .type(torch.FloatTensor).unsqueeze(1)

            label = idx % len(self.subsets)
            return y[label], label, y

else:
    from tqdm import tqdm
    
    import numpy as np
    dataset=np.zeros((100000, 32, 32), np.uint8)
    for id in tqdm(range(100000)):
        
        import cv2
        x1, y1, x2, y2 = np.random.randint(0, 31, (4, ))
        cv2.line(dataset[id], (x1, y1), (x2, y2), (1, ))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loader = torch.utils.data.DataLoader(
        dataset=torch.from_numpy(dataset).to(torch.float).view(-1, 1, 32, 32),
        batch_size=args.batch,
        shuffle=True,
    )
    
    for _ in tqdm(range(args.epoch)):

        mse = 0
        
        for x in loader:
            x = x.to(device)
            pred = model(x)

            optimizer.zero_grad()
            loss = torch.mean((x-pred)**2)
            loss.backward()
            optimizer.step()
            
            mse += loss.item()
        
        print(mse)

    torch.save(model.state_dict(), "draw_cifar.pth")