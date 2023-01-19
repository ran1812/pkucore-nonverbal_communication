from utils import *

class E(nn.Module):
    
    def __init__(self, nz):
        super().__init__()

        self.fc = Linear(784,512)
        self.fc2 = nn.Linear(512, nz)

    def forward(self, x):
        return self.fc2(self.fc(x.view(-1, 784)))

class D(nn.Module):
    
    def __init__(self, nz):
        super().__init__()

        self.fc = Linear(nz, 512)
        self.fc2 = nn.Linear(512, 784)

    def forward(self, z):
        return torch.sigmoid(self.fc2(self.fc(z)).view(-1, 1, 28, 28))

class VAE(nn.Sequential):
    
    def __init__(self, nz):
        super().__init__()
        self.e = E(nz)
        self.d = D(nz)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nz", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--load", type=bool, default=False)
    args = parser.parse_args()

    device = torch.device("cuda:0")

    model = VAE(args.nz)
    model.to(device)

    if args.load:
        model.load_state_dict(torch.load("draw.pth"))

        import matplotlib.pyplot as plt
        def draw(imgs):
            fig, ax = plt.subplots(1, 10)
            for i, img in enumerate(imgs):
                ax[i].imshow(img)
                ax[i].set_xticks([])
                ax[i].set_yticks([])
            plt.savefig("draw.jpg", dpi=300)

        def gen(n):
            with torch.no_grad():
                x = model.d(torch.rand((n, 4)).to(device))
                draw(x.cpu().squeeze())

        import code
        code.InteractiveConsole(locals()).interact()

    else:
        from tqdm import tqdm

        import numpy as np
        dataset=np.zeros((100000, 28, 28), np.uint8)
        for id in tqdm(range(100000)):

            import cv2
            x1, y1, x2, y2 = np.random.randint(0, 27, (4, ))
            cv2.line(dataset[id], (x1, y1), (x2, y2), (1, ), thickness=4)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = torch.utils.data.DataLoader(
            dataset=torch.from_numpy(dataset).to(torch.float).view(-1, 1, 28, 28),
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

        torch.save(model.state_dict(), "draw.pth")