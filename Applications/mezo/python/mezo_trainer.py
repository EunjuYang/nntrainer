import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import random
import numpy as np

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MNIST_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

class MeZO:
    def __init__(self, model, lr=1e-5, epsilon=1e-3, base_seed=42):
        self.model = model
        self.lr = lr
        self.epsilon = epsilon
        self.base_seed = base_seed
        self.t = 0 # Step counter

    @torch.no_grad()
    def _perturb_parameters(self, seed, scale):
        """Subroutine: PerturbParameters(θ, ϵ, s)"""
        torch.manual_seed(seed)
        for p in self.model.parameters():
            # Generate z ~ N(0, 1) for each parameter
            z = torch.randn_like(p)
            p.add_(z, alpha=scale)

    @torch.no_grad()
    def step(self, x, y, criterion):
        # 1. Sample random seed s for this timestep t
        current_s = self.base_seed + self.t

        # 2. θ ← PerturbParameters(θ, ϵ, s)
        self._perturb_parameters(current_s, self.epsilon)
        loss_plus = criterion(self.model(x), y)

        # 3. θ ← PerturbParameters(θ, -2ϵ, s)
        self._perturb_parameters(current_s, -2 * self.epsilon)
        loss_minus = criterion(self.model(x), y)

        # 4. θ ← PerturbParameters(θ, ϵ, s) ▷ Reset parameters
        self._perturb_parameters(current_s, self.epsilon)

        # 5. projected_grad ← (ℓ+ − ℓ−)/(2ϵ)
        projected_grad = (loss_plus - loss_minus) / (2 * self.epsilon)

        # 6. Apply descent: θi ← θi − ηt ∗ projected_grad ∗ z
        # Note: We reset the RNG with seed s again inside this call
        self._perturb_parameters(current_s, -self.lr * projected_grad)

        self.t += 1
        return (loss_plus + loss_minus) / 2

def run_training():
    set_global_seed(42)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters per Algorithm 1 logic
    # Without normalization, keep LR low to maintain stability
    EPOCHS = 100
    BATCH_SIZE = 64
    LR = 1e-5 
    EPSILON = 1e-3 

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), 
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), 
                             batch_size=BATCH_SIZE)

    model = MNIST_MLP().to(DEVICE)
    optimizer = MeZO(model, lr=LR, epsilon=EPSILON, base_seed=42)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = optimizer.step(x, y, criterion)
            total_loss += loss.item()
            pbar.set_postfix({"L": f"{loss.item():.3f}"})

        # Eval
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(1) == y).sum().item()
        
        print(f"End Epoch {epoch+1} | Accuracy: {100 * correct / 10000:.2f}%")

if __name__ == "__main__":
    run_training()