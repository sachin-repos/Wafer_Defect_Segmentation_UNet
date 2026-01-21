import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
BATCH_SIZE = 16
EPOCHS = 5  # Reduced because we generate fresh data every batch
STEPS_PER_EPOCH = 100 # How many batches per epoch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f">>> [SYSTEM] Nuclear Option Activated. Device: {DEVICE}")

# --- 1. MINI U-NET (Lighter/Faster) ---
class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.e1 = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU())
        
        # Bottleneck
        self.b = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU()) # 128 because cat(64, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d2 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU()) # 64 because cat(32, 32)
        
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        p1 = self.pool(e1)
        e2 = self.e2(p1)
        p2 = self.pool(e2)
        
        b = self.b(p2)
        
        u1 = self.up1(b)
        u1 = torch.cat((u1, e2), dim=1)
        d1 = self.d1(u1)
        
        u2 = self.up2(d1)
        u2 = torch.cat((u2, e1), dim=1)
        d2 = self.d2(u2)
        
        return self.final(d2)

# --- 2. ON-THE-FLY GENERATOR (No Disk I/O) ---
def get_batch(batch_size):
    images = []
    masks = []
    
    for _ in range(batch_size):
        # Create empty 224x224
        img = np.zeros((224, 224), dtype=np.float32)
        mask = np.zeros((224, 224), dtype=np.float32)
        
        # Draw Random Line (Scratch)
        x1, y1 = np.random.randint(20, 200, 2)
        x2, y2 = np.random.randint(20, 200, 2)
        
        # Simple line drawing algorithm (Bresenham-ish)
        # We cheat and just draw a few points to simulate a scratch
        # purely for speed
        t = np.linspace(0, 1, 50)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        for ix, iy in zip(x, y):
            try:
                img[int(iy), int(ix)] = 1.0
                mask[int(iy), int(ix)] = 1.0
                # Make it thicker
                img[int(iy)+1, int(ix)] = 1.0
                mask[int(iy)+1, int(ix)] = 1.0
            except: pass
            
        # Add Noise
        noise = np.random.normal(0, 0.1, (224, 224))
        img = img + noise
        
        images.append(img)
        masks.append(mask)
    
    # Convert to Tensor [B, 1, 224, 224]
    tx = torch.tensor(np.array(images)).unsqueeze(1).float()
    ty = torch.tensor(np.array(masks)).unsqueeze(1).float()
    return tx, ty

# --- 3. TRAINING LOOP ---
model = MiniUNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

print(">>> Starting High-Speed Training...")

for epoch in range(EPOCHS):
    epoch_loss = 0
    loop = tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch+1}")
    
    for _ in loop:
        # Generate Data INSTANTLY
        images, masks = get_batch(BATCH_SIZE)
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

# Save
torch.save(model.state_dict(), "unet_wafer_segmenter.pth")
print(">>> DONE. Saved 'unet_wafer_segmenter.pth'")