import torch
import numpy as np
import torch.nn as nn
from PIL import Image

# --- 1. DEFINE MODEL ---
class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU())
        self.b = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d2 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU())
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

# --- 2. CREATE TEST SAMPLE ---
def create_test_sample():
    img = np.zeros((224, 224), dtype=np.float32)
    # Donut
    cx, cy = 112, 112
    r = 40
    y, x = np.ogrid[:224, :224]
    mask_ring = ((x - cx)**2 + (y - cy)**2 <= r**2) & ((x - cx)**2 + (y - cy)**2 >= (r-5)**2)
    img[mask_ring] = 1.0
    # Scratch
    for i in range(60):
        img[50+i, 50+i] = 1.0
    # Noise
    noise = np.random.normal(0, 0.2, (224, 224))
    img = img + noise
    return img

# --- 3. RUN INFERENCE ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MiniUNet().to(device)
try:
    model.load_state_dict(torch.load("unet_wafer_segmenter.pth"))
    print(">>> Model loaded successfully!")
except:
    print(">>> ERROR: Could not load model.")
    exit()

model.eval()

raw_img = create_test_sample()
input_tensor = torch.tensor(raw_img).unsqueeze(0).unsqueeze(0).float().to(device)

with torch.no_grad():
    output = model(input_tensor)
    pred_mask = torch.sigmoid(output).cpu().numpy().squeeze()

# --- 4. SAVE USING PIL (No Matplotlib) ---
def array_to_img(arr):
    # Fix: Ensure array is float before doing math
    arr = arr.astype(np.float32)
    
    # Avoid division by zero if the image is perfectly solid color
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
    else:
        arr = arr * 255 # Just scale it up if it's constant
        
    return Image.fromarray(arr.astype(np.uint8))

# Convert Input (Grayscale)
img_input = array_to_img(raw_img)

# Convert Prediction (Boolean -> Float -> Image)
# This .astype(np.float32) fixes the error you just saw
img_pred = array_to_img((pred_mask > 0.5).astype(np.float32)) 

# Create a Combined Image (Side by Side)
total_width = img_input.width + img_pred.width
combined = Image.new('L', (total_width, img_input.height))
combined.paste(img_input, (0, 0))
combined.paste(img_pred, (img_input.width, 0))

combined.save("segmentation_result.png")
print(">>> Success! Saved 'segmentation_result.png' (Left: Input, Right: Prediction)")