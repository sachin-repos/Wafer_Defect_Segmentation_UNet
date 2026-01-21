import numpy as np
import os
from tqdm import tqdm
from PIL import Image, ImageDraw

# Configuration
IMG_SIZE = 224
CENTER = int(IMG_SIZE / 2)
WAFER_RADIUS = int(IMG_SIZE / 2) - 2 
NUM_SAMPLES = 2000
OUTPUT_DIR = "synthetic_data"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")

print(f">>> creating directories at: {os.path.abspath(OUTPUT_DIR)}")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

def create_wafer_with_defect(index):
    # 1. Background setup
    img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=0)
    mask = Image.new('L', (IMG_SIZE, IMG_SIZE), color=0)
    
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)
    
    # Draw the Wafer Disk (Base Silicon) - distinct from background
    bbox_wafer = [0, 0, IMG_SIZE, IMG_SIZE]
    draw_img.ellipse(bbox_wafer, fill=30) 
    
    # Randomly choose defect type (0 to 7)
    defect_type = np.random.randint(0, 8)
    
    # --- 0. SCRATCH (Line) ---
    if defect_type == 0:
        x1, y1 = np.random.randint(20, 200, 2)
        x2, y2 = np.random.randint(20, 200, 2)
        width = int(np.random.randint(1, 3))
        draw_img.line([(x1, y1), (x2, y2)], fill=255, width=width)
        draw_mask.line([(x1, y1), (x2, y2)], fill=255, width=width)
        
    # --- 1. DONUT (Center Ring) ---
    elif defect_type == 1:
        cx, cy = np.random.randint(80, 140, 2)
        r = int(np.random.randint(20, 40))
        bbox = [cx-r, cy-r, cx+r, cy+r]
        draw_img.ellipse(bbox, outline=255, width=2)
        draw_mask.ellipse(bbox, outline=255, width=2)
        
    # --- 2. LOC (Random Blob) ---
    elif defect_type == 2:
        cx, cy = np.random.randint(50, 170, 2)
        r = int(np.random.randint(5, 15))
        bbox = [cx-r, cy-r, cx+r, cy+r]
        draw_img.ellipse(bbox, fill=255)
        draw_mask.ellipse(bbox, fill=255)

    # --- 3. EDGE-RING (Critical yield killer) ---
    elif defect_type == 3:
        r = int(np.random.randint(WAFER_RADIUS - 15, WAFER_RADIUS - 5))
        bbox = [CENTER-r, CENTER-r, CENTER+r, CENTER+r]
        draw_img.ellipse(bbox, outline=255, width=3)
        draw_mask.ellipse(bbox, outline=255, width=3)

    # --- 4. EDGE-LOC (Blob near edge) ---
    elif defect_type == 4:
        theta = np.random.uniform(0, 2*np.pi)
        r_pos = np.random.randint(WAFER_RADIUS - 25, WAFER_RADIUS - 5)
        cx = int(CENTER + r_pos * np.cos(theta))
        cy = int(CENTER + r_pos * np.sin(theta))
        blob_r = int(np.random.randint(8, 15))
        bbox = [cx-blob_r, cy-blob_r, cx+blob_r, cy+blob_r]
        draw_img.ellipse(bbox, fill=255)
        draw_mask.ellipse(bbox, fill=255)

    # --- 5. CENTER (Blob at absolute center) ---
    elif defect_type == 5:
        r = int(np.random.randint(15, 30))
        bbox = [CENTER-r, CENTER-r, CENTER+r, CENTER+r]
        draw_img.ellipse(bbox, fill=255)
        draw_mask.ellipse(bbox, fill=255)

    # --- 6. RANDOM (Scattered dots) ---
    elif defect_type == 6:
        # Generate 10-20 small random dots
        num_dots = np.random.randint(10, 20)
        for _ in range(num_dots):
            dx, dy = np.random.randint(20, 200, 2)
            dr = np.random.randint(1, 4) # Tiny dots
            bbox = [dx-dr, dy-dr, dx+dr, dy+dr]
            draw_img.ellipse(bbox, fill=255)
            draw_mask.ellipse(bbox, fill=255)

    # --- 7. NEAR-FULL (Massive coverage) ---
    elif defect_type == 7:
        # A huge blob that covers almost everything
        r = int(WAFER_RADIUS * 0.85) # 85% of wafer size
        # Slightly offset center so it looks organic
        offset_x = np.random.randint(-10, 10)
        offset_y = np.random.randint(-10, 10)
        bbox = [CENTER+offset_x-r, CENTER+offset_y-r, CENTER+offset_x+r, CENTER+offset_y+r]
        draw_img.ellipse(bbox, fill=255)
        draw_mask.ellipse(bbox, fill=255)
    
    # 2. Add HEAVY Noise (Simulate real dirty data)
    img_np = np.array(img)
    noise = np.random.normal(0, 40, img_np.shape) 
    s_and_p = np.random.rand(*img_np.shape)
    
    img_noisy_np = img_np + noise
    img_noisy_np[s_and_p > 0.98] = 200 
    
    img_noisy_np = np.clip(img_noisy_np, 0, 255).astype(np.uint8)
    img_final = Image.fromarray(img_noisy_np)
    
    # 3. Save
    img_final.save(os.path.join(IMG_DIR, f"sample_{index}.png"))
    mask.save(os.path.join(MASK_DIR, f"sample_{index}.png"))

print(">>> Generating Complete 8-Class Synthetic Data...")
for i in tqdm(range(NUM_SAMPLES)):
    create_wafer_with_defect(i)

print(">>> Done! You now have a dataset covering the full WM-811K spectrum.")