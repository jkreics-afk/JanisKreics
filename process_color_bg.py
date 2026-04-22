import cv2
import numpy as np
from PIL import Image
import io, pathlib

src = pathlib.Path(r"C:\Users\B12\Desktop\CLAUDE_MY") / "Jānis Kreics_019b.jpg"
dst = r"C:\Users\B12\Desktop\CLAUDE_MY\Janis_Kreics_019b_lightbg.jpg"

pil_img = Image.open(io.BytesIO(src.read_bytes()))
img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

h, w = img.shape[:2]

# GrabCut segmentation
rect = (30, 20, w - 60, h - 30)
mask = np.zeros((h, w), np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_RECT)

# Force corners/edges as background
margin = 50
mask[:margin, :] = cv2.GC_BGD
mask[-20:, :] = cv2.GC_BGD
mask[:, :margin] = cv2.GC_BGD
mask[:, -margin:] = cv2.GC_BGD

cv2.grabCut(img, mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)

# Binary foreground mask
fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

# Clean up mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

# Feather edges
alpha = cv2.GaussianBlur(fg_mask.astype(np.float32), (21, 21), 0)[:, :, np.newaxis]

# Target: light warm gray background (#d8d8d8 in BGR)
new_bg_color = np.array([210, 210, 210], dtype=np.uint8)  # light neutral gray
new_bg = np.full_like(img, new_bg_color)

# Composite
result = (img.astype(np.float32) * alpha + new_bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)

result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
result_pil.save(dst, quality=95)
print(f"Saved: {dst}")
