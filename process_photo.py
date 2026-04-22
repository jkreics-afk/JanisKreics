from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

src = r"C:\Users\B12\Desktop\CLAUDE_MY\Jānis Kreics_019b.jpg"
dst = r"C:\Users\B12\Desktop\CLAUDE_MY\Janis_Kreics_019b_bw.jpg"

img = Image.open(src)

# Convert to grayscale
gray = img.convert("L")

arr = np.array(gray, dtype=np.float32)

# Apply a gentle S-curve: lift shadows/midtones (background), preserve deep darks (clothing)
# Background in original is ~128 gray — we want it near 220+
# Use gamma correction + levels to achieve this

# Step 1: Lift midtones with gamma < 1 (brightens without blowing out highlights)
gamma = 0.55
lifted = np.power(arr / 255.0, gamma) * 255.0

# Step 2: Stretch levels so output range covers full 0-255
in_black, in_white = 0, 255
out_black, out_white = 10, 255
stretched = np.clip((lifted - in_black) / (in_white - in_black) * (out_white - out_black) + out_black, 0, 255)

result = Image.fromarray(stretched.astype(np.uint8))

# Slight sharpness boost for professional crispness
result = ImageEnhance.Sharpness(result).enhance(1.3)
# Slight contrast boost to keep subject looking defined
result = ImageEnhance.Contrast(result).enhance(1.15)

result.save(dst, quality=95)
print(f"Saved to: {dst}")
