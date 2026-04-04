
from PIL import Image
import math
import numpy as np

# List your image paths (22 PNGs)

image_paths = []
for i in np.arange(6.4, 7.4, 0.1):
    age = str(round(i,1)).replace(".", "_")
    print(age)
    image_paths.append(f"/Users/dan/Code/FYP/Data/Images/BPASS/imf_chab100/Coeff/updated/plot-bin-imf_chab100.Age{age}yrs.png" )

# Open images
images = [Image.open(p) for p in image_paths]

# Optional: resize all images to the same size
# (important if they differ)
width, height = images[0].size
images = [img.resize((width, height)) for img in images]

# Grid settings
images_per_row = 3
rows = math.ceil(len(images) / images_per_row)

# Create blank canvas
combined_width = width * images_per_row
combined_height = height * rows
combined_image = Image.new('RGB', (combined_width, combined_height))

# Paste images
for index, img in enumerate(images):
    row = index // images_per_row
    col = index % images_per_row
    
    x = col * width
    y = row * height
    
    combined_image.paste(img, (x, y))

# Save result
combined_image.save("combined1.png")