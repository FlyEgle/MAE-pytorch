import cv2 
import torch 
import torch.nn as nn
import numpy as np 
from PIL import Image 


image = Image.open("/data/jiangmingchao/data/lr_1.jpg")
image = np.array(image)
image = cv2.resize(image, (224, 224))

image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
print(image_tensor.shape)

image_tensor = image_tensor / 255.
print(max(image_tensor))
conv = nn.Conv2d(3, 32, 16, 16)
output_tensor = conv(image_tensor)
print(output_tensor.shape)

unconv = nn.ConvTranspose2d(32, 3, 16, 16)

restore_tensor = unconv(output_tensor)

print(restore_tensor.shape)

restore_image = restore_tensor.squeeze(0).permute(1,2,0).detach().numpy()
# cv2.imwrite("./restore_image.jpg", restore_image*255)
restore_image = restore_image * 255
restore_image = restore_image.astype(np.uint8)
# restore_image.save("./restore_image.jpg")

image = Image.fromarray(restore_image)
image.save("./restore_image.jpg")
