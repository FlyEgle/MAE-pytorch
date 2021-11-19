import torch
from PIL import Image 
from torchvision.transforms import transforms
from torch.cuda.amp import autocast as autocast
import numpy as np 
from model.Transformers.VIT.mae import MAEVisionTransformers as MAE
from loss.mae_loss import build_mask

    
image = Image.open("E:/imagnet_val/val/ILSVRC2012_val_00010980.JPEG")
raw_image = image.resize((224, 224))
raw_image.save("./src_image.jpg")
raw_tensor  = torch.from_numpy(np.array(raw_image))
print(raw_tensor.shape)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

image = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)]
)(image)
image_tensor = image.unsqueeze(0)

ckpt = torch.load("F:/业务数据/hago数据/0701-0705/0716/pr_result/vit-mae_losses_0.20791142220139502.pth", map_location="cpu")['state_dict']
for key, value in ckpt.items():
    pass 


model = MAE(
    img_size = 224,
    patch_size = 16,  
    encoder_dim = 192,
    encoder_depth = 12,
    encoder_heads = 3,
    decoder_dim = 512,
    decoder_depth = 8,
    decoder_heads = 16, 
    mask_ratio = 0.75
)


# a = torch.randn(2, 3, 224, 224)
# a = a.cuda()
# model.cuda()
# with autocast():
#     b, _ = model(a)
# print(b.shape)

print(model)
model.load_state_dict(ckpt, strict=True)
model.cuda()
model.eval()
image_tensor = image_tensor.cuda()
with torch.no_grad():
    with autocast():
        output, mask_index = model(image_tensor)
        print(output.shape)
        
output_image = output.squeeze(0)
output_image = output_image.permute(1,2,0).cpu().numpy()
output_image = output_image * std + mean
output_image = output_image * 255
import cv2 
output_image = output_image[:,:,::-1]
cv2.imwrite("./output_image1.jpg", output_image)


mask_map = build_mask(mask_index, patch_size=16, img_size=224)

non_mask = 1 - mask_map 
non_mask = non_mask.unsqueeze(-1)

non_mask_image = non_mask * raw_tensor


mask_map = mask_map * 127
mask_map = mask_map.unsqueeze(-1)

print(torch.min(mask_map))

non_mask_image  += mask_map 

# print(non_mask_image)
non_mask_image = non_mask_image.cpu().numpy()
print(non_mask_image.shape)
cv2.imwrite("mask_image.jpg", non_mask_image[:,:,::-1])

print(output_image)
    
        
        
        
        

        