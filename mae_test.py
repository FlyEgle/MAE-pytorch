import torch
import argparse
from PIL import Image 
from torchvision.transforms import transforms
from torch.cuda.amp import autocast as autocast
import numpy as np 
from model.Transformers.VIT.mae import MAEVisionTransformers as MAE
from loss.mae_loss import build_mask


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='F:/业务数据/vit-mae_losses_0.20102281799793242.pth', type=str)
parser.add_argument('--test_image', default='E:/imagnet_val/val/ILSVRC2012_val_00039841.JPEG', type=str)

args = parser.parse_args()


image = Image.open(args.test_image)
raw_image = image.resize((224, 224))
raw_image.save("output/src_image.jpg")
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

ckpt = torch.load(args.ckpt, map_location="cpu")['state_dict']

model = MAE(
    img_size = 224,
    patch_size = 16,  
    encoder_dim = 768,
    encoder_depth = 12,
    encoder_heads = 12,
    decoder_dim = 512,
    decoder_depth = 8,
    decoder_heads = 16, 
    mask_ratio = 0.75
)


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
cv2.imwrite("output/output_image.jpg", output_image)


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
cv2.imwrite("output/mask_image.jpg", non_mask_image[:,:,::-1])

print(output_image)
    
        
        
        
        

        