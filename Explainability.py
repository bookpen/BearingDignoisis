from utils.utils import prepare_model
from timm.models.vision_transformer import vit_base_patch16_224
import os
import torch
import cv2
import numpy as np
from torchvision import transforms
import PIL.Image as Image

from model.baselines.ViT.ViT_LRP import vit_base_patch16_224
from model.Explainability import LRP

import matplotlib.pyplot as plt
arg = {"model_name":vit_base_patch16_224,
       "pretrain_model":"output_vit/model.pth",
       "epoch":60,
       "output_dir":"output",
       "net_name":"vit"
       }

model = prepare_model(arg)

# model.load_state_dict(torch.load("output_vit/model.pth"))
model = model.cuda()
model.eval()

attribution_generator = LRP(model)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

use_thresholding=False
def generate_visualization(original_image, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    if use_thresholding:
      transformer_attribution = transformer_attribution * 255
      transformer_attribution = transformer_attribution.astype(np.uint8)
      ret, transformer_attribution = cv2.threshold(transformer_attribution, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      transformer_attribution[transformer_attribution == 255] = 1

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

transform = transforms.Compose([
    transforms.ToTensor(),
])

# data/split_data/test/Bearing1_5/14g.jpg
# data/split_data/test/Bearing1_5/18b.jpg
# data/split_data/test/Bearing1_2/12b.jpg
# data/split_data/test/Bearing1_2/27g.jpg
image_to_show = 'data/split_data/test/Bearing1_2/12b.jpg'
image = Image.open(image_to_show)
img_bad = transform(image)
output = model(img_bad.unsqueeze(0).cuda())
print(output,image_to_show)
a = generate_visualization(img_bad)
plt.imshow(a)
plt.xticks([])
plt.yticks([])
# plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.gca().set_aspect('equal')
plt.show()

print("over")




