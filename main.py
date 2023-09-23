import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from torchvision.models import resnet34,GoogLeNet,vgg16,AlexNet

import os
import tqdm

from model.Dataset import ImageData
from Validation import valid
import json
from utils.utils import prepare_model
# 预训练模型下载
# https://pytorch.org/vision/0.8/models.html
# os.path.join("model","resnet","resnet34.pth"),
# os.path.join("model","VGG16","vgg16.pth"),
# os.path.join("model","GoogLeNet","googlenet.pth"),
# "model/ViT_base/vit_base.pth"
arg = {"model_name":GoogLeNet,
       "pretrain_model":"output_vit/model.pth",
       "epoch":40,
       "output_dir":"output_vit",
       "net_name":"vit"
       }

os.makedirs(arg["output_dir"],exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
])


trainset = ImageData("data/split_data/train",transform=transform)
testset = ImageData("data/split_data/test",transform=transform)
valset = ImageData("data/split_data/val",transform=transform)

train_dataloader = DataLoader(trainset,batch_size=64,shuffle=True)
test_dataloader = DataLoader(testset,batch_size=64,shuffle=True)
val_dataloader = DataLoader(valset,batch_size=64,shuffle=True)

model = prepare_model(arg)

loss_ce = torch.nn.CrossEntropyLoss()

optim = torch.optim.Adam(model.head.parameters(),lr=1e-7)

# val_json = {"epoch"}

for i in range(40,arg["epoch"]+40):
    model.train()
    for j,batch in enumerate(tqdm.tqdm(train_dataloader)):
        x,label = batch
        x, label = x.cuda(), label.cuda()
        pred = model(x)
        if arg["net_name"] == "googlenet":
            loss = loss_ce(pred.logits,label)
        else:
            loss = loss_ce(pred,label)
        loss.backward()
        optim.step()
        optim.zero_grad()
        # model.zero_grad()
        # if j % 20 ==0:
        #     print(loss.item())
    confusion,prec,recall,acc = valid(model, test_dataloader, val_dataloader, "val")
    with open(os.path.join(arg["output_dir"],"val_epoch{}.json".format(i)),"w") as f:
        f.write(json.dumps({"prec":prec,"recall":recall,"acc":acc,"confusion":confusion}))
    if acc>0.92:
        torch.save(model.state_dict(), os.path.join(arg["output_dir"], "model.pth"))
    else:
        break

confusion,prec,recall,acc = valid(model, test_dataloader, val_dataloader, "test")
with open(os.path.join(arg["output_dir"], "test.json"), "w") as f:
    f.write(json.dumps({"prec":prec,"recall":recall,"acc":acc,"confusion":confusion}))

# save model
# torch.save(model.state_dict(),os.path.join(arg["output_dir"],"model.pth"))

