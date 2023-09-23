from torchvision.models import resnet34,GoogLeNet,vgg16,AlexNet
import torch
from torch import nn

# from timm.models.vision_transformer import vit_base_patch16_224
from model.baselines.ViT.ViT_LRP import vit_base_patch16_224
from model.modules.layers_ours import *
def prepare_model(arg):
    model = arg["model_name"]()
    if arg["net_name"] == "resnet34":
        weight = torch.load(arg["pretrain_model"])
        model.load_state_dict(weight)
        for i in model.parameters():
            i.requires_grad=False
        in_feature = model.fc.in_features
        model.fc = nn.Linear(in_feature,5)
        print("model finish loading")
        model.cuda()
        return model
    elif arg["net_name"] == "vgg16":
        weight = torch.load(arg["pretrain_model"])
        model.load_state_dict(weight)
        for i in model.parameters():
            i.requires_grad=False
        in_feature = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_feature,5)
        print("model finish loading")
        model.cuda()
        return model
    elif arg["net_name"] == "googlenet":
        weight = torch.load(arg["pretrain_model"])
        model.load_state_dict(weight)
        for i in model.parameters():
            i.requires_grad=False
        in_feature = model.fc.in_features
        model.fc = nn.Linear(in_feature,5)
        print("model finish loading")
        model.cuda()
        return model
    elif arg["net_name"] == "vit":
        model = vit_base_patch16_224(pretrained=False)
        in_feature = model.head.in_features
        model.head = Linear(in_feature,5)
        model.load_state_dict(torch.load(arg["pretrain_model"]))
        # for name, para in model.named_parameters():
        #     # 除head, pre_logits外，其他权重全部冻结
        #     if "head" not in name and "pre_logits" not in name:
        #         para.requires_grad_(False)
        # model.head.requires_grad_(True)
        print("model finish loading")
        model.cuda()
        return model