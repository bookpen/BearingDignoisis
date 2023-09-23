import json
import os
from Validation import cal_prec_recall
import torch

# output_dir = "output_vit"
# for i in os.listdir(output_dir):
#     if "test" in i or "val" in i:
#         with open(os.path.join(output_dir,i),"r")as f:
#             info = json.loads(f.read())
#         info["confusion"] = torch.tensor(info["confusion"]).T
#         info["prec"],info["recall"],info["accuracy"]=cal_prec_recall(info["confusion"])
#         info["confusion"] = info["confusion"].tolist()
#         with open(os.path.join(output_dir,i),"w")as f:
#             f.write(json.dumps({"prec":info["prec"],
#                                 "recall":info["recall"],
#                                 "acc":info["accuracy"],
#                                 "confusion":info["confusion"]}))
#         print(info)
import os
import matplotlib.pyplot as plt
import json
def get_acc(output_dir):
    epoch_precision = {}
    for i in os.listdir(output_dir):
        if "val" in i:
            num = int(i.split("epoch")[1].split(".")[0])
            with open(os.path.join(output_dir,i),"r") as f:
                epoch_precision[num] = json.load(f)

    precision = [epoch_precision[i]["prec"] for i in epoch_precision]
    recall = [epoch_precision[i]["recall"] for i in epoch_precision]
    acc = [epoch_precision[i]["acc"] for i in epoch_precision]
    return precision,recall,acc
# F1 = [2*p*r/(p+r) for p,r in zip(precision,recall)]

_,_,vit_acc = get_acc("output_vit")
_,_,vgg_acc = get_acc("output_vgg")
_,_,resnet_acc = get_acc("output_resnet")
_,_,googlenet_acc = get_acc("output_googlenet")

plt.plot(vit_acc[:len(vgg_acc)],label="ViT",linestyle="-")
plt.plot(vgg_acc,label="VGG16",linestyle=":")
plt.plot(googlenet_acc,label="GoogLeNet",linestyle="--")
plt.plot(resnet_acc,label="ResNet",linestyle="solid")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
# plt.title(output_dir.split("_")[-1])
plt.show()

