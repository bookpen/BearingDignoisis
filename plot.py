import json
import os
import matplotlib.pyplot as plt


output_dir = "output_resnet"
epoch_precision = {}
for i in os.listdir(output_dir):
    if "val" in i:
        num = int(i.split("epoch")[1].split(".")[0])
        with open(os.path.join(output_dir,i),"r") as f:
            epoch_precision[num] = json.load(f)
    if "test" in i:
        with open(os.path.join(output_dir,i),"r") as f:
            test = json.load(f)
        test_prec = test["prec"]
        test_recall = test["recall"]

precision = [epoch_precision[i]["prec"] for i in epoch_precision]
recall = [epoch_precision[i]["recall"] for i in epoch_precision]
acc = [epoch_precision[i]["acc"] for i in epoch_precision]

# F1 = [2*p*r/(p+r) for p,r in zip(precision,recall)]
plt.plot(precision,label="precision",linestyle=":")
plt.plot(recall,label="recall",linestyle="-")
# plt.plot(F1,label="F1",linestyle="--")
plt.plot(acc,label="acc")

plt.xlabel("epoch")
plt.legend()
plt.title(output_dir.split("_")[-1])
plt.show()

# plt.close()