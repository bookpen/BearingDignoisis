import torch
import tqdm

def cal_prec_recall(confusion):
    prec = []
    recall = []
    acc = 0
    for i in range(5):
        acc+=confusion[i,i]
        if confusion[i,i]!=0:
            prec.append(confusion[i,i]/confusion[:,i].sum())
            recall.append(confusion[i, i] / confusion[i, :].sum())
        else:
            if confusion[:, i].sum() == 0:
                prec.append(1)
            else:
                prec.append(0)
            if confusion[i, :].sum() == 0:
                recall.append(1)
            else:
                recall.append(0)
    accuracy = acc/confusion.sum()
    prec = sum(prec)/len(prec)
    recall = sum(recall)/len(recall)
    return prec.item(),recall.item(),accuracy.item()

def valid(model,testloader,valloader,flag):
    model.eval()
    if flag=="test":
        loader = testloader
    if flag=="val":
        loader = valloader
    confusion = torch.zeros(5,5)
    for i,batch in enumerate(tqdm.tqdm(loader)):
        x,label = batch
        x,label = x.cuda(),label.cuda()
        pred = model(x)
        pred = torch.argmax(pred,dim=1)
        for i,j in zip(label,pred):
            confusion[j,i] += 1
    prec,recall,acc = cal_prec_recall(confusion)
    return confusion.tolist(),prec,recall,acc


