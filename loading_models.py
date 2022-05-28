from data_loader import *
from training_functions import *
import copy

def load(model, name):
    
    resnet = model
    resnet.fc =  nn.Linear(512, 2)
    resnet.load_state_dict(torch.load(f"outputs/model_{name}.pt"))
    resnet.eval()

    return resnet


def predict(data_val, model):
    score_val=0
    for image, label in data_val:
        image = image.to(device)
        label = label.to(device)
        y_pred = model(image.float())
        val, index_ = torch.max(y_pred, axis=1)
        score_val += torch.sum(index_ == label.data).item()
    return score_val/(len(data_val)*batch_size)


