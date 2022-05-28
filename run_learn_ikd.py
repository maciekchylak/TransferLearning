from training_functions import *
from loading_models import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Training:\n")
import time
start_time = time.time()

model=load(models.resnet34(pretrained=True),"teacher")
training_ikd(trainloader,testloader, resnet18, model, 0.7, epochs = 10,intervals=10, id='ikd')
print("--- %s seconds ---" % (time.time() - start_time))