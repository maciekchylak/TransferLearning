from training_functions import *
from loading_models import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Training:\n")
import time
start_time = time.time()

model=load(models.resnet34(pretrained=True),"teacher")
training_ikd(trainloader,testloader, resnet18, model, 0.9, epochs = 30,intervals=30, id='ikd_unifrom_90')
print("--- %s seconds ---" % (time.time() - start_time))