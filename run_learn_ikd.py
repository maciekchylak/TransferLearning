from training_functions import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Training:\n")
import time
start_time = time.time()

training(trainloader, resnet18, resnet34, 0.5, epochs = 10,intervals=10)
save_model(10, resnet18,"ikd")
print("--- %s seconds ---" % (time.time() - start_time))