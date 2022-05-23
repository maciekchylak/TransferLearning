from training_functions import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Training:\n")
import time
start_time = time.time()

training(trainloader, resnet18, resnet34, 0.8, epochs=1,intervals=1)

print("--- %s seconds ---" % (time.time() - start_time))
