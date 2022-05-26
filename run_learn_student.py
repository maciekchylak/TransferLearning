from training_functions import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Training:\n")
import time
start_time = time.time()

training_teacher(trainloader, resnet18, epochs=1)
save_model(1, resnet18,"student")
print("--- %s seconds ---" % (time.time() - start_time))