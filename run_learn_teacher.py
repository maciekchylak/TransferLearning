from training_functions import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Training:\n")
import time
start_time = time.time()

training_teacher(trainloader, resnet34, epochs=50)
save_model(50, resnet34,"teacher")
print("--- %s seconds ---" % (time.time() - start_time))
