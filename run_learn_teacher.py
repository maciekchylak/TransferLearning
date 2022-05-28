from training_functions import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Training:\n")
import time
start_time = time.time()

#training_model(trainloader,testloader, resnet34, epochs=50, id='teacher')
training_model(trainloader,testloader, resnet34, epochs=30, id='teacher_1')
print("--- %s seconds ---" % (time.time() - start_time))
