from cgi import test
from plots import *
from loading_scores import *
from loading_models import *

print(predict(testloader,load(models.resnet34(pretrained=True),"teacher")))
print(predict(testloader,load(models.resnet18(pretrained=True),"student")))
