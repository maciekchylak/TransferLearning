from cgi import test
from plots import *
from loading_scores import *
from loading_models import *

val_score_teacher=(predict(testloader,load(models.resnet34(pretrained=True),"teacher")))
val_score_student=(predict(testloader,load(models.resnet18(pretrained=True),"student")))
val_score_ikd_10=(predict(testloader,load(models.resnet18(pretrained=True),"ikd_unifrom_10")))
val_score_ikd_30=(predict(testloader,load(models.resnet18(pretrained=True),"ikd_unifrom_30")))
val_score_ikd_50=(predict(testloader,load(models.resnet18(pretrained=True),"ikd_unifrom_50")))
val_score_ikd_70=(predict(testloader,load(models.resnet18(pretrained=True),"ikd_unifrom_70")))
val_score_ikd_90=(predict(testloader,load(models.resnet18(pretrained=True),"ikd_unifrom_90")))

accuracy_pstart_plot([val_score_ikd_10,val_score_ikd_30,val_score_ikd_50,val_score_ikd_70,val_score_ikd_90],val_score_teacher,val_score_student)

