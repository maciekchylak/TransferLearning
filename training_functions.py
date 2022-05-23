import random
import ssl

import numpy as np
import pandas as pd
import torchvision.models as models
import torchvision
import torch.nn as nn
import torch
import tensorflow as tf
from torch.optim.lr_scheduler import MultiStepLR

from data_loader import *

ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def hybrid_blocks(student, teacher):
    '''
    Function used to get BasicBlocks from ResNet class model
    '''
    student_layers = [student.layer1, student.layer2, student.layer3, student.layer4]
    teacher_layers = [teacher.layer1, teacher.layer2, teacher.layer3, teacher.layer4]

    student_blocks = []
    teacher_blocks = []

    for i in range(len(student_layers)):
        teacher_blocks += list(np.array_split(teacher_layers[i], len(student_layers[i]))) # divide teacher blocks into n list, where n is number of student blocks
        student_blocks += [el for el in student_layers[i]]

    return student_blocks, teacher_blocks

def forward(x, student, teacher, a_all):
    '''
    Forward function for hybrid ResNet
    '''
    def _forward_blocks(x, student_blocks, teacher_blocks, a_all):
        '''
        Forward function containing only hybrid blocks predicitons
        '''
        len_teacher_blocks = len(teacher_blocks)
        len_student_blocks = len(student_blocks)
        assert len_teacher_blocks == len_student_blocks   # check if size of blocks is the same
        tmp_x = x
        for i in range(len_student_blocks): # hybrid block
            if a_all[i] == 1: # student path
                tmp_x = student_blocks[i].forward(tmp_x)

            if a_all[i] == 0: # teacher path
                for j in range(len(teacher_blocks[i])):
                    tmp_x = teacher_blocks[i][j].forward(tmp_x)

        return tmp_x, a_all

    student_blocks, teacher_blocks = hybrid_blocks(student, teacher)

    softmax = nn.Softmax(dim=1)

    tmp_x = x     # forward pipeline
    tmp_x = student.conv1(tmp_x)
    tmp_x = student.bn1(tmp_x)
    tmp_x = student.relu(tmp_x)
    tmp_x = student.maxpool(tmp_x)
    tmp_x, a_all = _forward_blocks(tmp_x, student_blocks, teacher_blocks, a_all)
    tmp_x = student.avgpool(tmp_x)
    tmp_x = torch.flatten(tmp_x, 1)
    tmp_x = student.fc(tmp_x)
    output = softmax(tmp_x)

    return output

def training(data, student, teacher, p, epochs = 200,intervals=200):
    # dodałem parametr intervals:
    # jeśli intervals = epochs     mamy Uniform schedule
    # jesli intervals = 1          mamy Linear growth schedule
    # jesli 1 < intervals < epochs mamy Review schedule, gdzie intervals oznacza liczbę "powtórek"
    loss_function = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(student.parameters(), lr=0.001)
    #optimizer(SGD) i modyfikacja learning rate(MultiStepLR) z artykułu
    optimizer = optim.SGD(student.parameters(), lr=0.1, weight_decay=0.0001, momentum=0.9)
    train_loss = []
    train_score = []
    x=np.linspace(p, 1, int(epochs/intervals))
    print(f"x = {x}")
    p_all=np.tile(x,intervals)
    print(f"p_all = {p_all}")
    for e in range(epochs):
        print(f"\nEpoch no. {e}")
        score = 0
        loss = 0
        student_blocks, teacher_blocks = hybrid_blocks(student, teacher)
        #a_all = [np.random.binomial(1, p) for i in range(len(student_blocks))]
        a_all = [np.random.binomial(1, p_all[e]) for i in range(len(student_blocks))]   # hybrid block building schema
        print(f"p_all[e] = {p_all[e]}")
        print(f"a_all = {a_all}")
        #a_all =[1,0,1,1,1,1,1,1]
        for block, a in zip(student_blocks,a_all):
            if a==0:
                for param in block.parameters():
                    param.requires_grad=False
            else:
                for param in block.parameters():
                    param.requires_grad=True
        for image, label in data:
            student_blocks, teacher_blocks = hybrid_blocks(student, teacher)
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            y_pred = forward(image, student, teacher, a_all)
            loss = loss_function(y_pred, label)
            loss.backward()
            optimizer.step()
            val, index_ = torch.max(y_pred, axis=1)
            score += torch.sum(index_ == label.data).item()
            loss += loss.item()
            print('step')
        epoch_score = score / len(data)
        epoch_loss = loss / len(data)
        train_loss.append(epoch_loss)
        train_score.append(epoch_score)
        print("Training loss: {}, accuracy: {}".format(epoch_loss, epoch_score))
        return train_loss, train_score

def training_teacher(data,teacher, epochs = 200):
    # dodałem parametr intervals:
    # jeśli intervals = epochs     mamy Uniform schedule
    # jesli intervals = 1          mamy Linear growth schedule
    # jesli 1 < intervals < epochs mamy Review schedule, gdzie intervals oznacza liczbę "powtórek"
    loss_function = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(student.parameters(), lr=0.001)
    #optimizer(SGD) i modyfikacja learning rate(MultiStepLR) z artykułu
    optimizer = optim.SGD(teacher.parameters(), lr=0.1, weight_decay=0.0001, momentum=0.9)
    train_loss = []
    train_score = []
    for e in range(epochs):
        print(f"\nEpoch no. {e}")
        score = 0
        loss = 0

        for image, label in data:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            y_pred = teacher(image)
            loss = loss_function(y_pred, label)
            loss.backward()
            optimizer.step()
            val, index_ = torch.max(y_pred, axis=1)
            score += torch.sum(index_ == label.data).item()
            loss += loss.item()

        epoch_score = score / len(data)
        epoch_loss = loss / len(data)
        train_loss.append(epoch_loss)
        train_score.append(epoch_score)
        print("Training loss: {}, accuracy: {}".format(epoch_loss, epoch_score))
        return train_loss, train_score

def save_model(epochs, model):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                }, 'outputs/model.pth')


resnet34 = models.resnet34(pretrained=True)
resnet18 = models.resnet18(pretrained=True)
resnet18.fc =  nn.Linear(512, 2)
resnet34.fc = nn.Linear(512, 2)
