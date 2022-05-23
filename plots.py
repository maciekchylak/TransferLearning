import matplotlib.pyplot as plt
import numpy as np


def accuracy_epochs_plot(score, p):
    epochs = np.linspace(1, len(score), len(score))

    plt.figure(figsize=(10, 8), dpi=80)
    plt.xticks(np.arange(1, len(epochs)+1, step = 1))
    plt.yticks(np.arange(int(min(score) * 10) / 10, 1, step=0.1))
    plt.grid(color='black', linestyle='-', linewidth=0.1)

    plt.plot(epochs, score, color='blue')

    plt.title(f'Accuracy of the IAKD model with p = {p} in successive epochs', fontsize=16)
    plt.xlabel('epochs', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)

def accuracy_pstart_plot(acc_iakd, acc_teacher, acc_student):
    p = np.linspace(0, 0.9, 10)

    plt.figure(figsize=(10, 8), dpi=80)
    plt.xticks(np.arange(0, 1, step = 0.1))
    plt.yticks(np.arange(int(min(acc_iakd) * 10) / 10, 1, step=0.1))
    plt.grid(color='black', linestyle='-', linewidth=0.1)

    plt.plot(p, acc_iakd, color='blue')
    plt.axhline(y=acc_teacher, color='red', linestyle='--')
    plt.axhline(y=acc_student, color='green', linestyle='--')

    plt.legend(['IAKD', 'Teacher', 'Student'], fontsize = 14)
    plt.title(f'Accuracy of the model in successive p', fontsize=16)
    plt.xlabel('p_start', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)


    