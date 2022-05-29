def load_val_score(id):
    txt_file = open(f"training_outputs/val_score__{id}.txt", "r")
    file_content = txt_file.read()

    content_list = file_content.split("_")
    txt_file.close()
    val_score=[float(i) for i in content_list[:-1]]
    return val_score


def load_train_loss(id):
    txt_file = open(f"training_outputs/train_loss__{id}.txt", "r")
    file_content = txt_file.read()

    content_list = file_content.split("_")
    txt_file.close()
    train_loss=[float(i) for i in content_list[:-1]]
    return train_loss

def load_train_score(id):
    txt_file = open(f"training_outputs/train_score__{id}.txt", "r")
    file_content = txt_file.read()

    content_list = file_content.split("_")
    txt_file.close()
    train_score=[float(i) for i in content_list[:-1]]
    return train_score