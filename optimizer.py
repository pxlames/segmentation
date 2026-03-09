import torch.optim as optim



def create_optimizer(parameters, mode="SGD", lr=0.001, momentum=0.9, weight_decay=0.0005, beta1=0.5, beta2=0.999):
    if mode == "sgd":
        return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif mode == "adam":
        return optim.Adam(parameters, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)