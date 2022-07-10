

import torchvision


import torchvision.transforms as T

def torch_model_info(model,optimizer):
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def tensor_to_PIL(tensor,normlized=True,mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)):
    if normlized:
        tensor=inverse_normalize(tensor,mean,std)

    transform = T.ToPILImage()
    return transform(tensor)

