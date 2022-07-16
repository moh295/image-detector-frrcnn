import cv2
import numpy as np

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
def nurmolize_numpy(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img
def tensor_to_numpy_cv2(teosor):
    numpy=teosor.permute(1, 2, 0).numpy()
    numpy=nurmolize_numpy(numpy, 0, 255, np.uint8)
    numpy=cv2.cvtColor(numpy, cv2.COLOR_BGR2RGB)
    return numpy

def image_resize(img,scale):
    width = int(img.shape[1] * scale )
    height = int(img.shape[0] * scale)
    dim = (width, height)
    return  cv2.resize(img, dim, interpolation=cv2.INTER_AREA)