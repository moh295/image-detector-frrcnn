import cv2
import numpy as np
import torchvision.transforms as T

# Dictionary containing some colors
colors = {'blue': (0, 0, 255), 'green': (0, 255, 0), 'red': (255, 0, 0), 'orange': (223, 70, 14),
          'yellow': (255, 255, 0),
          'magenta': (255, 0, 255), 'cyan': (0, 255, 255), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220), 'red 1': (255, 82, 82),
          'red 2': (255, 82, 82), 'red 3': (255, 82, 82)}
font = cv2.FONT_HERSHEY_SIMPLEX

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
    if scale == 1: return img
    width = int(img.shape[1] * scale )
    height = int(img.shape[0] * scale)
    dim = (width, height)
    return  cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def overlap(box1,box2):
    x1_a,y1_a,x2_a,y2_a=box1
    x1_b,y1_b,x2_b,y2_b=box2
    iou = 0.0
    # determine the coordinates of the intersection rectangle
    x_left = max(x1_b, x1_a)
    y_top = max(y1_b, y1_a)
    x_right = min(x2_b, x2_a)
    y_bottom = min(y2_b, y2_a)

    if x_right >= x_left and y_bottom >= y_top:
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        box1_area = (x2_b - x1_b) * (y2_b - y1_b)
        box2_area = (x2_a - x1_a) * (y2_a - y1_a)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
    # print('iou',iou,box1,box2)
    return iou

def my_nms(boxes,scores):
    if len(boxes)==0 :return False
    keep_inx = np.array([True] * len(boxes))
    # compare the size, scoer and overlap of all the boxes
    for box1 in range(len(boxes) - 1):
        for box2 in range(box1 + 1, len(boxes)):
            # to speed up don't compare with the removed ones
            if keep_inx[box1] and keep_inx[box2]:
                #if boxes overlap iou>0.5
                if overlap(boxes[box1],boxes[box2])>0:
                    #chose the smaller one if no much difference in score
                    if box_size(boxes[box1]) > box_size(boxes[box2]) \
                            and scores[box1]<scores[box2]*4:
                    # if scores[box1] > scores[box2]:
                        keep_inx[box1] = False
                    else:
                        keep_inx[box2] = False
    return keep_inx



def box_size(box):


    width = box[2] - box[0]
    hight = box[3] - box[1]
    return width * hight
