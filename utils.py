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

def re_labeling_old(target):
    # no contact 0 self=1, other person =2 portable object=3 , non portable =4

    labels_dict = {'Hand_free_R': 1, 'Hand_free_L': 2, 'Hand_cont_R': 3, 'Hand_cont_L': 4,
                   'person_R': 5, 'person_L': 6, 'person_LR': 7, 'portable_R': 8,
                   'portable_L': 9, 'portable_LR': 10, 'non-portable_R': 11, 'non-portable_L': 12,
                   'non-portable_LR': 13}
    boxes = []
    labels = []
    hands_temp_info = []
    objects_temp_inf = False

    num_of_ojb=0
    # print('filename',target['annotation']['filename'])
    for lb in target['annotation']['object']:
        num_of_ojb+= 1
        # print('obj',num_of_ojb)

        # starting with hands annotations
        # extract some target object annotations from the hand annotations e.g (person, portable object
        if lb['name'] == 'hand':
            contact_state = int(lb['contactstate'])
            side = int(lb['handside'])

            if side == 0:
                # free right hand
                if contact_state == 0:
                    labels.append(labels_dict['Hand_free_R'])
                # contact to person right hand
                elif contact_state == 1 or contact_state == 2:
                    labels.append(labels_dict['Hand_cont_R'])
                    objects_temp_inf = labels_dict['person_R']
                # right hand contact with portable object
                elif contact_state == 3:
                    labels.append(labels_dict['Hand_cont_R'])
                    objects_temp_inf = labels_dict['portable_R']
                else:
                    labels.append(labels_dict['Hand_cont_R'])
                    objects_temp_inf = labels_dict['non-portable_R']
            else:
                # free left hand
                if contact_state == 0:
                    labels.append(labels_dict['Hand_free_L'])
                # contact to person left hand
                elif contact_state == 1 or contact_state == 2:
                    labels.append(labels_dict['Hand_cont_L'])
                    objects_temp_inf = labels_dict['person_L']
                # left hand contact with portable object
                elif contact_state == 3:
                    labels.append(labels_dict['Hand_cont_L'])
                    objects_temp_inf = labels_dict['portable_L']
                else:
                    labels.append(labels_dict['Hand_cont_L'])
                    objects_temp_inf = labels_dict['non-portable_L']

            box = [None] * 4
            xmin = int(lb['bndbox']['xmin'])
            ymin = int(lb['bndbox']['ymin'])
            xmax = int(lb['bndbox']['xmax'])
            ymax = int(lb['bndbox']['ymax'])
            box[0] = xmin
            box[1] = ymin
            box[2] = (xmax if xmax - xmin > 0 else xmin + 1)
            box[3] = (ymax if ymax - ymin > 0 else ymin + 1)
            boxes.append(box)
            hands_temp_info.append([box, labels[-1], objects_temp_inf])

        # object annotations
        elif lb['name'] == 'targetobject':
            box = [None] * 4
            xmin = int(lb['bndbox']['xmin'])
            ymin = int(lb['bndbox']['ymin'])
            xmax = int(lb['bndbox']['xmax'])
            ymax = int(lb['bndbox']['ymax'])
            box[0] = xmin
            box[1] = ymin
            box[2] = (xmax if xmax - xmin > 0 else xmin + 1)
            box[3] = (ymax if ymax - ymin > 0 else ymin + 1)
            boxes.append(box)
            max_overlap = 0
            max_overalp_at = 0
            is_obj_contact_L= int(lb['contactleft'])==1
            is_obj_contact_R= int(lb['contactright'])==1
            is_obj_contact_LR= is_obj_contact_L and is_obj_contact_R
            # print('is_obj_contact_L',is_obj_contact_L,'is_obj_contact_R',is_obj_contact_R,'is_obj_contact_LR',is_obj_contact_LR)
            # in case object in contact with both hands label with one of the following : portable_LR , non-portable_LR ,person_LR
            if is_obj_contact_LR:
                for inx in range(len(hands_temp_info)):
                    h_bbx, h_label, objec_on_hand = hands_temp_info[inx]
                    if objec_on_hand:
                        bbx_overlap = (overlap(h_bbx, box))
                        if bbx_overlap > 0 and bbx_overlap > max_overlap:
                            max_overlap = bbx_overlap
                            max_overalp_at = inx

                _, _, objec_on_hand = hands_temp_info[max_overalp_at]
                if objec_on_hand == labels_dict['person_R'] or objec_on_hand == labels_dict['person_L']:
                    labels.append(labels_dict['person_LR'])
                elif objec_on_hand == labels_dict['portable_R'] or objec_on_hand == labels_dict['portable_L']:
                    labels.append(labels_dict['portable_LR'])
                else:
                    labels.append(labels_dict['non-portable_LR'])

            #objet on contact with right hand only
            elif is_obj_contact_R:

                for inx in range(len(hands_temp_info)):
                    h_bbx, h_label, objec_on_hand = hands_temp_info[inx]
                    # if hand label is R contact state
                    if objec_on_hand and h_label==labels_dict['Hand_cont_R']:
                        bbx_overlap = (overlap(h_bbx, box))
                        if bbx_overlap > max_overlap:
                            max_overlap = bbx_overlap
                            max_overalp_at = inx
                _, _, objec_on_hand = hands_temp_info[max_overalp_at]

                labels.append(objec_on_hand)


            # objet on contact with left hand only
            elif is_obj_contact_L:
                for inx in range(len(hands_temp_info)):
                    h_bbx, h_label, objec_on_hand = hands_temp_info[inx]
                    # if hand label is L contact state
                    if objec_on_hand and h_label == h_label==labels_dict['Hand_cont_L']:
                        bbx_overlap = overlap(h_bbx, box)
                        if  bbx_overlap > max_overlap:
                            max_overlap = bbx_overlap
                            max_overalp_at = inx
                _, _, objec_on_hand = hands_temp_info[max_overalp_at]
                labels.append(objec_on_hand)
            # print('max_overalp_at', max_overalp_at, hands_temp_info)


    return boxes, labels

def re_labeling(target):
    # no contact 0 self=1, other person =2 portable object=3 , non portable =4
    labels_dict = {'Hand_free': 1, 'Hand_cont': 2, 'object': 3,'person':4}
    boxes = []
    labels = []
    hands_temp_info = []
    objects_temp_inf = False

    num_of_ojb = 0
    # print('filename',target['annotation']['filename'])
    for lb in target['annotation']['object']:

        num_of_ojb += 1
        # print('obj',num_of_ojb)

        # starting with hands annotations
        # extract some target object annotations from the hand annotations e.g (person, portable object
        if lb['name'] == 'hand':
            contact_state = int(lb['contactstate'])
            side = int(lb['handside'])

            # free hand
            if contact_state == 0:
                labels.append(labels_dict['Hand_free'])
            # contact to person  hand
            elif contact_state == 1 or contact_state == 2:
                labels.append(labels_dict['Hand_cont'])
                objects_temp_inf = labels_dict['person']
            #  hand contact with  object

            else:
                labels.append(labels_dict['Hand_cont'])
                objects_temp_inf = labels_dict['object']



            box = [None] * 4
            xmin = int(lb['bndbox']['xmin'])
            ymin = int(lb['bndbox']['ymin'])
            xmax = int(lb['bndbox']['xmax'])
            ymax = int(lb['bndbox']['ymax'])
            box[0] = xmin
            box[1] = ymin
            box[2] = (xmax if xmax - xmin > 0 else xmin + 1)
            box[3] = (ymax if ymax - ymin > 0 else ymin + 1)
            boxes.append(box)
            hands_temp_info.append([box,side,objects_temp_inf])

        # object annotations
        elif lb['name'] == 'targetobject':
            box = [None] * 4
            xmin = int(lb['bndbox']['xmin'])
            ymin = int(lb['bndbox']['ymin'])
            xmax = int(lb['bndbox']['xmax'])
            ymax = int(lb['bndbox']['ymax'])
            box[0] = xmin
            box[1] = ymin
            box[2] = (xmax if xmax - xmin > 0 else xmin + 1)
            box[3] = (ymax if ymax - ymin > 0 else ymin + 1)
            boxes.append(box)
            max_overlap = 0
            max_overalp_at = 0
            is_obj_contact_L = int(lb['contactleft']) == 1
            is_obj_contact_R = int(lb['contactright']) == 1
            is_obj_contact_LR = is_obj_contact_L and is_obj_contact_R
            # print('is_obj_contact_L',is_obj_contact_L,'is_obj_contact_R',is_obj_contact_R,'is_obj_contact_LR',is_obj_contact_LR)
            # in case object in contact with both hands label with one of the following : portable_LR , non-portable_LR ,person_LR
            if is_obj_contact_LR:
                for inx in range(len(hands_temp_info)):
                    h_bbx, side, objec_on_hand = hands_temp_info[inx]
                    if objec_on_hand:
                        bbx_overlap = (overlap(h_bbx, box))
                        if bbx_overlap > 0 and bbx_overlap > max_overlap:
                            max_overlap = bbx_overlap
                            max_overalp_at = inx

                _, _ ,objec_on_hand = hands_temp_info[max_overalp_at]
                if objec_on_hand == labels_dict['person']:
                    labels.append(labels_dict['person'])
                else:
                    labels.append(labels_dict['object'])

            # objet on contact with right hand only
            elif is_obj_contact_R:

                for inx in range(len(hands_temp_info)):
                    h_bbx, side, objec_on_hand= hands_temp_info[inx]
                    # if hand label is R contact state
                    if objec_on_hand and side==0:
                        bbx_overlap = (overlap(h_bbx, box))
                        if bbx_overlap > max_overlap:
                            max_overlap = bbx_overlap
                            max_overalp_at = inx
                _, _, objec_on_hand = hands_temp_info[max_overalp_at]

                labels.append(objec_on_hand)


            # objet on contact with left hand only
            elif is_obj_contact_L:
                for inx in range(len(hands_temp_info)):
                    h_bbx, side, objec_on_hand = hands_temp_info[inx]
                    # if hand label is L contact state
                    if objec_on_hand and side==1:
                        bbx_overlap = overlap(h_bbx, box)
                        if bbx_overlap > max_overlap:
                            max_overlap = bbx_overlap
                            max_overalp_at = inx
                _, _,objec_on_hand = hands_temp_info[max_overalp_at]
                labels.append(objec_on_hand)
            # print('max_overalp_at', max_overalp_at, hands_temp_info)

    return boxes, labels

