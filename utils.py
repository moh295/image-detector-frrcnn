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

def re_labeling_13c(target):
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

            if side == 1:
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

def re_labeling_4c(target):
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
                    if objec_on_hand and side==1:
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
                    if objec_on_hand and side==0:
                        bbx_overlap = overlap(h_bbx, box)
                        if bbx_overlap > max_overlap:
                            max_overlap = bbx_overlap
                            max_overalp_at = inx
                _, _,objec_on_hand = hands_temp_info[max_overalp_at]
                labels.append(objec_on_hand)
            # print('max_overalp_at', max_overalp_at, hands_temp_info)
    return boxes, labels

def annutaion_2_classes(numpy_image,boxes,classes,scores,output_scale):
    labels_dict = ['targetobject', 'hand']
    obj_prob_thresh = 0.38
    hand_prob_thresh = 0.60

    kept_indices = scores > obj_prob_thresh
    boxes = boxes[kept_indices]
    classes = classes[kept_indices]
    scores = scores[kept_indices]

    for bbox, cls, prob in zip(boxes.tolist(), classes.tolist(), scores.tolist()):
        bbox = np.array(bbox)*output_scale
        bbox=bbox.astype(int)
        category = labels_dict[cls - 1]
        intensity = int(200 - 200 * prob)
        color = (intensity, intensity, 255) if cls == 1 else (225, intensity, intensity)
        if cls == 1:
            cv2.rectangle(numpy_image, bbox[:2], bbox[2:4], color, 2)
            cv2.putText(numpy_image, f'{category:s} {prob:.3f}', bbox[:2], font, 1, color, 2, cv2.LINE_AA)
        elif prob > hand_prob_thresh:
            cv2.rectangle(numpy_image, bbox[:2], bbox[2:4], color, 2)
            cv2.putText(numpy_image, f'{category:s} {prob:.3f}', bbox[:2], font, 1, color, 2, cv2.LINE_AA)

    return numpy_image

def annutaion_13_classes(numpy_image,boxes,classes,scores,output_scale):


    labels_dict = {'Hand_free_R': 1, 'Hand_free_L': 2, 'Hand_cont_R': 3, 'Hand_cont_L': 4,
                   'person_R': 5, 'person_L': 6, 'person_LR': 7, 'portable_R': 8,
                   'portable_L': 9, 'portable_LR': 10, 'non-portable_R': 11, 'non-portable_L': 12,
                   'non-portable_LR': 13}

    obj_prob_thresh = 0.15
    hand_prob_thresh = 0.30
    kept_indices = scores > obj_prob_thresh
    boxes = boxes[kept_indices]
    classes = classes[kept_indices]
    scores = scores[kept_indices]

    for bbox, cls, prob in zip(boxes.tolist(),classes.tolist(),scores.tolist()):
        bbox = np.array(bbox) * output_scale
        bbox = bbox.astype(int)
        category = list(labels_dict.keys())[cls - 1]

        color_intensity = int(200 - 200 * prob)
        Free_hand = cls < 3
        Contact_hand = 2 < cls < 5
        color = (225, color_intensity, color_intensity) if Free_hand else (color_intensity, 255, color_intensity)
        hand = Free_hand or Contact_hand
        if hand and prob > hand_prob_thresh:
            cv2.rectangle(numpy_image, bbox[:2], bbox[2:4], color, 2)
            cv2.putText(numpy_image, f'{category:s} {prob:.3f}', bbox[:2] + 20, font, 1, color, 2, cv2.LINE_AA)
        elif not hand and prob > obj_prob_thresh:
            color = (color_intensity, color_intensity, 255)
            cv2.rectangle(numpy_image, bbox[:2], bbox[2:4], color, 2)
            cv2.putText(numpy_image, f'{category:s} {prob:.3f}', bbox[:2] + 20, font, 1, color, 2, cv2.LINE_AA)
    return numpy_image

def annutaion_4_classes(numpy_image,boxes,classes,scores,output_scale):

    labels_dict = {'Hand_free': 1, 'Hand_cont': 2, 'object': 3,'person':4}
    obj_prob_thresh = 0.1
    hand_prob_thresh = 0.25
    # obj acceptable_size will be hand_max * max_size_ratio
    max_size_ratio= 10

    kept_indices = scores > obj_prob_thresh
    boxes = boxes[kept_indices]
    classes = classes[kept_indices]
    scores = scores[kept_indices]
    #hand list
    h_boxes = boxes[classes <3]
    h_scores = scores[classes <3]
    h_cls=classes[classes <3]
    h_kept = my_nms(h_boxes, h_scores)
    h_boxes = h_boxes[h_kept]
    h_scores = h_scores[h_kept]
    h_cls = h_cls[h_kept]
    max_hand_box=max([box_size(box) for box in h_boxes]) if len(h_boxes) else False

    #object
    o_boxes=boxes[classes==3]
    o_scores=scores[classes==3]
    o_kept = my_nms(o_boxes, o_scores)
    o_boxes = o_boxes[o_kept]
    o_scores = o_scores[o_kept]

    #preson list
    p_boxes = boxes[classes == 4]
    p_scores = scores[classes == 4]
    p_kept = my_nms(p_boxes, p_scores)
    p_boxes = p_boxes[p_kept]
    p_scores = p_scores[p_kept]

    for bbox, cls, prob in zip(h_boxes,h_cls,h_scores):
        bbox = np.array(bbox) * output_scale
        bbox = bbox.astype(int)
        category = list(labels_dict.keys())[cls - 1]
        color_intensity = int(200 - 200 * prob)
        Free_hand = cls ==1
        color = (225, color_intensity, color_intensity) if Free_hand else (color_intensity, 255, color_intensity)
        if  prob > hand_prob_thresh:
            cv2.rectangle(numpy_image, bbox[:2], bbox[2:4], color, 2)
            cv2.putText(numpy_image, f'{category:s} {prob:.3f}', bbox[:2] + 20, font, 1, color, 2, cv2.LINE_AA)

    for bbox, prob in zip(o_boxes, o_scores):
        category='object'
        bbox = np.array(bbox) * output_scale
        bbox = bbox.astype(int)

        if max_hand_box:
            acceptable_size=box_size(bbox)<max_hand_box*max_size_ratio
        else:
            acceptable_size=True

        color_intensity = int(200 - 200 * prob)
        color = (color_intensity, color_intensity, 255) if acceptable_size else (0, color_intensity, color_intensity)
        if prob > obj_prob_thresh:
            cv2.rectangle(numpy_image, bbox[:2], bbox[2:4], color, 2)
            cv2.putText(numpy_image, f'{category:s} {prob:.3f}', bbox[:2] + 20, font, 1, color, 2, cv2.LINE_AA)

    for bbox , prob in zip(p_boxes,p_scores):
        category = 'pesone'
        bbox = np.array(bbox) * output_scale
        bbox = bbox.astype(int)
        color_intensity = int(200 - 200 * prob)
        color = (color_intensity, color_intensity, 150)
        if prob > obj_prob_thresh:
            cv2.rectangle(numpy_image, bbox[:2], bbox[2:4], color, 2)
            cv2.putText(numpy_image, f'{category:s} {prob:.3f}', bbox[:2] + 20, font, 1, color, 2, cv2.LINE_AA)
    return numpy_image

def box_size(box):
    width = box[2] - box[0]
    hight = box[3] - box[1]
    return width * hight

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
                    if box_size(boxes[box1]) < box_size(boxes[box2]) \
                            and scores[box1]*1.5>scores[box2]:
                        keep_inx[box2] = False
                    else:
                        keep_inx[box1] = False
    return keep_inx
