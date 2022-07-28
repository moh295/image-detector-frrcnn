from utils import *
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
    for lb in target['annotation']['object']:
        num_of_ojb+= 1
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

def annutaion_4_classes(numpy_image,boxes,classes,scores,output_scale,grasp_tracker=False):

    labels_dict = {'H_free': 1, 'H_cont': 2, 'object': 3,'person':4}
    peson_prob_thresh=0.3
    obj_prob_thresh = 0.1
    hand_prob_thresh = 0.51
    # obj acceptable_size will be hand_max * max_size_ratio
    max_size_ratio= 10
    kept_indices = scores > peson_prob_thresh
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
    # preson list
    p_boxes = boxes[classes == 4]
    p_scores = scores[classes == 4]

    #filtring box with overframe tracking
    if grasp_tracker:
        o_kept= grasp_tracker.track(h_boxes,h_scores,o_boxes,o_scores)

    else:
        o_kept = my_nms(o_boxes, o_scores)


   #boxes after filtring

    o_boxes = o_boxes[o_kept]
    o_scores = o_scores[o_kept]
    # p_boxes = p_boxes[p_kept]
    # p_scores = p_scores[p_kept]

    #anutate non tracked hands
    for bbox, cls, prob in zip(h_boxes,h_cls,h_scores):
        bbox = np.array(bbox) * output_scale
        bbox = bbox.astype(int)
        category = list(labels_dict.keys())[cls - 1]
        color_intensity = int(200 - 200 * prob)
        color = (225, color_intensity, color_intensity)
        if  prob > hand_prob_thresh:
            cv2.rectangle(numpy_image, bbox[:2], bbox[2:4], color, 2)
            cv2.putText(numpy_image, f'{category:s} {prob:.3f}', bbox[:2] +[0,20], font, 1, color, 2, cv2.LINE_AA)



    #draw tracked hand-obj
    if grasp_tracker:
        for track in grasp_tracker.record:

            category='trk #'+ str(track.nb_trk_frame)+'- s'+str(track.last_seen)
            bbox = np.array(track.hand_bbx) * output_scale
            bbox = bbox.astype(int)
            color_intensity = int(200 - 200 * track.hand_score)
            if track.nb_trk_frame > 2:
                if track.last_seen <2:
                    color = (color_intensity, 255, color_intensity)
                else:
                    color=(0,0,0)
            else:color=(255, 255, 255)

            cv2.rectangle(numpy_image, bbox[:2], bbox[2:4], color, 2)
            cv2.putText(numpy_image, f'{category:s} ', (bbox[0],bbox[3] - 20), font, 1, color, 2, cv2.LINE_AA)
            # cv2.putText(numpy_image, f' {track.hand_score:.3f}', bbox[:2] + 20, font, 1, color, 2, cv2.LINE_AA)

            #tracked obj
            color_intensity = int(200 - 200 * track.obj_score)
            if track.nb_trk_frame > 2:
                if track.last_seen < 2:
                    color = (color_intensity, color_intensity, 255)
                else:
                    color = (0, 0, 0)
            else:
                color = (255, 255, 255)
            bbox = np.array(track.obj_bbx) * output_scale
            bbox = bbox.astype(int)
            cv2.rectangle(numpy_image, bbox[:2], bbox[2:4], color, 2)
            cv2.putText(numpy_image, f'{category:s} ', (bbox[0],bbox[3] - 20), font, 1, color, 2, cv2.LINE_AA)
            cv2.putText(numpy_image, f' {track.obj_score:.3f}', bbox[:2] +[-10,30], font, 1, color, 2, cv2.LINE_AA)
    else:

        # drow all object

        for bbox, prob in zip(o_boxes, o_scores):
            category = 'object'
            bbox = np.array(bbox) * output_scale
            bbox = bbox.astype(int)

            if max_hand_box:
                acceptable_size = box_size(bbox) < max_hand_box * max_size_ratio
            else:
                acceptable_size = True

            color_intensity = int(200 - 200 * prob)
            color = (255, 255, color_intensity)
            if prob > obj_prob_thresh:
                cv2.rectangle(numpy_image, bbox[:2], bbox[2:4], color, 2)
                cv2.putText(numpy_image, f'{category:s} {prob:.3f}', bbox[:2] +[0,20] , font, 1, color, 2, cv2.LINE_AA)

    # for bbox , prob in zip(p_boxes,p_scores):
    #     category = 'person'
    #     bbox = np.array(bbox) * output_scale
    #     bbox = bbox.astype(int)
    #     color_intensity = int(200 - 200 * prob)
    #     color = (color_intensity, 255, 255)
    #     if prob > peson_prob_thresh:
    #         cv2.rectangle(numpy_image, bbox[:2], bbox[2:4], color, 2)
    #         cv2.putText(numpy_image, f'{category:s} {prob:.3f}', bbox[:2] + 20, font, 1, color, 2, cv2.LINE_AA)
    return numpy_image
