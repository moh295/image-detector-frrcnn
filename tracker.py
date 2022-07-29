from utils import *

class Grasp:
    def __init__(self,hand_bbx,hand_score,obj_bbx,obj_score):
        self.hand_bbx=hand_bbx
        self.obj_bbx=obj_bbx
        self.last_seen=0
        self.hand_score=hand_score
        self.obj_score=obj_score
        self.obj_birth=obj_bbx
        self.nb_trk_frame = 0

class Grasp_tracker:
    def __init__(self):
        self.record=[]
        self.last_seen_thr=4
        self.iou_obj_thr=0.0 #overlp objects in the same frame
        self.ho_iou_thr=0.01 #hand to object minmum ovelap
        self.h_iou_over_frames=0.1 #hands overlap between frames
        self.min_iou_diff=0.5 # iou(h_F1 ,h_F2) -iou(obj_F1,obj_F2) ideal =0
        self.box_change_thr =0.78 # 0.0 - 1.0 where 0.0 no change
    def add(self,hand_bbx,hand_score,obj_bbx,obj_score):
        obj=Grasp(hand_bbx,hand_score,obj_bbx,obj_score)
        self.record.append(obj)
    def update(self,idx,hand_bbx,hand_score,obj_bbx,obj_score):
        self.record[idx].hand_bbx=hand_bbx
        self.record[idx].obj_bbx=obj_bbx
        self.record[idx].hand_score=hand_score
        self.record[idx].obj_score=obj_score
        self.record[idx].nb_trk_frame+=1
        self.record[idx].last_seen=0

    # remove h-o which lost track for more than "last_seen_thr" frames
    def clean(self,tracked_hand_idx):
        temp=[]

        for i in range(len(self.record)):
            added=False
            for idx in tracked_hand_idx:
                if idx != -1 and idx==i:
                    added=True
                    temp.append(self.record[i])
            if not added and self.record[i].last_seen<self.last_seen_thr:
                self.record[i].last_seen+=1
                temp.append(self.record[i])
        self.record = temp.copy()

    def hand_track(self,h_bbox):

        # not tracked hand has value of -1 the others will have the index of the corresponding hand in the record
        tracked_hand_idx=[-1]*len(h_bbox)

        #iou list between each hand to the ones in previous frame
        tracked_iou=[0]*len(h_bbox)

        for i in range(len(h_bbox)):
            max_iou = 0
            max_iou_at = 0
            iou_found = False

            for j in range(len(self.record)):
                iou=overlap(h_bbox[i],self.record[j].hand_bbx)
                if iou >self.h_iou_over_frames and iou>max_iou :
                    iou_found=True
                    max_iou=iou
                    max_iou_at=j
            if iou_found:
                tracked_hand_idx[i]=max_iou_at
                tracked_iou[i]=max_iou

        return  tracked_hand_idx, tracked_iou


    def find_iou_diff(self,h_boxes ,tracked_hand_idx, h1_h2_iou,o_boxes):
        '''

        this funtion finds two things first object overlap with hand array of indexes (hand_on_obj_idx)
        it has the length of o_boxes and contained the indexes of h_boxes
        seconds : in case the hand is tracked (tracked_hand_idx) !=-1 it (traked hands means both hands and object were deteced and saved privouse frame and the
        hand of the current frame is overlaping with the one from befor)
        the function will find the iou_opt_list which is a list of hand-object
         the minimization value calculated with h1_h2_iou and o1_o2_iou

        :param h_boxes:
        :param tracked_hand_idx:
        :param h1_h2_iou:
        :param o_boxes:
        :return: iou_opt_list,hand_on_obj_idx
        '''

        iou_opt_list = [False] * len(o_boxes)
        hand_on_obj_idx = [-1] * len(o_boxes)
        for i in range(len(o_boxes)):
            max_iou=0
            max_iou_at=0
            opt_iou=1
            opt_iou_at=0
            opt_iou_found=False
            iou_found=False
            for j in range(len(h_boxes)):
                # check if obj overlap with hand
                iou = overlap(h_boxes[j], o_boxes[i])

                if iou > self.ho_iou_thr:
                    #select the maximum overlap hand in case optim wasn't found yet
                    if iou > max_iou and not opt_iou_found:
                        max_iou=iou
                        max_iou_at = j
                        iou_found = True

                    # if this object overlap with tracked hand (hand detected in frame 1 and 2)
                    #find the optimal iou diff
                    if tracked_hand_idx[j] != -1:
                        #find iou between obj in frame 1 and obj  in frame 2
                        obj1_obj2_iou = overlap(o_boxes[i], self.record[tracked_hand_idx[j]].obj_bbx)
                        # the diffrence of iou between hands and objects will indicate if they move togather if iou_opt close to zero

                        change_ratio=chang_ratio(h1_h2_iou[j],obj1_obj2_iou)
                        iou_diff=change_ratio
                        if iou_diff<self.min_iou_diff and iou_diff<opt_iou:
                            opt_iou_found=True
                            iou_opt_list[i] = iou_diff
                            opt_iou=iou_diff
                            opt_iou_at = j

            #for obj overlap with tracked hand chose the hand that preduce min(iou_diff)
            if opt_iou_found:
                hand_on_obj_idx[i] = opt_iou_at
            #for obj ovrlap with non-tracked hand
            elif iou_found:
                hand_on_obj_idx[i] = max_iou_at
        return iou_opt_list,hand_on_obj_idx

    def remvoe_overlap_with_iou_opt(self,o_boxes,hand_on_obj_idx,iou_opt_list,o_scores):
        '''
        the function will clean up the object boxes base on some criteria , if the object dosen't fine any hand to attahced with
        if there is another overlap obj box that has more iou_opt (min(iou_diff)
        :param o_boxes:
        :param hand_on_obj_idx:
        :param iou_opt_list:
        :param o_scores:
        :return:  keep_obj
        '''

        if len(o_boxes) == 0: return False
        keep_obj = np.array([True] * len(o_boxes))
        # remvoe objects which not attached to hand
        for i in range(len(o_boxes)):
            if hand_on_obj_idx[i] == -1:
                keep_obj[i] = False
        # remvoing ojb boxes with minumzing the differnce beteween iou(hand_in_frame1, hand_in_frame2) and iou(box_in_frame_1,box_in_frame2)
        for box1 in range(len(o_boxes) - 1):
            for box2 in range(box1 + 1, len(o_boxes)):
                # to speed up don't compare with the removed ones
                if keep_obj[box1] and keep_obj[box2]:
                    # if boxes two boxes overlap and they belong to the same hands
                    #and the hand where they overlap is traked hand (if not tracked iou_opt_list[box1]=False)

                    if overlap(o_boxes[box1], o_boxes[box2]) > self.iou_obj_thr and hand_on_obj_idx[box1]==hand_on_obj_idx[box2] and iou_opt_list[box1]:
                        # chose the one with minimum iou_diff and not very lower ins score tho
                        if iou_opt_list[box1] > iou_opt_list[box2] and o_scores[box1] < o_scores[box2] * 5 and box_size(o_boxes[box1])*1.5>box_size(o_boxes[box2]):
                            keep_obj[box1] = False
                        else:
                            keep_obj[box2] = False
        return keep_obj

    def track(self, h_boxes,h_scores,o_boxes,o_scores):
        # if len(h_boxes)==0 or len(o_boxes)==0:return False
        tracked_hand_idx, h1_h2_iou = self.hand_track(h_boxes)
        record_size=len(self.record)
        #if nothing was tracked
        if not record_size:
            #remove overlp object based on min size and score
            keep_obj= my_nms(o_boxes,o_scores)
            o_boxes=o_boxes[keep_obj]
            o_scores=o_scores[keep_obj]
            #find hand object overalp , the function will retun a list of index sized as the objects list
            #it contains hand index for in case it's in contact with the corrsponding object
            _, hand_on_obj_idx = self.find_iou_diff(h_boxes, tracked_hand_idx, None, o_boxes)

            #adding new grasp (hand graping an object) to the list
            for i in range(len(o_boxes)):
                if hand_on_obj_idx[i] !=-1:
                    idx=hand_on_obj_idx[i]
                    self.add(h_boxes[idx],h_scores[idx], o_boxes[i],o_scores[i])

        #if there is some recored
        else:
            iou_opt_list, hand_on_obj_idx =self.find_iou_diff( h_boxes,tracked_hand_idx, h1_h2_iou,o_boxes)
            keep_obj=self.remvoe_overlap_with_iou_opt(o_boxes,hand_on_obj_idx,iou_opt_list,o_scores)
            o_boxes=o_boxes[keep_obj]
            o_scores=o_scores[keep_obj]
            hand_on_obj_idx=np.array(hand_on_obj_idx)[keep_obj]

            for i in range(len(tracked_hand_idx)):
                #if hand-obj was tracked update the record
                if tracked_hand_idx[i] !=-1:
                    for j in range(len(o_boxes)):
                        if hand_on_obj_idx[j]== i:
                            idx=tracked_hand_idx[i]
                            # if there is no big change in the size

                            if boxes_change_ratio(o_boxes[j],self.record[idx].obj_bbx) <self.box_change_thr:
                                self.update(tracked_hand_idx[i],h_boxes[i],h_scores[i],o_boxes[j],o_scores[j])
                            else:tracked_hand_idx[i] =-1

            # hand-obj detected for the first time add them to the list

            for i in range(len(tracked_hand_idx)):
                if tracked_hand_idx[i] == -1:
                    for j in range(len(o_boxes)):
                        if hand_on_obj_idx[j]== i:
                            self.add(h_boxes[i],h_scores[i],o_boxes[j],o_scores[j])

                #if hand is empty clean its tracked_hand_idx
                else:
                    has_and_obj = False
                    for j in range(len(o_boxes)):
                        if hand_on_obj_idx[j] == i:
                            has_and_obj = True
                    if not has_and_obj:
                        tracked_hand_idx[i]=-1
                    #remove h-o which lost track for more than "last_seen_thr" frames
        self.clean(tracked_hand_idx)
        return keep_obj