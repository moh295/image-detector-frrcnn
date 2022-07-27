from utils import *


class Grasp:
    def __init__(self,hand_bbx,obj_bbx):
        self.hand_bbx=hand_bbx
        self.obj_bbx=obj_bbx
        self.last_seen=0
        x=int((hand_bbx[2] - hand_bbx[0])/2)
        y=int((hand_bbx[3] - hand_bbx[1])/2)
        self.picked_point=[x,y]
        self.picked_hand_size=box_size(hand_bbx)
        self.nb_trk_frame = 0


class Grasp_tracker:
    def __init__(self):
        self.record=[]
        self.last_seen_thr=3
    def add(self,hand_bbx,obj_bbx):
        obj=Grasp(hand_bbx,obj_bbx)
        self.record.append(obj)
    def update(self,idx,hand_bbx,obj_bbx):
        self.record[idx].hand_bbx=hand_bbx
        self.record[idx].obj_bbx=obj_bbx
        self.record[idx].nb_trk_frame+=1



    def remove(self,idx):
        self.record.pop(idx)

    def hand_track(self,h_bbox):
        # not tracked hand has value of -1 the others will have the index of the corresponding hand in the record
        hand_idx=[-1]*len(h_bbox)
        #iou list between each hand to the ones in previous frame
        tracked_iou=[0]*len(h_bbox)
        max_iou = 0
        max_iou_at = 0
        iou_found = False
        for i in range(len(h_bbox)):
            for j in range(len(self.record)):
                iou=overlap(h_bbox[i],self.record[j].hand_bbx)
                if iou >0 and iou>max_iou:
                    iou_found=True
                    max_iou=iou
                    max_iou_at=j
            if iou_found:
                hand_idx[i]=max_iou_at
                tracked_iou[i]=max_iou

        return  hand_idx, tracked_iou


    def find_iou_diff(self,h_boxes ,hand_idx, h1_h2_iou,o_boxes):


        iou_opt_list = [1] * len(o_boxes)
        hand_on_obj_idx = [-1] * len(o_boxes)
        for i in range(len(o_boxes)):

            max_iou=0
            max_iou_at=0
            iou_found=False
            for j in range(len(h_boxes)):
                # check if obj overlap with hand
                iou = overlap(h_boxes[j], o_boxes[i])
                if iou > 0 and iou>max_iou:
                    iou_found=True
                    max_iou=iou
                    max_iou_at=j
                    # if this object overlap with tracked hand (hand detected in frame 1 and 2)
                    if hand_idx[j] != -1:
                        obj1_obj2_iou = overlap(o_boxes[i], self.record[hand_idx[j]].obj_bbx)
                        iou_opt_list[i] = abs(h1_h2_iou[j] - obj1_obj2_iou)

            if iou_found:
                hand_on_obj_idx[i] = max_iou_at

        return iou_opt_list,hand_on_obj_idx

    def remvoe_overlap_with_iou_opt(self,o_boxes,hand_on_obj_idx,iou_opt_list,o_scores):
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
                    # if boxes overlap iou>0.5
                    if overlap(o_boxes[box1], o_boxes[box2]) > 0:
                        # chose the one with minmum iou_diff and not very lower ins score tho
                        if iou_opt_list[box1] > iou_opt_list[box2] and o_scores[box1] < o_scores[box2] * 5 and box_size(box1)*2>box_size(box2):
                            keep_obj[box1] = False
                        else:
                            keep_obj[box2] = False

        return keep_obj

    def track(self, h_boxes,o_boxes,o_scores):
        if len(h_boxes)==0 or len(o_boxes)==0:return False
        record_size=len(self.record)
        #if nothing was tracked
        if not record_size:
            #remove overlp object based on min size and score
            keep_obj= my_nms(o_boxes,o_scores)
            #find hand object overalp
            hand_idx=[-1]*len(h_boxes)
            _, hand_on_obj_idx = self.find_iou_diff(h_boxes, hand_idx, None, o_boxes)

            #adding new grasp (and graping an object) to the list
            for i in range(len(o_boxes)):
                if hand_on_obj_idx[i] !=-1:
                    self.add(h_boxes[hand_on_obj_idx[i]], o_boxes[i])
                    print('new ho was init-ed')

        #if there is some recored
        else:

            hand_idx, h1_h2_iou = self.hand_track(h_boxes)
            # print('step 1',hand_idx,h1_h2_iou)
            iou_opt_list, hand_on_obj_idx =self.find_iou_diff( h_boxes,hand_idx, h1_h2_iou,o_boxes)
            print('step 2',iou_opt_list, hand_on_obj_idx)
            keep_obj=self.remvoe_overlap_with_iou_opt(o_boxes,hand_on_obj_idx,iou_opt_list,o_scores)
            print('step 3',keep_obj)
            o_boxes=o_boxes[keep_obj]

            for i in range(len(hand_idx)):
                #if hand-obj was tracked update the record
                if hand_idx[i] !=-1:
                    for j in range(len(o_boxes)):
                        if hand_on_obj_idx[j]== i:
                            self.update(hand_idx[i],h_boxes[i],o_boxes[j])
                            print('hand was tracked and updated')

                #hand-obj detected for the first time
                else:
                    for j in range(len(o_boxes)):
                        if hand_on_obj_idx[j]== i:
                            self.add(h_boxes[i],o_boxes[j])
                            print('new ho was added')


            #remove h-o which lost track for more than "last_seen_thr" frames
            temp=[]
            for idx in hand_idx:
                if idx!=-1:
                    temp.append(self.record[idx])
            self.record=temp

        return keep_obj






