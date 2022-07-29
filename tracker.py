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
    def add(self,obj):
        hand_bbx, hand_score, obj_bbx, obj_score= obj
        obj=Grasp(hand_bbx,hand_score,obj_bbx,obj_score)
        self.record.append(obj)
    def update(self,idx,obj):
        hand_bbx, hand_score, obj_bbx, obj_score = obj
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

    def hand_obj_pair(self,h_boxes,h_scores,o_boxes,o_scores,one2one=False):
        pairs = []
        if not len(h_boxes): return pairs
        #for each object take max iou hand
        if one2one:
            for o, os in zip(o_boxes, o_scores):
                max_iou=0
                max_at=0
                for i in range(len(h_boxes)):
                    iou=overlap(h_boxes[i], o)
                    if iou>self.ho_iou_thr and iou >max_iou :
                        max_iou=iou
                        max_at=i
                pairs.append([h_boxes[max_at],h_scores[max_at],o,os])
        #creat a pair for each hand overlap with object
        else:
            for h,hs in zip(h_boxes,h_scores):
                for o,os in zip (o_boxes,o_scores):
                    if overlap(h,o)>self.ho_iou_thr:
                        pairs.append([h,hs,o,os])

        return pairs

    def tracked_pairs(self,pairs):

        tracked_pairs=[]
        new_pairs=[]
        pairs_link=[]
        idx=range(len(self.record))
        for h,hs,o,os in pairs:
            for r,i in zip(self.record,idx):
                h1_h1_iou = overlap(h, r.hand_bbx)
                o1_o2_iou = overlap(o, r.obj_bbx)
                if o1_o2_iou>0 and h1_h1_iou >0:
                    pairs_link.append(i)
                    tracked_pairs.append([h,hs,o,os,h1_h1_iou,o1_o2_iou])
                else:
                    new_pairs.append([h,hs,o,os])


        return new_pairs,tracked_pairs ,pairs_link


    def selec_opt_paires(self,pairs):
        if len(pairs) == 0: return False
        keep_me = np.array([True] * len(pairs))

        # remvoing ojb boxes with minumzing the differnce beteween iou(hand_in_frame1, hand_in_frame2) and iou(box_in_frame_1,box_in_frame2)
        for option1 in range(len(pairs) - 1):
            for option2 in range(option1 + 1, len(pairs)):
                #unpack
                h1,hs1,o1,os1,h1_ious,o1_ious=pairs[option1]
                h2,hs1,o2,os2,h2_ious,o2_ious=pairs[option2]
                # to speed up don't compare with the removed ones
                if keep_me[option1] and keep_me[option2]:

                    # if boxes two boxes overlap
                    # and the hand where they overlap is traked hand (if not tracked iou_opt_list[box1]=False)

                    if overlap(o1, o2) > self.iou_obj_thr  :
                        # chose the one with minimum iou_diff and not very lower ins score tho
                        if o1_ious > o2_ious and os1 < os2 * 5 :
                            keep_me[option1]  = False
                        else:
                            keep_me[option2]  = False
        return keep_me


    def track(self, h_boxes,h_scores,o_boxes,o_scores):

        record_size=len(self.record)
        #if nothing was tracked
        if not record_size:
            #remove overlp object based on min size and score
            keep_obj= my_nms(o_boxes,o_scores)
            o_boxes=o_boxes[keep_obj]
            o_scores=o_scores[keep_obj]
            pairs=self.hand_obj_pair(h_boxes,h_scores,o_boxes,o_scores,one2one=True)
            if len(pairs):
                for obj in pairs:
                    self.add(obj)

        #if there is some recored
        else:
            pairs_link = []
            pairs=self.hand_obj_pair(h_boxes,h_scores,o_boxes,o_scores)
            if len(pairs):
                new_pairs,tracked_pairs,pairs_link=self.tracked_pairs(pairs)
                keep_those=self.selec_opt_paires(tracked_pairs)
                opt_pairs=np.array(tracked_pairs,dtype=object)[keep_those]
                pairs_link=np.array(pairs_link)[keep_those]
                for idx,pairs in zip(pairs_link,opt_pairs):

                    self.update(idx,pairs[:4])
                    for new in new_pairs:
                        self.add(new)


            #remove h-o which lost track for more than "last_seen_thr" frames
            self.clean(pairs_link)
