import torch
from torchvision import models
from timeit import default_timer as timer
from datetime import timedelta
import cv2
import numpy as np
from utils import image_resize
class FrRCNN:
    def __init__(self,checkpoint='data/torch_trained_fasterrcnn_100p.pth'):
        # self.labels_dict = ['targetobject', 'hand']
        self.labels_dict = {'Hand_free_R': 1, 'Hand_free_L': 2, 'Hand_cont_R': 3, 'Hand_cont_L': 4,
                       'person_R': 5, 'person_L': 6, 'person_LR': 7, 'portable_R': 8,
                       'portable_L': 9, 'portable_LR': 10, 'non-portable_R': 11, 'non-portable_L': 12,
                       'non-portable_LR': 13}

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # loading model
        self.model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False).to(device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=torch.device(device)))
        self.model.eval()

    def predict(self,
                save_dir,
                tensors, images, count=1,
                save=False, show=False,
                show_vid=False, save_vid=False,output_scale=1):
        # apply model on images and save the result

        font = cv2.FONT_HERSHEY_SIMPLEX
        # Dictionary containing some colors
        colors = {'blue': (0, 0, 255), 'green': (0, 255, 0), 'red': (255, 0, 0), 'orange': (223, 70, 14),
                  'yellow': (255, 255, 0),
                  'magenta': (255, 0, 255), 'cyan': (0, 255, 255), 'white': (255, 255, 255), 'black': (0, 0, 0),
                  'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
                  'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220), 'red 1': (255, 82, 82),
                  'red 2': (255, 82, 82), 'red 3': (255, 82, 82)}

        obj_prob_thresh = 0.15
        hand_prob_thresh = 0.30
        cnt = count
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        start = timer()

        start_pred = timer()
        tensors = list(tensor.to(device) for tensor in tensors)
        # print('prediction started')
        predictions = self.model(tensors)
        end = timer()
        elapsed = timedelta(seconds=end - start_pred)
        print(f'prediction takes {elapsed}')
        path_to_output_image = save_dir

        for data, image in zip(predictions, images):
            detection_bboxes, detection_classes, detection_probs = data['boxes'].cpu().detach().numpy(), \
                                                                   data['labels'].cpu().detach().numpy(), data[
                                                                       'scores'].cpu().detach().numpy()

            kept_indices = detection_probs > obj_prob_thresh
            detection_bboxes = detection_bboxes[kept_indices]
            detection_classes = detection_classes[kept_indices]
            detection_probs = detection_probs[kept_indices]
            draw = np.copy(image)
            for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
                bbox = np.array(bbox).astype(int)
                category = list(self.labels_dict.keys())[cls - 1]

                color_intensity = int(200 - 200 * prob)
                Free_hand=cls <3
                Contact_hand=2<cls <5
                color =  (225, color_intensity, color_intensity) if Free_hand else (color_intensity, 255, color_intensity)
                hand=Free_hand or Contact_hand
                if hand and prob > hand_prob_thresh:
                    cv2.rectangle(draw, bbox[:2], bbox[2:4], color, 2)
                    cv2.putText(draw, f'{category:s} {prob:.3f}', bbox[:2]+20, font, 1, color, 2, cv2.LINE_AA)
                elif not hand and prob > obj_prob_thresh:
                    color = (color_intensity, color_intensity, 255)
                    cv2.rectangle(draw, bbox[:2], bbox[2:4], color, 2)
                    cv2.putText(draw, f'{category:s} {prob:.3f}', bbox[:2]+20, font, 1, color, 2, cv2.LINE_AA)

            if save:

                cv2.imwrite(path_to_output_image + str(cnt) + '.png', image_resize(draw,output_scale))
                print(f'Output image is saved to {path_to_output_image}{cnt}.png')
            cnt += 1
            if show:
                cv2.imshow('frame', image_resize(draw,output_scale))
                cv2.waitKey(0)
            if show_vid:
                cv2.imshow('output image', image_resize(draw,output_scale))
                cv2.waitKey(1)
            if save_vid:
                self.video.write(image_resize(draw,output_scale))
                print('image added to video ', draw.shape)

        end = timer()
        elapsed = timedelta(seconds=end - start)
        print(f'full prediction process takes {elapsed} [+ annotation ,saving..]')

    def create_video(self, width, height, fps, output):
        # Define the codec and create VideoWriter object
        print('initiating video ...')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
        self.video  = cv2.VideoWriter(output, fourcc, fps, (width, height))


    def realse_vid(self):
        print('releasing video ..')
        self.video.release()
