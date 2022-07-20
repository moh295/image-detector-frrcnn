import torch
from torchvision import models
from timeit import default_timer as timer
from datetime import timedelta
from utils import *
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


            draw = np.copy(image_resize(image,output_scale))

            # draw=annutaion_13_classes(draw,detection_bboxes,detection_classes,detection_probs,output_scale)
            draw = annutaion_2_classes(draw, detection_bboxes, detection_classes, detection_probs, output_scale)


            if save:
                cv2.imwrite(path_to_output_image + str(cnt) + '.png', image_resize(draw,output_scale))
                print(f'Output image is saved to {path_to_output_image}{cnt}.png')
            cnt += 1
            if show:
                cv2.imshow('frame', draw)
                cv2.waitKey(0)
            if show_vid:
                cv2.imshow('output image', draw)
                cv2.waitKey(1)
            if save_vid:
                self.video.write(draw)
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
