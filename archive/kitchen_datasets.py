from hoa import load_detections, DetectionRenderer
import PIL.Image
from typing import Union, List
from pathlib import Path
import PIL.Image

import cv2

def np_to_PIL(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(img)


class read_vid:
    def __init__(self,path):
        self.cap = cv2.VideoCapture(path)
        # get total number of frames

        self.totalFrames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
    def __getitem__(self,idx:int)-> PIL.Image.Image:

        # check for valid frame number
        if idx >= 0 & idx <= self.totalFrames:
            # set frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret: return np_to_PIL(frame)




class LazyFrameLoader:
    def __init__(self, path: Union[Path, str], frame_template: str = 'frame_{:010d}.jpg'):
        self.path = Path(path)
        self.frame_template = frame_template

    def __getitem__(self, idx: int) -> PIL.Image.Image:
        return PIL.Image.open(str(self.path / self.frame_template.format(idx + 1)))


# Adjust these to the where-ever your detections and frames are stored.

# detections_root should point to a folder with the structure
# detections_root
# |-- PXX
# |   |--- PXX_YY.pkl
detections_root = Path('data/detections')

# frame_root shout point to a folder with the structure
# frames_root
# |-- PXX
# |   |-- PXX_YY
# |   |   |-- frame_zzzzzzzzzz.jpg
frames_root = Path('data/frames')

video_id = 'P01_107'
participant_id = video_id[:3]
video_detections = load_detections(detections_root / participant_id / (video_id + '.pkl'))
max_frame_idx = len(video_detections) - 1
print(max_frame_idx)
frame_idx = 17
# video_detections[frame_idx]

# frames = LazyFrameLoader('data/frames/'+video_id)

vid_path='data/frames/'+video_id+'.MP4'
print(vid_path)
frames = read_vid(vid_path)
# frames[0].show()
renderer = DetectionRenderer(hand_threshold=0.1, object_threshold=0.1)
# renderer.render_detections(frames[frame_idx], video_detections[frame_idx])
for i in range(max_frame_idx):
    print(i)
    try:
        output=renderer.render_detections(frames[i], video_detections[i])
        # output.show()
        output.save(f"data/output/{i}.png")
    except: print(f'error in frame{i} annotation')




