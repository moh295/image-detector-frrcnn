
#ssh machines
# id D22rFbRcj61648572524
puplic ip
ssh guillermo@45.158.142.229
EYA61yxxn

#id 4hIc24bZXK1644913824
local ip
ssh guillermo@192.168.188.124
Mb6TNx0hR8


#id ZzTe4LcpaA1657108798
puplic ip
ssh guillermo@45.158.142.230
vdP93907qi
# hand object dataset download
wget http://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/downloads/pascal_voc_format.zip
#transfer files
zip -r /media/workspace/hand_object_datasets/output.zip /media/workspace/hand_object_datasets/output
scp -r guillermo@192.168.178.143:/media/workspace/hand_object_datasets/torch_trained_fasterrcnn_100p.pth torch_trained_fasterrcnn_100p.pth
scp -r guillermo@45.158.142.229:/media/workspace/hand_object_datasets/retrain_fasterrcnn_80k.pth retrain_fasterrcnn_80k.pth
scp -r guillermo@45.158.142.229:/media/workspace/hand_object_datasets/retrain_fasterrcnn_80k_4c.pth retrain_fasterrcnn_80k_4c.pth
scp -r guillermo@45.158.142.229:/media/workspace/hand_object_datasets/output.mp4 output.mp4
scp -r guillermo@45.158.142.230:/media/workspace/hand_object_datasets/output_ho3_13c_200in_100out.mp4 output_ho3_13c_200in_100out.mp4
scp -r guillermo@45.158.142.230:/media/workspace/hand_object_datasets/base_4c base_4c
screen -dRaA -S torch
screen -dRaA -S test

# github access
eval $(ssh-agent);ssh-add /home/guillermo/.ssh/id_ed25519_new
git remote add origin https://github.com/moh295/image-detector-frrcnn.git

# build base
git pull; docker build . -t image-detector-frrcnn-base -f DockerfileBase
docker run -it --privileged -v /media/workspace/hand_object_datasets:/App/data --shm-size 50G image-detector-frrcnn-base


# run docker
git pull ; docker build . -t image-detector-frrcnn; docker run -it --privileged -v /media/workspace/hand_object_datasets:/App/data --shm-size 50G image-detector-frrcnn
python3 live_demo.py --batch 1  --checkpoint data/torch_trained_fasterrcnn_20x10p-30p.pth
python3 inference.py --checkpoint data/torch_trained_fasterrcnn_100p.pth --scale 1


trtexec --onnx=faster_rcnn.onnx --saveEngine=frrcnn_engine.trt
python3 -c "import torch ;print(torch.cuda.is_available)"


ideal camera res 480x 640 scaled to 0.6 @30fps

python3 inference_mp4_video.py --video data/ho3.mp4 --output data/output_ho3_13c_60in_200out.mp4 --checkpoint data/retrain_fasterrcnn_80k_8ep_13c.pth ;python3 inference_mp4_video.py --video data/ho3.mp4 --input_scale 2 --output_scale 1 --output data/output_ho3_13c_200in_100out.mp4 --checkpoint data/retrain_fasterrcnn_80k_8ep_13c.pth




python3 inference_mp4_video.py --video data/ho3.3.mp4 --checkpoint data/torch_trained_fasterrcnn_100p.pth --input_scale 1 --output_scale 1 --fps 60
python3 inference_mp4_video.py --video data/ho3.3.mp4 --checkpoint data/torch_trained_fasterrcnn_100p.pth --input_scale 0.3 --output_scale 1

python3 inference_mp4_video.py   --output data/base_4c.mp4;
python3 inference_mp4_video.py  --video data/ho3.mp4 --input_scale 1.2 --output_scale 2.5 --output data/ho3_4c_120-250.mp4 ;
python3 inference_mp4_video.py  --video data/ho2.mp4 --output data/ho2_4c_600-200.mp4;
python3 inference_mp4_video.py --video data/ho3.3.mp4 --output_scale 1 --output data/ho3.3_4c.mp4

scp -r guillermo@45.158.142.230:/media/workspace/hand_object_datasets/base_4c.mp4 base_4c.mp4
scp -r guillermo@45.158.142.230:/media/workspace/hand_object_datasets/ho2_4c_600-200.mp4 ho2_4c_600-200.mp4
scp -r guillermo@45.158.142.230:/media/workspace/hand_object_datasets/ho3_4c_120-250.mp4 ho3_4c_120-250.mp4
scp -r guillermo@45.158.142.230:/media/workspace/hand_object_datasets/ho3.3_4c.mp4 ho3.3_4c.mp4
