
#ssh machines
D22rFbRcj61648572524
puplic ip
ssh guillermo@45.158.142.229
EYA61yxxn

4hIc24bZXK1644913824
local ip
ssh guillermo@192.168.188.124
Mb6TNx0hR8


ZzTe4LcpaA1657108798
puplic ip
ssh guillermo@45.158.142.230
vdP93907qi
# hand object dataset download
wget http://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/downloads/pascal_voc_format.zip
#transfer files
zip -r /media/workspace/hand_object_datasets/output.zip /media/workspace/hand_object_datasets/output
scp -r guillermo@192.168.178.143:/media/workspace/hand_object_datasets/torch_trained_fasterrcnn_100p.pth torch_trained_fasterrcnn_100p.pth
scp -r guillermo@45.158.142.229:/media/workspace/hand_object_datasets/torch_trained_fasterrcnn_100p.pth  torch_trained_fasterrcnn_100p.pth
scp -r guillermo@45.158.142.229:/media/workspace/hand_object_datasets/retrain_fasterrcnn_80k.pth retrain_fasterrcnn_80k.pth


screen -dRaA -S torch

eval $(ssh-agent);ssh-add /home/guillermo/.ssh/id_ed25519_new
git remote add origin https://github.com/moh295/image-detector-frrcnn.git

git pull; docker build . -t image-detector-frrcnn-base -f DockerfileBase

rm images.zip; zip -r images.zip output;aws s3 cp images.zip s3://systemimages;rm output/*.png

git pull ; docker build . -t image-detector-frrcnn; docker run -it --privileged -v /media/workspace/hand_object_datasets:/App/data --shm-size 50G image-detector-frrcnn
python3 live_demo.py --batch 1  --checkpoint data/torch_trained_fasterrcnn_20x10p-30p.pth
python3 inference.py --checkpoint data/torch_trained_fasterrcnn_100p.pth --scale 1


trtexec --onnx=faster_rcnn.onnx --saveEngine=frrcnn_engine.trt
python3 -c "import torch ;print(torch.cuda.is_available)"


