FROM image-detector-frrcnn-base
WORKDIR /App
COPY . /App

#ENTRYPOINT ["/bin/bash"]
#ENTRYPOINT ["python3","live_demo.py"]
ENTRYPOINT ["python3","train.py"]
#ENTRYPOINT ["python3","train.py","--checkpoint", "data/torch_trained_fasterrcnn.pth"]