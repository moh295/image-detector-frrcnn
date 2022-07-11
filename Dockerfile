FROM image-detector-frrcnn-base
WORKDIR /App
COPY . /App

#ENTRYPOINT ["/bin/bash"]
#ENTRYPOINT ["python3","live_demo.py"]
ENTRYPOINT["python3","train","--checkpoint torch_trained_fasterrcnn.pth"]