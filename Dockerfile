#FROM image-detector-frrcnn-base
FROM ho-demo-cash

WORKDIR /App
COPY . /App
#RUN pip3 install pandas tensorboard
ENTRYPOINT ["/bin/bash"]
#ENTRYPOINT ["python3","live_demo.py"]
#ENTRYPOINT ["python3","train.py"]
# ENTRYPOINT ["python3","train.py","--checkpoint", "data/torch_trained_fasterrcnn_20x10p-30p.pth"]