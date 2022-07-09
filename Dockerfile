FROM simple-classifier
WORKDIR /App

# #to dockerbase -- simple-classifier (v1)
# RUN pip3 install cython
# RUN pip3 install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
# # end -- simple-classifier (v1)

COPY . /App
#ENTRYPOINT ["python3","app_fasterrcnn_mobilenet_torchvision.py"]
#ENTRYPOINT ["python3","app_fasterrcnn_mobilenet_training.py"]
ENTRYPOINT ["python3","export_images.py"]