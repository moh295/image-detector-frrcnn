FROM image-detector-frrcnn-base
WORKDIR /App
COPY . /App

#ENTRYPOINT ["/bin/bash"]
ENTRYPOINT ["python3","rename_image_folder.py"]