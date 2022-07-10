FROM image-detector-frrcnn-base
WORKDIR /App
COPY . /App

#ENTRYPOINT ["/bin/bash"]
ENTRYPOINT ["python3","video_to_images.py]