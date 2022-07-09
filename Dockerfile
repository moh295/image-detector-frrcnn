FROM image-detector-frrcnn-base
WORKDIR /App
COPY . /App

ENTRYPOINT ["/bin/bash"]