# image-detector-frrcnn

- image detectetor with boning boxes using faser RCNN with backbone mobilnet frn large 320
- datasetloder PSCAL VOC2007 format

- inference :
    -
    ````
  python3 inference --images [image path] --batch [batch size] --output [output path]
    
  ````
  example:
  ````
  python3 inference --/
  
- trainning :
  -
    ````
    train --data [data set folder with PASCAL VOC2007 format] --batch [batch size] --checkpoint [pretrained weight] --epoch [epoch]