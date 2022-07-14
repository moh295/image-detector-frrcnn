import cv2
import argparse
print(cv2.__version__)

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        #vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line
        success,image = vidcap.read()
        print ('Read a new frame: ', success)

        #name the files with (0000) four digits index
        strCount = str(count)
        strCount = '0' * (4 - len(strCount)) + strCount

        try:
            # resized = resize(image, 500, 375)
            cv2.imwrite( pathOut + strCount+"_frame.png", image)     # save frame as JPEG file
            print(pathOut + strCount+"_frame.png was saved")
            count = count + 1
        except:pass

def resize(src,w,h):
    width = int(w)
    height = int(h)
    dim = (width, height)

    # resize image
    return  cv2.resize(src, dim, interpolation=cv2.INTER_AREA)

if __name__=="__main__":

    video_path= 'data/input.mp4'
    image_folder='data/images/'
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="path to video",default=video_path)
    a.add_argument("--output", help="path to images folder ",default=image_folder)
    args = a.parse_args()
    print(args)

    extractImages(args.input, args.output)