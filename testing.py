import numpy as np
import cv2
import ctypes
import time

user32 = ctypes.windll.user32
scl,scb = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
print(scl,scb) #screen metrics


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized


cap = cv2.VideoCapture(0)
w = cap.set(3, scl/2)
h = cap.set(4, scb/2)

timeout = time.time() + 3

img_path = './GER.png'      #CHANGE IMAGES HERE.
logo = cv2.imread(img_path, -1)
watermark = image_resize(logo, height= 200)
watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)
hard = 0.6

while(time.time() <= timeout):
    # Capture frame-by-frame
        ret, framest = cap.read()
        frame = cv2.flip(framest, +1)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        cv2.imshow('frame',frame)
        print("work")
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

while (True):
    
        ret, framest = cap.read()
        frame = cv2.flip(framest, +1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        frame_h, frame_w, frame_c = frame.shape
    # overlay with 4 channels BGR and Alpha
        overlay = np.zeros((frame_h, frame_w, 4), dtype='uint8')
        watermark_h, watermark_w, watermark_c = watermark.shape
    # replace overlay pixels with watermark pixel values
    
        for i in range(0, watermark_h):
            for j in range(0, watermark_w):
                if watermark[i,j][3] != 0:
                    offset = 25
                    h_offset = frame_h - watermark_h - offset
                    w_offset = frame_w - watermark_w - offset
                    overlay[h_offset + i, w_offset+ j] = watermark[i,j]
                

        cv2.addWeighted(overlay, hard, frame, 1.0, 0, frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    #out.write(frame)
    # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
#out.release()
cv2.destroyAllWindows() 
