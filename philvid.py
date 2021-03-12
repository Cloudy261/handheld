from perlin_noise import PerlinNoise
import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

def ocr_core(frame):
    """
    This function will handle the core OCR processing of images.
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(frame)
    text = pytesseract.image_to_string(im_pil)  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    return text

os.chdir('.')
parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("framerate")
parser.add_argument("total_frames")
args = parser.parse_args()

input_pic = args.input
output_video = args.output
framerate = int(args.framerate)
total_frames = int(args.total_frames)

bordersize = 75
rot_max = 5
show = True

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], borderValue=(0,0,0), flags=cv2.INTER_LINEAR)
  return result


video_count = 1

noise = PerlinNoise()
print('hi phil')
print(os.listdir(input_pic))
#cap = cv2.VideoCapture('/Users/claudi/Downloads/RPReplay_Final1615234696.MP4')
for fr in os.listdir(input_pic):
    if fr[-4:] == 'jpeg' or fr[-3:] == 'jpg':
        fr = 'in/' + fr
        
        frame = cv2.imread(fr, 1)
        text = ocr_core(frame)
        print(text)
        text.replace(' ', '_')
        out =  output_video + text + '.mp4'
        video_count += 1
        height, width, channels = frame.shape
        print((width + 2 * bordersize , height + 2 * bordersize))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out, fourcc , framerate, (width + 2 * bordersize , height + 2 * bordersize))

        noise1 = PerlinNoise(octaves=3)
        noise2 = PerlinNoise(octaves=6)
        noise3 = PerlinNoise(octaves=12)
        noise4 = PerlinNoise(octaves=24)
        width = 1
        xpix, ypix = total_frames, width
        picl = []
        #cv2.imshow('frame', dest_img)

        for i in range(xpix):

            noise_val =         noise1([i/xpix, 1/ypix])
            noise_val += 0.5  * noise2([i/xpix, 1/ypix])
            noise_val += 0.25 * noise3([i/xpix, 1/ypix])
            noise_val += 0.125* noise4([i/xpix, 1/ypix])
            picl.append(noise_val)
            
            
        pic = [(int(element * bordersize)) for element in picl]
        pic2 = pic[::-1]
        rot = [(element * rot_max) for element in picl]
        print('offset done')
        print(pic)
        print(pic2)


        for i in tqdm(range(total_frames)):
            offset = pic[i]
            offset2 = pic2[i]
            rotation = rot[i]
            #ret,frame = cap.read()
            height, width, channels = frame.shape
            row, col = frame.shape[:2]
            border = cv2.copyMakeBorder(
                frame,
                top=bordersize - offset,
                bottom=bordersize + offset,
                left=bordersize - offset2,
                right=bordersize + offset2,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )

            border = rotate_image(border, rotation)
            if show:
                cv2.imshow('frame', border)
                if cv2.waitKey(1) &0XFF == ord('c'):
                    break
            out.write(border)
            #print(str(i) + '-ter y-offset: ' + str(offset) + ', x-offset: ' + str(offset2) + ', rotation: {:.2f}°'.format(rotation))
        out.release()


#cap.release()
cv2.destroyAllWindows()
    
    
    
    
    