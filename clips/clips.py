import os
import cv2

import argparse

# convert clip (as mp4) to frames (as jpg) every [delay] frames
def clip_to_frames(path, delay=100, scale=1.0):
    if not os.path.exists(path):
        raise ValueError("Path name ({}) does not exist".format(path))
    name = os.path.splitext(path)[0]
    video = cv2.VideoCapture(path)
    success, image = video.read()
    count = 0
    while success:
        if count % delay == 0:
            print(count)
            new_x, new_y = image.shape[1]*scale, image.shape[0]*scale
            image = cv2.resize(image, (int(new_x), int(new_y)))
            cv2.imwrite("{}_frame{}.jpg".format(name, count), image)
        success, image = video.read()
        count += 1
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File name of clip.")
    parser.add_argument("-d", "--delay", type=int, default=100, help="Frame delay.")
    parser.add_argument("-s", "--scale", type=float, default=1.0, help="Image scaling factor.")
    args = parser.parse_args()

    print("Transferring clip to frames...")
    clip_to_frames(args.file, args.delay, args.scale)
    print("Success.")
