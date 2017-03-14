import numpy as np
import csv
import os
import subprocess

import cv2
import scipy

"""
Downlaods yt-bb dataset and stores into csv_dir
Downsamples each image to 256x256x3
"""

# CONSTANTS:
SAVE_DIR = '/data/efros/ahliu/yt-bb/'
CSV_DIR = '/data/efros/ahliu/yt_bb_detection_train.csv'
DELIM = '=' # _ and - are both used in yt-id
temp_dir = SAVE_DIR+'temp_vid.mp4'
IMAGE_SIZE = 256
LOG_DIR = '/home/ahliu/logs/294-131_logs'

current_id = ""
cap = None

f = open(CSV_DIR, "r")
reader = csv.reader(f)
f = open(LOG_DIR, 'w')
f.write("\n\n =========\nSTARTING NEW PIPELINE\n \n\n")

for row in reader:
    if row[5] == "absent":
        # Object is missing from the frames
        continue 
    yt_id = row[0] # youtube-video-id
    time = int(row[1]) # in ms
    category = row[3] # Human Readable String
    xmin, xmax, ymin, ymax = row[6], row[7], row[8], row[9] #Floats
    
    save_name = SAVE_DIR + yt_id + DELIM + str(time) + DELIM + \
                category + DELIM + xmin + DELIM + \
                xmax + DELIM +ymin+DELIM + ymax + '.jpg'
            
    try:
        if current_id != yt_id:
            if os.path.exists(temp_dir):
                os.remove(temp_dir)
            current_id = yt_id
            subprocess.check_call(['youtube-dl', '-f', 'best[ext=mp4]', '-o', 
                                  temp_dir, 'http://youtu.be/'+yt_id])
            cap = cv2.VideoCapture(temp_dir)

        cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, time)
        ret, frames = cap.read()
        frames = cv2.resize(frames, (256, 256)) 
        cv2.imwrite(save_name, frames)
        print "Wrote %s" % save_name
    except subprocess.CalledProcessError as e:
        f.write("Error downloading video: %s\n\n" % e)
        continue
    except cv2.error as e:
        f.write("Error saving video: %s\n\n" % e)
        continue

f.close()