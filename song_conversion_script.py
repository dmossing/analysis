#!/usr/bin/env python
import os
import sys

def run(number):
    print('converting to WAV...')
    os.system('ffmpeg -i %s.mp3 -vn -acodec pcm_s16le -ac 1 -ar 44100 -f wav %s.wav'%(number,number))
    #os.system('cp %s.wav ~/Documents/data/birbs/birdsong/'%number)
    return number+'.wav'

if __name__ == "__main__":
    number = str(sys.argv[1])
    _ = run(number)
