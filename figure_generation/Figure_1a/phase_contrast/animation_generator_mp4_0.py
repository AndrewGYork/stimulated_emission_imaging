import os
from subprocess import check_call


print("Converting to mp4...", end='')
convert_command = [
    'ffmpeg', '-y',
    '-r', '10',
    '-f', 'image2',
    '-i', 'img%d.png',
    '-movflags', 'faststart',
    # See here: https://trac.ffmpeg.org/wiki/Encode/H.264#Encodingfordumbplayers
    '-pix_fmt', 'yuv420p',
    '-vcodec', 'libx264',
    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', # Ensures divisible by 2
    '-preset', 'veryslow',
    '-crf', '25',
    'fig_1a_video.mp4']
try:
    with open('conversion_messages.txt', 'wt') as f:
        f.write("So far, everthing's fine...\n")
        f.flush()
        check_call(convert_command, stderr=f, stdout=f)
        f.flush()
    os.remove('conversion_messages.txt')
except: # This is unlikely to be platform independent :D
    print("MP4 conversion failed. Is ffmpeg installed?")
    raise
print('done.')

