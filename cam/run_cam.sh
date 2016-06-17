#! /bin/bash

for i in `seq 1 2` 
do
    cur_date=$(date +"%Y%m%d%H%M%S")
    video_name=${pi_video_dir}/${cur_date}.mp4
    raspivid -o - -t $video_time -w 320 -h 320 -hf -fps 15 | ffmpeg -r 15 -i - -vcodec copy -f mp4 $video_name
    echo $cur_date
done
