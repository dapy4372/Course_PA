#! /bin/bash

pi_video_dir=/home/pi/es_finalproject/video/
pi_image_dir=/home/pi/es_finalproject/image/
pi_table_dir=/home/pi/es_finalproject/image/

pc=user@192.168.1.190
pc_video_dir=/home/user/for_fun/video
pc_image_dir=/home/user/for_fun/image
pc_table_dir=/home/user/for_fun/image

#for i in `seq 1 5` do
#    python writer.py
rsync -a --ignore-existing -e "ssh" $pi_video_dir $pc:$pc_video_dir &
rsync -a --ignore-existing -e "ssh" $pi_image_dir $pc:$pc_image_dir &
rsync -a --ignore-existing -e "ssh" $pi_table_dir $pc:$pc_table_dir &
#done
