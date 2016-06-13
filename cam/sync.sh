#! /bin/bash

pi_data_dir=/home/pi/data
pc_data_dir=/home/user/data

pi_video_dir=$pi_data_dir/video/
pi_image_dir=$pi_data_dir/image/
pi_table_dir=$pi_data_dir/image/

pc_video_dir=$pc_data_dir/video
pc_image_dir=$pc_data_dir/image
pc_table_dir=$pc_data_dir/image

pc=user@192.168.1.190

rsync -a --ignore-existing -e "ssh" $pi_video_dir $pc:$pc_video_dir
rsync -a --ignore-existing -e "ssh" $pi_image_dir $pc:$pc_image_dir 
rsync -a --ignore-existing -e "ssh" $pi_table_dir $pc:$pc_table_dir
