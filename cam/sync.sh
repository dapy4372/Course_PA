#! /bin/bash

rsync -a --ignore-existing -e "ssh" $pi_video_dir $pc:$pc_video_dir
rsync -a --ignore-existing -e "ssh" $pi_image_dir $pc:$pc_image_dir 
rsync -a --ignore-existing -e "ssh" $pi_table_dir $pc:$pc_table_dir
