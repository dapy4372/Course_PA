#! /bin/bash

rsync -avP --ignore-existing -e "ssh" $pi_video_dir $pc:$pc_video_dir
rsync -avP --ignore-existing -e "ssh" $pi_image_dir $pc:$pc_image_dir 
rsync -avP --ignore-existing -e "ssh" $pi_table_dir $pc:$pc_table_dir
scp $pi_data_dir/grp_list $pc:$pc_data_dir
