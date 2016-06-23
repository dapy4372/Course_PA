#! /bin/bash
while read line
do
    $DIR/cap.py $pi_video_dir/$line.mp4 $video_time
    $DIR/grp.py $pi_image_dir $pi_data_dir/grp_list
    $DIR/sync.sh
    rm -f $pi_video_dir/$line.mp4
    rm -f $pi_image_dir/$line*.jpg
    rm -f $pi_table_dir/$line*.txt
done
