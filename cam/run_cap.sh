#! /bin/bash
while read line
do
    python $DIR/cap.py $line $video_time
    $DIR/sync.sh
    rm -f $pi_video_dir/$line.mp4
    rm -f $pi_image_dir/$line*.png
    rm -f $pi_table_dir/$line*.txt
done
