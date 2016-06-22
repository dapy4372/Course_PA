#! /bin/bash
while read line
do
    python $DIR/cap.py $pi_video_dir/$line.mp4 $video_time
    $DIR/sync.sh
    rm -f $pi_video_dir/$line.mp4
    rm -f $pi_image_dir/$line*.jpg
    rm -f $pi_table_dir/$line*.txt
done
