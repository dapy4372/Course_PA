while read line
do
    python cap.py $line $video_time
    ./sync.sh
    rm -f ./data/video/$line.mp4
    rm -f ./data/image/$line*.png
    rm -f ./data/table/$line*.tab
done
