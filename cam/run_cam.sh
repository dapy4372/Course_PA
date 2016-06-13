video_dir=./data/video

for i in `seq 1 3` 
do
    video_name=${video_dir}/$(date +"%Y%m%d%H%M%S").mp4
    raspivid -o - -t $video_time -w 320 -h 320 -hf -fps 15 | ffmpeg -r 15 -i - -vcodec copy -f mp4 $video_name
    echo $video_name
done
