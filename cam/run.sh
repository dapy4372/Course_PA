export DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

export video_time=100000

export pi_data_dir=/home/pi/data
export pc_data_dir=/home/user/data

export pi_video_dir=$pi_data_dir/video/
export pi_image_dir=$pi_data_dir/image/
export pi_table_dir=$pi_data_dir/table/

export pc_video_dir=$pc_data_dir/video
export pc_image_dir=$pc_data_dir/image
export pc_table_dir=$pc_data_dir/table

export pc=user@192.168.1.190

[ -d $pi_video_dir ] || mkdir $pi_video_dir
[ -d $pi_image_dir ] || mkdir $pi_image_dir
[ -d $pi_table_dir ] || mkdir $pi_table_dir

ssh $pc '[ -d $pc_video_dir ] || mkdir $pc_video_dir'
ssh $pc '[ -d $pc_image_dir ] || mkdir $pc_image_dir'
ssh $pc '[ -d $pc_table_dir ] || mkdir $pc_table_dir'

$DIR/run_cam.sh | $DIR/run_cap.sh
