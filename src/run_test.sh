datasetFilename=../pkl/nnet_dataset_without_preprocessing.pkl
mkdir -p ../log ../result ../model
s=1
dw=1024
dd=5
b=32
dh=1.
di=1.
lr=0.005
nowdate=project_nnet_without_preprocessing\_$(date +"%Y%m%d")
echo $nowdate $s $dw $dd $b $lr $dh $di $datasetFilename
stdbuf -o0 python 4_dnn.py $nowdate $s $dw $dd $b $lr $dh $di $datasetFilename  | tee ../log/$nowdate\_s_$s\_dw_$dw\_dd_$dd\_b_$b\_lr_$lr\_dh_$dh\_di_$di\.log
