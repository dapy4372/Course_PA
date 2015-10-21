datasetFilename=../pkl/nnet_dataset.pkl
mkdir -p ../log ../result ../model
s=1
b=32
lr=0.05
dd=3
di=0.1
dh=0.2
nowdate=nnet\_$(date +"%Y%m%d")

for dw in 1024 ; do
    for b in 128 64 32 ; do
        stdbuf -o0 python 4_dnn.py $nowdate $s $dw $dd $b $lr $dh $di $datasetFilename  | tee ../log/$nowdate\_s_$s\_dw_$dw\_dd_$dd\_b_$b\_lr_$lr\_dh_$dh\_di_$di\.log &
        stdbuf -o0 python 4_dnn.py $nowdate $s $dw $(($dd+1)) $b $lr $dh $di $datasetFilename  | tee ../log/$nowdate\_s_$s\_dw_$dw\_dd_$(($dd+1))\_b_$b\_lr_$lr\_dh_$dh\_di_$di\.log
        wait
    done
done
