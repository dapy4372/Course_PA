datasetFilename=../pkl/fbank_dataset.pkl
mkdir -p ../log ../result ../model
s=1
m=0.98
dw=1024
dd=7
b=32
dh=0.8
di=0.8
lr=0.07
nowdate=fbank\_$(date +"%Y%m%d")
echo $nowdate $s $dw $dd $b $lr $dh $di $datasetFilename
stdbuf -o0 python 4_dnn.py $nowdate $s $m $dw $dd $b $lr $dh $di $datasetFilename  | tee ../log/$nowdate\_s_$s\_m_$m\_dw_$dw\_dd_$dd\_b_$b\_lr_$lr\_dh_$dh\_di_$di\.log
