datasetFilename=../pkl/fbank_dataset.pkl
mkdir -p ../log ../result ../model
s=1
m=0.98
dw=4096
dd=4
b=32
dh=0.5
di=0.8
lr=0.03
nowdate=fbank\_$(date +"%Y%m%d")
echo $nowdate $s $dw $dd $b $lr $dh $di $datasetFilename
stdbuf -o0 python dnn.py $nowdate $s $m $dw $dd $b $lr $dh $di $datasetFilename  | tee ../log/$nowdate\_s_$s\_m_$m\_dw_$dw\_dd_$dd\_b_$b\_lr_$lr\_dh_$dh\_di_$di\.log
