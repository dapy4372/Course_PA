datasetFilename=../pkl/fbank_no_valid_dataset.pkl
outputFilename=$1\_only_trainset
mkdir -p ../log ../result ../model
stdbuf -o0 python 7_dnn_only_trainset.py $datasetFilename $outputFilename | tee ../log/$outputFilename\.log
