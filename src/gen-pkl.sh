mkdir -p ../map ../pkl ../fbank_valid ../model ../log ../result ../result/smoothed_test_result  \
        ../result/smoothed_valid_result  ../result/test_result  ../result/valid_result ../result/final_result

echo '... generate map'
python ./data_prepare/0_gen-map.py

echo '... generate int label'
python ./data_prepare/1_gen-intlab.py

echo '... pick  validation set'
python ./data_prepare/2_pick_valid.py

echo '... make pkl file'
python ./data_prepare/3_make_pkl.py

echo '... preprocessing pkl file'
python ./data_prepare/4_preprocessing.py

echo 'done'
