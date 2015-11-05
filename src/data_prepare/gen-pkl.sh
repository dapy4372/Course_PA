mkdir -p ../../map ../../pkl ../../fbank_valid ../../model ../../log ../../result \
         ../../result/smoothed_test_result ../../result/smoothed_valid_result  \
         ../../result/test_result  ../../result/valid_result ../../result/final_result

echo '... generate map'
python ./0_gen-map.py

echo '... generate int label'
python ./1_gen-intlab.py

echo '... pick  validation set'
python ./2_pick_valid.py

echo '... make pkl file'
python ./3_make_pkl.py

#echo '... preprocessing pkl file'
#python ./4_preprocessing.py

echo 'done'
