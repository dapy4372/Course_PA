model_dir=/share/MLDS/glove_normalized_maxout_model/
modelname=glove_normalized_ionly_False_lonly_True_ifdim_0_iidim_0_lfdim_1800_dropout_0.2_activation_maxout_unit_1024_1024_1024
model=${model_dir}${modelname}.json
qfile=./data/question_wordvector/glove_normalized_300_test.pkl.gz
cfile=./data/choice_wordvector/glove_normalized_1500_test.pkl.gz
ifile=./data/image_feature/caffenet_4096_test.pkl.gz
ifdim=4096
lfdim=1800
ionly=False
lonly=True
w1=${model_dir}${modelname}_valid_01_epoch_040_loss_0.133_error_0.215.hdf5
w2=${model_dir}${modelname}_valid_02_epoch_040_loss_0.132_error_0.214.hdf5
w3=${model_dir}${modelname}_valid_03_epoch_040_loss_0.132_error_0.214.hdf5
w4=${model_dir}${modelname}_valid_04_epoch_040_loss_0.132_error_0.221.hdf5
w5=${model_dir}${modelname}_valid_05_epoch_040_loss_0.132_error_0.217.hdf5
w6=${model_dir}${modelname}_valid_06_epoch_040_loss_0.132_error_0.210.hdf5
w7=${model_dir}${modelname}_valid_07_epoch_040_loss_0.132_error_0.217.hdf5
w8=${model_dir}${modelname}_valid_08_epoch_040_loss_0.132_error_0.219.hdf5

#python evaluateLSTMandMLP.py -model ${model} -idim ${idim} -ldim ${ldim} -w ${w1} -qf ${qfile} -cf ${cfile} -if ${ifile} -predict_type train
for w in $w1 $w2 $w3 $w4 $w5 $w6 $w7 $w8 $w9 $w10
do
    python evaluateLSTMandMLP.py -model ${model} -ionly ${ionly} -lonly ${lonly} -ifdim ${ifdim} -lfdim ${lfdim} -w ${w} -qf ${qfile} -cf ${cfile} -if ${ifile} -predict_type test
done
#model=./data/models/glove_weightedsum_idim_4096_ldim_1800_dropout_0.4_unit_1024_1024_1024.json
#qfile=./data/question_wordvector/glove_weightedsum_300_test.pkl.gz
#cfile=./data/choice_wordvector/glove_sum_v2_1500_test.pkl.gz
#ifile=./data/image_feature/caffenet_4096_test.pkl.gz
#w=./data/models/glove_weightedsum_idim_4096_ldim_1800_dropout_0.4_unit_1024_1024_1024_valid_07_epoch_100_loss_3.142_error_0.259.hdf5
#python evaluateLSTMandMLP.py -model ${model} -idim ${idim} -ldim ${ldim} -w ${w} -qf ${qfile} -cf ${cfile} -if ${ifile} -predict_type test







