dropout=0.5
cross_valid=1
language_dim=1800
image_dim=4096
image_feature=$1
question_feature=$2
choice_feature=$3
epochs=100

python trainLSTMandMLP.py -u 512 512 512 \
                          -dropout ${dropout} \
                          -ldim ${language_dim} \
                          -idim ${image_dim} \
                          -cross_valid ${cross_valid} \
                          -qf ${question_feature} \
                          -cf ${choice_feature} \
                          -if ${image_feature} \
                          -epochs ${epochs}
