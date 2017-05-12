TRAIN_LABEL=/home/roylu/share/MLDS_data/label/train_int.lab
TEST_PROB=/home/roylu/share/mlds_hw3_upload/for_hmm_test_prob_input
HMM_OUTPUT=./hmmoutput
all: hmm.cpp 
	g++ -g hmm.cpp -o hmm
run:
	python count.py ${TRAIN_LABEL} trainset_count_prob
	./hmm trainset_count_prob ${TEST_PROB} ${HMM_OUTPUT}
	python ../shrink/post_shrink setting.txt
clean:
	rm -f hmm
