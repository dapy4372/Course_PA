mkdir -p ../map ../pkl

echo '...generate map'
python 0_gen-map.py

echo '...generate int label'
python 1_gen-intlab.py

echo '...create pkl file'
python 2_create_pkl.py

echo '...preprocessing pkl file'
python 3_preprocessing.py

echo 'done'
