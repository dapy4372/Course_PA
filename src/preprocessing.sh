source ./path
ORIDATA=data

echo "copy feat from binary to txt..."
#test -e $ORIDATA/train.ark || 
copy-feats ark:$source_tr.ark ark,t:$ORIDATA/train.ark 
#test -e $ORIDATA/dev.ark || 
copy-feats ark:$source_dv.ark ark,t:$ORIDATA/dev.ark   
#test -e $ORIDATA/test.ark || 
copy-feats ark:$source_ts.ark ark,t:$ORIDATA/test.ark

echo "copy label from uchar to int..."
uchar-to-int32 ark:$source_tr.lab ark,t:$ORIDATA/train.lab || exit 1
uchar-to-int32 ark:$source_dv.lab ark,t:$ORIDATA/dev.lab || exit 1
uchar-to-int32 ark:$source_ts.lab ark,t:$ORIDATA/test.lab || exit 1

#echo "tranfer format..."
#python format.py

#echo "generate .gz file..."
#python gen-gz.py
