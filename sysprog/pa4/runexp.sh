n=$1
>&2 echo 'seg_size = '${n}
./merger ${n} < test_${n}
>&2 echo ""
>&2 echo 'seg_size = '$((${n}/2))
./merger $((${n}/2)) < test_${n}
>&2 echo ""
>&2 echo 'seg_size = '$((${n}/5))
./merger $((${n}/5)) < test_${n}
>&2 echo ""
>&2 echo 'seg_size = '$((${n}/10))
./merger $((${n}/10)) < test_${n}
>&2 echo ""
>&2 echo 'seg_size = '$((${n}/25))
./merger $((${n}/25)) < test_${n}
>&2 echo ""
>&2 echo 'seg_size = '$((${n}/100))
./merger $((${n}/100)) < test_${n} 
>&2 echo 'seg_size = '$((${n}/100))
./merger 2 < test_${n} 
