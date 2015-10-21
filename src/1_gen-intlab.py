DEBUG = False

if DEBUG:
    phone_48_int_map = '../map/phone_48_int.map'
    state_lab = '../data/state_label/train.lab'
    phone_lab = '../data_for_debug/train_small_y.lab'
    state_int_lab = '../data/state_label/train_int.lab'
    phone_int_lab = '../data_for_debug/train_small_int.lab'
else:
    phone_48_int_map = '../map/phone_48_int.map'
    state_lab = '../data/state_label/train.lab'
    phone_lab = '../data/label/train.lab'
    state_int_lab = '../data/state_label/train_int.lab'
    phone_int_lab = '../data/label/train_int.lab'

f1 = open(phone_48_int_map, 'r')
f2 = open(state_lab, 'r')
f3 = open(phone_lab, 'r')
f4 = open(state_int_lab, 'w')
f5 = open(phone_int_lab, 'w')

d_map = {}

for i in f1:
    i = i.strip()
    i = i.split()
    d_map[i[0]] = i[1]
"""
for i in f2:
    i = i.strip()
    i = i.split(',')
    if i[1] in d_map:
        tmp = i[0] + ',' + str(d_map[i[1]]) + '\n'
        f4.write(tmp)
    else:
        print "error"
"""

for i in f3:
    i = i.strip()
    i = i.split(',')
#    print i[1]
    if i[1] in d_map:
        tmp = i[0] + ',' + str(d_map[i[1]]) + '\n'
        f5.write(tmp)
    else:
        print error
