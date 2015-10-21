import operator
state_48_39_ori_map = '../data/phones/state_48_39.map'
phone_48_39_ori_map = '../data/phones/48_39.map'
state_label_map = '../map/state_label.map'
phone_48_39_int_map = '../map/phone_48_39_int.map'
phone_48_int_map = '../map/phone_48_int.map'

f1 = open(state_48_39_ori_map, 'r')
f2 = open(phone_48_39_ori_map, 'r')
f3 = open(state_label_map, 'w')
f4 = open(phone_48_39_int_map, 'w')
f6 = open(phone_48_int_map, 'w')
d_s_48 = {}
d_s_48_39 = {}
d_48_39 = {}

"""
# state label: state to 48
# state label: 48 to 30
for i in f1:
    i = i.strip()
    i = i.split()
    d_s_48[i[0]] = i[1]
    d_s_48_39[i[1]] = i[2]
"""
d_39_int = {}
# phone label map: 39 to int
ii = 0
for i in f2:
    i = i.strip()
    i = i.split()
    if i[1] not in d_39_int:
        d_39_int[i[1]] = ii
        ii += 1

f2.seek(0)

# phone label: 48 to 39
for i in f2:
    i = i.strip()
    i = i.split()
    tmp = i[0] + '\t' + i[1] + '\t' + str(d_39_int[i[1]]) + '\n'
    f4.write(tmp) 

f2.seek(0)

ii = 0
for i in f2:
    i = i.strip()
    i = i.split()
    tmp = i[0] + '\t' + str(ii) + '\n'
    ii += 1
    f6.write(tmp) 

"""
# state label map: 39 to int
ii = 0
for i in f1:
    i = i.strip()
    i = i.split()
    if i[0] not in state_label_map:
        state_label_map[i[0]] = ii
        ii += 1

sorted_map = sorted(state_label_map.items(), key=operator.itemgetter(1))
for i in xrange(len(sorted_map)):
    tmp = str(i) + ' ' + (sorted_map[i][0]) + '\n'
    f3.write(tmp)
"""
"""
# phone label map: 39 to int
ii = 0
for i in f2:
    i = i.strip()
    i = i.split()
    if i[1] not in 39_int:
        39_int[i[1]] = ii
        ii += 1

sorted_map = sorted(phone_label_map.items(), key=operator.itemgetter(1))
for i in xrange(len(sorted_map)):
    f4.write(tmp)
"""
