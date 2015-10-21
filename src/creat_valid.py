from operator import itemgetter, attrgetter
import utils
featureType     = 'fbank'
trainArkDir     = '../data/' + featureType + '/train.ark'
trainLabelDir   = '../data/label/train_int.lab'
#trainArkDir     = '../data_for_debug/train_small_x.ark'
#trainLabelDir   = '../data_for_debug/train_small_int.lab'
RATIO = 10
trainXName = '../data/fbank_valid/train.ark'
trainYName = '../data/fbank_valid/train.lab'
validXName = '../data/fbank_valid/valid.ark'
validYName = '../data/fbank_valid/valid.lab'

f1 = open(trainArkDir, 'rb')
f2 = open(trainLabelDir, 'rb')
f3 = open(trainXName, 'wb')
f4 = open(trainYName, 'wb')
f5 = open(validXName, 'wb')
f6 = open(validYName, 'wb')

print '...open file'

maleNum = 0
femaleNum = 0
prevName = ''
x = []
for i in f1:
    i = i.strip()
    i = i.split(' ')
    name, ID, num = i[0].split('_')
    feature = i[1:]
    x.append([name, ID, int(num), feature])
    if(name != prevName):
        if name[0] == 'm':
            maleNum += 1
        if name[0] == 'f':
            femaleNum += 1
    prevName = name

y = []
for i in f2:
    i = i.strip()
    i = i.split(',')
    name, ID, num = i[0].split('_')
    label = i[1]
    y.append([name, ID, int(num), label])
print '...sort'
x = sorted(x, key = itemgetter(0,1,2))
y = sorted(y, key = itemgetter(0,1,2))


print '...create valid'
validFemaleNum = femaleNum / RATIO
validMaleNum = maleNum / RATIO
speakerInterval = utils.findSpeakerInterval(y)

validSpeakerList = []

validSetX = []
validSetY = []
trainSetX = []
trainSetY = []

curValidFemaleNum = 0
curValidMaleNum = 0
for i in xrange(len(speakerInterval)):
    if (speakerInterval[i][2] not in validSpeakerList) and (curValidFemaleNum < validFemaleNum):
        validSetX += x[speakerInterval[i][0]:speakerInterval[i][1]]
        validSetY += y[speakerInterval[i][0]:speakerInterval[i][1]]
        curValidFemaleNum += 1
        validSpeakerList.append(speakerInterval[i][2])

    elif( (speakerInterval[i][2] not in validSpeakerList) and (curValidMaleNum < validMaleNum) ):
        validSetX += x[speakerInterval[i][0]:speakerInterval[i][1]]
        validSetY += y[speakerInterval[i][0]:speakerInterval[i][1]]
        curValidMaleNum += 1
        validSpeakerList.append(speakerInterval[i][2])
    else:
        trainSetX += x[speakerInterval[i][0]:speakerInterval[i][1]]
        trainSetY += y[speakerInterval[i][0]:speakerInterval[i][1]]

def writeFile(fx, fy, x, y):
    for i in xrange(len(x)):
        fx.write(x[i][0]+'_'+x[i][1]+'_'+str(x[i][2])+' '+ ' '.join(x[i][3])+'\n')
        fy.write(y[i][0]+'_'+y[i][1]+'_'+str(y[i][2])+','+ y[i][3]+'\n')

writeFile(f3,f4,trainSetX,trainSetY)
writeFile(f5,f6,validSetX,validSetY)

