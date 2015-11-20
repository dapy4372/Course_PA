from utils import namepick, readFile, readSetting, readFile_shrink
import sys


def findEndIndxofGroup(name, label):
    curName = namepick(name[0])
    endIndxGroup = []
    for i in xrange(len(name)):
        if(curName != namepick(name[i])):
            endIndxGroup.append(i)
    endIndxGroup.append(len(name))
    return endIndxGroup 

def correctLabel(endIndxGroup, name, label, MAX_ALLOW = 3):
    currentFlag     = label[0]
    currentOdd      = label[len(name)/2]
    currentContinue = 0
    hitNumber       = 0
    for i in xrange(len(name)):
      if i == 0:
        currentContinue += 1
      elif i == (len(name) - 1): 
        currentContinue += 1
        for j in xrange(currentContinue):
          label[i-j] = currentFlag
      else:
        if label[i] == currentFlag:
          currentContinue += 1
        elif label[i] == currentOdd:
          hitNumber += 1
          currentContinue += 1
          if hitNumber == MAX_ALLOW:
            for j in xrange(currentContinue-MAX_ALLOW):
              label[i-j-MAX_ALLOW] = currentFlag
            currentContinue = MAX_ALLOW
            hitNumber   = 0
            currentOdd  = label[i-1]
            currentFlag = label[i]
        else:
          currentOdd      = label[i]
          hitNumber       = 1
          currentContinue += 1
        
        
    return label

def writeFile(filename, name, label):
    f = open(filename, 'w')
    for i in xrange(len(name)):
        f.write(name[i] + ',' + label[i])
    f.close()

def findParameters(title, parameter):
    preLabelFilename = parameter[(title.index('preLabelFilename'))].strip('\n')
    maxAllow         = int(parameter[title.index('maxAllow')])
    smoothFilename = parameter[(title.index('smoothFilename'))].strip('\n')
    outputFilename = parameter[(title.index('outputFilename'))].strip('\n')
    twoMap = parameter[(title.index('twoMap'))].strip('\n')
    threeMap = parameter[(title.index('threeMap'))].strip('\n')

    return preLabelFilename, maxAllow, smoothFilename, outputFilename, twoMap, threeMap

def readMap(filename):
    f = open(filename, 'r')
    word = []
    alph = []
    for i in f:
        part = i.split(',')
        word.append(part[0])
        alph.append(part[2])
    f.close()
    word = clearLabel(word)
    alph = clearLabel(alph)
    return word, alph
# This function reads the 49-39 map #
# from HW #1                        #
def readMap_old(filename):
    f = open(filename, 'r')
    label = []
    for i in f:
        part = i.split(',')
        label.append(part[1])
    f.close()
    label = clearLabel(label)
    return label
# This function converts the 48 dim output #
# to 39 dim                                #
def remap_48_39(name, label, cur_map):
  for i in xrange(len(name)):
    for j in xrange(len(label[i])):
      label[i][j] = cur_map[int(label[i][j])]
  return label

def remap_39_alph(name, label, word, alph):
  for i in xrange(len(name)):
    for j in xrange(len(label[i])):
      label[i][j] = alph[word.index(label[i][j])]
  return label

def shrinkPhonemes(name, label):
    curLabel    = 'dummy'
    curName     = 'dummy'
    j = 0
    returnNameList = []
    returnLabelList = []
    dummyLabel = []
    for i in xrange(len(name)):
        if(i == 0): # initialization
            curLabel = label[i]
            curName  = name[i]
        elif(i == (len(name)-1)):
            returnNameList.append(str(curName))
            dummyLabel.append(label[i])
            returnLabelList.append(dummyLabel)
        else:
            if(curName == name[i]):
                if(curLabel == label[i]):
                    j += 1
                else:
                    dummyLabel.append(curLabel)
                    curLabel = label[i]
            else:
                # Write in the name info and update current name
                returnNameList.append(str(curName))
                curName = name[i]
                dummyLabel.append(label[i])
                returnLabelList.append(dummyLabel)
                dummyLabel = []

    return returnNameList, returnLabelList

def writeFile2(filename, namelist, labellist):
    f = open(filename, 'w')
    f.write('id,phone_sequence\n')
    for i in xrange(len(namelist)):
        dummy_str = ''
#        f.write(name[i] + ',')
        for j in xrange(len(labellist[i])):
            current = str(labellist[i][j]).split('\n')
            dummy_str += current[0]
#          f.write(labellist[i][j])
        f.write(namelist[i] + ',' + dummy_str + '\n')
#        f.write('\n')
    f.close

def clearLabel(label):
    for i in xrange(len(label)):
        label[i] = label[i].replace("\n", "")
    return label
''' 
if __name__ == '__main__':
  name, label = readFile(preShrunkFilename)

  twoMap     = readMap_old(twoMapName)
  word, alph = readMap(threeMapName)

  name, label = shrinkPhonemes(name, label)

  label = remap_48_39(name, label, twoMap)
  label = remap_39_alph(name, label, word, alph)

  writeFile2(ShrunkFilename + '_remap', name, label)
  '''

if __name__ == '__main__':
    settingFile = sys.argv[1]
    title, parameter    = readSetting(settingFile)
    preLabelFilename, maxAllow, smoothFilename, outputFilename, two_map, three_map = findParameters(title, parameter)
    
    name, label = readFile(preLabelFilename)
    endIndxGroup = findEndIndxofGroup(name = name, label = label)
    label = correctLabel(endIndxGroup = endIndxGroup, name = name, label = label, MAX_ALLOW = maxAllow)
    writeFile(filename = smoothFilename, name = name, label = label)

    new_name, new_label = readFile_shrink(smoothFilename)
    twoMap = readMap_old(two_map)
    word, alph = readMap(three_map)
    
    new_name, new_label = shrinkPhonemes(new_name, new_label)
    label = remap_48_39(new_name, new_label, twoMap)
    label = remap_39_alph(new_name, new_label, word, alph)

    writeFile2(outputFilename, new_name, new_label)
