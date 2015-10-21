from utils import namepick, readFile
import sys
preLabelFilename = sys.argv[1]
correctLabelDir = '../tmp/'

def findEndIndxofGroup(name, label):
    curName = namepick(name[0])
    endIndxGroup = []
    for i in xrange(len(name)):
        if(curName != namepick(name[i])):
            endIndxGroup.append(i)
    endIndxGroup.append(len(name))
    return endIndxGroup 

def correctLabel(endIndxGroup, name, label):
    indxOfGroup = 0
    curEnd = endIndxGroup[0]
    firstLabelOfcurGroup = True
    for i in xrange(1, len(name)):
        if firstLabelOfcurGroup:
            firstLabelOfcurGroup = False
            continue
        else:
            if i == ( endIndxGroup[indxOfGroup] - 1 ):
                indxOfGroup += 1
                firstLabelOfcurGroup = True
                continue
            if( label[i - 1] == label[i + 1] and label[i - 1] != label[i] ):
                label[i] = label[i - 1]
        if i == len(name) - 2:
            break
    return label

def writeFile(filename, name, label):
    f = open(filename, 'w')
    for i in xrange(len(name)):
        f.write(name[i] + ',' + label[i])
    f.close()

if __name__ == '__main__':
    name, label = readFile(preLabelFilename)
    endIndxGroup = findEndIndxofGroup(name = name, label = label)
    label = correctLabel(endIndxGroup = endIndxGroup, name = name, label = label)
    writeFile(filename = findOutputFilename(oriFilename = preLabelFilename), name = name, label = label)
        
