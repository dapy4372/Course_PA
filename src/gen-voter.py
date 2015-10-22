import dnn

modelList = ''


def genWidthList(D, Ws):
    WsNum = len(Ws)
    Wlist = []
    if WsNum != D:
        Wlist = [Ws[0]] * ( D - WsNum + 1)
    for i in xrange(WsNum - 1):
        Wlist.append(Ws[i+1])
    return Wlist

def genVote():
    modelFilename = []
    f1 = open('modelList', 'rb')
    for i in xrange()
        modelFilename.append(f1.readline)
    
    for i in xrange(3):
        parentModelFileName = modelFilename[i]
        model = readModel(parentModelFileName)
        for j in xrange(1:4):
          childModelFileName = parentModelFileName + '-' + j
          childWidth = model[0]
          childDeep = model[1]
          childParams = model[2]
          if i == 1:
              childWidth[childDeep] *= 2
          elif i == 3:
              childWidth[childDeep] /= 2
          childParams[childDeep*2] = None
          childParams[childDeep*2+1] = None
          childModel = [childWidth, childDeep, childParams]
          writeModel(childModelFileName, childModel)

def readModel(fileName):
    # read model in file, return a list: [WidthList, Deep, Params]

def writeModel(fileName, model)
    # write model into a file
