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

def genTree():
    modelFilename = []
    f1 = open('modelList', 'rb')
    for i in xrange()
        modelFilename.append(f1.readline)
    
    parent = []
    for i in xrange(3):
        parent.append(readModel(modelFilename[i]))
    readModel()
    genWidthList
