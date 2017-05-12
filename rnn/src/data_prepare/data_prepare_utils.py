import cPickle

def namepick(name):
    part = name.split('_')
    return (part[0] + '_' + part[1]), int(part[2])

def makePkl(pkl, filename):
    f = open(filename, 'wb')
    cPickle.dump(pkl, f, protocol=2)
    f.close()
