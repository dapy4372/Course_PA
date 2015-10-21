import sys
import os
from utils import pickResultFilename
import postprocessing as pp
import transformIntToLabel as titl

withPPresultDir = '../result/with_postprocessing_result/'
withoutPPresultDir = '../result/without_postprocessing_result/'
tmpDir = '../tmp/'

resultFilename = sys.argv[1]
resultName = pickResultFilename(resultFilename)

# with postprocessing
name, label = pp.readFile(resultFilename)
endIndxGroup = pp.findEndIndxofGroup(name = name, label = label)
label = pp.correctLabel(endIndxGroup = endIndxGroup, name = name, label = label)
pp.writeFile(filename = tmpDir + resultName, name = name, label = label)
titl.transform(beforeTransformFilename = tmpDir + resultName, afterTransformFilename = withPPresultDir + resultName)
#os.remove(tmpDir + resultName)

# without postprocessing
titl.transform(beforeTransformFilename = resultFilename, afterTransformFilename = withoutPPresultDir + resultName)
