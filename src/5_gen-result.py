import sys
import os
from utils import pickResultFilename
import postprocessing as pp
import transformIntToLabel as titl

resultFilename = sys.argv[1]
Filename = pickResultFilename(resultFilename)
withPostprocessingFilename = '../result/with_postprocessing_result/' + Filename
withoutPostprocessingFilename = '../result/without_postprocessing_result/' + Filename
tmpFilename = './' + Filename

# with postprocessing
name, label = pp.readFile(resultFilename)
endIndxGroup = pp.findEndIndxofGroup(name = name, label = label)
label = pp.correctLabel(endIndxGroup = endIndxGroup, name = name, label = label)
pp.writeFile(filename = tmpFilename, name = name, label = label)
titl.transform(beforeTransformFilename = tmpFilename, afterTransformFilename = withPostprocessingFilename)
os.remove(tmpFilename)

# without postprocessing
titl.transform(beforeTransformFilename = resultFilename, afterTransformFilename = withoutPostprocessingFilename)
