#!/usr/bin/python

import os

PROJECT_PATH = '/afs/ir/users/m/g/mganjoo/projects/vsm-learning/'

matlabCommand = """
module load MATLAB-R2012b
matlab -nodesktop <<EOD
cd {projectPath:s};
trainParams.wordDataset     = {wordDataset!r};
trainParams.batchFilePrefix = {batchFilePrefix!r};
trainParams.maxPass         = {maxPass:d};
trainParams.maxIter         = {maxIter:d};
trainParams.cReg            = {cReg:.3E};
trainParams.outputPath      = {outputPath!r};
train;
EOD
"""

jobNum = 1
for iter in [ 3, 4, 6, 8 ]:
    jobName = "vsm{0:03d}".format(jobNum)
    outputPath = PROJECT_PATH + "savedParams-%s/" % jobName

    arguments = {
        'wordDataset':     'icml',
        'batchFilePrefix': 'default_batch',
        'maxPass':         70,
        'maxIter':         iter,
        'cReg':            1E-3,
        'projectPath':     PROJECT_PATH,
        'outputPath':      outputPath
    }

    qsubCommand = "cat <<EOF | qsub -V -m be -M mganjoo@stanford.edu -N {0} -o {1} -e {2}{3}EOF"

    os.system(qsubCommand.format(jobName, outputPath + "out", outputPath + "err", matlabCommand.format(**arguments)))
    jobNum = jobNum + 1
