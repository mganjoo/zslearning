#!/usr/bin/python

import os

matlab_command = """
module load MATLAB-R2012b
matlab -nodesktop <<EOD
cd /afs/ir/users/m/g/mganjoo/projects/vsm-learning;
trainParams.wordDataset     = {wordDataset!r};
trainParams.batchFilePrefix = {batchFilePrefix!r};
trainParams.maxPass         = {maxPass:d};
trainParams.maxIter         = {maxIter:d};
trainParams.cReg            = {cReg:.3E};
trainParams.outputPath      = {outputPath!r};
train;
EOD
"""

job_num = 1
for iter in [ 3, 4, 5, 6 ]:
    job_name = "vsm{0:03d}".format(job_num)

    arguments = {
        'wordDataset':     'icml',
        'batchFilePrefix': 'default_batch',
        'maxPass':         70,
        'maxIter':         iter,
        'cReg':            1E-3,
        'outputPath':      'savedParams-%s' % job_name
    }

    os.system("cat <<EOF | qsub -N {0}{1}EOF".format(job_name, matlab_command.format(**arguments)))
    job_num = job_num + 1
