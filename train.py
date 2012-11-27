#!/usr/bin/env python

import os
import argparse
import sys

# The project path is always the directory of the script
PROJECT_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser(description="Run MATLAB training job")
parser.add_argument('--wordset', help='the word vector dataset to use')
parser.add_argument('--trainset', help='the prefix of the image batches to use')
parser.add_argument('--maxPass', help='the number of passes through the training data', type=int)
parser.add_argument('--maxIter', help='the number of iterations over a training batch', type=int)
parser.add_argument('--wordReg', help='the regularization param for words', type=float)
parser.add_argument('--imageReg', help='the regularization param for images', type=float)
parser.add_argument('--outputPath', help='the output path for the parameters')
args = parser.parse_args()

matlabArguments = {
    'wordset':     args.wordset or 'icml',
    'trainset':    args.trainset or 'mini_batch',
    'maxPass':     args.maxPass or 3,
    'maxIter':     args.maxIter or 2,
    'wordReg':     args.wordReg or 1E-3,
    'imageReg':    args.imageReg or 1E-6,
    'outputPath':  args.outputPath or os.path.join(PROJECT_PATH, 'savedParams'),
    'projectPath': PROJECT_PATH,
}

matlabCommand = """
matlab -nodisplay -nodesktop <<EOD
cd {projectPath:s};
trainParams.wordDataset     = {wordset!r};
trainParams.batchFilePrefix = {trainset!r};
trainParams.maxPass         = {maxPass:d};
trainParams.maxIter         = {maxIter:d};
trainParams.wReg            = {wordReg:.3E};
trainParams.iReg            = {imageReg:.3E};
trainParams.outputPath      = {outputPath!r};
train;
EOD
"""

# Execute the MATLAB command
os.system(matlabCommand.format(**matlabArguments))
