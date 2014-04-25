#!/usr/bin/env python

import re
import glob
import os

if __name__ == "__main__":
    outf = None
    for fname in glob.glob("out*.log"):
        dname = os.path.splitext(fname)[0]
        with open(fname, 'r') as f:
            if os.path.exists(dname):
                continue
            os.makedirs(dname)
            # Get current directory and change
            for line in f:
                pmatch = re.match('<BEGIN_EXPERIMENT ([^\s]+)>', line)
                if pmatch:
                    experimentName = pmatch.group(1)
                    outf = open(os.path.join(dname, '%s.log' % experimentName), 'w')
                    files = []
                    files.append(open(os.path.join(dname, 'train-%s.txt' % experimentName), 'w'))
                    files.append(open(os.path.join(dname, 'valid-%s.txt' % experimentName), 'w'))
                    files.append(open(os.path.join(dname, 'test-%s.txt' % experimentName), 'w'))
                    counter = 0
                pmatch = re.match('<END_EXPERIMENT>', line)
                if pmatch:
                    outf.close()
                if outf and not(outf.closed):
                    outf.write(line)
                    pmatch = re.match('^Accuracy: ([0-9\.]+)', line)
                    if pmatch:
                        files[counter].write("%f\n" % float(pmatch.group(1)))
                        counter = (counter + 1) % 3
