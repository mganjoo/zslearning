#!/usr/bin/env python

import re

if __name__ == "__main__":
    outf = None
    with open('out.log', 'r') as f:
        for line in f:
            pmatch = re.match('<BEGIN_EXPERIMENT ([^\s]+)>', line)
            if pmatch:
                experimentName = pmatch.group(1)
                outf = open('%s.log' % experimentName, 'w')
                files = []
                files.append(open('trainr-%s.txt' % experimentName, 'w'))
                files.append(open('validr-%s.txt' % experimentName, 'w'))
                files.append(open('testr-%s.txt' % experimentName, 'w'))
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
