import sys


def vprint(*args,**kwargs):
    VERBOSE = False
    if VERBOSE: print(*args,**kwargs)

class Logger(object):
    def __init__(self, fname="logfile.log"):
        self.terminal = sys.stdout
        self.log = open(fname, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
