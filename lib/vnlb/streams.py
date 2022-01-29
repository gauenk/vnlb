
import torch
from easydict import EasyDict as edict

def init_streams(curr_stream,nstreams,device):

    # -- create streams --
    torch.cuda.synchronize()
    streams = [torch.cuda.default_stream()]
    streams += [torch.cuda.Stream(device=device,priority=0) for s in range(nstreams-1)]
    wait_streams(streams,[streams[curr_stream]])
    if nstreams > 0:
        for s in streams: s.synchronize()

    # -- stream buffers --
    bufs = edict()
    bufs.l2 = [None,]*nstreams
    bufs.ave = [None,]*nstreams
    bufs.inds = [None,]*nstreams
    bufs.noisyView = [None,]*nstreams
    # bufs.indexView = [None,]*nstreams
    bufs.maskView = [None,]*nstreams
    bufs.patchesView = [None,]*nstreams
    bufs.indsView = [None,]*nstreams
    bufs.distsView = [None,]*nstreams
    bufs.accessView = [None,]*nstreams

    return bufs,streams

def wait_streams(waiting,waitOn):

    # -- create events for each stream --
    events = []
    for stream in waitOn:
        event = torch.cuda.Event(blocking=False)
        event.record(stream)
        # stream.record_event(event)
        events.append(event)

    # -- issue wait for all streams in waitOn and all events --
    for stream in waiting:
        for event in events:
            # stream.wait_event(event)
            event.wait(stream)

def divUp(a,b): return (a-1)//b+1

def get_nbatches(t,h,w,bsize):
    nelems = t*h*w
    nbatches = divUp(nelems,bsize)
    return nbatches

def get_hw_batches(h,w,bsize):
    hbatches = torch.arange(0,h,bsize)
    wbatches = torch.arange(0,w,bsize)
    return hbatches,wbatches

def view_batch(tensor,h_start,w_start,size):
    hslice = slice(h_start,h_start+size)
    wslice = slice(w_start,w_start+size)
    return tensor[...,hslice,wslice]

def vprint(*args,**kwargs):
    VERBOSE = False
    if VERBOSE: print(*args,**kwargs)


