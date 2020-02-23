import numpy as np
import multiprocessing as mp

'''
    Global parameters
'''
class GP:
    g = np.array([0.0,-1.0,0.0])
    k = 1.0
    dt = 0.001
    L1 = 1.0
    L2 = 1.0
    N1 = 128 
    N2 = 128
    TN = N1*N2
    DL = L1/float(N1)
    R0 = np.array([-0.5,0.5,0.0])
    E1 = np.array([0.0,1.0,0.0])
    E2 = np.array([0.0,-1.0,0.0])
    idx2ij = dict()
    ij2idx = dict()
    adjlist = dict()

'''
    Initialize particles 
'''
def init():
    GP.idx2ij.clear()
    GP.ij2idx.clear()
    GP.adjlist.clear()
    posl = []
    vell = []
    accl = []
    naccl = []
    idx = 0
    for i in range(GP.N1):
        for j in range(GP.N2):
            pos = GP.R0 + GP.DL*(float(i))*GP.E1 + GP.DL*(float(j))*GP.E2
            vel = np.array([0.0,0.0,0.0])
            acc = np.array([0.0,0.0,0.0])
            nacc = np.array([0.0,0.0,0.0])
            posl.append(pos)
            vell.append(vel)
            accl.append(acc)
            naccl.append(nacc)
            GP.ij2idx[(i,j)] = idx
            GP.idx2ij[idx] = (i,j)
            idx += 1

    DI = [+1,0,-1,0]
    DJ = [0,+1,0,-1]
    for i in range(GP.N1):
        for j in range(GP.N2):
            idx = GP.ij2idx[(i,j)]
            GP.adjlist[idx] = set()
            for (di,dj) in zip(DI,DJ):
                ni = i+di
                nj = j+dj
                if ni>-1 and ni<GP.N1 and nj>-1 and nj<GP.N2:
                    GP.adjlist[idx].add(GP.ij2idx[(ni,nj)])

    return posl,vell,accl,naccl

'''
    Vanilla, single-process implementation
'''
def verlet_mono(Tt = 100):
    posl,vell,accl,naccl = init()
    for t in range(Tt):
        # update pos
        for idx in range(GP.TN):
            posl[idx] = posl[idx] + vell[idx]*GP.dt+0.5*accl[idx]*GP.dt**2
        # update n_acc
        for idx in range(GP.TN):
            nacc = GP.g
            pos = posl[idx]
            for n_idx in GP.adjlist[idx]:
                dr = posl[n_idx] - pos
                dr_norm = np.linalg.norm(dr)
                nacc += GP.k*(1.0-(GP.DL/dr_norm))*dr
        # update vel
        for idx in range(GP.TN):
            vell[idx] = vell[idx] + 0.5*GP.dt*(accl[idx]+naccl[idx])
        # acc <- nacc
        accl = naccl

def chunk_processor(posl,vell,accl,naccl,id_range,conn):
    posl = np.frombuffer(posl.get_obj()).reshape(GP.TN,3)
    vell = np.frombuffer(vell.get_obj()).reshape(GP.TN,3)
    accl = np.frombuffer(accl.get_obj()).reshape(GP.TN,3)
    naccl = np.frombuffer(naccl.get_obj()).reshape(GP.TN,3)

    while(True):
        for idx in range(id_range[0],id_range[1]):
            posl[idx] = posl[idx] + vell[idx]*GP.dt+0.5*accl[idx]*GP.dt**2
        # posl[id_range[0]] += np.array([9999.,9999.,9999.])
        conn.send(time.time())
        rdy = conn.recv()
       
        if rdy!=1:
            raise Exception("No READY signal!")

        for idx in range(id_range[0],id_range[1]):
            nacc = GP.g
            pos = posl[idx]
            for n_idx in GP.adjlist[idx]:
                dr = posl[n_idx] - pos
                dr_norm = np.linalg.norm(dr)
                nacc += GP.k*(1.0-(GP.DL/dr_norm))*dr

        conn.send(time.time())
        rdy = conn.recv()
        if rdy!=1:
            raise Exception("No READY signal!")


        for idx in range(id_range[0],id_range[1]):
            vell[idx] = vell[idx] + 0.5*GP.dt*(accl[idx]+naccl[idx])

        conn.send(time.time())
        return

import time
psize = 128


def verlet_multi(Tt = 1,PSIZE = 8):
    master_proc_times = []
    processes_update_pos = []
    processes_update_acc = []
    processes_update_vel = []

    t0 = time.time()

    for i in range(PSIZE):
        processes_update_pos.append([0.0,0.0])
        processes_update_vel.append([0.0,0.0])
        processes_update_acc.append([0.0,0.0])
    
    posl, vell, accl, naccl = init()
    posl = np.stack(posl)
    vell = np.stack(vell)
    accl = np.stack(accl)
    naccl = np.stack(naccl)
    posl = mp.Array('d', posl.flatten())
    accl = mp.Array('d', accl.flatten())
    vell = mp.Array('d', vell.flatten())
    naccl = mp.Array('d', naccl.flatten())
    processes = []
    conns = []
    t1 = time.time()
    master_proc_times.append(t1-t0)
    for pi in range(PSIZE):
        ir = (int((GP.TN/PSIZE)*pi),int((GP.TN/PSIZE)*(pi+1)))
        p_con,c_con = mp.Pipe()
        conns.append(p_con)
        proc = mp.Process(target=chunk_processor,args=(posl,vell,accl,naccl,ir,c_con,))
        processes.append(proc)
        processes_update_pos[pi][0] = time.time()
        proc.start()
    
    for t in range(Tt):
        for pi in range(PSIZE):
            rettime = conns[pi].recv()
            processes_update_pos[pi][1] = time.time()
        # print(f"POS updated {t}")
        for pi in range(PSIZE):
            processes_update_vel[pi][0] = time.time()
            conns[pi].send(1)

        for pi in range(PSIZE):
            rettime = conns[pi].recv()
            processes_update_vel[pi][1] = time.time()
        # print("ACC updated")
        for pi in range(PSIZE):
            processes_update_acc[pi][0] = time.time()
            conns[pi].send(1)

        for pi in range(PSIZE):
            rettime = conns[pi].recv()
            processes_update_acc[pi][1] = time.time()
        # print("VEL updated")
        # Let's plot

    import matplotlib.pyplot as plt

    def timelines(y, xstart, xstop, color='b'):
        """Plot timelines at y from xstart to xstop with given color."""   
        plt.hlines(y, xstart, xstop, color, lw=4)
        plt.vlines(xstart, y+0.03, y-0.03, color, lw=2)
        plt.vlines(xstop, y+0.03, y-0.03, color, lw=2)  

    plt.xlabel('time')
    timelines(PSIZE,t0,t1,'black')
    for i in range(PSIZE):
        timelines(PSIZE-i,processes_update_pos[i][0],processes_update_pos[i][1],'b')
        timelines(PSIZE-i,processes_update_vel[i][0],processes_update_vel[i][1],'g')
        timelines(PSIZE-i,processes_update_acc[i][0],processes_update_acc[i][1],'b')
    plt.show()

verlet_multi(1,psize)