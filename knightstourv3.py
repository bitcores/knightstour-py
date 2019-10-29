from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import sys
import numpy as np 
import time
from random import shuffle

# starting from 0,4 needs at least 22 threads
# starting in the middle (3,3 3,4, 4,3 4,4) needs at least 49 threads

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

def checkbounds(pos, move):
    return (pos[0] + move[0] >= 0) and (pos[0] + move[0] < 8) and (pos[1] + move[1] >= 0) and (pos[1] + move[1] < 8)

def checkvalid(move, history):
    for a in history:
        if a[0] == move[0] and a[1] == move[1]:
            return False
    return True

def check_steps(npos):
    nlist = allmoves[:]
    blist = []
    for step in nlist:
            if checkbounds(npos, step) and checkvalid([npos[0]+step[0], npos[1]+step[1]], move_list):
                blist.append(step)             
    return blist

def rec_move(pos, step_moves=None):
    global found_result
    global move_list

    if len(move_list) == 64:
        # complete tour found, signal functions to exit
        found_result = True
        return True

    # if step_moves is not passed, first iteration, build a valid move list
    if step_moves == None:
        step_moves = check_steps(pos)
    
    next_list = []
    for i,step in enumerate(step_moves):
        next_step = [pos[0] + step[0], pos[1] + step[1]]
        next_list.append([next_step, check_steps(next_step)])
        
    next_list.sort(key=len, reverse=True)
    
    for step in next_list:
        if step == None:
            # reached the end of the list, no possible steps, return to previous
            return True    
        else:
            # call the next recursion
            move_list.append(step[0])
            #print ('next', len(move_list), step[0], step[1])
            rec_move(step[0], step[1])

        if not found_result:
            move_list.pop(-1)
        else:
            break
    
    return True


allmoves = [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]]

if rank == 0:
    st = time.time()
    s_point = [0,4]
    if len(sys.argv) == 3:
        s_point[0] = int(sys.argv[1])
        s_point[1] = int(sys.argv[2])  
    
    f_move = []
    s_move = []
    
    for a in allmoves:
        if checkbounds(s_point, a):
            f_move.append([s_point[0]+a[0], s_point[1]+a[1]])

    for m in f_move:
        for a in allmoves:
            if checkbounds(m, a) and checkvalid([m[0]+a[0], m[1]+a[1]], [s_point, a]):
                s_move.append([s_point, m, [m[0]+a[0], m[1]+a[1]]])

    # the starting positions have now been successfully generated
    # let the threading begin
    print("Starting Knight Tour from point " , s_point , " with " , len(s_move) , " threads.")
    move_npa = np.array_split(s_move, size)
    
else:
    move_npa = None

move_npa = comm.scatter(move_npa, root=0)
if move_npa.size == 0:
    sys.exit()

else:
    nt = time.time()
    move_list = move_npa[0].tolist()
    found_result = False

    # start the recursive function
    rec_move(move_list[-1])


    if found_result:
        print('Thread ' , rank , ' found result @ ', time.time() - nt)
        print(move_list)
    else:
        print('Thread ' , rank , ' found no result')

    if rank == 0:
        print('Entire run complete @ ', time.time() - st)

        
    
