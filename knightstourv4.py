from mpi4py import MPI
import sys
import numpy as np 
import time
import datetime
import argparse
from random import shuffle

# starting from 0,4 needs at least 22 threads
# starting in the middle (3,3 3,4, 4,3 4,4) needs at least 48 threads

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
    global rank
    global nt
    global move_list
    global filename
    global fileout
    global rv_sort

    if len(move_list) == 64:
        # complete tour found, return and keep going (let's find more routes)
        output = "Thread " + str(rank) + " found result @ " + str(time.time() - nt)
        print(output)
        if not fileout:
            print(move_list)
        else:
            f = open(filename, "a+")
            f.write(str(output) + "\n")
            f.write(str(move_list) + "\n")
            f.close()
        return True

    # if step_moves is not passed, first iteration, build a valid move list
    if step_moves == None:
        step_moves = check_steps(pos)
    
    next_list = []
    for i,step in enumerate(step_moves):
        next_step = [pos[0] + step[0], pos[1] + step[1]]
        next_list.append([next_step, check_steps(next_step)])
    
    # reverse True for follow path with most branches, False for least    
    next_list.sort(key=len, reverse=rv_sort)
    
    for step in next_list:
        if step == None:
            # reached the end of the list, no possible steps, return to previous
            return True    
        else:
            # call the next recursion
            move_list.append(step[0])
            #print ('next', len(move_list), step[0], step[1])
            rec_move(step[0], step[1])

        # keep going, find more routes
        move_list.pop(-1)
    
    return True


allmoves = [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]]

parser = argparse.ArgumentParser(description='Run a Knights Tour.')
parser.add_argument('y', nargs='?', type=int, default='0', help='start position y')
parser.add_argument('x', nargs='?', type=int, default='4', help='start position x')
parser.add_argument('-o', '--fileout', action='store_true', default=False, help='turn on file output')
parser.add_argument('-r', '--reverse', action='store_true', default=False, help='sort path lists in reverse')

args = parser.parse_args()

fileout = args.fileout
rv_sort = args.reverse

if rank == 0:
    st = time.time()
    s_point = [0,4]
    if len(sys.argv) > 2:
        s_point[0] = args.y
        s_point[1] = args.x
    
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
    threads = len(s_move)
    print('Starting Knight Tour from point ' , s_point , ' with ' , threads , ' threads and file ouput ', fileout, ' and reverse sort ' , rv_sort)
    move_npa = np.array_split(s_move, size)
    
else:
    move_npa = None

move_npa = comm.scatter(move_npa, root=0)
if move_npa.size == 0:
    print('Thread ' , rank , ' exiting with no task.')
    sys.exit()

else:
    move_list = move_npa[0].tolist()
    nt = time.time()
    if fileout:
        dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = "ktr_" + dt + "_thread_" + str(rank) + ".txt"
        f = open(filename, "w+")
        f.write("Knight Tour started at " + dt + " from " + str(move_list) + "\n")
        f.close()

    # start the recursive function
    rec_move(move_list[-1])

    # at this point it is very unlikely that you will ever reach these
    # because backtracking takes exponential time
    print('Thread ' , rank , ' completed all paths @ ', time.time() - nt)
    complete = comm.gather(True, root=0)

    if rank == 0:
        if len(complete) >= threads:
            print('Entire run complete @ ', time.time() - st)

        
    
