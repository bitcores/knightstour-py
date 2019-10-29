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

def make_step_moves(pos):
    if pos[0] < 2:
        temp = moves_top_sector[:4]
        shuffle(temp)
        temp = temp + moves_top_sector[4:]
        step_moves.append(temp)
    elif pos[0] > 5:
        temp = moves_bottom_sector[:4]
        shuffle(temp)
        temp = temp + moves_bottom_sector[4:]
        step_moves.append(temp)
    else:
        temp = moves_midl_sector[:]
        shuffle(temp)
        step_moves.append(temp)
    
    return True


moves_top_sector = [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]]
moves_midl_sector = [[1, 2], [-1, 2], [1, -2], [-1, -2], [2, 1], [-2, 1], [2, -1], [-2, -1]]
moves_bottom_sector = [[1, 2], [1, -2], [-1, 2], [-1, -2], [-2, 1], [-2, -1], [2, 1], [2, -1]]

if rank == 0:
    st = time.time()
    s_point = [0,4]
    if len(sys.argv) == 3:
        s_point[0] = int(sys.argv[1])
        s_point[1] = int(sys.argv[2])  
    
    f_move = []
    s_move = []
    
    for a in moves_top_sector:
        if checkbounds(s_point, a):
            f_move.append([s_point[0]+a[0], s_point[1]+a[1]])

    for m in f_move:
        for a in moves_top_sector:
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
    step_moves = [None]
    no_result = False

    while len(move_list) < 64:
        while len(step_moves) < len(move_list):
            make_step_moves(move_list[(len(step_moves) - len(move_list))])

        l_pos = move_list[-1]
        n_move = step_moves[-1][0]

        if checkbounds(l_pos, n_move) and checkvalid([l_pos[0]+n_move[0], l_pos[1]+n_move[1]], move_list):
            move_list.append([l_pos[0]+n_move[0], l_pos[1]+n_move[1]])
        else:
            check_next = False
            while not check_next:
                if len(step_moves) == 1:
                    # returned to the beginning step without finding a path
                    break

                if len(step_moves[-1]) > 1:
                    step_moves[-1].pop(0)
                    check_next = True              
                elif len(step_moves[-1]) == 1:
                    move_list.pop(-1)
                    step_moves = step_moves[:-1]                          
                
            if not check_next:
                print('Thread ' , rank , ' found no result')
                no_result = True
                # returned to the beginning step without finding a path, no  result
                break

    if not no_result:
        print('Thread ' , rank, ' found result @ ', time.time() - nt)
        print(move_list)

    if rank == 0:
        print('Entire run complete @ ', time.time() - st)

        
    
