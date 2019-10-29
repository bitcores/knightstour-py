from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import sys
import numpy as np 
import time

# run with -n 23 because there are 22 branches and 1 message/output handler with rank 22

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

def getmove(pos, stepno):
    if pos[0] < 2:
        return moves_top_sector[stepno]
    elif pos[0] > 5:
        return moves_bottom_sector[stepno]
    return moves_midl_sector[stepno]


moves_top_sector = [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]]
moves_midl_sector = [[1, 2], [-1, 2], [1, -2], [-1, -2], [2, 1], [-2, 1], [2, -1], [-2, -1]]
moves_bottom_sector = [[1, 2], [1, -2], [-1, 2], [-1, -2], [-2, 1], [-2, -1], [2, 1], [2, -1]]

if rank == 0:
    st = time.time()
    s_point = [0,4]
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
    move_npa = np.array_split(s_move, size)
    
else:
    move_npa = None

move_npa = comm.scatter(move_npa, root=0)
if move_npa.size == 0 and rank > 22:
    sys.exit()

if rank == 22:
    while True:
        output = comm.recv(source=ANY_SOURCE, tag=1101)
        print(output)

else:
    nt = time.time()
    move_list = move_npa[0].tolist()
    step_list = [0]
    no_result = False
    #print(move_list[-1])
    while len(move_list) < 64:
        l_pos = move_list[-1]
        n_move = getmove(l_pos, step_list[-1])
        #print('Thread ', rank, ' at move ' , len(move_list))
        if checkbounds(l_pos, n_move) and checkvalid([l_pos[0]+n_move[0], l_pos[1]+n_move[1]], move_list):
            move_list.append([l_pos[0]+n_move[0], l_pos[1]+n_move[1]])
            step_list.append(0)
        else:
            check_next = False
            while not check_next:
                step_list[-1] = step_list[-1] + 1
                
                if step_list[-1] >= 7:
                    step_list.pop(-1)
                    move_list.pop(-1)
                    if len(step_list) == 0:
                        # returned to the beginning step without finding a path
                        break
                    elif len(step_list) <= 5:
                        # search takes exponential amounts of time so we will check progress
                        # by looking for return to early points in the list
                        output = "Incrementing " , step_list[:5]
                        comm.send(output, dest=22, tag=1101)
                else:
                    check_next = True         
                
            if not check_next:
                print('No result at ', len(move_list) , ' steps')
                no_result = True
                # returned to the beginning step without finding a path, no  result
                break

    if not no_result:
        print('Thread ' , rank, ' found result @ ', time.time() - nt)
        print(move_list)

    if rank == 0:
        print('Entire run complete @ ', time.time() - st)

        
    
