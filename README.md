# knightstour-py
Knights Tour implementation in python with mpi4py for threading

Solving a Knights Tour problem through backtracking is a problem that takes exponentially more time the larger the grid, and thus more steps. One way to speed up finding routes would be to run many concurrent threads which would reduce the amount of time required to test EVERY path to 1/n. It's still not comprehensive or even the best way of attacking the problem, but it is fairly simple.

knightstourv1.py is the first version which just uses loops and always starts from 0,4. It used a doped move list to speed up results but each thread would stop on the first route found and it would always take the same paths.

knightstourv2.py is a modified version of v1 which randomizes the paths and can accept input of starting co-ordinates. Each thread would still stop on the first route found.

knightstourv3.py was modified to use recursion instaed of loops. It actually implements a reverse Warnsdorff's rule (see Wikipedia article on Knight's Tour) to look ahead a step and decide which move to make next based on which step has the most valid moves. It would still stop on the first route found.

knightstourv4.py was modified from v3 and adds new runtime options (-o for output to file, -r for reverse Warndorff's rule) and threads will continue running and spitting out routes as found.

The text files contain all the results found over about a 13 hour period of running the script from a 0,4 starting position. Interestingly, thread_8 and thread_15 are all closed tours after this period.

