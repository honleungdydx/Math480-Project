%python

import math
import numpy
from cvxopt import matrix, solvers
from timeit import Timer
import random

# Subroutines for Weiszfeld (See the report for the definition of these functions)

# Euclidean distance
def dist(x,y):
    return numpy.linalg.norm(x-numpy.array(y))

# Define function beta
def beta(x,a,eta):
    b = 0
    for i in range(0,len(a)):
        if dist(x,a[i]) > 0:
            b += eta[i]/dist(x,a[i])
    return 1/b

# Define function Tilde_T
def Tilde_T(x,a,eta):
    t = numpy.array([0]*len(a[0]))
    for i in range(0,len(a)):
        if dist(x,a[i]) > 0:
            t = t + (eta[i]/dist(x,a[i]))*numpy.array(a[i])
    return beta(x,a,eta)*t

# Define function eta
def eta_function(x,a,eta):
    e = 0
    for i in range(0,len(a)):
        if dist(x,a[i]) == 0:
           e = eta[i]
    return e

# Define function r
def r(x,a,eta):
    s = numpy.array([0]*len(a[0]))
    for i in range(0,len(a)):
        if dist(x,a[i])>0:
            s = s + (eta[i]/dist(x,a[i]))*(numpy.array(a[i])-x)
    return numpy.linalg.norm(s)

# Define the iteration mapping T
def T(x,a,eta):
    if eta_function(x,a,eta) == 0 and r(x,a,eta) == 0:
        return Tilde_T(x,a,eta)
    elif eta_function(x,a,eta) > r(x,a,eta) and r(x,a,eta) == 0:
        return x
    else:
        return max([1- (eta_function(x,a,eta)/r(x,a,eta)),0.])*Tilde_T(x,a,eta) + \
        min([1,eta_function(x,a,eta)/r(x,a,eta)])*x

"""
Modified Weiszfeld algorithm
Input: y = initial point
       a = list of given points
       eta = weights
Output: A list including the number of iterations, minimum value and the
        Fermat point.
"""
def weiszsolve(y,a,eta):
    t = len(a)
    z = 0
    for i in range(0,t):
        if (len(a[i-1]) != len(a[0])):
            z += 1
    if (z!=0):
        print "The points may have different dimensions."
    x = numpy.array(y)
    count = 0
    while (r(x,a,eta)>0 and r(x,a,eta)>eta_function(x,a,eta)) and count <= 99:
        x = T(x,a,eta)
        count += 1
    L = [count]
    L.append(sum([dist(x,a[i]) for i in range(0,len(a))]))
    L.append(x.tolist())
    return L

# Subroutines for SOCP

"""
Solving the Fermat-Torricelli problem using second order cone programming
(SOCP) formulation.
Input: a = list of given points
       eta = weights
Output: The output from the solver in cvxopt
"""
def socpsolve(a,eta):
    t = len(a)
    z = 0
    for i in range(0,t):
        if (len(a[i-1]) != len(a[0])):
            z += 1
    if (z!=0):
        print "The points may have different dimensions."
    m = len(eta)
    d = len(a[0])
    for i in range(1,d+1):
        eta.insert(0,0.)
    c = matrix(eta)
    l = [0.]*(d-1)
    l.insert(0,-1.)
    l.insert(0,0.)
    L = [l]
    for i in range(1,d):
        l = [0.]*(d-1)
        l.insert(0,-1.)
        l.insert(0,0.)
        l[i],l[i+1] = l[i+1],l[i]
        L.append(l)
    p = [0.]*(d)
    p.insert(0,-1.)
    L.append(p)
    for i in range(1,m):
        L.append([0.]*(d+1))
    G = [matrix(L)]
    for i in range(d,d+m-1):
        L[i],L[i+1] = L[i+1],L[i]
        G += [matrix(L)]
    a[0].insert(0,0.)
    h = [matrix([j*(-1) for j in a[0]])]
    for i in range(1,m):
        a[i].insert(0,0.)
        h += [matrix([j*(-1) for j in a[i]])]
    sol = solvers.socp(c, Gq = G, hq = h)
    return sol

# Subroutines for SDP

"""
Solving the Fermat-Torricelli problem using semidefinite programming
(SDP) formulation.
Input: b = list of given points
       eta = weights
Output: The output from the solver in cvxopt
"""
def sdpsolve(b,eta):
    m = len(b)
    z = 0
    for i in range(0,m):
        if (len(b[i-1]) != len(b[0])):
            z += 1
    if (z!=0):
        print "The points may have different dimensions."
    a = []
    for i in range(0,m):
        a.append(b[i][0])
        a.append(b[i][1])
    eta.insert(0,0.)
    eta.insert(0,0.)
    eta = matrix(eta)
    L = [[-1.,0.,0.,1.],
        [0.,-1.,-1.,0.],
        [-1.,0.,0.,-1.],
        [0.,0.,0.,0.]]
    for i in range(1,m-1):
        L.append([0.,0.,0.,0.])
    G = [matrix(L)]
    for j in range(2,m+1):
        L[j], L[1+j] = L[1+j], L[j]
        G += [matrix(L)]
    h = [matrix([[-a[0],-a[1]],[-a[1],a[0]]])]
    for k in range(1,m):
        h += [matrix([[-a[2*k],-a[2*k+1]],[-a[2*k+1],a[2*k]]])]
    return solvers.sdp(eta, Gs=G, hs=h)

# Other subroutines

"""
Plotting some points on a plane or 3D space.
Input: list = a list of m+1 points in R2 or R3.
Output: First m points in blue and the last point in red.
"""
def plotting(list):
    print "Blue points are given."
    print "The red point is the Fermat-Torricelli point."
    AA = point(list[0],size=50)
    for i in range(1,len(list)-1):
        AA += point(list[i],size=50)
    AA += point(list[len(list)-1],size=80,rgbcolor = 'red')
    for j in range(0,len(list)-1):
        AA += line([list[j], list[len(list)-1]],rgbcolor = 'green')
    AA.show()

"""
Generate random points using normal distribution.
Input: d = dimension
       n = number of points in R^d to be generated
       mu = mean of each random coordinate
       sigma = standard deviation of each random coordinate
Output: A list of random points
"""
def gen_points(n, d, mu, sigma):
    points = []
    for i in range(0, n):
        p = [0.] * d
        for j in range(0, d):
            p[j] = random.normalvariate(mu, sigma)
        points.append(p)
    return points


# Main process

"""
Solving the Fermat-Torricelli Problem with Weights, using three methods.
Input: points = a list of points
       weights = a list of weights, length of weights is the number of points
       method = either Weiszfeld ("weisz"), SOCP ("socp") or SDP ("sdp")
Output: data = The input data and the method
        iterations = number of iterations in the process, cannot exceed 100
        fermat_point = the Fermat point
        min_val = minimum objective value
        how_long = clock time
        image = a picture showing the given points and the Fermat point;
                only works in R2 or R3 (need Java)
"""
class FermatTorri(object):
    def __init__(self, points, weights, method):
        self.points = points
        self.weights = weights
        self.method = method
        if self.method == "weisz":
            self.sol1 = weiszsolve(self.points[0],self.points,self.weights)
        elif self.method == "socp":
            self.sol2 = socpsolve(self.points,self.weights)
            for i in self.points:
                i.remove(0)
            for j in range(0,len(self.points[0])):
                self.weights.remove(0)
        elif self.method == "sdp":
            self.sol3 = sdpsolve(self.points,self.weights)
            for j in range(0,len(self.points[0])):
                self.weights.remove(0)
        else: print "Unavailable method!"

    def data(self):
        if self.method == "weisz":
            print "Modified Weiszfeld algorithm"
        elif self.method == "socp":
            print "Second Order Cone Programming"
        elif self.method == "sdp":
            print "Semidefinite Programming"
        print "The " + str(len(self.points)) + " points are: " + str(self.points)
        print "The " + str(len(self.weights)) + " weights are: " + str(self.weights)

    def iterations(self):
        if self.method == "weisz":
            return self.sol1[0]
        elif self.method == "socp":
            return(self.sol2['iterations'])
        elif self.method == "sdp":
            return(self.sol3['iterations'])

    def fermat_point(self):
        if self.method == "weisz":
            return self.sol1[2]
        elif self.method == "socp":
             d = len(self.points[0])
             return([self.sol2['x'][i] for i in range(0,d)])
        elif self.method == "sdp":
             return([self.sol3['x'][0],self.sol3['x'][1]])

    def min_val(self):
        if self.method == "weisz":
            return self.sol1[1]
        elif self.method == "socp":
             return(self.sol2['primal objective'])
        elif self.method == "sdp":
             return(self.sol3['primal objective'])

    def how_long(self):
        if self.method == "weisz":
            t1 = Timer(lambda:self.sol1)
            return t1.timeit(number = 10000)
        elif self.method == "socp":
             t2 = Timer(lambda:self.sol2)
             return t2.timeit(number = 10000)
        elif self.method == "sdp":
             t3 = Timer(lambda:self.sol3)
             return t3.timeit(number = 10000)

    def image(self):
        if self.method == "weisz":
            p = self.sol1[2]
        elif self.method == "socp":
            d = len(self.points[0])
            p = [self.sol2['x'][i] for i in range(0,d)]
        elif self.method == "sdp":
            p = [self.sol3['x'][0],self.sol3['x'][1]]
        self.points.append(p)
        plotting(self.points)
        self.points.remove(p)

"""
Test the three methods.
Input: n = number of datasets
       m = number of points in R2 for each dataset
       mu,sigma = mean and standard deviation of each random coordinate
                  (following normal distribution)
Output: Median clock time and number of iterations of each method
"""
def test_algorithm(n,m,mu,sigma):
    d = 2
    weights = [1.]*m
    iteration_list1 = [0.]*n
    iteration_list2 = [0.]*n
    iteration_list3 = [0.]*n
    time_list1 = [0.]*n
    time_list2 = [0.]*n
    time_list3 = [0.]*n
    for j in range(0,n):
        points = gen_points(m,d,mu,sigma)
        sol1 = FermatTorri(points,weights,"weisz")
        iteration_list1[j] = sol1.iterations()
        time_list1[j] =  sol1.how_long()
        sol2 = FermatTorri(points,weights,"socp")
        iteration_list2[j] = sol2.iterations()
        time_list2[j] =  sol2.how_long()
        sol3 = FermatTorri(points,weights,"sdp")
        iteration_list3[j] = sol3.iterations()
        time_list3[j] =  sol3.how_long()
    print "Weiszfeld algorithm median number of iterations: " + str(numpy.median(numpy.array(iteration_list1)))
    print "Weiszfeld algorithm median time: " + str(numpy.median(numpy.array(time_list1)))
    print "SOCP median number of iterations: " + str(numpy.median(numpy.array(iteration_list2)))
    print "SOCP median time: " + str(numpy.median(numpy.array(time_list2)))
    print "SDP median number of iterations: " + str(numpy.median(numpy.array(iteration_list3)))
    print "SDP median time: " + str(numpy.median(numpy.array(time_list3)))




︡4b26b134-4733-49e7-80d7-8cb9c440eaa1︡{"stdout":"'\\nModified Weiszfeld algorithm\\nInput: y = initial point\\n       a = list of given points\\n       eta = weights\\nOutput: A list including the number of iterations, minimum value and the\\n        Fermat point.\\n'\n'\\nSolving the Fermat-Torricelli problem using second order cone programming\\n(SOCP) formulation.\\nInput: a = list of given points\\n       eta = weights\\nOutput: The output from the solver in cvxopt\\n'\n'\\nSolving the Fermat-Torricelli problem using semidefinite programming\\n(SDP) formulation.\\nInput: b = list of given points\\n       eta = weights\\nOutput: The output from the solver in cvxopt\\n'\n'\\nPlotting some points on a plane or 3D space.\\nInput: list = a list of m+1 points in R2 or R3.\\nOutput: First m points in blue and the last point in red.\\n'\n'\\nGenerate random points using normal distribution.\\nInput: d = dimension\\n       n = number of points in R^d to be generated\\n       mu = mean of each random coordinate\\n       sigma = standard deviation of each random coordinate\\nOutput: A list of random points\\n'\n'\\nSolving the Fermat-Torricelli Problem with Weights, using three methods.\\nInput: points = a list of points\\n       weights = a list of weights, length of weights is the number of points\\n       method = either Weiszfeld (\"weisz\"), SOCP (\"socp\") or SDP (\"sdp\")\\nOutput: data = The input data and the method\\n        iterations = number of iterations in the process, cannot exceed 100\\n        fermat_point = the Fermat point\\n        min_val = minimum objective value\\n        how_long = clock time\\n        image = a picture showing the given points and the Fermat point;\\n                only works in R2 or R3 (need Java)\\n'\n'\\nTest the three methods.\\nInput: n = number of datasets\\n       m = number of points in R2 for each dataset\\n       mu,sigma = mean and standard deviation of each random coordinate\\n                  (following normal distribution)\\nOutput: Median clock time and number of iterations of each method\\n'\n"}︡
︠f0bbd349-8796-4e99-a86c-4faa2a548f95︠
%python
A = FermatTorri([[0.,0.],[1.,0.],[3.,1.],[-1.,4.]],[1.5,1.,1.,1.],"socp")
︡e0e6ce33-de03-4f30-b570-bdc191921426︡
︠4fc4ba17-dee8-42b0-ae7e-cafaef646bb4︠
A.data()
︡88765ed1-a065-4985-bbf6-49f9ba153292︡{"stdout":"Second Order Cone Programming\nThe 4 points are: [[0.0, 0.0], [1.0, 0.0], [3.0, 1.0], [-1.0, 4.0]]\nThe 4 weights are: [1.5, 1.0, 1.0, 1.0]\n"}︡
︠1c5e999c-5cc3-4283-ab83-2a2e731c5730︠
A.fermat_point()
︡4a15c5d2-7de5-4235-9220-2a6319b4f000︡{"stdout":"[0.6171234918123726, 0.2789962153648668]\n"}︡
︠43679da3-cd8f-453d-a386-70c9f245c974︠
A.iterations()
︡7f7baa77-ade6-45c6-952e-374503c598ca︡{"stdout":"10\n"}︡
︠2334284b-97f1-42ca-96ee-b44b3b70c846︠
A.image()
︡9a9447e6-110f-4f3d-bb7d-8cae0686a299︡{"stdout":"Blue points are given.\nThe red point is the Fermat-Torricelli point.\n"}︡{"file":{"show":true,"uuid":"5ff9ddc6-6095-45e0-a826-756bb977a228","filename":"/mnt/home/kBrwPcrD/.sage/temp/compute1a/32274/tmp_4xN1ya.png"}}︡
︠35a0f7a4-257b-4934-ab28-b41a4933631f︠
A.min_val()

︡db328d30-abad-4164-ba93-723fcbd7fac6︡{"stdout":"8.036411107527076\n"}︡
︠5c115fce-ac0c-47ef-9a7d-7a8d1d751dcf︠
A.how_long()
︡4f83a8c6-357e-4f78-907d-912ef0a26e6c︡{"stdout":"0.0024628639221191406\n"}︡
︠def411eb-919b-4881-bfb4-94a16c81d007︠
B = FermatTorri([[0.,0.,1.],[1.,0.,-1.],[3.,1.,3.]],[1.,1.,1.],"weisz")

︡9b32d4e9-490a-47d5-882f-5c9623d1a37e︡
︠e2a4b0ca-1f2b-4ba0-afe6-55d5cb52af9c︠
B.iterations()
︡5d81ea60-92ad-4df3-9607-d4733ea4f324︡{"stdout":"100\n"}︡
︠170cdb08-0cf2-400e-95c0-718451f3fb5ci︠
B.fermat_point()
︠bef51bbe-3166-4569-a091-c7f792455d56︠
B.image()
︡38f3ca9a-8fe8-48ef-b467-e31fcce724a6︡{"stdout":"Blue points are given.\nThe red point is the Fermat-Torricelli point.\n"}︡
︠242c53d7-b33b-4a02-84cd-d60dfd150853︠
C = FermatTorri([[0.,0.,1.],[1.,0.],[3.,1.,3.]],[1.,1.,1.],"weisz")
︡40cc30b6-1844-45f7-a90e-353de4890315︡{"stdout":"The points may have different dimensions.\n"}︡{"stderr":"Error in lines 1-1\nTraceback (most recent call last):\n  File \"/mnt/home/kBrwPcrD/.sagemathcloud/sage_server.py\", line 412, in execute\n    exec compile(block, '', 'single') in namespace, locals\n  File \"\", line 1, in <module>\n  File \"\", line 7, in __init__\n  File \"\", line 11, in weiszsolve\n  File \"\", line 4, in r\n  File \"\", line 2, in dist\nValueError: operands could not be broadcast together with shapes (3) (2) \n"}︡
︠0746703b-8b16-48e1-9828-a7ee10a68b13︠
%python

C = FermatTorri([[0.,0.,0.,0.,0.],[1.,0.,0.,0.,0.],[3.,1.,0.,0.,0.],[-1.,4.,0.,0.,0.]],[1.5,1.,1.,1.],"socp")
︡c0ce2e22-2090-48c0-9f9d-2950cb535fa5︡{"stderr":"Error in lines 1-1\nTraceback (most recent call last):\n  File \"/mnt/home/kBrwPcrD/.sagemathcloud/sage_server.py\", line 412, in execute\n    exec compile(block, '', 'single') in namespace, locals\n  File \"\", line 1, in <module>\n  File \"\", line 9, in __init__\n  File \"\", line 31, in socpsolve\n  File \"/usr/local/sage/sage-5.10.rc1/local/lib/python2.7/site-packages/cvxopt/coneprog.py\", line 3552, in socp\n    = ds)\n  File \"/usr/local/sage/sage-5.10.rc1/local/lib/python2.7/site-packages/cvxopt/coneprog.py\", line 705, in conelp\n    raise ValueError(\"Rank(A) < p or Rank([G; A]) < n\")\nValueError: Rank(A) < p or Rank([G; A]) < n\n"}︡
︠dd0f1985-2bb2-4649-91c5-682216ca3d09︠
%python

test_algorithm(3,3,0,5)
︡9ad6d7c9-f664-421b-9f30-160cc275c28d︡{"stdout":"     pcost       dcost       gap    pres   dres   k/t"}︡{"stdout":"\n 0: -1.6254e-15 -1.3510e-15  3e+01  1e+00  3e-16  1e+00\n 1:  1.5708e+01  1.5967e+01  6e+00  2e-01  9e-16  5e-01\n 2:  1.7307e+01  1.7325e+01  4e-01  2e-02  7e-16  3e-02\n 3:  1.7408e+01  1.7416e+01  1e-01  4e-03  5e-15  1e-02\n 4:  1.7433e+01  1.7440e+01  6e-02  3e-03  4e-15  9e-03\n 5:  1.7451e+01  1.7452e+01  4e-03  2e-04  2e-15  6e-04\n 6:  1.7453e+01  1.7453e+01  2e-04  8e-06  5e-15  3e-05\n 7:  1.7453e+01  1.7453e+01  2e-06  9e-08  3e-14  4e-07\nOptimal solution found.\n     pcost       dcost       gap    pres   dres   k/t\n 0:  4.9801e-16 -4.1731e-16  3e+01  1e+00  4e-16  1e+00\n 1:  1.5652e+01  1.5948e+01  7e+00  3e-01  3e-16  5e-01\n 2:  1.7290e+01  1.7309e+01  4e-01  2e-02  9e-16  3e-02\n 3:  1.7401e+01  1.7409e+01  1e-01  5e-03  6e-16  1e-02\n 4:  1.7432e+01  1.7438e+01  7e-02  3e-03  6e-16  8e-03\n 5:  1.7451e+01  1.7451e+01  5e-03  2e-04  1e-15  7e-04\n 6:  1.7453e+01  1.7453e+01  2e-04  1e-05  3e-16  3e-05\n 7:  1.7453e+01  1.7453e+01  5e-06  2e-07  4e-16  6e-07\n 8:  1.7453e+01  1.7453e+01  1e-07  6e-09  7e-16  2e-08\nOptimal solution found.\n     pcost       dcost       gap    pres   dres   k/t"}︡
︠3721982f-c967-4fdf-92bb-1df3af8f1e42i︠
%python
B = FermatTorri([[0.,0.,1.],[1.,0.,-1.],[3.,1.,0.]],[1.5,1.,1.],"socp")
test_algorithm(3,4,0,5)
︠2d4c6448-d3a3-488d-8c39-299e20481eb7i︠
test_algorithm(3,4,0,5)
%python
test_algorithm(5,4,0,5)
︠37729d53-95ab-41f2-acbb-689bb6d5f263i︠
test_algorithm(2,4,0,5)
︠d551041b-499a-4598-b16a-59bab9efd7fb︠

