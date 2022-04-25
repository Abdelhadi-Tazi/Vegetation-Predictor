from Point import Point
from math import sqrt, ceil


# ***********************************************
# Functions used to preprocess the sample
# ***********************************************


def Merge_x(L1: list, L2: list):
    i1 = i2 = 0
    L = []
    while i1 < len(L1) and i2 < len(L2):
        if L1[i1].x < L2[i2].x or (L1[i1].x == L2[i2].x and L1[i1].y < L2[i2].y) :
            L.append(L1[i1])
            i1 += 1
        else:
            L.append(L2[i2])
            i2 += 1
    if i1 == len(L1):
        for k in range(i2, len(L2)):
            L.append(L2[k])
    else:
        for k in range(i1, len(L1)):
            L.append(L1[k])
    return L

def Merge_y(L1: list, L2: list):
    i1 = i2 = 0
    L = []
    while i1 < len(L1) and i2 < len(L2):
        if L1[i1].y < L2[i2].y or (L1[i1].y == L2[i2].y and L1[i1].x < L2[i2].x) :
            L.append(L1[i1])
            i1 += 1
        else:
            L.append(L2[i2])
            i2 += 1
    if i1 == len(L1):
        for k in range(i2, len(L2)):
            L.append(L2[k])
    else:
        for k in range(i1, len(L1)):
            L.append(L1[k])
    return L

def Order_i(L: list, i: int):
    """Orders a list of points by increasing i-th coordinate"""
    if len(L) <= 1:
        return L
    else:
        L1 = Order_i(L[0::2], i)
        L2 = Order_i(L[1::2], i)
        if i == 0:
            return Merge_x(L1, L2) 
        else:
            return Merge_y(L1, L2)    

def dist(p: Point, q: Point):
    """Computes the euclidean distance between two points."""
    return sqrt((p.x - q.x)**2 + (p.y - q.y)**2)

def lexico(p: Point, q: Point):
    """Returns True if p comes before q in lexicographic order"""
    return p.x < q.x or (p.x == q.x and p.y < q.y)

def DistClosestPair_aux(Lx: list, Ly: list):
    """Here we suppose that Lx is sorted in order of increasing x-value and Ly in increasing y-value.
    In case of ties, the points are ordered by their other coordinate.
    We also suppose that Lx and Ly represent the same set of points, which contains at least two elements."""
    
    # base cases
    if len(Lx) == 2:
        return dist(Lx[0], Lx[1])
    
    elif len(Lx) == 3:
        return min(dist(Lx[0], Lx[1]), dist(Lx[0], Lx[2]), dist(Lx[1], Lx[2]))
    
    # if we split a list of more than 4 elements, we get two lists of at least 2 elements
    else:
        m = ceil(len(Lx)/2)
        # splitting phase
        Qx = Lx[:m]
        Rx = Lx[m:]
        Qy = []
        Ry = []
        for i in range(len(Ly)):
            if lexico(Ly[i], Qx[-1]): # uses lexicographic order
                Qy.append(Ly[i])
            else:
                Ry.append(Ly[i])
        
        # recursion
        qmin = DistClosestPair_aux(Qx, Qy)
        rmin = DistClosestPair_aux(Rx, Ry)
        
        # combining phase
        dmin = min(qmin, rmin)
        l = Qx[-1].x
        S = [p for p in Ly if l - dmin <= p.x and p.x <= l + dmin]
        for i in range(len(S)):
            for j in range(1, min(7, len(S) - i - 1)):
                dmin = min(dmin, dist(S[i], S[i+j]))
        return dmin

def DistClosestPair(L: list):
    Lx = Order_i(L, 0)
    Ly = Order_i(L, 1)
    return DistClosestPair_aux(Lx, Ly), Lx, Ly