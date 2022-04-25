from Point import Point
from ClosestPair import dist
from math import ceil, acos, pi, sin, cos
import numpy as np

# ***********************************************
# Functions used to work with grids and neighborhoods
# ***********************************************

def grid_nb(p_center: Point, p: Point, r: float, n_shell: int, n_sector: int):
    """ Determines the index of the sector in which p is located:
     - r is the radius of the smallest concentric circle
     - n in the number of concentric circles
     - p_center is the centerpoint of the neighborhood being studied """
    
    if dist(p, p_center) > n_shell * r or dist(p, p_center) == 0:
        return -1
    
    shell = ceil(dist(p, p_center) / r) - 1
    
    if p.y >= p_center.y:
        sector = n_sector * acos((p.x - p_center.x)/dist(p, p_center)) // (2 * pi)
    
    else:
        sector = n_sector * (1 - acos((p.x - p_center.x) / dist(p, p_center)) / (2 * pi) ) // 1
    return int(shell * n_sector + sector)

def get_params(gridnb: int, n_shell: int, n_sector: int):
    """ The argument n_shell is here useless but we keep it to maintain harmony, the result should verify shell < n_shell """
    shell = gridnb // n_sector
    sector = gridnb - shell * n_sector
    return shell, sector

def representative_point(gridnb: int, p: Point, r: float, n_shell: int, n_sector: int):
    """ Returns the coordinates of the representative point of the grid of the neighborhood of p """
    shell, sector = get_params(gridnb, n_shell, n_sector)
    
    d = (shell + 0.5) * r
    theta = (sector + 0.5) * 2 * pi / n_sector
    
    return Point(p.x + d * cos(theta), p.y + d * sin(theta))

def neighborhood(p0: Point, S: list, r: float, n_shell: int, n_sector: int):
    """Returns two lists:
       - N contains at idx k the list of the relative positions of the points in grid k of the neighborhood of p0
       - Hist_N is the histogram of N"""
    N = []
    Hist_N = []
    for _ in range(n_shell * n_sector):
        N.append([])
        Hist_N.append(0)
    for q in S:
        k = grid_nb(p0, q, r, n_shell, n_sector)
        if k >= 0:
            Hist_N[k] += 1
            N[k].append(Point(q.x - p0.x, q.y - p0.y))
    return N, Hist_N

def neighborhood2(p: Point, S: dict, r: float, len_cell: float, n_shell: int, n_sector: int):
    """Returns only Hist_N, the histogram of the neighborhood"""
    Hist_N = []
    for _ in range(n_shell * n_sector):
        Hist_N.append(0)
    
    kxmin, kxmax = int((p.x - r) // len_cell), int((p.x + r) // len_cell)
    kymin, kymax = int((p.y - r) // len_cell), int((p.y + r) // len_cell)
    for x_cell in range(kxmin, kxmax + 1):
        for y_cell in range(kymin, kymax + 1):
            if (x_cell, y_cell) in S:
                for q in S[(x_cell, y_cell)]:
                    k = grid_nb(p, q, r, n_shell, n_sector)
                    if k >= 0:
                        Hist_N[k % (n_sector * n_shell)] += 1
    return Hist_N

def dist_n(N1: list, N2: list, determined: list, n_shell: int, n_sector: int, sigma: float = 0.5) -> float:
    """ sigma is an empirical value, anything between 0.1 and 10 seems to work, in the article they use 0.5 """
    M = np.array([[L1 - L2 for L1, L2 in zip(N1, N2)]])
    _, n = M.shape
    for k in range(n):
        M[0, k] = M[0, k] * determined[k]
    A = np.zeros((n, n))
    for i in range(n):
        shell_i, sector_i = get_params(i, n_shell, n_sector)
        for j in range(n):
            shell_j, sector_j = get_params(j, n_shell, n_sector)
            A[i, j] = np.exp(- sigma * (abs(shell_i - shell_j) + abs(sector_i - sector_j)))
    return np.dot(np.dot(M, A), np.transpose(M))

