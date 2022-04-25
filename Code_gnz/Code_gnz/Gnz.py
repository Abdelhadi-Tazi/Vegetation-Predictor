#%%
# generic imports
from math import sqrt, ceil
from typing import Iterable
import numpy as np
import random as rd
import heapq
import tkinter as tk
import matplotlib.pyplot as plt
import os

#%% imports from files in this project

from GUI import Sample, draw_circle, Display, Display2, Display_dict, Display_dict2
import Point
from Point import Point
from ClosestPair import dist, DistClosestPair
from Neighborhood import dist_n, neighborhood, neighborhood2, representative_point

#%%

# ***********************************************
# Other functions used by the algorithm
# ***********************************************

def is_close(p: Point, dL: dict, dmin: float, len_cell: float):
    """returns True if there is a point in dL which is less than dmin away from p, dL is a dictionary with coordinates of cells as keys and lists of points as values"""
    # We first determine which cells overlay the circle of center p and radius dmin
    kxmin, kxmax = int((p.x - dmin) // len_cell), int((p.x + dmin) // len_cell)
    kymin, kymax = int((p.y - dmin) // len_cell), int((p.y + dmin) // len_cell)
    for x_cell in range(kxmin, kxmax + 1):
        for y_cell in range(kymin, kymax + 1):
            if (x_cell, y_cell) in dL: # for each of these cells, if they are part of the map then we check if there is a point too close to p
                for q in dL[(x_cell, y_cell)]:
                    if dist(p, q) < dmin:
                        return True
    return False

def motif_counter(p: Point, dL: dict, dmin: float, len_cell: float, nb_N: int):
    """returns True if there is a point in L which is less than dmin away from p, and a counter of the different neighborhoods copied less than 5*dmin away, the counter is only fully correct if the boolean returned is False, but in the case where it is True we do not make use of the counter anyway"""
    motif_counter = [0] * nb_N

    kxmin, kxmax = int((p.x - dmin) // len_cell), int((p.x + dmin) // len_cell)
    kymin, kymax = int((p.y - dmin) // len_cell), int((p.y + dmin) // len_cell)
    for x_cell in range(kxmin, kxmax + 1):
        for y_cell in range(kymin, kymax + 1):
            if (x_cell, y_cell) in dL: # for each of these cells, if they are part of the map then we check if there is a point too close to p
                for q in dL[(x_cell, y_cell)]:
                    if dist(p, q) < 5 * dmin:
                        if q.neighborhood >= 0:
                            motif_counter[q.neighborhood] += 1
    return motif_counter


def penalize(nb_instances: int, distance: float):
    return (nb_instances + 1) * distance


# ***********************************************
# Functions for analysis of performance
# ***********************************************

def calc_density(L: Iterable[Point], r: float, xmin: int, ymin: int, xmax: int, ymax: int):
    r = ceil(r)
    D = np.zeros((xmax - xmin + 1, ymax - ymin + 1))
    for p in L:
        a, b = int(p.x), int(p.y)
        for x in range(max(xmin, a - r), min(xmax, a + r + 1)):
            for y in range(max(ymin, b - r), min(ymax, b + r + 1)):
                d = dist(p, Point(x,y))
                if d <= r:
                    D[x, y] += 1 - d/r
    return D

def calc_density_point(p: Point, L: Iterable[Point], r: float):
    count = 0
    for q in L:
        if dist(p, q) <= r:
            count += 1 - dist(p, q)/r
    return count

#%%

# ***********************************************
# Main generalization algorithm
# ***********************************************

# Initialization
def init_neighborhoods(S: Iterable[Point], dmin: float = None, ignore_border = True, n_shell = 3, n_sector = 8):
    # ***********************************************
    # Pre-processing of the sample: producing dmin and hist
    # ***********************************************

    dclosestpair, Sx, Sy = DistClosestPair(S)
    if dmin == None:
        dmin = dclosestpair
    
    # Coordinates of the edges of the sample
    xmins = Sx[0].x
    xmaxs = Sx[-1].x
    ymins = Sy[0].y
    ymaxs = Sy[-1].y
    
    sample_neighborhoods, sample_histograms = [], []
    if ignore_border:
        # we eliminate the elements close to the edges to avoid border effects because their neighborhood hasn't been fully described in the sample
        for p in Sx:
            if (p.x < xmaxs - 3*dmin and p.x > xmins + 3*dmin and p.y > ymins + 3*dmin and p.y < ymaxs - 3*dmin):
                n_p, h_p = neighborhood(p, Sx, dmin, n_shell, n_sector)
                sample_neighborhoods.append(n_p)
                sample_histograms.append(h_p)

    else:
        for p in Sx:
            n_p, h_p = neighborhood(p, Sx, dmin, n_shell, n_sector)
            sample_neighborhoods.append(n_p)
            sample_histograms.append(h_p)
    
    return dclosestpair, sample_histograms, sample_neighborhoods

def init_dilate(sample_histograms, sample_neighborhoods, density: np.ndarray, n_shell = 3, n_sector = 8):
    """
    Initialization of the map: first point and neighborhood copied.

    Density is modified by dilating/contracting the reference length dmin.
    """
    # Generates a map of assets that ranges from 0 to xmax and 0 to ymax
    xmax, ymax = np.shape(density)

    nb_N = len(sample_histograms)

    first_elem = rd.randint(0, nb_N - 1)
    p0 = Point(xmax//2, ymax//2)
    L = [p0]                    # list of points on the map
    heap = []                   # heap of points to be examined, with priority the distance to the first point examined: in this case the center
    
    density_factor = 1 / sqrt(density[p0.x, p0.y])

    for grid in range(n_shell * n_sector):
        for p in sample_neighborhoods[first_elem][grid]:
            q = Point(p0.x + p.x * density_factor, p0.y + p.y * density_factor, p.type)
            d = dist(p0, q)
            L.append(q)
            heapq.heappush(heap, (d, q))
    
    return L, heap


def gen_dilate(S: Iterable[Point], density: np.ndarray, display = True, dmin: float = None, ignore_border = True, n_shell = 3, n_sector = 8):
    """
    Density is modified by dilating/contracting the reference length dmin.
    Inputs:
        - sample S in the shape of a rectangle
        - density parameter : a matrix of size (xmax, ymax), which determines the size of the generated map
    """
    # Generates a map of assets that ranges from 0 to xmax and 0 to ymax
    xmax, ymax = np.shape(density)

    if display:
        root = tk.Tk()
        root.title("Generated assets")

        canvas = tk.Canvas(root, width = xmax + 10, height = ymax + 10)
        canvas.pack(side="top", fill="both", expand=True)
        r = 2
        
        def click(event):
            x, y = event.x, event.y
            print(x, y)
        
        root.bind('<Button-1>', click)
    
    # Pre-processing of the sample: producing dclosestpair and histograms
    dclosestpair, sample_histograms, sample_neighborhoods = init_neighborhoods(S, dmin, ignore_border, n_shell, n_sector)
    if dmin == None:
        dmin = dclosestpair

    nb_N = len(sample_neighborhoods)

    # Initialization of the map: first point and neighborhood copied
    L, heap = init_dilate(sample_histograms, sample_neighborhoods, density, n_shell, n_sector)


    len_cell = max(xmax/10, 5 * dmin)
    kx, ky = xmax // len_cell, ymax // len_cell
    dL = {}
    done = {}                 # points whose neighborhood should not evolve anymore, reference for is_close
    dheap = {}
    coord_heap = []

    for i in range(int(kx) + 1):
        for j in range(int(ky) + 1):
            dL[(i, j)] = []
            done[(i, j)] = []
            dheap[(i, j)] = []

    p0 = Point(xmax//2, ymax//2)

    done[(p0.x // len_cell, p0.y // len_cell)].append(p0)
    for p in L:
        i, j = p.x // len_cell, p.y // len_cell
        dL[(i, j)].append(p)
        if (p.x, p.y) != (p0.x, p0.y):
            if len(dheap[(i, j)]) == 0:
                heapq.heappush(coord_heap, (dist(p, p0), (i, j)))
            heapq.heappush(dheap[(i, j)], (dist(p, p0), p))
        if display:
            draw_circle(canvas, p.x, p.y, r)
            canvas.update()


    # ***********************************************
    # Main loop
    # ***********************************************

    print('Initialization: done')
    while len(coord_heap) >  0: # d_last_element < dmax and 
        _, (i, j) = heapq.heappop(coord_heap)
        while len(dheap[(i, j)]) > 0:
            d, p = heapq.heappop(dheap[(i, j)])
            density_factor = 1 / max(density[int(p.x), int(p.y)], 0.01)

            Hist_N = neighborhood2(p, dL, dmin * density_factor, len_cell, n_shell, n_sector)
            idx_neighbor = 0

            determined = []
            for grid in range(n_shell * n_sector):
                p_rep_grid = representative_point(grid, p, dmin * density_factor, n_shell, n_sector)
                determined.append(is_close(p_rep_grid, done, 3 * dmin * density_factor, len_cell))
            
            motifs = motif_counter(p, done, dmin, len_cell, nb_N)
            dist_neighbor = penalize(motifs[0], dist_n(Hist_N, sample_histograms[0], determined, n_shell, n_sector))
            for k in range(1, nb_N):
                distance = penalize(motifs[k], dist_n(Hist_N, sample_histograms[k], determined, n_shell, n_sector))
                if distance < dist_neighbor:
                    idx_neighbor = k
                    dist_neighbor = distance
            for grid in range(n_shell * n_sector):
                if not determined[grid]:
                    for q in sample_neighborhoods[idx_neighbor][grid]:
                        p1 = Point(p.x + q.x * density_factor, p.y + q.y * density_factor, q.type)
                        if not is_close(p1, dL, dclosestpair * 0.75 * density_factor, len_cell):
                            if p1.x < xmax and p1.y < ymax and 0 <= p1.x and 0 <= p1.y:
                                i2, j2 = p1.x // len_cell, p1.y // len_cell
                                dL[(i2, j2)].append(p1)
                                if display:
                                    draw_circle(canvas, p1.x, p1.y, r)
                                    canvas.update()
                                d2 = dist(p0, p1)
                                if len(dheap[(i2, j2)]) == 0:
                                    heapq.heappush(coord_heap, (d2, (i2, j2)))
                                heapq.heappush(dheap[(i2, j2)], (d2, p1))

            done[(p.x // len_cell, p.y // len_cell)].append(p)

    print('Generalization process... done')
    return dL


#%%
if __name__ == '__main__':
    dir_path = "" # ENTER PATH TO FOLDER HERE

    # regular sample
    S = [Point(p[0]/2, p[1]/2) for p in [(12, 7), (6, 17), (7, 41), (22, 32), (22, 19), (33, 10), (65, 9), (50, 20), (42, 35), (34, 48), (19, 50), (14, 63), (12, 79), (30, 77), (36, 61), (53, 61), (53, 47), (65, 33), (70, 22), (80, 5), (101, 4), (92, 23), (83, 38), (70, 52), (67, 74), (55, 77), (38, 84), (20, 102), (11, 96), (11, 115), (28, 125), (37, 109), (47, 99), (74, 95), (61, 87), (59, 116), (45, 116), (42, 140), (22, 135), (10, 127), (11, 143), (55, 130), (55, 140), (72, 129), (77, 146), (89, 140), (88, 125), (76, 111), (95, 103), (99, 118), (112, 123), (110, 138), (126, 144), (127, 128), (138, 109), (126, 112), (114, 104), (112, 78), (106, 85), (93, 87), (83, 74), (86, 56), (97, 68), (111, 64), (103, 44), (110, 25), (117, 10), (136, 6), (131, 20), (121, 34), (119, 49), (132, 53), (130, 67), (130, 83), (124, 93), (141, 89), (139, 37), (140, 135)]]

    # sample with clusters
    S2 = [Point(x/2, y/2) for (x,y) in np.load(dir_path + "/Samples/sample_bosquets.npy")]

    dmin, _, _ = init_neighborhoods(S)
    dir_path += "/Density_examples"
    # launch simulation on 10 density maps produced by neural network
    for nb_file in range(10):
        if nb_file >= 10:
            filename = "00" + str(nb_file) + ".npy"
        else:
            filename = "000" + str(nb_file) + ".npy"
        
        path = os.path.join(dir_path,filename)
        print(filename)
        M1 = np.load(path)
        I, J, _ = np.shape(M1)
        M = np.zeros((4 * I, 4 * J))
        for i in range(4 * I):
            for j in range(4 * J):
                M[i, j] = M1[i//4, j//4, 0] / 4
        
        L = gen_dilate(S2, M, display = False, dmin = 24)
        L2 = []
        for li in L.values():
            L2 += li
        D_calc = calc_density(L2, 5 * dmin, 0, 0, 4 * I, 4 * J)
        im = Display_dict2(L, 4 * I, 4 * J)
        plt.imshow(im)
        plt.show()
        plt.imshow(D_calc)
        plt.show()
        plt.imshow(M)
        plt.show()
