import multipers as mp
import multipers.grids as mpg
import numpy as np
import gudhi as gd
from multipers.ml.convolutions import KDE as KernelDensity
import dionysus as d
from itertools import combinations
from collections import defaultdict


# =========================================================================
# From a point cloud to a bifiltration
# =========================================================================

def get_bifiltration(cloud):
    bifiltration = gd.RipsComplex(points=cloud).create_simplex_tree()
    bifiltration = mp.SimplexTreeMulti(bifiltration, num_parameters=2)

    codensity = - KernelDensity(bandwidth=0.2).fit(cloud).score_samples(cloud) 
    bifiltration.fill_lowerstar(codensity, parameter=1)
    return bifiltration


# =========================================================================
# From a bifiltration to a grid & intervals
# =========================================================================

# The four functions below are used to construct a grid and intervals from a bifiltration.
def is_smaller(node1, node2):
    if node1[0] <= node2[0] and node1[1] <= node2[1]:
        return True
    else:
        return False


def add_source(pt, points):
    for other in points:
        if is_smaller(other, pt) and other != pt:
            return False
    return True
    

def add_sink(pt, points):
    for other in points:
        if is_smaller(pt, other) and other != pt:
            return False
    return True


def get_src_snk(points):
    src = []
    snk = []
    for pt in points:
        if add_source(pt, points):
            src.append(pt)
        if add_sink(pt, points):
            snk.append(pt)
    return (src, snk)


# Based on a given bifiltration, this function creates a 4 x 4 grid
# To reduce computational cost, we've chosen a resolution parameter of 4 for now.
def get_grid(bifiltration, number):
    grid = mpg.compute_grid(bifiltration, strategy='regular', resolution=number)

    return grid


# We will generate all (1, 1)- and (2, 1)-intervals on a given grid. 
def get_int_2_1(grid):
    grid = [(coor_x, coor_y) for coor_x in grid[0] for coor_y in grid[1]]
    list_int = []

    # (1, 1)-intervals
    for c in grid:
        list_int.append(([c], [c]))
    for c in combinations(grid, 2):
        if is_smaller(c[0], c[1]):
            list_int.append(([c[0]], [c[1]]))
        elif is_smaller(c[1], c[0]):
            list_int.append(([c[1]], [c[0]]))

    # (2, 1)-intervals
    for c in combinations(grid, 3):
        # if is_convex(c, grid) and is_connected(c):
        tmp = get_src_snk(c)
        if len(tmp[0]) == 2 and len(tmp[1]) == 1:
            list_int.append((tmp[0], tmp[1]))
    return list_int


# =========================================================================
# to compute GRIs
# =========================================================================

def find_upper_fence(maximal_elements):
    # maximal_elements = sorted([p for p in points if not parent_map[p]], key=lambda coor: coor[0])
    upper_fence = [maximal_elements[0]]

    for i in range(1, len(maximal_elements)):
        meet = (maximal_elements[i-1][0], maximal_elements[i][1])
        upper_fence += [meet, maximal_elements[i]]
    
    return upper_fence


def find_lower_fence(minimal_elements):
    # minimal_elements = sorted([p for p in points if not child_map[p]], key=lambda coor: coor[0], reverse=True)
    minimal_elements = sorted(minimal_elements, key=lambda coor: coor[0], reverse=True)
    lower_fence = [minimal_elements[0]]

    for i in range(1, len(minimal_elements)):
        join = (minimal_elements[i-1][0], minimal_elements[i][1])
        lower_fence += [join, minimal_elements[i]]
    
    return lower_fence

def get_bifiltration_on_grid(bifiltration, grid): 
    points = [(x, y) for x in grid[0] for y in grid[1]]
    bifil_on_grid = {pt: [] for pt in points}

    for complex, coordinate in bifiltration:
        for pt in points:
            if is_smaller(coordinate, pt):
                bifil_on_grid[pt].append(tuple(complex))

    return bifil_on_grid


def get_ZZpersistence(bifil_on_grid, interval, dimension):
    # to construct boundary ZZ filtration

    lower_fence = find_lower_fence(interval[0])
    upper_fence = find_upper_fence(interval[1])

    ZZ_coordinates = lower_fence + upper_fence
    
    # return a collection of complexes
    complexes = []
    for i in ZZ_coordinates:
        complexes += bifil_on_grid[i]

    complexes = list(set(complexes))
    
    # generate inputs for Dionysus 2
    f = d.Filtration(complexes)

    in_out = {key: [] for key in complexes}

    for complex in bifil_on_grid[ZZ_coordinates[0]]:
        in_out[complex].append(.1)

    step = .2

    for i in range(1, len(ZZ_coordinates)):
        enter_or_leave = list(set(bifil_on_grid[ZZ_coordinates[i]]) ^ set(bifil_on_grid[ZZ_coordinates[i - 1]]))

        for complex in enter_or_leave:
            in_out[complex].append(step)
        
        step += .1

    times = []

    for complex in complexes:
        times.append(in_out[complex])

    zz, dgms, cells = d.zigzag_homology_persistence(f, times)

    for i, dgm in enumerate(dgms):
        if i == dimension:
            return dgm
        
    return f"We can't find any {dimension}-dimensional cycle!"


# to check how many fully supported bar exists, i.e., GRIs
def count_fullbar(dgm):
    count = 0
    fullbar = d._dionysus.DiagramPoint(0.1, float('inf'))
    for bar in dgm:
        if bar == fullbar:
            count += 1

    return count


# =========================================================================
# to calculate Mobius inversion
# =========================================================================

def is_contained(interval_1, interval_2):
    for src_1 in interval_1[0]:
        src = False
        for src_2 in interval_2[0]:
            if is_smaller(src_2, src_1):
                src = True
                break
        if not src:
            return False
    for snk_1 in interval_1[1]:
        snk = False
        for snk_2 in interval_2[1]:
            if is_smaller(snk_1, snk_2):
                snk = True
                break
        if not snk:
            return False
    return True


def generate_poset(intervals):
    num_intervals = len(intervals)

    poset = [[-1 for j in range(num_intervals)] for i in range(num_intervals)]

    for i in range(num_intervals):
        poset[i][i] = 1
    
    for i in range(num_intervals):
        for j in range(num_intervals):
            if poset[i][j] == -1:
                if is_contained(intervals[i], intervals[j]):
                    poset[i][j] = 1
                    poset[j][i] = 0
                else:
                    poset[i][j] = 0
                    if is_contained(intervals[j], intervals[i]):
                        poset[j][i] = 1
                    else:
                        poset[j][i] = 0
    return poset


# poset[index] = ([children], [parents])
def generate_tree(intervals, poset):
    num_intervals = len(intervals)
    tree = [[[], []] for _ in range(num_intervals)]

    for i in range(num_intervals):
        possible_parents = [index for index in range(num_intervals) if poset[i][index] == 1 and index != i]
        
        check = np.array([0 for _ in range(num_intervals)])
        for possible_parent in possible_parents:
            check += np.array(poset[possible_parent])
        for index, value in enumerate(check):
            if value == 1:
                tree[i][1].append(index)
                tree[index][0].append(i)
            
    return tree


def mobius_inversion(int_1, int_2, poset, tree, matrix):
    if matrix[int_1][int_2] != None:
        return matrix[int_1][int_2]
    else:
        value = 0
        visit = [False for _ in range(len(matrix))]
        visit[int_2] = True

        children = [child for child in tree[int_2][0] if poset[int_1][child]]
        
        while children:
            for child in children:
                if not visit[child]:
                    visit[child] = True
                    value -= mobius_inversion(int_1, child, poset, tree, matrix)
            
            next_children = []
            for child in children:
                next_children += [_ for _ in tree[child][0] if poset[int_1][_]]
            
            children = list(set(next_children))
            
        matrix[int_1][int_2] = value

        return matrix[int_1][int_2]


def mobius_matrix(intervals):
    num_intervals = len(intervals)
    mu = [[None for j in range(num_intervals)] for i in range(num_intervals)]

    for i in range(num_intervals):
        mu[i][i] = 1
    
    poset = generate_poset(intervals)
    tree = generate_tree(intervals, poset)

    for i in range(num_intervals):
        for j in range(num_intervals):
            if not poset[i][j]:
                mu[i][j] = 0
            else:
                mobius_inversion(i, j, poset, tree, mu)

    return mu