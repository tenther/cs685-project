#!/usr/bin/env python
import argparse
from collections import defaultdict, namedtuple
import cv2
import heapq
import math
import numpy as np
import os
import pdb
import pickle
import random
import rrt

#HeapNode = namedtuple('HeapNode', ['estimated', 'cost', 'state', 'path', 'deleted'])

class HeapNode(object):
    def __init__(self, estimated, cost, state, path):
        self.estimated = estimated
        self.cost     = cost
        self.state    = state
        self.path     = path
        self.deleted  = False

    def __lt__(self, other):
        self.estimated < other.estimated

class AStarHeap(object):
    def __init__(self):
        self.heap = []
        self.state_nodes = {}
        self.size = 0

    def heappush(self, node):
        heapq.heappush(self.heap, node)
        self.state_nodes[node.state] = node
        self.size += 1
    
    def exists_worse(self, state, estimated):
        if state in self.state_nodes and self.state_nodes[state].estimated > estimated:
            return True
        return False

    def exists(self, state):
        return state in self.state_nodes

    def replace(self, new_node):
        self.state_nodes[new_node.state].deleted = True
        del(self.state_nodes[new_node.state])
        self.size -= 1
        self.heappush(new_node)

    def heappop(self):
        while self.heap[0].deleted:
            heapq.heappop(self.heap)
        self.size -= 1
        node = heapq.heappop(self.heap)
        del(self.state_nodes[node.state])
        return node

    def getsize(self):
        return self.size

def A_star_search(start_node, goal_node, nodes_x, nodes_y, edges_idx):

    def node_distance(n0, n1):
        return math.sqrt((nodes_x[n0] - nodes_x[n1])**2 + (nodes_y[n0] - nodes_y[n1])**2)

    goal_distances = np.sqrt(np.power(nodes_x - nodes_x[goal_node], 2) + np.power(nodes_y - nodes_y[goal_node], 2))
    edges = defaultdict(set)
    for i in range(len(edges_idx)):
        edges[edges_idx[i][0]].add(edges_idx[i][1])
        edges[edges_idx[i][1]].add(edges_idx[i][0])

    frontier = AStarHeap()
    frontier.heappush(HeapNode(goal_distances[start_node],0,start_node, [start_node]))
    explored = set()

    while True:
        if not frontier.getsize():
            return None
        node = frontier.heappop()
        if node.state == goal_node:
            return node
        explored.add(node.state)
        for child in edges[node.state]:
            newnode = HeapNode(node.cost + goal_distances[child], node.cost + node_distance(node.state, child), child, node.path + [child])
            if (not (newnode.state in explored or frontier.exists(newnode.state))):
                frontier.heappush(newnode)
            elif frontier.exists_worse(newnode.state, newnode.estimated):
                frontier.replace(newnode)
    return None


class PathFinder(object):
    def __init__(self, directory):
        self.dir            = directory
        self.nodes_x        = None
        self.nodes_y        = None
        self.edges_idx      = None
        self.free           = None
        self.config         = None
        self.min_x          = None
        self.max_x          = None
        self.min_y          = None
        self.max_y          = None
        self.px_per_meter   = None
        self.padding_meters = None
        self.pc             = None

    def load(self):
        rrt_file_name = os.path.join(self.dir, 'rrt.npz')
        config_file_name = os.path.join(self.dir, 'config.pickle')

        npfile         = np.load(rrt_file_name)
        self.nodes_x   = npfile['arr_0']
        self.nodes_y   = npfile['arr_1']
        self.edges_idx = npfile['arr_2']
        self.free      = npfile['arr_3']

        with open(config_file_name, 'rb') as config_file:
            self.config = pickle.load(config_file)
        self.min_x = self.config['min_x']
        self.max_x = self.config['max_x']
        self.min_y = self.config['min_y']
        self.max_y = self.config['max_y']
        self.px_per_meter = self.config['px_per_meter']
        self.padding_meters = self.config['padding_meters']
        self.pc = rrt.PointConverter(self.min_x, self.max_x, self.min_y, self.max_y, self.px_per_meter, self.padding_meters, self.free)

    def find(self, x0, y0, x1, y1):
        pc = self.pc
        nodes_x = self.nodes_x
        nodes_y = self.nodes_y

        if not pc.free_point(x0, y0):
            raise("starting point ({},{}) is not free".format(x0, y0))

        if not pc.free_point(x1, y1):
            raise("starting point ({},{}) is not free".format(x1, y1))

        start_node = self.node_closest_to_point(x0, y0)
        goal_node  = self.node_closest_to_point(x1, y1)
        solution = A_star_search(start_node, goal_node, nodes_x, nodes_y, self.edges_idx)

        lines =  [np.array([pc.y_to_pixel(y0), pc.x_to_pixel(x0)])]
        lines += [np.array([pc.y_to_pixel([nodes_y[node]]), pc.x_to_pixel(nodes_x[node])]) for node in solution.path]
        lines += [np.array([pc.y_to_pixel(y1), pc.x_to_pixel(x1)])]

        return solution, lines

    def node_closest_to_point(self, x, y):
        distances = np.argsort(np.sqrt(np.power(self.nodes_x - x, 2) + np.power(self.nodes_y - y, 2)))
        node = None
        for i in range(distances.shape[0]):
            j = distances[i]
            if rrt.line_check(self.pc.point_to_pixel((self.nodes_x[j], self.nodes_y[j])), self.pc.point_to_pixel((x, y)), self.free):
                node = j
                break
        if not node:
            raise Exception("Could not find a clear path from ({},{}) to a node".format(x,y))
        return node

def main(directory, x0, y0, x1, y1):

    path_finder = PathFinder(directory)
    print("Loading...")
    path_finder.load()
    pc = path_finder.pc
    print("Finding...")
    solution, lines = path_finder.find(x0, y0, x1, y1)
    
    if solution:
        print("Writing solution...")
        floormap_file_name = os.path.join(directory, 'floormap.png')
        floormap  = cv2.imread(floormap_file_name)

        floormap_with_path_file_name = os.path.join(directory, 'floormap_with_path.png')
        for i in range(len(lines) - 1):
            cv2.line(floormap, (lines[i][1], lines[i][0]), (lines[i+1][1], lines[i+1][0]), (0,255,0), 5)

        cv2.circle(floormap, tuple([lines[0][1], lines[0][0]]),  50, (0, 255, 0), thickness=10)
        cv2.circle(floormap, tuple([lines[-1][1], lines[-1][0]]),50, (255, 0, 0), thickness=10)
        cv2.imwrite(floormap_with_path_file_name, floormap)
    else:
        print("No path found")

if __name__ == "__main__":    
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dir', type=str, default='gibson-data/dataset/Allensville')
    parser.add_argument('-x0', type=float, default=-0.591084)
    parser.add_argument('-y0', type=float, default=7.3339)
    parser.add_argument('-x1', type=float, default=5.93709)
    parser.add_argument('-y1', type=float, default=-0.421058)
    args = parser.parse_args()
    main(args.dir, args.x0, args.y0, args.x1, args.y1)
    
