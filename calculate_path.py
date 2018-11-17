#!/usr/bin/env python
import argparse
import rrt
import cv2
import numpy as np
import os
import pdb
import pickle
import random

def node_closest_to_point(pc, nodes_x, nodes_y, x, y, free):
    distances = np.argsort(np.sqrt(np.power(nodes_x - x, 2) + np.power(nodes_y - y, 2)))
    node = None
    for i in range(distances.shape[0]):
        j = distances[i]
        if rrt.line_check(pc.point_to_pixel((nodes_x[j], nodes_y[j])), pc.point_to_pixel((x, y)), free):
            node = j
            break
    if not node:
        raise Exception("Could not find a clear path from ({},{}) to a node".format(x,y))
    return node

def main(dir, x0, y0, x1, y1):
    floormap_file_name = os.path.join(dir, 'floormap.png')
    rrt_file_name = os.path.join(dir, 'rrt.npz')
    config_file_name = os.path.join(dir, 'config.pickle')

    floormap  = cv2.imread(floormap_file_name)
    npfile    = np.load(rrt_file_name)
    nodes_x   = npfile['arr_0']
    nodes_y   = npfile['arr_1']
    edges_idx = npfile['arr_2']
    free      = npfile['arr_3']

    with open(config_file_name, 'rb') as config_file:
        config    = pickle.load(config_file)
    min_x = config['min_x']
    max_x = config['max_x']
    min_y = config['min_y']
    max_y = config['max_y']
    px_per_meter = config['px_per_meter']
    padding_meters = config['padding_meters']

    pc = rrt.PointConverter(min_x, max_x, min_y, max_y, px_per_meter, padding_meters, free)

    if not pc.free_point(x0, y0):
        raise("starting point ({},{}) is not free".format(x0, y0))

    if not pc.free_point(x1, y1):
        raise("starting point ({},{}) is not free".format(x1, y1))

    start_node = node_closest_to_point(pc, nodes_x, nodes_y, x0, y0, free)
    end_node   = node_closest_to_point(pc, nodes_x, nodes_y, x1, y1, free)
    print("start_node={}, end_node={}".format(start_node, end_node))
    return


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
    
