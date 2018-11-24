import bresenham
from collections import defaultdict
import cv2
import math
import matplotlib.pyplot as plt
import meshcut
import numpy as np
import pdb
import random
import sys

class PointConverter(object):

    def __init__(self,
                 min_x,
                 max_x,
                 min_y,
                 max_y,
                 px_per_meter,
                 padding_meters,
                 free,
                 ):
        self.min_x          = min_x
        self.max_x          = max_x
        self.min_y          = min_y
        self.max_y          = max_y
        self.px_per_meter   = px_per_meter
        self.padding_meters = padding_meters
        self.free           = free

    def random_x(self):
        return random.random() * (self.max_x - self.min_x) + self.min_x
        
    def random_y(self):
        return random.random() * (self.max_y - self.min_y) + self.min_y

    def x_to_pixel(self, x):
        return int((x - self.min_x) * self.px_per_meter + self.px_per_meter * self.padding_meters / 2.0)

    def y_to_pixel(self, y):
        return int((y - self.min_y) * self.px_per_meter + self.px_per_meter * self.padding_meters / 2.0)

    def point_to_pixel(self, p):
        return self.x_to_pixel(p[0]), self.y_to_pixel(p[1])

    def random_point(self, ):
        x = self.random_x()
        y = self.random_y()
        return x,y

    def free_point(self, x,y):
        x_px = self.x_to_pixel(x)
        y_px = self.y_to_pixel(y)
        # check image boundaries due to rounding errors
        return (x_px >= 0 and x_px < self.free.shape[1] and
                y_px >= 0 and y_px < self.free.shape[0] and
                self.free[y_px, x_px] == 255)

    def random_free_point(self, ):
        while True:
            x,y = self.random_point()
            if self.free_point(x,y):
                return x, y

def load_obj(fn):
    verts = []
    faces = []
    with open(fn) as f:
        for line in f:
            if line[:2] == 'v ':
                verts.append(list(map(float, line.strip().split()[1:4])))
            if line[:2] == 'f ':
                face = [int(item.split('/')[0]) for item in line.strip().split()[-3:]]
                faces.append(face)
    verts = np.array(verts)
    faces = np.array(faces) - 1
    return verts, faces

def get_cross_section(verts, faces):
    z =  np.min(verts[:,-1]) + 0.5 # 0.5 is the height of husky, actually it should be 0.37
    # cut the mesh with a surface whose value on z-axis is plane_orig, and its normal is plane_normal vector
    #print('verts: {}'.format(verts[0]))
    cross_section = meshcut.cross_section(verts, faces, plane_orig=(0, 0, z), plane_normal=(0, 0, 1))
    return cross_section

def cross_section_bounds(cross_section, padding_meters):
    return (min([x[:,0].min() for x in cross_section]) - padding_meters/2.0,
            max([x[:,0].max() for x in cross_section]) + padding_meters/2.0,
            min([x[:,1].min() for x in cross_section]) - padding_meters/2.0,
            max([x[:,1].max() for x in cross_section]) + padding_meters/2.0,)

def fill_in_gaps(lines):

    lines = [line for line in lines if len(line) > 1]
    endpoints = np.array([[l[0], l[-1]]  for l in lines]).reshape((len(lines)*2, 2))
    endpoint_equals = endpoints.reshape(endpoints.shape[0],1,2) == endpoints.reshape(1,endpoints.shape[0],2)
    #points_with_matches = dict.fromkeys([x for x in np.argwhere(np.sum((np.sum(endpoint_equals, axis=2)==2), axis=1)==2)])

    endpoint_distances = np.sqrt(np.sum(np.power((endpoints.reshape(endpoints.shape[0],1,2) - endpoints.reshape(1,endpoints.shape[0],2)), 2), axis=2))
    #all_sorted_endpoint_indexes = np.dstack(np.unravel_index(np.argsort(endpoint_distances.ravel()), endpoint_distances.shape))
    sorted_endpoint_indexes = np.argsort(endpoint_distances)

    connections = {}
    fix_up_lines = []

    for i in range(len(lines)):
        for j in [i * 2, i * 2 + 1]:
            if j not in connections:
                for p in sorted_endpoint_indexes[j]:
                    if p != j:
                        connections[j] = p
                        connections[p] = j
                        l0 = j//2
                        idx0 = 0
                        if j%2 == 1:
                            idx0 = -1
                        l1 = p//2
                        idx1 = 0
                        if p%2 == 1:
                            idx1 = -1
                        fix_up_lines.append(np.array([lines[l0][idx0], lines[l1][idx1]]))
                        break
    return fix_up_lines

def make_free_space_image(cross_section_2d, px_per_meter, padding_meters):
    min_x, max_x, min_y, max_y = cross_section_bounds(cross_section_2d, padding_meters)

    image_height = int(np.ceil(px_per_meter * (max_x - min_x) + px_per_meter * padding_meters))
    image_width  = int(np.ceil(px_per_meter * (max_y - min_y) + px_per_meter * padding_meters))

    x_adj = -min_x * px_per_meter + px_per_meter * padding_meters / 2.0
    y_adj = -min_y * px_per_meter + px_per_meter * padding_meters / 2.0

    lines = [np.round(x * px_per_meter + np.array([x_adj,y_adj]),0).astype(np.int32) for x in cross_section_2d]

    image = np.zeros((image_width, image_height, 3), np.uint8)
    cv2.polylines(image, lines, False, (255, 255, 255), thickness=3)

    # Fill in gaps for Allensville
    # fix_up_lines = [
    #     np.array([lines[0][-1], lines[2][-1]]),
    #     np.array([lines[2][0], lines[4][-1]]),
    #     np.array([lines[20][0], lines[20][-1]]),
    #     np.array([lines[4][0], lines[6][-1]]),
    #     np.array([lines[6][0], lines[7][-1]]),
    #     np.array([lines[5][0], lines[7][0]]),
    #     np.array([lines[8][0], lines[8][-1]]),
    #     np.array([lines[16][0], lines[16][-1]]),
    #     np.array([lines[10][0], lines[10][-1]]),
    #     np.array([lines[19][0], lines[19][-1]]),
    #     ]
    cv2.imwrite('before_fill_in_gaps.png', image)
    cv2.polylines(image, fill_in_gaps(lines), False, (255, 255, 255), thickness=3)
    cv2.imwrite('after_fill_in_gaps.png', image)

    if False:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2

        for i, line in enumerate(lines):
            if len(line) > 1:
                for end in [0,-1]:
                    cv2.putText(
                        image,
                        "{}.{}".format(i,end), 
                        tuple(line[end]),
                        font, 
                        fontScale,
                        fontColor,
                        lineType)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = np.zeros((image_width + 2, image_height + 2), np.uint8)
    cv2.floodFill(gray, mask, (2000,2000), 255)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(gray,kernel,iterations = 30)
    return image, erosion

def line_check(p0, p1, free):
    for p in bresenham.line_points(p0, p1):
        if free[p[1], p[0]] != 255:
            return False
    return True

def make_rrt(cross_section, padding_meters, px_per_meter, num_nodes, epsilon, free):
#    pdb.set_trace()
    min_x, max_x, min_y, max_y = cross_section_bounds(cross_section, padding_meters)

    pc = PointConverter(min_x, max_x, min_y, max_y, px_per_meter, padding_meters, free)

    starting_point = pc.random_free_point()
    nodes_x = np.zeros(num_nodes, np.float32)
    nodes_y = np.zeros(num_nodes, np.float32)
    nodes_x[0] = starting_point[0]
    nodes_y[0] = starting_point[1]
    edges = np.full((num_nodes-1, 2), -1)
    edges_from_px = []
    edges_to_px = []

    for i in range(1, num_nodes):
        if i%1000 == 0:
            print("{}/{}".format(i, num_nodes))
        while True:
            next_node = pc.random_point()
            distances = np.sqrt(np.power(nodes_x[:i] - next_node[0], 2) + np.power(nodes_y[:i] - next_node[1], 2))
            closest_point_idx = np.argmin(distances)
            theta = math.atan2(next_node[1] - nodes_y[closest_point_idx], next_node[0] - nodes_x[closest_point_idx])
            node = (nodes_x[closest_point_idx] + np.cos(theta) * epsilon,
                    nodes_y[closest_point_idx] + np.sin(theta) * epsilon)

            p0 = pc.point_to_pixel((nodes_x[closest_point_idx], nodes_y[closest_point_idx]))
            p1 = pc.point_to_pixel(node)

            if pc.free_point(*node) and line_check(p0, p1, free):
                nodes_x[i] = node[0]
                nodes_y[i] = node[1]
                edges[i-1][0] = closest_point_idx
                edges[i-1][1] = i
                edges_from_px.append(p0)
                edges_to_px.append(p1)
                break

    # Connect leaf nodes to closest points
    leaf_index = np.setdiff1d(edges[:,1], edges[:,0])
    leaf_nodes = np.stack((nodes_x[leaf_index], nodes_y[leaf_index]), axis=1)
    distances = np.sqrt(np.sum(np.power((leaf_nodes.reshape(leaf_nodes.shape[0],1,2) -
                                         leaf_nodes.reshape(1,leaf_nodes.shape[0],2)), 2), axis=2))
    #all_sorted_endpoint_indexes = np.dstack(np.unravel_index(np.argsort(endpoint_distances.ravel()), endpoint_distances.shape))
    sorted_endpoint_indexes = np.argsort(distances)
    
    connect_leaves = False
    if connect_leaves:
        new_edges = defaultdict(set)
        for i in range(len(leaf_nodes)):
            node0_idx = leaf_index[i]
            p0 = pc.point_to_pixel((nodes_x[node0_idx], nodes_y[node0_idx]))
            for j in list(sorted_endpoint_indexes[i,1:]):
                node1_idx = leaf_index[j]
                if node1_idx not in new_edges[node0_idx]:
                    p1 = pc.point_to_pixel((nodes_x[node1_idx], nodes_y[node1_idx]))
                    if line_check(p0, p1, free):
                        new_edges[node0_idx].add(node1_idx)
                        new_edges[node1_idx].add(node0_idx)
                        edges_from_px.append(pc.point_to_pixel((nodes_x[node0_idx], nodes_y[node0_idx])))
                        edges_to_px.append(pc.point_to_pixel((nodes_x[node1_idx], nodes_y[node1_idx])))
                        break
        new_edge_array = np.array([[i, j] for i in new_edges for j in new_edges[i]])
        edges = np.vstack((edges, new_edge_array))
    
    return edges_from_px, edges_to_px, nodes_x, nodes_y, edges
