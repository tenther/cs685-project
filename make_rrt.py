#!/usr/bin/env python
import rrt
import cv2
import numpy as np
import os
import pickle

def main(mesh_file_name, px_per_meter, padding_meters, num_nodes, epsilon):
    verts, faces = rrt.load_obj(mesh_file_name)

    cross_section = rrt.get_cross_section(verts, faces)

    cross_section_2d = [c[:,0:2] for c in cross_section]

    floor_map, free = rrt.make_free_space_image(cross_section_2d, px_per_meter, padding_meters)

    edges_from_px, edges_to_px, nodes_x, nodes_y, edges = rrt.make_rrt(cross_section_2d, padding_meters, px_per_meter, num_nodes, epsilon, free)

    for i in range(len(edges_from_px)):
        cv2.line(floor_map, edges_from_px[i], edges_to_px[i], (0, 0, 255), thickness=5)

    min_x, max_x, min_y, max_y = rrt.cross_section_bounds(cross_section_2d, padding_meters)

    mesh_dir = os.path.dirname(mesh_file_name)
    free_file_name = os.path.join(mesh_dir, 'free.png')
    floormap_file_name = os.path.join(mesh_dir, 'floormap.png')
    rrt_file_name = os.path.join(mesh_dir, 'rrt.npz')
    config_file_name = os.path.join(mesh_dir, 'config.pickle')

    cv2.imwrite(floormap_file_name, floor_map)
    print("Wrote {}".format(floormap_file_name))
    np.savez(rrt_file_name, nodes_x, nodes_y, edges, free)
    print("Wrote {}".format(rrt_file_name))
    with open(config_file_name, 'wb') as config_file:
        pickle.dump({
            'px_per_meter': px_per_meter,
            'padding_meters': padding_meters,
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y,
            }, config_file)
    print("Wrote {}".format(config_file_name))
    cv2.imwrite(free_file_name, free)
    print("Wrote {}".format(free_file_name))

if __name__ == "__main__":    
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mesh_name', type=str, default='gibson-data/dataset/Allensville/mesh_z_up.obj')
    parser.add_argument('-n', '--num_nodes', type=int, default=5000)
    parser.add_argument('-e', '--epsilon', type=float, default=.1)
    args = parser.parse_args()
    main(args.mesh_name, 500, 2.0, args.num_nodes, args.epsilon)
