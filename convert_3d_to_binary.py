#!/usr/bin/env python
import cv2
import matplotlib.pyplot as plt
import meshcut
import numpy as np
import pdb
import sys

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

def main(mesh_file_name):
    verts, faces = load_obj(mesh_file_name)
    z =  np.min(verts[:,-1]) + 0.5 # 0.5 is the height of husky, actually it should be 0.37
    # cut the mesh with a surface whose value on z-axis is plane_orig, and its normal is plane_normal vector
    #print('verts: {}'.format(verts[0]))
    cross_section = meshcut.cross_section(verts, faces, plane_orig=(0, 0, z), plane_normal=(0, 0, 1))

    cross_section_2d = [c[:,0:2] for c in cross_section]
    min_x = min([x[:,0].min() for x in cross_section_2d])
    max_x = max([x[:,0].max() for x in cross_section_2d])

    min_y = min([x[:,1].min() for x in cross_section_2d])
    max_y = max([x[:,1].max() for x in cross_section_2d])

    px_per_meter = 500

    image_height = int(np.ceil(px_per_meter * (max_x - min_x) + px_per_meter))
    image_width  = int(np.ceil(px_per_meter * (max_y - min_y) + px_per_meter))

    x_adj = -min_x * px_per_meter + px_per_meter/2
    y_adj = -min_y * px_per_meter + px_per_meter/2

    lines = [np.round(x * px_per_meter + np.array([x_adj,y_adj]),0).astype(np.int32) for x in cross_section_2d]
    

    image = np.zeros((image_width, image_height, 3), np.uint8)
    cv2.polylines(image, lines, False, (255, 255, 255), thickness=3)

    # Fill in gaps
    fix_up_lines = [
        np.array([lines[0][-1], lines[2][-1]]),
        np.array([lines[2][0], lines[4][-1]]),
        np.array([lines[20][0], lines[20][-1]]),
        np.array([lines[4][0], lines[6][-1]]),
        np.array([lines[6][0], lines[7][-1]]),
        np.array([lines[5][0], lines[7][0]]),
        np.array([lines[8][0],  lines[8][-1]]),
        np.array([lines[16][0],  lines[16][-1]]),
        np.array([lines[10][0],  lines[10][-1]]),
        np.array([lines[19][0], lines[19][-1]]),
        ]
    cv2.polylines(image, fix_up_lines, False, (255, 255, 255), thickness=3)

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

    cv2.imwrite('output.png',gray)

    

#    cv2.imwrite('output.png',image)

if __name__ == "__main__":    
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mesh_name', type=str, default='gibson-data/dataset/Allensville/mesh_z_up.obj')
    args = parser.parse_args()
    main(args.mesh_name)

