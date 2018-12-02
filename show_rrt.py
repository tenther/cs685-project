#!/usr/bin/env python
import argparse
import cairo
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
import numpy as np
import os
import pdb

import rrt

def draw(da, ctx, *args):
    pf, width, height, scale = args

    ctx.set_source_rgb(1,1,1)
    ctx.rectangle(0,0,width,height)
    ctx.fill()
    ctx.set_source_rgb(0, 0, 1.0)
    ctx.set_line_width(1)
#    ctx.set_tolerance(0.1)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)

    p2p = pf.pc.scaled_point_to_pixel

    ctx.save()
    for idx, line in enumerate(pf.cross_section_2d):
#        if line.shape[0] > 1:
            line = [p2p((x[0], x[1]), scale) for x in line]
            ctx.new_path()
            ctx.move_to(line[0][0], line[0][1])
            for p in line[1:]:
                ctx.line_to(p[0], p[1])
            ctx.stroke()
    ctx.restore()
    return 

def main(directory, scale):
#    pdb.set_trace()
    pf = rrt.PathFinder(directory)
    pf.load()
    bounds = pf.get_bounds()

    win = Gtk.Window()
    win.connect('destroy', Gtk.main_quit)
    width = pf.pc.x_to_pixel(bounds.max_x - bounds.min_x)*scale
    height = pf.pc.x_to_pixel(bounds.max_y - bounds.min_y)*scale
    win.set_default_size(width, height)

    drawingarea = Gtk.DrawingArea()
    win.add(drawingarea)
    drawingarea.connect('draw', draw, pf, width, height, scale)
    win.show_all()
    Gtk.main()

if __name__ == "__main__":    
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dir', type=str, default='gibson-data/dataset/Allensville')
    parser.add_argument('-s', '--scale', type=float, default=0.1)
    parser.add_argument('-x0', type=float, default=-0.591084)
    parser.add_argument('-y0', type=float, default=7.3339)
    parser.add_argument('-x1', type=float, default=5.93709)
    parser.add_argument('-y1', type=float, default=-0.421058)
    args = parser.parse_args()
    main(args.dir, args.scale, args.x0, args.y0, args.x1, args.y1)
    
