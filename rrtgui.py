#!/usr/bin/env python
import argparse
import cairo
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib
import numpy as np
import os
import pdb
import threading

import rrt

class RrtDisplay(Gtk.Window):
    def __init__(self):

        Gtk.Window.__init__(self, title="RRT Planner")
        self.set_default_size(500,500)

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL) 
        self.add(vbox)
        box = Gtk.Box(spacing=6)
        vbox.pack_start(box, False, True, 0)

        button1 = Gtk.Button("Load")
        button1.connect("clicked", self.on_folder_clicked)
        box.pack_start(button1, True, True, 0)

        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.connect('draw', self.draw)
        vbox.pack_start(self.drawing_area, True, True, 0)

        self.status_bar = Gtk.Statusbar()
        self.context_id = self.status_bar.get_context_id("rrt")
        vbox.pack_start(self.status_bar, False, True, 0)
        self.status_bar.push(self.context_id, "Pick a Gibson folder to load")

        self.scale = 0.1
        self.pf = None
        self.solution = None
        self.cross_section_2d = None
        self.floor_map = None
        self.free = None

    def load_object_file(self, object_file_name):
        px_per_meter = 500
        padding_meters = 0.5
        erosion_iterations = 5

        self.send_status_message("Loading object file")
        verts, faces = rrt.load_obj(object_file_name)
        self.send_status_message("Finding 2D cross-section")
        cross_section = rrt.get_cross_section(verts, faces)
        cross_section_2d = [c[:, 0:2] for c in cross_section]

        self.send_status_message("Creating free mask")
        _, free = rrt.make_free_space_image(
            cross_section_2d,
            px_per_meter,
            padding_meters,
            erosion_iterations=erosion_iterations)

        min_x, max_x, min_y, max_y = rrt.cross_section_bounds(cross_section_2d, padding_meters)

        self.pf = rrt.PathFinder(
            os.path.dirname(object_file_name),
            free=free,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            px_per_meter=px_per_meter,
            padding_meters=padding_meters,
            pc=rrt.PointConverter(
                (min_x, max_x, min_y, max_y),
                px_per_meter,
                padding_meters, free),
            cross_section_2d=cross_section_2d,
            )
        self.send_status_message("Loaded {}".format(os.path.basename(object_file_name)))
        self.drawing_area.queue_draw()
        return
        #GLib.idle_add(self.send_redraw)

    def on_folder_clicked(self, widget):
        dialog = Gtk.FileChooserDialog("Choose a folder", self,
            Gtk.FileChooserAction.SELECT_FOLDER,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
             "Select", Gtk.ResponseType.OK))
        dialog.set_default_size(800, 400)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            folder = dialog.get_filename()
            obj_file_name = os.path.join(folder, 'mesh_z_up.obj')
            if os.path.exists(obj_file_name):
                thread = threading.Thread(target=self.load_object_file, args=(obj_file_name,))
                thread.daemon = True
                thread.start()
            else:
                err_file_name = '.../' + '/'.join(obj_file_name.split('/')[-2:])
                GLib.idle_add(self.send_status_message, "File {} does not exist. Try another folder.".format(err_file_name))
        dialog.destroy()

    def send_status_message(self, message):
        self.status_bar.push(self.context_id, message)

    def send_redraw(self):
        self.drawing_area.queue_draw()
        
    def draw_map(self, da, ctx):
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(1)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)

        p2p = self.pf.pc.scaled_point_to_pixel

        for idx, line in enumerate(self.pf.cross_section_2d):
            line = [p2p((x[0], x[1]), self.scale) for x in line]
            ctx.new_path()
            ctx.move_to(line[0][0], line[0][1])
            for p in line[1:]:
                ctx.line_to(p[0], p[1])
            ctx.stroke()

    def draw_rrt(self, da, ctx):
        node_x = self.node_x
        node_y = self.node_y
        ctx.set_source_rgb(0, 0, 1.0)
        for src, dst in self.pf.edges_idx:
            ctx.new_path()
            ctx.move_to(node_x[src], node_y[src])
            ctx.line_to(node_x[dst], node_y[dst])
            ctx.stroke()

    def draw_solution(self, da, ctx):
        ctx.set_source_rgb(1.0, 0, 0)
        px_path = []
        node_x = self.node_x
        node_y = self.node_y
        for node in self.solution.path:
            p = (node_x[node], node_y[node])
            px_path.append(p)
        ctx.new_path()
        ctx.move_to(px_path[0][0], px_path[0][1])
        for p in px_path[1:]:
            ctx.line_to(p[0], p[1])
        ctx.stroke()

    def draw(self, da, ctx):
        ctx.set_source_rgb(1,1,1)
        ctx.rectangle(0,0,da.get_allocated_width(), da.get_allocated_height())
        ctx.fill()

        ctx.save()
        if self.pf:
            self.draw_map(da, ctx)

            if self.pf.nodes_x:
                x2px = self.pf.pc.x_to_pixel
                y2px = self.pf.pc.y_to_pixel
                self.node_x = node_x = [x2px(x)*self.scale for x in self.pf.nodes_x]
                self.node_y = node_y = [y2px(y)*self.scale for y in self.pf.nodes_y]
                self.draw_rrt(da,ctx)

                if self.solution:
                    self.draw_solution(da, ctx)
        ctx.restore()

        return 

if __name__=='__main__':
    win = RrtDisplay()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()
