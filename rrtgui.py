#!/usr/bin/env python
import argparse
import cairo
from collections import defaultdict
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib, Gdk
import math
import numpy as np
import os
import pdb
import sys
import threading
import traceback

import rrt

class RrtDisplay(Gtk.Window):
    def __init__(self):

        Gtk.Window.__init__(self, title="RRT Planner")
        self.set_default_size(500,500)

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL) 
        self.add(vbox)
        box = Gtk.Box(spacing=6)
        vbox.pack_start(box, False, True, 0)

        self.load_button = Gtk.Button("Load")
        self.load_button.connect("clicked", self.on_folder_clicked)
        box.pack_start(self.load_button, True, True, 0)

        self.rrt_button = Gtk.Button("Make RRT")
        self.rrt_button.connect("clicked", self.on_make_rrt_clicked)
        self.rrt_button.set_sensitive(False)
        box.pack_start(self.rrt_button, True, True, 0)

        self.find_button = Gtk.Button("Find Path")
        self.find_button.connect("clicked", self.on_find_path_clicked)
        self.find_button.set_sensitive(False)
        box.pack_start(self.find_button, True, True, 0)

        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.connect('draw', self.draw)
        self.drawing_area.set_events(
            self.drawing_area.get_events()
            | Gdk.EventMask.BUTTON_PRESS_MASK
            | Gdk.EventMask.BUTTON_RELEASE_MASK)
        
        self.connect("key-press-event", self.key_pressed)
        self.connect("key-release-event", self.key_release)
        self.drawing_area.connect('button-press-event', self.mouse_click)
        self.drawing_area.connect('button-release-event', self.mouse_release)

        vbox.pack_start(self.drawing_area, True, True, 0)

        self.status_bar = Gtk.Statusbar()
        self.context_id = self.status_bar.get_context_id("rrt")
        vbox.pack_start(self.status_bar, False, True, 0)
        self.status_bar.push(self.context_id, "Pick a Gibson folder to load")

        self.scale = 0.1
        self.epsilon = 0.1
        self.erosion_iterations = 5
        self.num_nodes = 5000
        self.pf = None
        self.solution = None
        self.cross_section_2d = None
        self.floor_map = None
        self.free = None
        self.object_file_name = None

        self.pressed = None
        self.path_start_point = None
        self.path_end_point = None
        self.path_start_px = None
        self.path_end_px = None

    def key_pressed(self, widget, event, data=None):
        key = Gdk.keyval_name(event.keyval)
        self.pressed = key
        print(key)

    def key_release(self, widget, event, data=None):
        self.pressed = None
        print("Released")        

    def mouse_click(self, widget, event):
        if self.pf and self.pf.pc:
            pixels = event.x, event.y
            map_point = self.pf.pc.pixel_to_point(
                (event.x/self.scale,
                 event.y/self.scale))
            if self.pressed == 'Shift_L':
                self.path_end_point = map_point
                self.path_end_px = pixels
                print("{}".format(self.path_end_px))
            else:
                self.path_start_point = map_point
                self.path_start_px = pixels
                print("{}".format(self.path_start_px))
            self.drawing_area.queue_draw()
            if self.path_start_point and self.path_end_point:
                self.find_button.set_sensitive(True)
                
    def mouse_release(self, widget, event):
        pass

    def load_object_file_worker(self, object_file_name):
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
        self.object_file_name = object_file_name
        self.object_file_loaded()
        return

    def object_file_loaded(self):
        self.load_button.set_sensitive(True)
        self.rrt_button.set_sensitive(True)
        self.send_status_message("Loaded {}".format(os.path.basename(self.object_file_name)))
        self.drawing_area.queue_draw()

    def load_object_file(self, obj_file_name):
        self.pf = None
        self.solution = None
        self.cross_section_2d = None
        self.floor_map = None
        self.free = None
        self.obj_file_name = None
        self.rrt_button.set_sensitive(False)
        self.drawing_area.queue_draw()
        self.rrt_button.set_sensitive(False)
        self.load_button.set_sensitive(False)

        thread = threading.Thread(target=self.load_object_file_worker, args=(obj_file_name,))
        thread.daemon = True
        thread.start()

    def on_find_path_clicked(self, widget):
        if not (self.path_start_point and self.path_end_point):
            GLib.idle_add(self.send_status_message,
                          "Both start and end points must be defined to find a path.")
            return
        solution, lines = self.pf.find(
            self.path_start_point[0],
            self.path_start_point[1],
            self.path_end_point[0],
            self.path_end_point[1])
        self.solution = solution
        self.send_redraw()
    
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
                self.load_object_file(obj_file_name)
            else:
                err_file_name = '.../' + '/'.join(obj_file_name.split('/')[-2:])
                GLib.idle_add(self.send_status_message, "File {} does not exist. Try another folder.".format(err_file_name))
        dialog.destroy()

    def make_rrt_worker(self):
        counter = 0
        def callback(nodes_x, nodes_y, edges):
            nonlocal counter
            counter += 1
            if counter%100 == 0:
                self.pf.nodes_x = nodes_x
                self.pf.nodes_y = nodes_y
                self.pf.edges_idx = np.copy(edges)
                self.send_redraw()

        self.send_status_message("Creating RRT")
        _, _, self.pf.nodes_x, self.pf.nodes_y, self.pf.edges_idx = rrt.make_rrt(
            self.pf.cross_section_2d,
            self.pf.padding_meters,
            self.pf.px_per_meter,
            self.num_nodes,
            self.epsilon,
            self.pf.free,
            new_edge_callback=callback,
            )

        edges = defaultdict(dict)
        edges_idx = self.pf.edges_idx
        nodes_x = self.pf.nodes_x
        nodes_y = self.pf.nodes_y
        for edge_idx in edges_idx:
            n0 = edge_idx[0]
            n1 = edge_idx[1]
            distance = math.sqrt((nodes_x[n0] - nodes_x[n1])**2 +
                                 (nodes_y[n0] - nodes_y[n1])**2)
            edges[n0][n1] = distance
            edges[n1][n0] = distance
        self.pf.edges = edges
        self.rrt_made()

    def rrt_made(self):
        self.rrt_button.set_sensitive(True)
        self.send_status_message("Created RRT")
        self.drawing_area.queue_draw()

    def make_rrt(self):
        self.load_button.set_sensitive(False)
        self.rrt_button.set_sensitive(False)
        thread = threading.Thread(target=self.make_rrt_worker,)
        thread.daemon = True
        thread.start()

    def on_make_rrt_clicked(self, widget):
        self.make_rrt()
        return

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
        ctx.set_line_width(5)
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

    def filled_circle(self, ctx, center, radius, color):
        ctx.set_source_rgb(*color)
        ctx.arc(
            center[0], 
            center[1], 
            radius,
            0,
            2*math.pi)
        ctx.fill()

    def draw(self, da, ctx):
        ctx.set_source_rgb(1,1,1)
        ctx.rectangle(0,0,da.get_allocated_width(), da.get_allocated_height())
        ctx.fill()

        ctx.save()
        if self.pf:
            self.draw_map(da, ctx)

            if self.pf.nodes_x is not None:
                x2px = self.pf.pc.x_to_pixel
                y2px = self.pf.pc.y_to_pixel
                self.node_x = node_x = [x2px(x)*self.scale for x in self.pf.nodes_x]
                self.node_y = node_y = [y2px(y)*self.scale for y in self.pf.nodes_y]
                self.draw_rrt(da,ctx)

                if self.path_start_px:
                    self.filled_circle(ctx, self.path_start_px, 10, (0, 1, 0))
                if self.path_end_px:
                    self.filled_circle(ctx, self.path_end_px, 10, (1, 0, 0))
                if self.solution:
                    self.draw_solution(da, ctx)
        ctx.restore()

        return 

if __name__=='__main__':
#    pdb.set_trace()
    win = RrtDisplay()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()
