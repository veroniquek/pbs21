
import numpy as np

import platform
import os
import bpy

# vertices of tetraeders
elements = []

# make mesh
vertices = []
edges = []
faces = []

filepath = os.getcwd()

#if (platform.system() == 'Darwin') or platform.system() == "Linux":
filepath = filepath + '/tetgen'
    # tODO add paths
# filepath = "C:/Users/Chris/Google Drive/z_eth/Master/Semester11/Physically-Based Simulation in Computer Graphics/Aufgaben/Project/my_code/"
filename = "ellell.1"


def get_vertices_edges_faces_elements():
    return np.array(vertices), np.array(edges), np.array(faces), elements


def create_mesh(filename="ellell.1"):
    with open(filepath + "/" + filename + ".node", 'r+') as f:
        lines = f.readlines()
        for line in lines[1:-1]:
            li = line.split(" ")

            li1 = []
            for i in li:
                if i != "":
                    li1.append(i)
            li = li1

            vertices.append((float(li[1]), float(li[2]), float(li[3])))

    with open(filepath + "/" + filename + ".ele", 'r+') as f:
        lines = f.readlines()
        print(len(lines[1:-1]))
        for line in lines[1:-1]:
            li = line.split(" ")
            li1 = []
            for i in li:
                if i != "":
                    li1.append(i)
            li = li1

            edges.append((int(li[1]), int(li[3])))
            edges.append((int(li[2]), int(li[3])))
            
            elements.append(list(map(int, li[1:])))

    with open(filepath + "/" + filename + ".face", 'r+') as f:
        lines = f.readlines()
        for line in lines[1:-1]:
            li = line.split(" ")
            li1 = []
            for i in li:
                if i != "":
                    li1.append(i)
            li = li1

            faces.append((int(li[1]), int(li[2]), int(li[3])))



    ## Delete the data from the previous iteration
    collection = bpy.data.collections.get('sim_collection')

    if not (collection is None):
        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        bpy.data.collections.remove(collection)

    new_mesh = bpy.data.meshes.new(filename + '_mesh')
    new_mesh.from_pydata(vertices, edges, faces)
    new_mesh.update()

    # make object from mesh
    new_object = bpy.data.objects.new(filename + '_object', new_mesh)

    # make collection
    new_collection = bpy.data.collections.new('sim_collection')
    bpy.context.scene.collection.children.link(new_collection)

    # add object to scene collection
    new_collection.objects.link(new_object)
    return new_mesh
