import numblend as nb
import taichi as ti
ti.init()

import numpy as np
import os
import bpy

import sys
import platform 
from importlib import reload

dir = os.getcwd()


dir1 = dir + '/src'
dir2 = dir + '/tetgen'
print("Running script:", __file__)
print("Running in:", dir)


if not dir1 in sys.path:
    sys.path.append(dir1)

if not dir2 in sys.path:
    sys.path.append(dir1)

import taichi_mesh
import discretization

# update algorithm_code in case it has been changed while blender was already running
reload(taichi_mesh)
reload(discretization)
import taichi_mesh
import discretization

nb.init()

def helper_objects(ground_height):
    collection = bpy.data.collections.get('help_collection')

    if not (collection is None):
        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        bpy.data.collections.remove(collection)

    new_mesh = bpy.data.meshes.new('ground_plane')
    x = [[10,10,ground_height], [-10,10,ground_height], [-10,-10,ground_height], [10,-10,ground_height]]
    e = [[0, 1], [1, 2], [2, 3], [3, 0]]
    f = [[0, 1, 2, 3]]

    new_mesh.from_pydata(x, e, f)
    new_mesh.update()

    # make object from mesh
    new_object = bpy.data.objects.new('ground_plane_object', new_mesh)

    # make collection
    new_collection = bpy.data.collections.new('help_collection')
    bpy.context.scene.collection.children.link(new_collection)

    # add object to scene collection
    new_collection.objects.link(new_object)



##############################################################################

# Change demo simulations here to reproduce them

FEPR = True
DEMO = 1

##############################################################################

@nb.add_animation
def main():
    ground_height = -5

    if DEMO == 0:
        meshObj = taichi_mesh.MeshObject(FEPR=FEPR, dt=1/30, damping=0.999, ground_height=ground_height, iterations_per_frame=1, filename="ellell.1", G=9.81, node_mass=5, E=1e3, nu=0.3, demo=0)

    elif DEMO == 1:
        # Fluent with FEPR at lower frame rate, explodes without FEPR
        meshObj = taichi_mesh.MeshObject(FEPR=FEPR, dt=1/20, damping=0.999, ground_height=ground_height, iterations_per_frame=1, filename="ellell.1", G=9.81, node_mass=5, E=1e3, nu=0.3, demo=1)

    elif DEMO == 2:
        if FEPR:
            # Timestep 1/30 works fine with FEPR
            meshObj = taichi_mesh.MeshObject(FEPR=True, dt=1/30, damping=0.999, ground_height=ground_height, iterations_per_frame=1, filename="test_mesh.1", G=9.81, node_mass=5, E=1e3, nu=0.3, demo=2)
        else:
            # Timestep 1/45 required otherwise the simulation explodes
            meshObj = taichi_mesh.MeshObject(FEPR=False, dt=1/45, damping=0.999, ground_height=ground_height, iterations_per_frame=1, filename="test_mesh.1", G=9.81, node_mass=5, E=1e3, nu=0.3, demo=2)

    elif DEMO == 3:
        # Fluent with FEPR at lower frame rate, explodes without FEPR
        meshObj = taichi_mesh.MeshObject(FEPR=FEPR, dt=1/30, damping=0.999, ground_height=ground_height, iterations_per_frame=1, filename="ellell.1", G=9.81, node_mass=5, E=1e3, nu=0.3, demo=3)

    # Does not work due to taichi limitations
    elif DEMO == 4:
        meshObj = taichi_mesh.MeshObject(FEPR=FEPR, dt=1/30, damping=0.999, ground_height=ground_height, iterations_per_frame=1, filename="bunny2.1", G=9.81, node_mass=5, E=1e3, nu=0.3, demo=1)



    helper_objects(ground_height)
       
    while True: 
        # step() should update vertices, edges, and faces
        meshObj.update_step()
        
        yield nb.mesh_update(meshObj.mesh, meshObj.x.to_numpy())

     