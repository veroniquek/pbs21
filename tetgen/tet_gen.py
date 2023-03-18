import pyvista as pv
import tetgen
import numpy as np

def main_sphere():
    pv.set_plot_theme('document')

    sphere = pv.Sphere()
    tet = tetgen.TetGen(sphere)
    nodes, elem = tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    grid = tet.grid
    grid.plot(show_edges=True)

    # get cell centroids
    cells = grid.cells.reshape(-1, 5)[:, 1:]
    cell_center = grid.points[cells].mean(1)

    # extract cells below the 0 xy plane
    mask = cell_center[:, 2] < 0
    cell_ind = mask.nonzero()[0]
    subgrid = grid.extract_cells(cell_ind)

    # advanced plotting
    plotter = pv.Plotter()
    plotter.add_mesh(subgrid, 'lightgrey', lighting=True, show_edges=True)
    plotter.add_mesh(sphere, 'r', 'wireframe')
    plotter.add_legend([[' Input Mesh ', 'r'],
                        [' Tessellated Mesh ', 'black']])
    plotter.show()


    cell_qual = subgrid.compute_cell_quality()['CellQuality']
    subgrid.plot(scalars=cell_qual, stitle='Quality', cmap='bwr', clim=[0, 1], flip_scalars=True, show_edges=True)
    
def main_mesh(filename, savename):
    pv.set_plot_theme('document')

    mesh = pv.read(filename)
    tet = tetgen.TetGen(mesh)
    nodes, elem = tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    grid = tet.grid
    print(nodes, elem)
    
    with open( savename + ".vertex_pos", 'w+' ) as f:
    #for i in range(0, (len(node_index) + len(face_index) + 2)):
        f.write( f"{len(nodes)}\n")
        for node in nodes:
            f.write( f"{node[0]} {node[1]} {node[2]}\n")
    
    with open( savename + ".elements", 'w+' ) as f:        
        f.write( f"{len(elem)}\n")
        for e in elem:
            f.write( f"3 {e[0]} {e[1]} {e[2]}\n")
            f.write( f"3 {e[1]} {e[2]} {e[3]}\n")
            f.write( f"3 {e[2]} {e[3]} {e[0]}\n")

    
    
    
    grid.plot(show_edges=True)

    # get cell centroids
    cells = grid.cells.reshape(-1, 5)[:, 1:]
    cell_center = grid.points[cells].mean(1)

    # extract cells below the 0 xy plane
    mask = cell_center[:, 2] < 0
    cell_ind = mask.nonzero()[0]
    subgrid = grid.extract_cells(cell_ind)

    # advanced plotting
    plotter = pv.Plotter()
    plotter.add_mesh(subgrid, 'lightgrey', lighting=True, show_edges=True)
    plotter.add_mesh(mesh, 'r', 'wireframe')
    plotter.add_legend([[' Input Mesh ', 'r'],
                        [' Tessellated Mesh ', 'black']])
    plotter.show()


    cell_qual = subgrid.compute_cell_quality()['CellQuality']
    subgrid.plot(scalars=cell_qual, stitle='Quality', cmap='bwr', clim=[0, 1], flip_scalars=True, show_edges=True)





if __name__ == "__main__":
    # for options see https://tetgen.pyvista.org/api.html


    #main_sphere()
    main_mesh('manifold_bunny.obj', 'tetra_bunny.off')