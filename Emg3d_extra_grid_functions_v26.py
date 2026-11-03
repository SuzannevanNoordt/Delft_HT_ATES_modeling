#!/usr/bin/env python
# coding: utf-8

# In[9]:


import emg3d
import empymod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

plt.style.use('bmh')

# Comment this out on Google Colab
#%matplotlib widget

#define interpolation functions
def find_distance2d(coord1, coord2):
    distance = (coord1[0]-coord2[0])**2+(coord1[1]-coord2[1])**2
    return distance
    #This is the distance squared but it should work the same

def interpolate2d(location, array):
    dis = []
    for i in np.arange(len(array)):
        dis.append(find_distance2d(location, array[i]))
    return np.argmin(dis)

def round_up(number):
    sign = number/np.abs(number)
    rounded_down = int(number)
    decimals = np.abs(number - rounded_down)
    if decimals > 0.49 and decimals < 0.51:
        rounded_up = rounded_down + 1*sign
    else:
        rounded_up = np.round(number)
    return int(rounded_up)

def point2index_xyz(grid, point, QC=False, Extra_QC=False):
    #Assumes that 

    x_spacing = min(grid.h[0])
    y_spacing = min(grid.h[1])
    z_spacing = min(grid.h[2])

    grid_center_i = int(np.ceil((grid.n_cells+grid.shape_cells[0]*grid.shape_cells[1]+ grid.shape_cells[0])/2))
    grid_center =  grid.cell_centers[grid_center_i]

    if Extra_QC ==True:
        print('x-spacing:', x_spacing)
        print('y-spacing:', y_spacing)
        print('z-spacing:', z_spacing)
        print('grid center: ', grid_center)

    ix_steps = round_up((point[0]-grid_center[0])/x_spacing)
    iy_steps = round_up((point[1]-grid_center[1])/y_spacing)
    iz_steps = round_up((point[2]-grid_center[2])/z_spacing)

    if Extra_QC ==True:
        print('x-steps:', ix_steps)
        print('y-steps:', iy_steps)
        print('z-steps:', iz_steps)

    ix = int(np.ceil(grid.shape_cells[0]/2))+ix_steps
    iy = int(np.ceil(grid.shape_cells[1]/2))+iy_steps
    iz = int(np.ceil(grid.shape_cells[2]/2))+iz_steps

    if Extra_QC ==True:
        print('ix:', ix)
        print('iy:', iy)
        print('iz:', iz)

    if grid.h[0][int(grid.shape_cells[0]/2)] != x_spacing:
        print('Warning: no homogeneous x-spacing')
    if grid.h[1][int(grid.shape_cells[1]/2)] != y_spacing:
        print('Warning: no homogeneous y-spacing')
    if grid.h[2][int(grid.shape_cells[2]/2)] != z_spacing:
        print('Warning: no homogeneous z-spacing')

    if QC==True:
        print('x: ', grid.cell_centers_x[ix])
        print('y: ', grid.cell_centers_y[iy])
        print('z: ', grid.cell_centers_z[iz])
    return ix, iy, iz

def find_cuboid_indeces_xyz(grid, x_extent, y_extent, zmin, zmax, QC=True):

    #To ensure that we select inside the cuboid in case the edge aligns with cell edges, we add a little to each value
    cornerpoint1 = [-x_extent+1, -y_extent+1, zmin+1]
    cornerpoint2 = [x_extent-1, y_extent-1, zmax-1]

    if QC==True:
        print(cornerpoint1, cornerpoint2)

    ix_min, iy_min, iz_min = point2index_xyz(grid, cornerpoint1, QC=True)
    ix_max, iy_max, iz_max = point2index_xyz(grid, cornerpoint2, QC=True)

    return ix_min, iy_min, iz_min, ix_max, iy_max, iz_max

def change_grid_values(values_orig, grid, x_extent, y_extent, zmin, zmax, new_value):
    #Assumes that cuboid will be created around x=0 and y=0
    #Only works for grids that are symmetric around the origin in the number of cells

    ix_min, iy_min, iz_min, ix_max, iy_max, iz_max = find_cuboid_indeces_xyz(grid, x_extent, y_extent, zmin, zmax)

    #QC the bounds
    print('x bounds: ', grid.cell_centers_x[ix_min], ' , ', grid.cell_centers_x[ix_max])
    print('y_bounds: ', grid.cell_centers_y[iy_min], ' , ', grid.cell_centers_y[iy_max])
    print('z_bounds: ', grid.cell_centers_z[iz_min], ' , ', grid.cell_centers_z[iz_max])

    # Set target resistivity of the cuboid
    values_new = values_orig
    values_new[ix_min:ix_max+1, iy_min:iy_max+1, iz_min:iz_max+1] = new_value

    return values_new

# In[ ]:




