# %% [markdown]
# # Emg3d pyGIMLI minimal example inversion
# 
# Requires
# - `pygimli>=1.5.2`
# - ipygimli.py
# 
# We use two regions to invert for, and print the models for each iteration (no jacobian visualization)

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

import emg3d
import ipygimli as ipygimli
from Emg3d_extra_grid_functions_v26 import *
import h5py

# %%
#Random number regenerator used to set the noise, seed guarantees the same noise is reproduced each time, can be adapted with different numbers to test different noise
rng_real = np.random.default_rng(seed=0)
rng_imag = np.random.default_rng(seed=1)

# %% [markdown]
# ## Create synthetic data
# 
# Model used in previous forward-modeling (v1), with a cuboid hot plume inside

# %% [markdown]
# ### Define survey

# %%
# Source, y-directed point dipole on the surface
src_coords = (0, -130, 0, 90.0, 0.0)
src = emg3d.surveys.txrx_coordinates_to_dict(emg3d.TxElectricDipole, src_coords)

# Source frequency
frequency = 1

# Receivers, vertical z-directed point dipoles
rec_x = np.arange(1, 41)*0
rec_y = np.array([75]*40)
rec_z = np.arange(1,41)*-7.5
rec_coords = (rec_x, rec_y, rec_z, 0.0, 90.0)
rec = emg3d.surveys.txrx_coordinates_to_dict(emg3d.RxElectricPoint, rec_coords)

# Layer boundaries
# We idealize the depth model here a bit, to have cells of 10m in depth
depth = [0, -50, -80, -120, -190]

# Layer resistivities in Ohm.m for the baseline
resistivities_base = np.array([1e8, 52.0, 38.0, 26.0, 29.5, 17.5]) # base incl. air (lower res, because of 3D modelling)

# %% [markdown]
# ### Create grids and models

# %%
# The fine grid, used for forward-modelling

f_grid = emg3d.construct_mesh(
        center=src_coords[:3],                                 # Center of wanted grid
        frequency=frequency,                            # Frequency we will use the grid for
        properties=[25, 1e8],                           # Reference resistivity
        domain=([-100, 100], [-170, 150], [-300, 25]),   # Domain in which we want precise results
        center_on_edge=True,
        min_width_limits = [5, 5, 2.5],
    )

f_grid  # mesh info
#grid.plot_grid() # plotting the grid

# %%
# The coarse grid, used for inverse modelling

c_grid = emg3d.construct_mesh(
        center=src_coords[:3],                                 # Center of wanted grid
        frequency=frequency,                            # Frequency we will use the grid for
        properties=[25, 1e8],                           # Reference resistivity
        domain=([-105, 105], [-150, 135], [-250, 5]),   # Domain in which we want precise results
        center_on_edge=True,
        min_width_limits = [15, 15, 5],
    )
 
c_grid  # mesh info
#grid.plot_grid() # plotting the grid

# %%
# Create baseline resistivity models for both grids

# Pre-allocate an array with number of cells corresponding to our grid
f_res_base = np.ones(f_grid.shape_cells)
c_res_base = np.ones(c_grid.shape_cells)

# Layered base model: Fill in layered resistivities into our grid and create model
dd = np.r_[np.inf, depth, -np.inf]
for i in range(len(resistivities_base)):
    f_res_base[:, :, (f_grid.cell_centers_z > dd[i+1]) & (f_grid.cell_centers_z <= dd[i])] = resistivities_base[i]
    c_res_base[:, :, (c_grid.cell_centers_z > dd[i+1]) & (c_grid.cell_centers_z <= dd[i])] = resistivities_base[i]

f_model_base = emg3d.Model(f_grid, property_x=f_res_base, mapping='Resistivity')
f_model_base_conductivity = emg3d.Model(f_grid, property_x=1/np.array(f_res_base), mapping='Conductivity')
c_model_base = emg3d.Model(c_grid, property_x=c_res_base, mapping='Resistivity')
c_model_base_conductivity = emg3d.Model(c_grid, property_x=1/np.array(c_res_base), mapping='Conductivity')

# %%
# Create target resistivity model for the fine grid

#Use predefined function to insert a cuboid inton the target layer

x_extent = 50
y_extent = 50
zmin = depth[4]
zmax = depth[3]
rho_target = 1 # emphasizing the contrast to make inversion easier

res_target = change_grid_values(f_res_base.copy(), f_grid, x_extent, y_extent, zmin, zmax, rho_target)

model_target = emg3d.Model(f_grid, property_x=res_target, mapping='Resistivity')
model_target_conductivity = emg3d.Model(f_grid, property_x=1/np.array(res_target), mapping='Conductivity')

# %%
# QC the target model on the y,z plane
f0 = f_grid.plot_slice(
    model_target.property_x,
    normal = 'X',
    pcolor_opts={'norm': LogNorm(vmin=1, vmax=100)},
    range_x=[-300, 300],
    range_y=[-300, 25],
#    slice_loc=0
)
fig = plt.gcf()
fig.suptitle('Resistivity model (Ohm.m) and survey layout')
axs = fig.get_children()
axs[1].plot(rec_coords[1], rec_coords[2], 'bv', markersize = 10)
axs[1].plot(src_coords[1], src_coords[2], 'r*', markersize = 20)
axs[1].grid(False)
fig.colorbar(f0[0], ax=axs[1], fraction=0.05, location='right')

plt.savefig("25nov_minex: QC1 - target model")

# %% [markdown]
# ### Create the simulated data
# 
# Code is based on sensitivity gradient notebook v1_7
# 
# We want to create a simulation class with the true model data + noise as observed data and the starting model as the model

# %%
# Parameterize the noise to be added

noise_floor = 1e-10
relative_error = 0.03
mean_noise = 0.0

# %%
survey = emg3d.surveys.Survey(
    name='Min_ex1: survey forward',  # Name of the survey
    sources=src,               # Source coordinates
    receivers=rec,             # Receiver coordinates
    frequencies=frequency,   # Two frequencies
    # data=data,               # Adding the observed data here will allow sensitivity calculation later, but we need to generate it first
)

# %%
# simulate with the true model to create the observed data

simulation = emg3d.simulations.Simulation(
            name='Minex1: simulation forward',    # A name for this simulation
            survey=survey,        # Our survey instance
            model=model_target_conductivity,          # The model
            gridding='same',      # Frequency- and source-dependent meshes
            max_workers=1,        # How many parallel jobs
            # solver_opts,        # Any parameter to pass to emg3d.solve
            #gridding_opts=gopts,  # Gridding options
            receiver_interpolation = "linear" #Needed for correct computation of the gradient
        )

simulation.compute(observed=True)
#The simulation should now contain observed data (+noise), synthetic data (without noise) and standard deviation values
#The survey class is also updated with this information

# %%
# Re-compute the noise and overwrite the data, based on empymod+pygimli testing

dobs = survey.data['synthetic']

# Add random white noise to dobs
std_dev_real = np.sqrt(noise_floor**2 + np.abs(relative_error*dobs.real)**2)
std_dev_imag = np.sqrt(noise_floor**2 + np.abs(relative_error*dobs.imag)**2)
noise_real = rng_real.normal(loc=0, scale=std_dev_real, size = dobs.shape)
noise_imag = rng_imag.normal(loc=0, scale=std_dev_imag, size = dobs.shape)
std_dev = std_dev_real + 1j*std_dev_imag
noise = noise_real + 1j*noise_imag
dobs += noise
dobs0 = dobs
noise0 = noise

# Overwrite values in survey
survey.data['observed'] = dobs
survey.standard_deviation = std_dev

# %%
survey2 = emg3d.surveys.Survey(
    name='Min_ex1: survey invert',  # Name of the survey
    sources=src,               # Source coordinates
    receivers=rec,             # Receiver coordinates
    frequencies=frequency,   # Two frequencies
    data=survey.data['observed'], # Adding the observed data here will allow sensitivity calculation later, but we need to generate it first
)
survey2.standard_deviation = survey.standard_deviation

# %%
simulation2 = emg3d.simulations.Simulation(
            name='Min_ex1: simulation invert',    # A name for this simulation
            survey=survey2,        # Our survey instance
            model=c_model_base_conductivity,          # The model
            gridding='same',      # Frequency- and source-dependent meshes
            max_workers=1,        # How many parallel jobs
            # solver_opts,        # Any parameter to pass to emg3d.solve
            #gridding_opts=gopts,  # Gridding options
            receiver_interpolation = "linear", #Needed for correct computation of the gradient
            tqdm_opts = False #To switch-off progress bar during inversion
        )

simulation2.compute(observed=False)

#We don't put observed=True in this case because this will overwrite the observed data saved inside the survey

# %%
# QC data
dobs = np.array(dobs0)[0,:,0]
noise = np.array(noise0)[0,:,0]
dhalf = np.array(survey2.data['synthetic'])[0,:,0]

# Plot it
fig, (ax2, ax3, ax4, ax5) = plt.subplots(1, 4, figsize=(12, 6), constrained_layout=True)
axes = [ax2, ax3, ax4, ax5]

plt.locator_params(axis='x', nbins=3)

#Plot measured data
labels = ['Amplitude (V/m)', 'Phase (°)', '|Real| (V/m)', '|Imag| (V/m)']
data_obs = [np.abs(dobs), np.angle(dobs,deg=True), abs(dobs.real), abs(dobs.imag)]
data0 = [np.abs(dobs-noise), np.angle(dobs-noise,deg=True), abs((dobs-noise).real), abs((dobs-noise).imag)]
data_half = [np.abs(dhalf), np.angle(dhalf,deg=True), abs(dhalf.real), abs(dhalf.imag)]
fig.suptitle("Min_ex1: inversion data QC incl. "+str(relative_error)+" relative error")
for j in np.arange(len(labels)):
    ax = axes[j]
    ax.set_title(labels[j])
    ax.plot(data0[j][:], rec_coords[2], 'C'+str(j)+'-', label = f"{frequency: 2.1f} Hz - no noise")
    ax.plot(data_obs[j][:], rec_coords[2], 'C'+str(j)+'.', label=f"{frequency: 2.1f} Hz - data")
    ax.plot(data_half[j][:], rec_coords[2], 'C'+str(j)+'--', label=f"{frequency: 2.1f} Hz - baseline")
    ax.legend(loc='upper left')
    ax.set_xscale('log')
    xmin, xmax = ax.get_xlim()
    ax.set_xlim([xmin*0.8, xmax*1.2])
    
ax3.set_xscale('linear')
ax3.set_xlim([179.5, 180])
#The noise looks very acceptable

plt.savefig("Min_ex1: QC2 - data going into inversion")
np.savetxt("Min_ex1: QC2 - data going into inversion", np.array([dobs, noise, dhalf]))

# %% [markdown]
# ## Define regions, instantiate Kernel and Inversion
# 
# We only invert for the value of the anomaly, not the location, by marking the region of the anomaly and using single=True in pyGIMLi's region manager

# %%
# Create region markers
markers = np.zeros(simulation2.model.shape, dtype=int)

# Create two regions to invert for, at the bottom and top of the target
ix_min, iy_min, iz_min, ix_max, iy_max, iz_max = find_cuboid_indeces_xyz(c_grid, 50, 50, -190, -150, QC=True)
markers[ix_min:ix_max+1, iy_min:iy_max+1, iz_min:iz_max+1] = 1            # Bottom part of target
ix_min, iy_min, iz_min, ix_max, iy_max, iz_max = find_cuboid_indeces_xyz(c_grid, 50, 50, -150, -120, QC=True)
markers[ix_min:ix_max+1, iy_min:iy_max+1, iz_min:iz_max+1] = 2       # Top part of target

# Instantiate the emg3d-kernel for pyGIMLi
fop = ipygimli.Kernel(simulation=simulation2, markers=markers, pgthreads=1)

# Instantiate the inversion and set some inversion parameters
INV = ipygimli.Inversion(fop=fop)
#INV.inv.setCGLSTolerance(1)  # is _absolute_, not _relative_
INV.inv.setMaxCGLSIter(10)

# Set the regions
INV.fop.setRegionProperties(1, single=True, limits=(0.01, 2)) # Target top half
INV.fop.setRegionProperties(2, single=True, limits=(0.01, 2)) # Target bottom half
INV.fop.setRegionProperties(0, background=True) # Background

# %%
# QC the inversion region corresponds to the wanted region
# #Focus on the y,z plane
f0 = c_grid.plot_slice(
    markers,
    normal = 'X',
    #pcolor_opts={'norm': LogNorm(vmin=10, vmax=100)},
    range_x=[-300, 300],
    range_y=[-300, 25],
#    slice_loc=0
)
fig = plt.gcf()
fig.suptitle('Min_ex1: Region markers and survey layout')
axs = fig.get_children()
axs[1].plot(rec_coords[1], rec_coords[2], 'bv', markersize = 10)
axs[1].plot(src_coords[1], src_coords[2], 'r*', markersize = 20)
axs[1].grid(False)
fig.colorbar(f0[0], ax=axs[1], fraction=0.05, location='right')

plt.savefig("Min_ex1: QC3 - marker regions used in inversion")

# %% [markdown]
# ## Run inversion

# %%
INV.run(
    #maxIter=2, # just to test
    lam=20,  # btw 1-100
    lambdaFactor=10,
    stopAtChi1=False,
    dPhi=1.0,
    #verbose=True,
    #debug=True,
    #startModel=model.property_x.ravel('F'),
    #isReference=True, 
)

# %% [markdown]
# # Plots

# %%
# Load relevant inversion parameters saved through post_step (ipygimli functionality)

inv_data = h5py.File("Minimal 1: simulation invert.h5",'r')

phi_history = []
for i in np.arange(len(INV.modelHistory)):
    phi_history.append(int(np.array(inv_data['invinfo'][str(i)]['phi'])))

print(phi_history)

# %%
fig, ax = plt.subplots()

ax.plot(INV.chi2History)
ax.tick_params(axis='y', labelcolor='C0')
ax.set_ylabel('Chi2_history', color = 'C0')
ax2 = ax.twinx()
ax2.plot(phi_history, c = 'red')
ymin, ymax = ax2.get_ylim()
ax2.set_ylim([ymin*1.1, ymax*1.1])
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylabel('Phi_history', c='red')
ax.set_title('Inversion information')

plt.savefig("Min_ex1: Inversion progress")

# %%
fig, ax = plt.subplots()

ax.plot(1/np.array(INV.modelHistory)[:,0], label='Top of target')
ax.plot(1/np.array(INV.modelHistory)[:,1], label='Bottom of target')
ax.set_ylabel('Target resistivity')
ax.set_xlabel('Iterations')
ax.set_title('Inversion model value')
xmin, xmax = ax.get_xlim()
ax.hlines(rho_target, xmin, xmax, colors='C3', label= 'True model')
ax.legend()

plt.savefig("Min_ex1: Convergence of parameters")

# %%
#Code adapted from plot_models function in sampledata.py

sim_amount = len(INV.chi2History)
n_rows = 3
n_columns = 3
depth = 150
vmin = 1e0
vmax = 100

popts = {'cmap': 'viridis', 'norm': LogNorm(vmin=vmin, vmax=vmax)} #Normalize(vmin=vmin, vmax=vmax)}
# popts2 = {'edgecolors': 'grey', 'linewidth': 0.5, 'cmap': 'Spectral_r',
#           'norm': LogNorm(vmin=0.1, vmax=1000)}
opts = {'v_type': 'CC', 'normal': 'X', 'range_x': [-300, 300], 'range_y': [-300, 25],}

fig, axs = plt.subplots(n_rows, n_columns, figsize=(n_columns*2, n_rows*2), tight_layout=True, sharex = True, sharey = True)

for i in np.arange(sim_amount-1):
    ax=axs[int((i-i%n_columns)/n_columns), i%n_columns]
    m_it = 1/fop.model2emg3d(INV.modelHistory[i])

    # Plotting the model
    out = c_grid.plot_slice(m_it, ax=ax, pcolor_opts=popts, **opts)
    ax.set_title(str(i))
    ax.plot(rec_coords[1], rec_coords[2], 'bv')
    ax.plot(src_coords[1], src_coords[2], 'r*')
    if i%n_columns != 0:
        ax.set_ylabel('')
        ax.set_yticklabels([])
        if i%n_columns == n_rows:
            plt.colorbar(out[0], ax=ax)#, fraction=.5, shrink=.8, aspect=30)
    
for i in np.arange(sim_amount-1-n_columns):
        ax=axs[int((i-i%n_columns)/n_columns), i%n_columns]
        ax.set_xlabel('')
        ax.set_xticklabels([])

    #for ax in axs.ravel():
    #    ax.axis('equal')
fig.suptitle('Resistivity models (Ohmm) per iteration')

plt.savefig("Min_ex1: Models per iteration")
np.savetxt("Min_ex1: Models per iteration", INV.modelHistory)


# %%
# QC data
dobs = np.array(dobs0)[0,:,0]
noise = np.array(noise0)[0,:,0]
response = simulation2.survey.data.synthetic[0,:,0]

# Plot it
fig, (ax2, ax3, ax4, ax5) = plt.subplots(1, 4, figsize=(12, 6), constrained_layout=True)
axes = [ax2, ax3, ax4, ax5]


plt.locator_params(axis='x', nbins=3)

#Plot measured data
labels = ['Amplitude (V/m)', 'Phase (°)', '|Real| (V/m)', '|Imag| (V/m)']
data_obs = [np.abs(dobs), np.angle(dobs,deg=True), abs(dobs.real), abs(dobs.imag)]
data0 = [np.abs(dobs-noise), np.angle(dobs-noise,deg=True), abs((dobs-noise).real), abs((dobs-noise).imag)]
data_inv = [np.abs(response), np.angle(response,deg=True), abs(response.real), abs(response.imag)]
fig.suptitle("v1 emg3d inversion result "+str(relative_error)+" relative error")
for j in np.arange(len(labels)):
    ax = axes[j]
    ax.set_title(labels[j])
    ax.plot(data0[j][:], rec_coords[2], 'C'+str(j)+'-', label = f"{frequency: 2.1f} Hz - no noise")
    ax.plot(data_obs[j][:], rec_coords[2], 'C'+str(j)+'.', label=f"{frequency: 2.1f} Hz - data")
    ax.plot(data_inv[j][:], rec_coords[2], color= 'orange', linestyle='--', label=f"{frequency: 2.1f} Hz - fitted")
    ax.legend(loc='upper left')
    ax.set_xscale('log')
    xmin, xmax = ax.get_xlim()
    ax.set_xlim([xmin*0.8, xmax*1.2])
    
ax3.set_xscale('linear')
ax3.set_xlim([179.5, 180])
#The noise looks very acceptable

plt.savefig("Min_ex1: Data fitting")


