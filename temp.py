# %% [markdown]
# # Demo for nonlinear inverse design of kirigami patterns

# %%
import numpy as np
import matplotlib.pyplot as plt
from Structure import *
from Utils import *
import scipy.optimize as scopt
import time

# %% [markdown]
# ## Define an objective function based on circularity

# %%
def boundary_residual_circle(interior_offsets_vector):
    
    global width 
    global height
    global target_phi
    global structure
    global boundary_points
    global corners
    global boundary_offsets
    global dual_bound_inds
    global reduced_dual_bound_inds
    global boundary_points_vector
    
    interior_offsets = np.reshape(interior_offsets_vector, (height, width))
    
    # update the pattern using the linear inverse design method with the input offset
    structure.linear_inverse_design(boundary_points_vector, corners, interior_offsets, boundary_offsets)
    structure.make_hinge_contact_points()

    # get the second contracted state
    deployed_points, deployed_hinge_contact_points = structure.layout(phi=0.0) 
    
    # assess the circularity using sum( (r-r_mean)^2 )
    distance_from_center = np.sqrt(np.square(np.array(deployed_points[reduced_dual_bound_inds,0]-np.mean(deployed_points[reduced_dual_bound_inds,0]))) + np.square(np.array(deployed_points[reduced_dual_bound_inds,1]-np.mean(deployed_points[reduced_dual_bound_inds,1]))))
    residuals = distance_from_center - np.mean(distance_from_center)
    
    # (Alternative way) assess the circularity using (L^2/(4*pi*A) - 1)^2
    # reduced_dual_bound_inds_shifted = reduced_dual_bound_inds[1:]
    # reduced_dual_bound_inds_shifted.append(reduced_dual_bound_inds[0])
    # edgelengths = np.sqrt(np.square(np.array(deployed_points[reduced_dual_bound_inds,0]-deployed_points[reduced_dual_bound_inds_shifted,0])) + np.square(np.array(deployed_points[reduced_dual_bound_inds,1]-deployed_points[reduced_dual_bound_inds_shifted,1])))
    # L = np.sum(edgelengths)
    # pgon = geom.Polygon(zip(deployed_points[reduced_dual_bound_inds,0], deployed_points[reduced_dual_bound_inds,1]))
    # A = pgon.area
    # residuals = L**2/(4*np.pi*A) - 1
    
    return residuals

# %% [markdown]
# ## Solve for an optimized pattern that gives a square-to-circle transformation

# %%
# set the pattern size
width = 8 
height = 8

# create a square kirigami structure and get the required information for the optimization
structure = MatrixStructure(num_linkage_rows=height, num_linkage_cols=width)
bound_linkage_inds = [structure.get_boundary_linkages(i) for i in range(4)]
bound_directions = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
boundary_points = []
corners = []
for i, bound in enumerate(bound_linkage_inds):
    local_boundary_points = []
    for j, linkage_ind in enumerate(bound):
        p = structure.is_linkage_parallel_to_boundary(linkage_ind[0], linkage_ind[1], i)
        if j == 0:
            corner = np.array([linkage_ind[1], -linkage_ind[0]]) + bound_directions[i]
            if not p:
                corner += bound_directions[(i-1)%4]
            corners.append(corner)
        if not p:
            point = np.array([linkage_ind[1], -linkage_ind[0]]) + bound_directions[i]
            local_boundary_points.append(point)
    boundary_points.append(np.vstack(local_boundary_points))
corners = np.vstack(corners)
boundary_offsets = [[0.0]*height, [0.0]*width, [0.0]*height, [0.0]*width]
boundary_points_vector = np.vstack(boundary_points)

# also get the reconfigured boundary node indices for optimizing the second contracted shape
structure.linear_inverse_design(boundary_points_vector, corners, np.reshape(np.zeros(width*height), (height, width)), boundary_offsets)
structure.make_hinge_contact_points()
deployed_points, deployed_hinge_contact_points = structure.layout(phi=0.0) 
dual_bound_inds = []
for bound_ind in range(4):
    dual_bound_inds.extend(structure.get_dual_boundary_node_inds(bound_ind))
reduced_dual_bound_inds = []
for i, ind in enumerate(dual_bound_inds):    
    next_i = (i+1) % len(dual_bound_inds)    
    next_ind = dual_bound_inds[next_i]    
    if norm(deployed_points[ind] - deployed_points[next_ind]) > 1e-10:
        reduced_dual_bound_inds.append(ind)
        
# perform the nonlinear optimization to find an optimal set of offset parameters 
# that gives a square-to-circle transformation
print("Optimization starts...")
start = time.time()
result = scopt.least_squares(boundary_residual_circle, np.zeros(width*height),bounds=(-np.ones(width*height),np.inf),xtol = 1e-4,verbose=2)
end = time.time()
print("Finished.")
print('Time taken = ' + str(end - start) + ' seconds')
print('Cost = ' + str(result.cost))
print('Optimal offset = ')
print(result.x)

# %% [markdown]
# ## Plot the optimized pattern

# %%
interior_offsets = np.reshape(result.x, (height,width)) # optimal offsets
structure.linear_inverse_design(np.vstack(boundary_points), corners, interior_offsets, boundary_offsets)
structure.assign_node_layers()
structure.assign_quad_genders()
structure.make_hinge_contact_points()

num_frames = 5
phis = np.linspace(np.pi, 0, num_frames)

panel_size = 10
fig, axs = plt.subplots(1, num_frames, figsize=(1.2*panel_size*num_frames, panel_size), sharey=True)

for ax_ind, phi in enumerate(phis):
    
    deployed_points, deployed_hinge_contact_points = structure.layout(phi)
    deployed_points = rotate_points(deployed_points, np.array([0, 0]), -(np.pi - phi)/2.0)
    
    deployed_points[:,0] = deployed_points[:,0] - (np.max(deployed_points[:,0])+np.min(deployed_points[:,0]))/2
    deployed_points[:,1] = deployed_points[:,1] - (np.max(deployed_points[:,1])+np.min(deployed_points[:,1]))/2
    
    plot_structure(deployed_points, structure.quads, structure.linkages, axs[ax_ind])
    axs[ax_ind].set_aspect('equal')
    
#     write_obj('Example_square-to-circle_' + str(ax_ind) + '.obj', deployed_points, structure.quads)

# %%
# %% [markdown]
# ## Show target vs. initial pattern, then create & save a GIF of the deployment

# %%
import matplotlib.pyplot as plt
from matplotlib import animation
import imageio
import os

# ---------- 1. Reference figure: target circle & undeployed pattern ----------
fig_ref, ax_ref = plt.subplots(1, 2, figsize=(12, 6))

# (left) undeployed pattern at phi = π
undeployed_pts, _ = structure.layout(phi=np.pi)
plot_structure(undeployed_pts, structure.quads, structure.linkages, ax_ref[0])
ax_ref[0].set_title("Flat pattern (φ = π)")
ax_ref[0].set_aspect('equal')
ax_ref[0].axis('off')

# (right) ideal target: unit circle
theta = np.linspace(0, 2*np.pi, 400)
ax_ref[1].plot(np.cos(theta), np.sin(theta))
ax_ref[1].set_title("Target shape (circle)")
ax_ref[1].set_aspect('equal')
ax_ref[1].axis('off')

plt.tight_layout()
plt.show()

# ---------- 2. Build animation frames ----------
num_frames = 40         # finer sweep for smoother animation
phis_anim = np.linspace(np.pi, 0, num_frames)

fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
ax_anim.set_aspect('equal')
ax_anim.axis('off')

def init():
    ax_anim.clear()
    ax_anim.set_aspect('equal')
    ax_anim.axis('off')
    return []

def animate(i):
    phi = phis_anim[i]
    ax_anim.clear()
    ax_anim.set_aspect('equal')
    ax_anim.axis('off')

    pts, _ = structure.layout(phi)
    # recentre
    pts = rotate_points(pts, np.array([0, 0]), -(np.pi - phi)/2.0)
    pts[:,0] -= (pts[:,0].max() + pts[:,0].min())/2
    pts[:,1] -= (pts[:,1].max() + pts[:,1].min())/2

    plot_structure(pts, structure.quads, structure.linkages, ax_anim)
    return []

ani = animation.FuncAnimation(fig_anim, animate, init_func=init,
                              frames=num_frames, interval=80, blit=True)

# ---------- 3. Save as GIF ----------
# First dump frames to PNGs (works reliably everywhere)
tmp_dir = "_tmp_frames"
os.makedirs(tmp_dir, exist_ok=True)
frame_files = []

for i in range(num_frames):
    animate(i)
    fname = os.path.join(tmp_dir, f"frame_{i:03d}.png")
    fig_anim.savefig(fname, bbox_inches="tight", pad_inches=0.05)
    frame_files.append(fname)

# stitch into GIF (6 fps → ~0.167 s per frame)
with imageio.get_writer("square_to_circle.gif", mode="I", duration=0.167) as writer:
    for fname in frame_files:
        writer.append_data(imageio.imread(fname))

# clean up
for fname in frame_files:
    os.remove(fname)
os.rmdir(tmp_dir)

print("Saved animation ➜ square_to_circle.gif")


# %%
# %% [markdown]
# ## 🔄  Inverse design for a heart-shaped contraction + animation

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scopt
from matplotlib import animation
import imageio, os, shutil, time

# ------------------------------------------------------------------
# 1.  Target geometry: parametric heart curve,
#     re-sampled to match the number of dual-boundary vertices
# ------------------------------------------------------------------
num_target_pts = len(reduced_dual_bound_inds)
t = np.linspace(0, 2*np.pi, num_target_pts, endpoint=False)

# classical cardioid-like heart (scaled later)
x_heart = 16*np.sin(t)**3
y_heart = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)

target_points_raw = np.vstack([x_heart, y_heart]).T
# isotropically scale so its RMS radius is 1 (gives optimizer a sensible scale)
target_points_raw -= target_points_raw.mean(axis=0)
target_points_raw /= np.sqrt((target_points_raw**2).sum(axis=1)).mean()

# make it global so the residual can see it
target_points = target_points_raw.copy()           # <<< global

# ------------------------------------------------------------------
# 2.  Residual function: match every dual-boundary node to the target point
#     (up to a uniform scale and translation, which we strip out)
# ------------------------------------------------------------------
def boundary_residual_heart(interior_offsets_vector):
    global width, height, structure
    global boundary_points_vector, corners, boundary_offsets
    global reduced_dual_bound_inds, target_points
    
    interior_offsets = np.reshape(interior_offsets_vector, (height, width))
    
    # forward solve
    structure.linear_inverse_design(boundary_points_vector, corners,
                                    interior_offsets, boundary_offsets)
    structure.make_hinge_contact_points()
    deployed_pts, _ = structure.layout(phi=0.0)
    
    # extract, centre
    P = deployed_pts[reduced_dual_bound_inds]
    P -= P.mean(axis=0)
    T = target_points.copy()
    
    # best uniform scale (no rotation –  heart has clear orientation already)
    scale = np.linalg.norm(P) / np.linalg.norm(T)
    residuals = (P - scale*T).ravel()
    return residuals

# ------------------------------------------------------------------
# 3.  Optimise interior offsets
# ------------------------------------------------------------------
print("\n💓  Optimising for heart target …")
t0 = time.time()
result = scopt.least_squares(boundary_residual_heart,
                             x0=np.zeros(width*height),
                             bounds=(-np.ones(width*height), np.inf),
                             xtol=1e-4, verbose=2)
print(f"… done in {time.time()-t0:.1f}s, cost = {result.cost:.3e}")

# push optimal offsets back into structure for visualisation
opt_offsets = np.reshape(result.x, (height, width))
structure.linear_inverse_design(boundary_points_vector, corners,
                                opt_offsets, boundary_offsets)
structure.assign_node_layers(); structure.assign_quad_genders()
structure.make_hinge_contact_points()

# ------------------------------------------------------------------
# 4.  Plot reference figure: flat pattern & target heart
# ------------------------------------------------------------------
fig_ref, ax_ref = plt.subplots(1, 2, figsize=(12, 6))

# (left) undeployed pattern at φ = π
flat_pts, _ = structure.layout(phi=np.pi)
plot_structure(flat_pts, structure.quads, structure.linkages, ax_ref[0])
ax_ref[0].set_title("Flat pattern (φ = π)")
ax_ref[0].set_aspect("equal"); ax_ref[0].axis("off")

# (right) target heart
ax_ref[1].plot(target_points[:,0], target_points[:,1], linewidth=2)
ax_ref[1].set_title("Target shape (heart)")
ax_ref[1].set_aspect("equal"); ax_ref[1].axis("off")
plt.tight_layout(); plt.show()

# ------------------------------------------------------------------
# 5.  Build animation frames (flat ➜ heart) & save GIF
# ------------------------------------------------------------------
num_frames = 40
phis_anim = np.linspace(np.pi, 0, num_frames)

fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
ax_anim.set_aspect("equal"); ax_anim.axis("off")

def draw_state(phi):
    ax_anim.clear(); ax_anim.set_aspect("equal"); ax_anim.axis("off")
    P, _ = structure.layout(phi)
    P = rotate_points(P, np.array([0, 0]), -(np.pi - phi)/2)
    P[:,0] -= (P[:,0].max()+P[:,0].min())/2
    P[:,1] -= (P[:,1].max()+P[:,1].min())/2
    plot_structure(P, structure.quads, structure.linkages, ax_anim)

def init(): draw_state(phis_anim[0]); return []
def animate(i): draw_state(phis_anim[i]); return []

ani = animation.FuncAnimation(fig_anim, animate, init_func=init,
                              frames=num_frames, interval=90, blit=True)

tmp_dir = "_tmp_frames_heart"
os.makedirs(tmp_dir, exist_ok=True)
pngs = []
for i, phi in enumerate(phis_anim):
    draw_state(phi)
    fn = os.path.join(tmp_dir, f"frame_{i:03d}.png")
    fig_anim.savefig(fn, bbox_inches="tight", pad_inches=0.0)
    pngs.append(fn)

with imageio.get_writer("square_to_heart.gif", mode="I", duration=0.15) as w:
    for fn in pngs: w.append_data(imageio.imread(fn))

shutil.rmtree(tmp_dir)       # clean
print("✅  Saved ➜ square_to_heart.gif")

# %%


# %%
# %% [markdown]
# ## ❤️  Square-to-Heart inverse design with ordering + rigid Procrustes fit

# %%
import numpy as np, time, os, shutil, imageio
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy.optimize as scopt
from scipy.linalg import svd

# ------------------------------------------------------------------
# 0.  (Optional) Finer pattern for sharper heart
# ------------------------------------------------------------------
# Comment these two lines if you want to keep the 4×4 mesh
width, height = 6, 6                       # <- higher resolution
structure = MatrixStructure(num_linkage_rows=height,
                            num_linkage_cols=width)

# -----  rebuild boundary-related data for the new resolution -----
bound_linkage_inds = [structure.get_boundary_linkages(i) for i in range(4)]
bound_directions = np.array([[-1.,0.], [0.,-1.], [1.,0.], [0.,1.]])
boundary_points, corners = [], []
for i, bound in enumerate(bound_linkage_inds):
    local = []
    for j, ind in enumerate(bound):
        p = structure.is_linkage_parallel_to_boundary(ind[0], ind[1], i)
        if j == 0:
            corner = np.array([ind[1], -ind[0]]) + bound_directions[i]
            if not p: corner += bound_directions[(i-1)%4]
            corners.append(corner)
        if not p:
            pt = np.array([ind[1], -ind[0]]) + bound_directions[i]
            local.append(pt)
    boundary_points.append(np.vstack(local))
corners = np.vstack(corners)
boundary_offsets = [[0.0]*height, [0.0]*width,
                    [0.0]*height, [0.0]*width]
boundary_points_vector = np.vstack(boundary_points)

# -----  first, create the zero-offset pattern to grab indices -----
structure.linear_inverse_design(boundary_points_vector, corners,
                                np.zeros((height,width)), boundary_offsets)
structure.make_hinge_contact_points()
deployed_pts, _ = structure.layout(phi=0.0)
dual_bound_inds = []
for b in range(4):
    dual_bound_inds.extend(structure.get_dual_boundary_node_inds(b))
reduced_dual_bound_inds = []
for i, ind in enumerate(dual_bound_inds):
    nxt = dual_bound_inds[(i+1)%len(dual_bound_inds)]
    if np.linalg.norm(deployed_pts[ind]-deployed_pts[nxt]) > 1e-10:
        reduced_dual_bound_inds.append(ind)

# ------------------------------------------------------------------
# 1.  Target heart curve (same number of points as boundary)
# ------------------------------------------------------------------
n_target = len(reduced_dual_bound_inds)
t = np.linspace(0, 2*np.pi, n_target, endpoint=False)
xH = 16*np.sin(t)**3
yH = (13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t))
T_raw = np.vstack([xH, yH]).T
T_raw -= T_raw.mean(0)                       # centre at origin
T_raw /= np.sqrt((T_raw**2).sum(1)).mean()   # unit RMS radius
target_points = T_raw                        # global for residual

# ------------------------------------------------------------------
# 2.  Residual with boundary re-ordering + full rigid Procrustes fit
# ------------------------------------------------------------------
def residual_heart(offset_vec):
    interior = offset_vec.reshape(height, width)
    structure.linear_inverse_design(boundary_points_vector, corners,
                                    interior, boundary_offsets)
    structure.make_hinge_contact_points()
    P_full, _ = structure.layout(phi=0.0)
    P = P_full[reduced_dual_bound_inds]

    # -----  order nodes by polar angle (counter-clockwise) -----
    ctr = P.mean(0)
    ang = np.arctan2(P[:,1]-ctr[1], P[:,0]-ctr[0])
    order = np.argsort(ang)
    P = P[order]

    # shift heart curve so its rightmost point is first (best guess)
    T = np.roll(target_points, -np.argmin(target_points[:,0]), axis=0)

    # -----  rigid + uniform-scale fit  -----
    Pc = P - P.mean(0)
    Tc = T - T.mean(0)
    C = Tc.T @ Pc
    U, _, Vt = svd(C)
    R = U @ Vt                    # 2×2 rotation / reflection
    s = (Pc*Pc).sum() / (Tc@R).ravel().dot(Pc.ravel())
    return (Pc - s*Tc@R).ravel()

# ------------------------------------------------------------------
# 3.  Non-linear least squares optimisation
# ------------------------------------------------------------------
print("💓  Optimising …")
t0 = time.time()
x0 = 0.1*np.random.randn(width*height)       # small random start
result = scopt.least_squares(residual_heart, x0,
                             bounds=(-2*np.ones(width*height), np.inf),
                             xtol=1e-4, verbose=2)
print(f"… done in {time.time()-t0:.1f}s; cost = {result.cost:.2e}")

opt_offsets = result.x.reshape(height, width)
structure.linear_inverse_design(boundary_points_vector, corners,
                                opt_offsets, boundary_offsets)
structure.assign_node_layers(); structure.assign_quad_genders()
structure.make_hinge_contact_points()

# ------------------------------------------------------------------
# 4.  Reference plot (flat pattern & target heart)
# ------------------------------------------------------------------
fig_ref, ax = plt.subplots(1,2, figsize=(12,6))
flat, _ = structure.layout(phi=np.pi)
plot_structure(flat, structure.quads, structure.linkages, ax[0])
ax[0].set_title("Flat pattern"); ax[0].set_aspect("equal"); ax[0].axis("off")
ax[1].plot(target_points[:,0], target_points[:,1], lw=2)
ax[1].set_title("Target heart"); ax[1].set_aspect("equal"); ax[1].axis("off")
plt.tight_layout(); plt.show()

# ------------------------------------------------------------------
# 5.  Animation: square → heart  (saved as GIF)
# ------------------------------------------------------------------
num_frames = 45
phis = np.linspace(np.pi, 0, num_frames)

fig_anim, ax_anim = plt.subplots(figsize=(6,6))
ax_anim.axis("off"); ax_anim.set_aspect("equal")

def draw(phi):
    ax_anim.clear(); ax_anim.axis("off"); ax_anim.set_aspect("equal")
    P,_ = structure.layout(phi)
    P = rotate_points(P, np.array([0,0]), -(np.pi-phi)/2)
    P[:,0] -= (P[:,0].max()+P[:,0].min())/2
    P[:,1] -= (P[:,1].max()+P[:,1].min())/2
    plot_structure(P, structure.quads, structure.linkages, ax_anim)

def init(): draw(phis[0]); return []
def animate(i): draw(phis[i]); return []

ani = animation.FuncAnimation(fig_anim, animate, init_func=init,
                              frames=num_frames, interval=90, blit=True)

tmp = "_frames_heart"
os.makedirs(tmp, exist_ok=True)
pngs = []
for i,phi in enumerate(phis):
    draw(phi)
    fname = f"{tmp}/f_{i:03d}.png"
    fig_anim.savefig(fname, bbox_inches="tight", pad_inches=0.0)
    pngs.append(fname)

with imageio.get_writer("square_to_heart.gif", mode="I", duration=0.12) as w:
    for f in pngs: w.append_data(imageio.imread(f))
shutil.rmtree(tmp)
print("✅  GIF saved as  square_to_heart.gif")

# %%



