import numpy as np
import tqdm, sys, os
from ..path_config import config 
from .polytrope import calc_pressure, calc_gamma
from ..main import dataclass
import tqdm
import tempfile
import shutil

def load_RAMSES(self, snap, path):
    sys.path.insert(0, config["user_osyris_path"])
    import osyris

    ds = osyris.Dataset(snap, path=path)
    ds.meta['unit_l'] = self.l_cgs
    ds.meta['unit_t'] = self.t_cgs
    ds.meta['unit_d'] = self.d_cgs
    ds.set_units()
    data = ds.load()
    data['hydro']['gamma'] = osyris.Array(calc_gamma(data['hydro']['density']._array ))

    return data, ds

dataclass.load_RAMSES = load_RAMSES

def load_DISPATCH(self, 
                  snap, 
                  path, 
                  loading_bar, 
                  lagrangian_forces,
                  shm,
                  verbose):
    if verbose > 0 and self.data_sphere_au != None:
        print(f'Only selecting patches for the combined dataset within {self.data_sphere_au:4.1f} au and with {self.lv_min} < level < {self.lv_max}')

    self.amr = {key: [] for key in ['pos', 'ds']}
    self.mhd = {key: [] for key in ['vel', 'B', 'p','d', 'P', 'm', 'gamma', 'phi']}
    if lagrangian_forces:
        self.force = {key: [] for key in ['inertia', 'lorentz', 'gradP', 'gravity']}

    sys.path.insert(0, config["user_dispatch_path"])
    import dispatch as dis

    if shm and (not os.path.isdir('/dev/shm')):
        print("Warning: /dev/shm folder does not exist or is not a folder. Disabling shared memory caching.")
        shm = False
    if shm and (not os.access('/dev/shm', os.W_OK)):
        print("Warning: No write access to /dev/shm. Disabling shared memory caching.")
        shm = False
    if shm:
        new_folder = tempfile.TemporaryDirectory(prefix='/dev/shm/')
        path_internal = new_folder.name
        source = os.path.join(path, '{:05d}'.format(snap)) # snapshot folder
        dest = os.path.join(path_internal, '{:05d}'.format(snap))
        _ = shutil.copytree(source, dest) # copy snapshot to shm
    else:
        path_internal = path

    sn = dis.snapshot(snap, '.', data = path_internal)

    #Load in sink data closest to the snapshot time
    try:
        sn_times = np.array([sink_out.time for sink_out in sn.sinks[self.sink_id]])
        sn_i = np.argmin(abs(sn.time - sn_times))
        self.sink_pos = sn.sinks[self.sink_id][sn_i].position.astype(self.dtype)
        self.sink_vel = sn.sinks[self.sink_id][sn_i].velocity.astype(self.dtype) 
        self.time = sn.sinks[self.sink_id][sn_i].time.astype(self.dtype) 
        self.sink_mass = sn.sinks[self.sink_id][sn_i].mass.astype(self.dtype) 
    except:
        if verbose > 0:
            print('Sink could not be loaded, using "core_pos" and "core_vel" as sink position and velocity')
        self.sink_pos = self.core_pos
        self.sink_vel = self.core_vel 
        self.time = sn.time
        self.sink_mass = None

    #Sort the patces according to their level
    if self.data_sphere_au == None:
        pp = [p for p in sn.patches if (p.level >= self.lv_min) & (p.level <= self.lv_max)]
    else:
        pp = [p for p in sn.patches 
              if (np.linalg.norm(np.array(np.meshgrid(p.xi, p.yi, p.zi, indexing='ij')) - self.sink_pos[:,None,None,None], axis = 0) < self.data_sphere_au / self.code2au).any()  
              and p.level >= self.lv_min and p.level <= self.lv_max]
        
    w = np.array([p.level for p in pp]).argsort()[::-1]
    sorted_patches = [pp[w[i]] for i in range(len(pp))]

    ncells = 0
    ocell = [0]
    plist = []
    pmask = []
    for p in tqdm.tqdm(sorted_patches, 
                       disable = not loading_bar, 
                       desc = 'Loading patches',):
        p.xyz = np.array(np.meshgrid(p.xi, p.yi, p.zi, indexing='ij'))
        
        nbors = [sn.patchid[i] for i in p.nbor_ids if i in sn.patchid]
        children = [ n for n in nbors if n.level == p.level + 1]
        leafs = [n for n in children if ((n.position - p.position)**2).sum() < ((p.size)**2).sum()/12]
        if len(leafs) == 8: continue
        
        if self.data_sphere_au == None:
            to_extract = np.ones(pp[0].n, dtype=bool)
        else:
            p.rel_xyz = p.xyz - self.sink_pos[:, None, None, None]
            p.rel_xyz[p.rel_xyz < -0.5] += 1
            p.rel_xyz[p.rel_xyz > 0.5] -= 1
            p.dist_xyz = np.linalg.norm(p.rel_xyz, axis = 0) 
            to_extract = p.dist_xyz < self.data_sphere_au / self.code2au
        for lp in leafs: 
            leaf_extent = np.vstack((lp.position - 0.5 * lp.size, lp.position + 0.5 * lp.size)).T
            covered_bool = ~np.all((p.xyz > leaf_extent[:, 0, None, None, None]) 
                                   & (p.xyz < leaf_extent[:, 1, None, None, None]), axis=0)
            to_extract *= covered_bool
        
        plist.append(p)
        pmask.append(to_extract)
        ncells += to_extract.sum()
        ocell.append(ncells)

    if loading_bar: print(f'Total number of cells loaded: {ncells}')
    ocell = np.array(ocell, dtype=np.int64)

    self.amr['pos'] = np.empty((3, ncells), dtype=self.dtype)
    self.amr['ds'] = np.empty((ncells,), dtype=self.dtype)
    self.mhd['vel'] = np.empty((3, ncells), dtype=self.dtype)
    self.mhd['p'] = np.empty((3, ncells), dtype=self.dtype)
    self.mhd['B'] = np.empty((3, ncells), dtype=self.dtype)
    self.mhd['d'] = np.empty((ncells,), dtype=self.dtype)
    self.mhd['P'] = np.empty((ncells,), dtype=self.dtype)
    self.mhd['m'] = np.empty((ncells,), dtype=self.dtype)
    self.mhd['gamma'] = np.empty((ncells,), dtype=self.dtype)
    self.mhd['phi'] = np.empty((ncells,), dtype=self.dtype)
    if lagrangian_forces:
        self.force['gradP'] = np.empty((3, ncells), dtype=self.dtype)
        self.force['lorentz'] = np.empty((3, ncells), dtype=self.dtype)
        self.force['inertia'] = np.empty((3, ncells), dtype=self.dtype)
        self.force['gravity'] = np.empty((3, ncells), dtype=self.dtype)

        if verbose > 0 : print('Including force balance from lagrangian momentum equation')
    
    for e,(p,m) in tqdm.tqdm(enumerate(zip(plist, pmask)), 
    disable = not loading_bar, 
    total = len(plist),
    desc = 'Extracting cell data from highest level patches'):
        self.amr['pos'][:, ocell[e]:ocell[e+1]] = p.xyz[:,m]
        self.amr['ds'][ocell[e]:ocell[e+1]] = p.ds[0]
        self.mhd['vel'][0, ocell[e]:ocell[e+1]] = p.var('ux')[m]
        self.mhd['vel'][1, ocell[e]:ocell[e+1]] = p.var('uy')[m]
        self.mhd['vel'][2, ocell[e]:ocell[e+1]] = p.var('uz')[m]
        self.mhd['p'][0, ocell[e]:ocell[e+1]] = p.var('px')[m]
        self.mhd['p'][1, ocell[e]:ocell[e+1]] = p.var('py')[m]
        self.mhd['p'][2, ocell[e]:ocell[e+1]] = p.var('pz')[m]
        self.mhd['B'][0, ocell[e]:ocell[e+1]] = p.var('Bx')[m]
        self.mhd['B'][1, ocell[e]:ocell[e+1]] = p.var('By')[m]
        self.mhd['B'][2, ocell[e]:ocell[e+1]] = p.var('Bz')[m]
        self.mhd['d'][ocell[e]:ocell[e+1]] = p.var('d')[m]
        self.mhd['m'][ocell[e]:ocell[e+1]] = self.mhd['d'][ocell[e]:ocell[e+1]]*np.prod(p.ds)
        self.mhd['P'][ocell[e]:ocell[e+1]] = calc_pressure(self.mhd['d'][ocell[e]:ocell[e+1]])
        self.mhd['gamma'][ocell[e]:ocell[e+1]] = calc_gamma(self.mhd['d'][ocell[e]:ocell[e+1]])
        self.mhd['phi'][ocell[e]:ocell[e+1]] = p.var('phi')[m]
        if lagrangian_forces:
            self.force['gradP'][:, ocell[e]:ocell[e+1]]   = _pressure_gradient_patch(p)[:, m]
            self.force['lorentz'][:, ocell[e]:ocell[e+1]] = _magnetic_force_patch(p)[:, m]
            self.force['inertia'][:, ocell[e]:ocell[e+1]]  = _intertia_patch(p, self.sink_vel)[:, m]
            self.force['gravity'][:, ocell[e]:ocell[e+1]]  = _gravity_patch(p, self.sink_mass, self.sink_pos, self.G_code)[:, m]
    if shm:
        new_folder.cleanup() # delete the temporary shm folder

dataclass.load_DISPATCH = load_DISPATCH

def _intertia_patch(p, sink_vel):
    # CODE UNITS !!!!
    #     
    ds  = np.asarray(p.ds) 
    vx = p.var('ux', all=True) - sink_vel[0]
    vy = p.var('uy', all=True) - sink_vel[1]
    vz = p.var('uz', all=True) - sink_vel[2]
    
    f = np.zeros((3,) + vx.shape)    
    dev_func = lambda v, axis : np.gradient(v, ds[axis], axis = axis) 
    for i, v_comp in enumerate([vx, vy, vz]):
        f[i,...] = vx * dev_func(v_comp, 0) + vy * dev_func(v_comp, 1) + vz * dev_func(v_comp, 2)

    # trim guard zones back to the interior
    ng = np.asarray(p.ng); n = np.asarray(p.n)
    sl = (slice(None),) + tuple(slice(ng[k], ng[k]+n[k]) for k in range(3))
    return f[sl] * p.var('d') #!!!! PLUS


def _magnetic_force_patch_old(p):
    # CODE UNITS !!!!

    ds  = np.asarray(p.ds)                   # cell size (code), per axisT
    Bx = p.var('Bx', all=True)                   # guard-padded, code (Gauss)
    By = p.var('By', all=True) 
    Bz = p.var('Bz', all=True) 

    # centred-difference curl on the padded arrays (== code's ddx, with physical spacing)
    Jx = np.gradient(Bz, ds[1], axis=1) - np.gradient(By, ds[2], axis=2)
    Jy = np.gradient(Bx, ds[2], axis=2) - np.gradient(Bz, ds[0], axis=0)
    Jz = np.gradient(By, ds[0], axis=0) - np.gradient(Bx, ds[1], axis=1)

    # Lorentz force density  +(∇×B)×B    [code units]
    fx = (Jy*Bz - Jz*By) 
    fy = (Jz*Bx - Jx*Bz) 
    fz = (Jx*By - Jy*Bx) 
    f  = np.stack([fx, fy, fz])

    # trim guard zones back to the interior
    ng = np.asarray(p.ng); n = np.asarray(p.n)
    sl = (slice(None),) + tuple(slice(ng[k], ng[k]+n[k]) for k in range(3))
    return - f[sl] #!!!! MINUS


def _magnetic_force_patch(p):
    # CODE UNITS !!!!

    Bx = p.var('Bx', all=0)                   # guard-padded, cgs (Gauss)
    By = p.var('By', all=0) 
    Bz = p.var('Bz', all=0) 

    gc = int((p.gn - p.n)[0] / 2)
    #________________________________________________________________________________________
    Jx_corner = (ddy(zdn(p.var('bz', all=True))) - ddz(ydn(p.var('by', all=True)))) / (p.ds[0])

    # [gn, gn-1, gn-1]
    Jx = 0.25 * (Jx_corner[:, :-1, :-1] + Jx_corner[:, 1:, :-1]
            + Jx_corner[:, :-1, 1:]  + Jx_corner[:, 1:, 1:])

    Jx_centered = Jx[gc:-gc, gc:-(gc - 1), gc:-(gc - 1)]

    #________________________________________________________________________________________
    Jy_corner = (ddz(xdn(p.var('bx', all=True))) - ddx(zdn(p.var('bz', all=True)))) / (p.ds[0])
    # [gn-1, gn, gn-1]
    Jy = 0.25 * (Jy_corner[:-1,  :, :-1] + Jy_corner[1:, :, :-1]
            + Jy_corner[ :-1, :, 1:]  + Jy_corner[1:, :, 1:])

    Jy_centered = Jy[gc:-(gc - 1), gc:-gc, gc:-(gc - 1)]

    #________________________________________________________________________________________
    Jz_corner = (ddx(ydn(p.var('by', all=True))) - ddy(xdn(p.var('bx', all=True)))) / (p.ds[0])
    # [gn-1, gn-1, gn]
    Jz = 0.25 * (Jz_corner[:-1, :-1,:] + Jz_corner[1:, :-1,:]
            + Jz_corner[:-1, 1:,:]  + Jz_corner[1:, 1:,:])

    Jz_centered = Jz[gc:-(gc - 1),gc:-(gc - 1), gc:-gc]
    #_______________________________________________________________________________________
    return np.stack([-(Jy_centered*Bz - Jz_centered*By), 
                     -(Jz_centered*Bx - Jx_centered*Bz), 
                     -(Jx_centered*By - Jy_centered*Bx)])  # = -(∇×B)×B, edge-centred calculation, trimmed to interior


def _pressure_gradient_patch(p):
    P = calc_pressure(p.var('d', all = True))
    ds  = np.asarray(p.ds)
    
    f = np.zeros((3,) + P.shape)
    dev_func = lambda v, axis : np.gradient(v, ds[axis], axis = axis) 
    for i in range(3):
        f[i,...] = dev_func(P, i)

        # trim guard zones back to the interior
    ng = np.asarray(p.ng); n = np.asarray(p.n)
    sl = (slice(None),) + tuple(slice(ng[k], ng[k]+n[k]) for k in range(3))
    return f[sl] #!!!! PLUS

def _gravity_patch(p, sink_mass, sink_pos, G):
    ds  = np.asarray(p.ds)
    phi = p.var('phi', all=True).copy()
    
    if sink_mass != None:
        dist2sink_22grid = np.array(np.meshgrid(p.x, p.y, p.z, indexing='ij')) - sink_pos[:,None, None, None]
        φ_sink = - G * sink_mass / np.linalg.norm(dist2sink_22grid, axis = 0)
        φ_sink[np.isnan(φ_sink)] = 0  
        phi += φ_sink
    
    f = np.zeros((3,) + phi.shape)
    dev_func = lambda v, axis : np.gradient(v, ds[axis], axis = axis) 
    for i in range(3):
        f[i,...] = dev_func(phi, i)

    ng, n = np.asarray(p.ng), np.asarray(p.n)
    sl = (slice(None),) + tuple(slice(ng[k], ng[k]+n[k]) for k in range(3))
    
    return - f[sl] * p.var('d') #!!!! MINUS                                             

def dnup(q, shift=1, axis=0):
    i = 0 if shift==1 else -1
    if axis == 0 and q.shape[0] > 1:
        f = (q+np.roll(q,shift,axis))*0.5
        f[i,:,:] = q[i,:,:]
    elif axis == 1 and q.shape[1] > 1:
        f = (q+np.roll(q,shift,axis))*0.5
        f[:,i,:] = q[:,i,:]
    elif axis == 2 and q.shape[2] > 1:
        f = (q+np.roll(q,shift,axis))*0.5
        f[:,:,i] = q[:,:,i]
    else:
        f = q
    return f

def xdn(f):
    return dnup(f,1,0)
def ydn(f):
    return dnup(f,1,1)
def zdn(f):
    return dnup(f,1,2)

def xup(f):
    return dnup(f,-1,0)
def yup(f):
    return dnup(f,-1,1)
def zup(f):
    return dnup(f,-1,2)

def ddx(f):
    return dnup(f,-1,0)-dnup(f,1,0)
def ddy(f):
    return dnup(f,-1,1)-dnup(f,1,1)
def ddz(f):
    return dnup(f,-1,2)-dnup(f,1,2) 