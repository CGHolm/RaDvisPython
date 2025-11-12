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

def load_DISPATCH(self, snap, path, loading_bar, verbose, shm=False):
    if verbose > 0 and self.data_sphere_au != None:
        print(f'Only selecting patches for the combined dataset within {self.data_sphere_au} au and with level > {self.lv_cut}')

    self.amr = {key: [] for key in ['pos', 'ds']}
    self.mhd = {key: [] for key in ['vel', 'B', 'p','d', 'P', 'm', 'gamma', 'phi']}

    sys.path.insert(0, config["user_dispatch_path"])
    import dispatch as dis

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
    sn_times = np.array([sink_out.time for sink_out in sn.sinks[self.sink_id]])
    sn_i = np.argmin(abs(sn.time - sn_times))

    self.sink_pos = sn.sinks[self.sink_id][sn_i].position.astype(self.dtype)
    self.sink_vel = sn.sinks[self.sink_id][sn_i].velocity.astype(self.dtype) 
    self.time = sn.sinks[self.sink_id][sn_i].time.astype(self.dtype) 
    self.sink_mass = sn.sinks[self.sink_id][sn_i].mass.astype(self.dtype) 

    #Sort the patces according to their level
    if self.data_sphere_au == None:
        pp = [p for p in sn.patches if p.level > self.lv_cut]
    else:
        pp = [p for p in sn.patches 
              if (np.linalg.norm(np.array(np.meshgrid(p.xi, p.yi, p.zi, indexing='ij')) - self.sink_pos[:,None,None,None], axis = 0) < self.data_sphere_au / self.code2au).any()  
              and p.level > self.lv_cut]
        
    w = np.array([p.level for p in pp]).argsort()[::-1]
    sorted_patches = [pp[w[i]] for i in range(len(pp))]

    for p in tqdm.tqdm(sorted_patches, disable = not loading_bar, desc = 'Loading patches'):
        p.m = p.var('d') * np.prod(p.ds)
        p.P = calc_pressure(p.var('d'))
        p.γ = calc_gamma(p.var('d'))
        p.xyz = np.array(np.meshgrid(p.xi, p.yi, p.zi, indexing='ij'))
        p.vel_xyz = np.concatenate([p.var(f'u'+axis)[None,...] for axis in ['x','y','z']], axis = 0)
        p.B =  np.concatenate([p.var(f'b'+axis)[None,...] for axis in ['x','y','z']], axis = 0)
        p.p = np.concatenate([p.var(f'p'+axis)[None,...] for axis in ['x','y','z']], axis = 0)
        

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
        
        self.amr['pos'].extend((p.xyz[:,to_extract].T).tolist())
        self.amr['ds'].extend((p.ds[0] * np.ones(to_extract.sum())))

        self.mhd['vel'].extend((p.vel_xyz[:,to_extract].T).tolist())
        self.mhd['p'].extend((p.p[:,to_extract].T).tolist())
        self.mhd['B'].extend((p.B[:,to_extract].T).tolist())
        self.mhd['d'].extend((p.var('d')[to_extract].T).tolist())
        self.mhd['P'].extend((p.P[to_extract].T).tolist())
        self.mhd['m'].extend((p.m[to_extract].T).tolist())     
        self.mhd['gamma'].extend((p.γ[to_extract].T).tolist())
        self.mhd['phi'].extend((p.var('phi')[to_extract].T).tolist())

    for key in self.amr:
        self.amr[key] = np.array(self.amr[key], dtype = self.dtype).T
    for key in self.mhd:
        self.mhd[key] = np.array(self.mhd[key], dtype = self.dtype).T

    if shm:
        new_folder.cleanup() # delete the temporary shm folder

dataclass.load_DISPATCH = load_DISPATCH
    
