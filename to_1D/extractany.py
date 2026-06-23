import numpy as np
from scipy.interpolate import interp1d
from ..main import dataclass
from ..path_config import config


def extract_1D(self, variables,
               weights,
               n_σH = 3,
               surface_extraction_pct = None,
               data_name = 'data1'):
    
    try: self.extract1D_ivs[data_name] = {}
    except: self.extract1D_ivs = {data_name: {}}

    def extend_func1D(all_values, all_points):
        ma = np.ma.masked_array(all_values, mask = np.isnan(all_values))
        f = interp1d(all_points[~ma.mask], all_values[~ma.mask], kind = 'linear', fill_value='extrapolate')
        return f(all_points)

    try: 
        self.H_1D
    except: 
        print('The function fit HΣ must be called prior to this function...')
        print('Calling now...')
        self.fit_HΣ(verbose = 1)

    r_plot = self.r_bins[:-1] + 0.5 * np.diff(self.r_bins)
    H_func = interp1d(r_plot, self.H_1D[:,0], fill_value='extrapolate')

    mask_r = (self.cyl_R > self.r_bins.min()) & (self.cyl_R < self.r_bins.max())

    if type(n_σH) == tuple or type(n_σH) == list:
        mask_h = (abs(self.cyl_z[mask_r]) > n_σH[0] * H_func(self.cyl_R[mask_r])) & ((abs(self.cyl_z[mask_r]) < n_σH[1] * H_func(self.cyl_R[mask_r])))
    
    if type(n_σH) == int or type(n_σH) == float:
        if not surface_extraction_pct:
            mask_h = abs(self.cyl_z[mask_r]) < n_σH * H_func(self.cyl_R[mask_r])
        else:
            mask_h = ((abs(self.cyl_z[mask_r]) < (1 + surface_extraction_pct) * n_σH * H_func(self.cyl_R[mask_r])) &
                (abs(self.cyl_z[mask_r]) > (1 - surface_extraction_pct) * n_σH * H_func(self.cyl_R[mask_r])))
        
    mask = np.zeros_like(mask_r, dtype = 'bool')
    mask[mask_r] = mask_h

    ds = self.amr['ds'][mask]
    R_coor = self.cyl_R[mask]
        
    values = {ivs: [] for ivs in variables}

    for i, ivs in enumerate(variables):
        if (ivs == np.array(['d', 'P'])).any():
            values[ivs] = self.mhd[ivs][mask]
        else:
            values[ivs] = getattr(self, ivs)[mask]

    weights_dict = {ivs: [] for ivs in variables}
    for i, ivs in enumerate(variables):
        w = weights[i]
        if w == 'mass': weights_dict[ivs] = self.m[mask]
        if w == 'volume': weights_dict[ivs] = (self.amr['ds']**3)[mask]
        if w == 'count' : weights_dict[ivs] = np.ones(mask.sum())
        if w == 'raw': continue            

    for i, ivs in enumerate(variables):
        if weights[i] == 'raw':
            raw_data = {key: [] for key in range(len(self.r_bins) - 1)}     
            bin_index = np.digitize(R_coor, bins = self.r_bins) 
            for bin in np.unique(bin_index):
                if bin == 0 or bin == len(self.r_bins): continue
                if (ivs == np.array(['d', 'P'])).any():
                    raw_data[bin - 1].extend(self.mhd[ivs][mask][bin_index == bin])
                else:
                    raw_data[bin - 1].extend(getattr(self, ivs)[mask][bin_index == bin])
                    
            self.extract1D_ivs[data_name][ivs] = np.array(raw_data)
            continue

        weight, _ = np.histogram(R_coor, bins = self.r_bins, weights = weights_dict[ivs])
        weighted_value, _ = np.histogram(R_coor, bins = self.r_bins, weights =  values[ivs] * weights_dict[ivs])
        weighted_value2, _ = np.histogram(R_coor, bins = self.r_bins, weights =  values[ivs]**2 * weights_dict[ivs])

        value = weighted_value / weight
        value2 = weighted_value2 / weight
        sigma_value = np.sqrt(value2 - value**2) 
        if np.isnan(np.concatenate((value, sigma_value))).any():
            value = extend_func1D(value, r_plot)
            sigma_value = extend_func1D(sigma_value, r_plot)
    
        
        self.extract1D_ivs[data_name][ivs] = np.vstack((value, sigma_value))

dataclass.extract_1D = extract_1D