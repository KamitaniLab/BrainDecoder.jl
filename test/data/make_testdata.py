import bdpy
import numpy as np


# Data settings
voxel_data = np.array([np.arange(10),
                       np.arange(10) + 10,
                       np.arange(10) + 20,
                       np.arange(10) + 30,
                       np.arange(10) + 40,
                       np.arange(10) + 50])
roi_v1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
roi_v2 = [0, 0, 0, 0, 0, 1, 1, 0, 1, 0]
roi_v3 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
runs = np.array([1, 1, 1, 2, 2, 2]).T
stimulus_name = np.array([2, 1, 3, 1, 3, 2]).T
vmap = {
    'stimulus_name': {
        1: 'stimulus_01',
        2: 'stimulus_02',
        3: 'stimulus_03',
    }
}

bdata = bdpy.BData()
bdata.add(voxel_data, 'VoxelData')
bdata.add_metadata('ROI_V1', roi_v1, description='ROI V1', where='VoxelData')
bdata.add_metadata('ROI_V2', roi_v2, description='ROI V2', where='VoxelData')
bdata.add_metadata('ROI_V3', roi_v3, description='ROI V3', where='VoxelData')
bdata.add(runs, 'Run')
bdata.add(stimulus_name, 'stimulus_name')
bdata.add_vmap('stimulus_name', vmap['stimulus_name'])

bdata.save('test.h5')
