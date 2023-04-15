'''
'''

# import meshio
import h5py as h5

import numpy as np
import torch

class Elements(torch.nn.Module):

     def __init__(self, element_pos, element_ind, bd_point_ind=None):
         super().__init__()

         self.element_pos = element_pos
         self.element_ind = element_ind
         self.bd_point_ind = bd_point_ind

         return

'''
Responsible for loading mesh structure.
'''
class MeshLoader():
    def __init__(self, mesh_file):

        self.mesh_file = mesh_file

    def load_mesh(self):

        elements = None

        try:
            with h5.File(self.mesh_file, 'r') as file:

                points = torch.from_numpy(file["points"][...].astype(np.float32))

                if 'elements' in file.keys():
                    element_pos = file['elements']["element_positions"][...]
                    element_ind = file['elements']["element_indices"][...]

                    if "boundary_point_indices" in file['elements'].keys():
                        bd_point_ind = file['elements']["boundary_point_indices"][...]
                    else:
                        bd_point_ind = None

                    elements = Elements(element_pos, element_ind, bd_point_ind)

        except Exception as e:
            raise e

        return points, elements
