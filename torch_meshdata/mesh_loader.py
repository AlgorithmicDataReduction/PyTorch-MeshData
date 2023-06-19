'''
'''

import pathlib
import meshio
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

        return
    
    def load_numpy(self):

        points = np.load(self.mesh_file).astype(np.float32)

        return points, None

    def load_hdf5(self):

        with h5.File(self.mesh_file, 'r') as file:

            points = file["points"][...].astype(np.float32)
            elements = None

            if 'elements' in file.keys():
                element_pos = file['elements']["element_positions"][...]
                element_ind = file['elements']["element_indices"][...]

                if "boundary_point_indices" in file['elements'].keys():
                    bd_point_ind = file['elements']["boundary_point_indices"][...]
                else:
                    bd_point_ind = None

                elements = Elements(element_pos, element_ind, bd_point_ind)

        return points, elements

    def load_meshio(self):
        
        mesh = meshio.read(self.mesh_file)

        element_pos = [0]
        element_ind = []

        for cell_block in mesh.cells:
            for cell in cell_block.data:
                element_pos.append(element_pos[-1]+len(cell))
                element_ind.extend(cell)

        if "boundary" in mesh.point_data.keys():
            bd_point_ind = np.nonzero(mesh.point_data["boundary"])
        else:
            bd_point_ind = None

        points = mesh.points
        elements = Elements(element_pos, element_ind, bd_point_ind)

        return points, elements

    def load_mesh(self):

        try:
            ext = pathlib.Path(self.mesh_file).suffix

            if ext == ".npy":
                points, elements = self.load_numpy()
            elif ext == ".hdf5":
                points, elements = self.load_hdf5()
            else:
                points, elements = self.load_meshio()

        except Exception as e:
            raise e

        return torch.from_numpy(points), elements
