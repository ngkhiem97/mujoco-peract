from lxml import etree
from stl import mesh
import numpy as np

# parse xml file
tree = etree.parse('saves/robot_2023_1_16_13_52_17.xml')

# extract data from xml
vertices = tree.xpath('//vertex/text()')
faces = tree.xpath('//face/text()')

# convert data to numpy arrays
vertices = np.array(vertices).reshape(-1, 3)
faces = np.array(faces).reshape(-1, 3)

# create stl object
stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        stl_mesh.vectors[i][j] = vertices[f[j],:]

# save stl file
stl_mesh.save('saves/robot_2023_1_16_13_52_17.stl')
