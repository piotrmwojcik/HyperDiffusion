import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the point cloud data from the .npy file
pc = np.load('/Users/piotrwojcik/Downloads/ssdnerf_baseline/460f2b6d8f4dc18d565895440030d853.obj.npy')

# Extract points (x, y, z) and occupancies (occs)
pts = pc[:, :3]      # Assuming the first three columns are x, y, z coordinates
occs = pc[:, 3]      # Assuming the fourth column is occupancy

# Filter points with x > 0
filled_points = pts[occs[:] > 0]

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot the filtered points
ax.scatter(filled_points[:, 0], filled_points[:, 1], filled_points[:, 2],
           c='blue', s=1, label='Filled (1)')

# Set plot limits
ax.set_xlim([-0.5, 0.5])
ax.set_ylim([-0.5, 0.5])
ax.set_zlim([-0.5, 0.5])

# Label the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()