import numpy as np
import trimesh
import mcubes
import mesh_to_sdf
from pyvirtualdisplay import Display  # Import the Display class

def create_uniform_grid(density=256, low_limit=-1.0, high_limit=1.0):
    """Create a uniform grid within a cube."""
    gap = (high_limit - low_limit) / density
    x = np.linspace(low_limit, high_limit, density+1)
    y = np.linspace(low_limit, high_limit, density+1)
    z = np.linspace(low_limit, high_limit, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)
    return grid, gap

def compute_sdf_for_grid(mesh, grid):
    """Compute SDF for each point in the grid."""
    sdf_values = mesh_to_sdf.mesh_to_sdf(mesh, grid)
    return sdf_values

def reconstruct_mesh_from_sdf(sdf_values, density, gap):
    """Reconstruct mesh from SDF values using marching cubes."""
    volume = sdf_values.reshape(density+1, density+1, density+1)
    verts, faces = mcubes.marching_cubes(volume, 0)
    verts *= gap
    verts -= 1.0  # Adjust to the original coordinate system
    mesh = trimesh.Trimesh(verts, faces)
    return mesh

def main():
    # Start a virtual display
    display = Display(visible=0, size=(1024, 768))
    display.start()

    # Load the mesh
    mesh_path = "/data7/haolin/TeethData/RD_3/watertight_meshes/100_24.obj"  # Replace with your mesh path
    mesh = trimesh.load(mesh_path)

    # Create a uniform grid
    density = 256
    grid, gap = create_uniform_grid(density)

    # Compute SDF for the grid
    sdf_values = compute_sdf_for_grid(mesh, grid)

    # Reconstruct the mesh from SDF values
    reconstructed_mesh = reconstruct_mesh_from_sdf(sdf_values, density, gap)

    # Export the reconstructed mesh
    reconstructed_mesh.export("reconstructed_mesh.obj")
    print("Reconstructed mesh saved as 'reconstructed_mesh.obj'")

    # Stop the virtual display
    display.stop()

if __name__ == "__main__":
    main()