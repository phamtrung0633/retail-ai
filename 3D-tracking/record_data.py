import numpy as np
def estimate_velocity_3d(timestamps, positions):
    # Ensure consistent shapes
    timestamps = np.asarray(timestamps)
    positions = np.asarray(positions)
    if len(timestamps) != positions.shape[0]:
        raise ValueError("Timestamps and positions must have the same number of samples.")

    # Design matrix for linear regression in each dimension
    design_matrix = np.vstack([np.ones_like(timestamps), timestamps]).T

    # Estimate velocities for each dimension independently
    velocities = np.zeros(3)
    for i in range(3):
        params, _, _, _ = np.linalg.lstsq(design_matrix, positions[:, i], rcond=None)
        velocities[i] = params[1]  # Extract slope coefficient (velocity)

    return velocities

positions = [[15, 17, 20], [16, 19, 22], [19, 21, 23], [21, 23, 25], [23, 25, 27]]
timestamps = [7, 8, 11.5]
timestamps2 = [7, 8, 11.5]
print(np.array(timestamps) + (np.array(timestamps2) * 2))