import numpy as np
import matplotlib.pyplot as plt

def homogeneous_model(nx, nz, velocity):
    vel = np.full((nz, nx), velocity, dtype=float)
    return vel

def two_layer_model(nx,nz,dz,interface_depth,velocity_top,velocity_bottom):
    vel = np.full((nz, nx),velocity_top,dtype=float)
    iz = int(interface_depth / dz)
    vel[iz:, :] = velocity_bottom
    return vel

def layered_model(nx,nz,dz,interfaces,velocities):
    '''
    Parameters
    nx, nz : int
        Grid size.
    dz : float
        Vertical grid spacing, m.
    interfaces : list
        Interface depths, m.
    velocities : list
        Layer velocities, m/s.

    Returns
    -------
    vel : ndarray
        2D velocity model.
    ''' 
    if len(velocities) != len(interfaces) + 1:
        raise ValueError("len(velocities) must equal len(interfaces) + 1")
    
    vel = np.full((nz, nx),velocities[0],dtype=float)
    for depth, velocity in zip(interfaces,velocities[1:]):
        iz = int(depth / dz)
        vel[iz:, :] = velocity
    return vel

def fault_model(nx,nz,dx,dz,interfaces,velocities,fault_x,fault_throw):
    if len(velocities) != len(interfaces) + 1:
        raise ValueError("len(velocities) must equal len(interfaces) + 1")
    vel = np.zeros((nz, nx), dtype=float)
    x = np.arange(nx) * dx
    z = np.arange(nz) * dz
    for ix in range(nx):
        # Left side: original interfaces
        if x[ix] < fault_x:
            current_interfaces = interfaces

        # Right side: interfaces displaced downward
        else:
             current_interfaces = [
                depth + fault_throw
                for depth in interfaces
            ]
             
        for iz in range(nz):
            depth = z[iz]
            layer = 0
            while (
                layer < len(current_interfaces)
                and depth >= current_interfaces[layer]
            ):
                layer += 1
            vel[iz, ix] = velocities[layer]
    return vel

def linear_gradient_model(nx,nz,dz,velocity_top,gradient):
    """
    Generate a velocity model increasing linearly with depth.
    """
    z = np.arange(nz) * dz
    velocity_z = velocity_top+ gradient * z
    vel = np.repeat(velocity_z[:, np.newaxis],nx,axis=1)
    return vel

def circular_anomaly_model(nx,nz,dx,dz,background_velocity,anomaly_velocity,center_x,center_z,radius):
    x = np.arange(nx) * dx
    z = np.arange(nz) * dz
    X, Z = np.meshgrid(x, z)
    vel = np.full((nz, nx),background_velocity,dtype=float)
    mask = (X - center_x)**2 + (Z - center_z)**2<= radius**2
    vel[mask] = anomaly_velocity
    return vel

def rectangular_anomaly_model(nx,nz,dx,dz,background_velocity,anomaly_velocity,x_min,x_max,z_min,z_max):
    vel = np.full((nz, nx),background_velocity,dtype=float)
    ix1 = int(x_min / dx)
    ix2 = int(x_max / dx)
    iz1 = int(z_min / dz)
    iz2 = int(z_max / dz)
    vel[iz1:iz2, ix1:ix2] = anomaly_velocity
    return vel

def dipping_layer_model(nx,nz,dx,dz,velocity_top,velocity_bottom,depth0,angle):
    vel = np.full((nz, nx),velocity_top,dtype=float)
    theta = np.deg2rad(angle)
    for ix in range(nx):
        x = ix * dx
        interface_depth = depth0 + x * np.tan(theta)
        iz = int(interface_depth / dz)
        if 0 <= iz < nz:
            vel[iz:, ix] = velocity_bottom
    return vel

def sinusoidal_interface_model(nx,nz,dx,dz,velocity_top,velocity_bottom,depth0,amplitude,wavelength):
    vel = np.full((nz, nx),velocity_top,dtype=float)
    for ix in range(nx):
        x = ix * dx
        interface_depth = depth0 + amplitude * np.sin(2.0 * np.pi * x / wavelength)
        iz = int(interface_depth / dz)
        if 0 <= iz < nz:
            vel[iz:, ix] = velocity_bottom
    return vel

def depression_model(nx,nz,dx,dz,velocity_top,velocity_bottom,depth0,center_x,depth_amplitude,width):
    vel = np.full((nz, nx),velocity_top,dtype=float)
    for ix in range(nx):
        x = ix * dx
        interface_depth = depth0+ depth_amplitude* np.exp(-(x - center_x)**2/ (2.0 * width**2))
        iz = int(interface_depth / dz)
        if 0 <= iz < nz:
            vel[iz:, ix] = velocity_bottom
    return vel

def salt_model(nx,nz,dx,dz,background_velocity=2500.0,salt_velocity=4500.0,center_x=None,center_z=None,radius_x=800.0,radius_z=1200.0):
    x = np.arange(nx) * dx
    z = np.arange(nz) * dz
    X, Z = np.meshgrid(x, z)
    if center_x is None:
        center_x = 0.5 * (nx - 1) * dx

    if center_z is None:
        center_z = 0.5 * (nz - 1) * dz

    vel = np.full((nz, nx),background_velocity,dtype=float)
    salt_mask = (((X - center_x) / radius_x) ** 2 + ((Z - center_z) / radius_z) ** 2 <= 1.0)
    vel[salt_mask] = salt_velocity
    return vel

def checkerboard_model(nx,nz,dx,dz,background_velocity=2500.0,velocity_perturbation=300.0,block_width=500.0,block_height=500.0):
    vel = np.zeros((nz, nx), dtype=float)
    block_nx = max(1, int(block_width / dx))
    block_nz = max(1, int(block_height / dz))
    for iz in range(nz):
        for ix in range(nx):
            block_x = ix // block_nx
            block_z = iz // block_nz
            if (block_x + block_z) % 2 == 0:
                vel[iz, ix] = background_velocity + velocity_perturbation
            else:
                vel[iz, ix] = background_velocity - velocity_perturbation
    return vel