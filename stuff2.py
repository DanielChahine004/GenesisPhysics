import genesis as gs
import numpy as np
import torch

gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=True)

plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(
    gs.morphs.URDF(
        file=r'C:\Users\h\Desktop\GenesisPhysics\onshape\urdf_output\robot.urdf'
    )
)
# scene.add_entity(
#     gs.morphs.MJCF(
#         file=r'C:\Users\h\Desktop\GenesisPhysics\onshape\my-robot\robot.xml',
#         fixed=False
#     )
# )

scene.build(n_envs=1)

print(f"Number of DOFs: {robot.n_dofs}")
print(f"Joint Names: {[j.name for j in robot.joints]}")

# 7 DOFs = [x, y, z, quat_w, quat_x, quat_y, quat_z] for freejoint + [hinge_angle]
# Wait, that's 8... Let me check the actual representation

# Get current DOF positions to see the structure
current = robot.get_dofs_position()
print(f"Current DOF shape: {current.shape}")
print(f"Current DOF values: {current}")

for i in range(10000):
    # Get current state
    dofs = robot.get_dofs_position()
    
    # Modify only the last DOF (hinge joint)
    hinge_angle = np.sin(i * 0.01) * 1.5
    dofs[-1] = hinge_angle
    
    # Convert to torch tensor if needed
    if isinstance(dofs, np.ndarray):
        dofs = torch.from_numpy(dofs).cuda()
    
    robot.control_dofs_position(dofs)
    scene.step()