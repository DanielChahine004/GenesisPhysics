import genesis as gs
import torch
from torch.utils.tensorboard import SummaryWriter

# 1. Initialize
writer = SummaryWriter("logs/robot_metrics")
gs.init(backend=gs.gpu)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())

robot = scene.add_entity(
    gs.morphs.URDF(
        file=r'C:\Users\h\Desktop\GenesisPhysics\onshape\urdf_output\robot.urdf',
        pos=(0, 0, 1.0),
        fixed=False # Floating base (adds 6 DOFs: 0-5)
    )
)

scene.build(n_envs=1)

# --- NAME ENFORCEMENT ---
target_names = ['revolute_1', 'revolute_2']
actuated_indices = []

# In Genesis 0.3.13, joints are accessed via the .joints list
for joint in robot.joints:
    if joint.name in target_names:
        actuated_indices.append(joint.dof_idx_local)

# Diagnostic if names aren't found
if len(actuated_indices) != len(target_names):
    all_names = [j.name for j in robot.joints]
    print(f"Available joints in URDF: {all_names}")
    raise ValueError(f"Could not find joints {target_names}. Check the names above.")

print(f"Mapped {target_names} to indices: {actuated_indices}")

# --- STABILIZATION ---
# Get limits for the whole robot (length 8)
all_lowers, all_uppers = robot.get_dofs_limit()

# Calculate safe midpoint for our specific joints (around 5.23 rad)
# We index the limit tensors using our found actuated_indices
init_pos = (all_lowers[actuated_indices] + all_uppers[actuated_indices]) / 2

# Snap the joints to the valid range BEFORE the first physics step
# This prevents the massive 'snap' force that causes the explosion
robot.set_dofs_position(init_pos, actuated_indices)

# Set gains only for these joints. 
# Applying KP to indices 0-5 (floating base) is a common cause of crashes.
robot.set_dofs_kp([150.0] * len(actuated_indices), actuated_indices)
robot.set_dofs_kv([15.0] * len(actuated_indices), actuated_indices)

# --- MAIN LOOP ---
for i in range(50000):
    scene.step()

    # Metrics: Linear velocity of the base link (link 0)
    links_vel = robot.get_links_vel() 
    x_vel = links_vel[0, 0, 0].item() 

    if i % 100 == 0:
        writer.add_scalar("Vel/X", x_vel, i)
        print(f"\rStep: {i} | X-Vel: {x_vel:.4f}")