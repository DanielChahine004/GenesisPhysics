import genesis as gs
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# --- 1. DEFINE THE MODEL CLASS ---
# This must be defined before it is instantiated
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh() # Keeps output in range [-1, 1]
        )

    def forward(self, obs):
        return self.net(obs)

# --- 2. INITIALIZE GENESIS ---
writer = SummaryWriter("logs/robot_policy")
gs.init(backend=gs.gpu)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())

n_envs = 12 # Running 12 robots in parallel
robot = scene.add_entity(
    gs.morphs.URDF(
        file=r'C:\Users\h\Desktop\GenesisPhysics\onshape\urdf_output\robot.urdf',
        pos=(0, 0, 1.0),
        fixed=False
    )
)

scene.build(n_envs=n_envs, env_spacing=(2.0, 2.0))

# --- 3. IDENTIFY AND STABILIZE JOINTS ---
target_names = ['revolute_1', 'revolute_2']

actuated_indices = []
for name in target_names:
    joint = robot.get_joint(name)
    idx = joint.dofs_idx_local
    if isinstance(idx, slice):
        # Expand slice (e.g., slice(0, 1) -> [0])
        actuated_indices.extend(range(idx.start, idx.stop))
    elif isinstance(idx, (list, tuple)):
        actuated_indices.extend(idx)
    else:
        actuated_indices.append(idx)

actuated_indices.sort()

# Now get limits using the flattened list of integers
all_lowers, all_uppers = robot.get_dofs_limit()
joint_lowers = all_lowers[actuated_indices]
joint_uppers = all_uppers[actuated_indices]

# Snap joints to midpoint
init_pos = (joint_lowers + joint_uppers) / 2
robot.set_dofs_position(init_pos, actuated_indices)
robot.set_dofs_kp([150.0] * len(actuated_indices), actuated_indices)
robot.set_dofs_kv([15.0] * len(actuated_indices), actuated_indices)

# --- 4. INSTANTIATE THE MODEL ---
# Observation: 2 joint pos + 2 joint vel + 3 base lin vel = 7 total
obs_dim = 7
action_dim = 2 
policy = PolicyNet(obs_dim, action_dim).to(gs.device)

# --- 5. LOOP ---
for i in range(100000):
    scene.step()

    # A. Gather Observations
    # Tensors are (n_envs, dim)
    dof_pos = robot.get_dofs_position()[:, actuated_indices]   
    dof_vel = robot.get_dofs_velocity()[:, actuated_indices]   
    base_v  = robot.get_links_vel()[:, 0, :]                  # Base link linear velocity
    
    # Concatenate into one observation vector for the network
    obs = torch.cat([dof_pos, dof_vel, base_v], dim=-1)

    # B. Inference (No-grad because we aren't training yet)
    with torch.no_grad():
        # Policy takes (12, 7) and returns (12, 2)
        actions = policy(obs) 

    # C. Scale Actions from [-1, 1] to [joint_lowers, joint_uppers]
    target_pos = joint_lowers + (actions + 1.0) * 0.5 * (joint_uppers - joint_lowers)

    # D. Apply Control
    robot.control_dofs_position(target_pos, actuated_indices)

    # E. Log Swarm Progress
    if i % 100 == 0:
        avg_x_vel = base_v[:, 0].mean().item()
        writer.add_scalar("Vel/Avg_X", avg_x_vel, i)
        print(f"\rStep: {i} | Avg X-Vel: {avg_x_vel:.4f}", end="")