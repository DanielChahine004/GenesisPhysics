import genesis as gs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# --- 1. STOCHASTIC ACTOR ---
class GroupActor(nn.Module):
    def __init__(self, single_obs_dim, action_dim, history_len=30):
        super().__init__()
        self.input_dim = single_obs_dim * history_len
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
        )
        self.mu = nn.Linear(128, action_dim)
        with torch.no_grad():
            self.mu.weight.fill_(0.0)
            self.mu.bias.fill_(0.0)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0)

    def forward(self, obs_history):
        x = self.net(obs_history)
        mu = torch.tanh(self.mu(x)) 
        std = torch.exp(self.log_std)
        return Normal(mu, std)

# --- 2. SETUP GENESIS ---
gs.init(backend=gs.gpu)

# FIX 1: Change dt to 0.02. 
# dt=0.02 / substeps=10 = 0.002s (2ms). This fixes the GJK Collision instability warning!
sim_options = gs.options.SimOptions(dt=0.02, substeps=10)
scene = gs.Scene(sim_options=sim_options, show_viewer=True)
scene.add_entity(gs.morphs.Plane())

robot = scene.add_entity(
    gs.morphs.URDF(
        file=r'C:\Users\h\Desktop\GenesisPhysics\onshape\urdf_output\robot.urdf',
        fixed=False
    )
)

n_envs = 16 
scene.build(n_envs=n_envs, env_spacing=(1.0, 1.0))

# FIX 2: Raise spawn height to 0.5 so the robot doesn't clip into the floor on frame 1
home_pos = torch.tensor([0.0, 0.0, 0.5], device=gs.device)
home_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)

target_names = ['revolute_1', 'revolute_2', 'revolute_3', 'revolute_4']

actuated_indices = []
for name in target_names:
    idx = robot.get_joint(name).dofs_idx_local
    actuated_indices.append(idx.start if isinstance(idx, slice) else idx[0])

# FIX 3: Clamp joint limits to valid radians (-3.14 to 3.14) just in case your URDF 
# has infinite limits. (Infinite limits cause NaN math errors during Action Selection).
all_lowers, all_uppers = robot.get_dofs_limit()
j_low = torch.clamp(all_lowers[actuated_indices], -3.14, 3.14)
j_high = torch.clamp(all_uppers[actuated_indices], -3.14, 3.14)
j_mid = (j_low + j_high) / 2.0

robot.set_dofs_kp([400.0] * len(actuated_indices), actuated_indices) 
robot.set_dofs_kv([20.0] * len(actuated_indices), actuated_indices)

# FIX 4: Force the robot into a safe, legal pose BEFORE the first scene.step()
robot.set_pos(home_pos.repeat(n_envs, 1))
robot.set_quat(home_quat.repeat(n_envs, 1))
robot.set_dofs_position(j_mid.repeat(n_envs, 1), actuated_indices)

# The physics engine can now step safely!
scene.step()

# --- 3. INDEXING ---
for i, link in enumerate(robot.links):
    print(f"Link Index {i}: {link.name}")

# Based on your print log, Link 4 is 'part_8_1'
part_3_idx = 4 

# --- 4. TRAINING HELPERS ---
def get_single_obs():
    # Use torch.nan_to_num to prevent initial NaNs from crashing the net
    d_pos = torch.nan_to_num(robot.get_dofs_position()[:, actuated_indices])
    d_vel = torch.clamp(torch.nan_to_num(robot.get_dofs_velocity()[:, actuated_indices]), -20, 20)
    
    # Base velocity (Link 0)
    b_vel = torch.clamp(torch.nan_to_num(robot.get_links_vel()[:, 0, :3]), -20, 20)
    
    # Part 3 position
    p3_pos = torch.nan_to_num(robot.get_links_pos()[:, part_3_idx, :]) 
    
    return torch.cat([d_pos, d_vel, b_vel, p3_pos], dim=-1)


# Now get the sample observation
sample_obs = get_single_obs()
single_obs_dim = sample_obs.shape[-1] 
action_dim = len(actuated_indices)
history_len = 50

# --- 5. TRAINING UPDATES ---
# Lower the learning rate
actor = GroupActor(single_obs_dim, action_dim, history_len).to(gs.device)
optimizer = optim.Adam(actor.parameters(), lr=3e-4) # Changed from 3e-3

def reset_history(envs_idx=None):
    # Create flat history buffer [batch, single_obs * history_len]
    init_obs = get_single_obs().repeat(1, history_len)
    return init_obs if envs_idx is None else init_obs[envs_idx]

def check_termination():
    p3_height = robot.get_links_pos()[:, part_3_idx, 2]
    return p3_height < 0.05 

# --- 5. TRAINING LOOP ---
for epoch in range(2000):
    log_probs, rewards, entropies = [], [], []
    
    # Global Initial Reset
    robot.set_pos(home_pos.repeat(n_envs, 1))
    robot.set_quat(home_quat.repeat(n_envs, 1))
    robot.zero_all_dofs_velocity()
    robot.set_dofs_position(j_mid.repeat(n_envs, 1), actuated_indices)
    obs_history = reset_history()
    
    for t in range(200):
        # 1. Action Selection
        dist = actor(obs_history)
        actions = dist.rsample()
        
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # 2. Step Simulation
        target_pos = j_low + (torch.tanh(actions) + 1.0) * 0.5 * (j_high - j_low)
        robot.control_dofs_position(target_pos, actuated_indices)
        scene.step()

        # 3. Individual Reset Logic
        terminated = check_termination()
        envs_to_reset = torch.where(terminated)[0]

        if len(envs_to_reset) > 0:
            robot.set_pos(home_pos.repeat(len(envs_to_reset), 1), envs_idx=envs_to_reset)
            robot.set_quat(home_quat.repeat(len(envs_to_reset), 1), envs_idx=envs_to_reset)
            robot.zero_all_dofs_velocity(envs_idx=envs_to_reset)
            robot.set_dofs_position(j_mid.repeat(len(envs_to_reset), 1), actuated_indices, envs_idx=envs_to_reset)
            
            # Reset history for dead environments
            fresh_hists = reset_history(envs_to_reset)
            obs_history = obs_history.clone() 
            obs_history[envs_to_reset] = fresh_hists

        # 4. Update History Buffer (Shift left and append new)
        new_obs = get_single_obs()
        obs_history = torch.cat([obs_history[:, single_obs_dim:], new_obs], dim=-1)

        # 5. Track rewards
        reward = torch.where(terminated, -1.0, 1.0)
        log_probs.append(log_prob)
        rewards.append(reward)
        entropies.append(entropy)

    # --- 6. UPDATE BRAIN ---
    log_probs = torch.stack(log_probs)
    rewards = torch.stack(rewards)
    entropies = torch.stack(entropies)

    baseline = rewards.mean(dim=1, keepdim=True) 
    advantage = (rewards - baseline) / (rewards.std() + 1e-6)
    loss = -(log_probs * advantage.detach()).mean() - (0.02 * entropies).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    optimizer.step()

    if epoch % 5 == 0:
        avg_survival = rewards.sum(dim=0).mean().item()
        print(f"Epoch {epoch} | Avg Survival Time: {avg_survival:.1f} steps")