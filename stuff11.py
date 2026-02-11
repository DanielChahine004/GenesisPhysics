import genesis as gs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# --- 1. STOCHASTIC ACTOR ---
class GroupActor(nn.Module):
    def __init__(self, single_obs_dim, action_dim, history_len=50):
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
sim_options = gs.options.SimOptions(dt=0.02, substeps=10) # Stable 2ms substep
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

# --- 3. CONFIGURATION & INDEXING ---
home_pos = torch.tensor([0.0, 0.0, 0.1], device=gs.device)
home_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)

# PER-JOINT INITIALIZATION CONFIG
# Options: 'min', 'max', 'mid'
joint_init_config = {
    'revolute_1': 'max',
    'revolute_2': 'max',
    'revolute_3': 'mid',
    'revolute_4': 'mid',
}

# --- 3. CONFIGURATION & INDEXING (Updated) ---
target_names = joint_init_config.keys()
actuated_indices = []
init_joint_list = []
low_list = []
high_list = []

all_lowers, all_uppers = robot.get_dofs_limit()

for name in target_names:
    idx = robot.get_joint(name).dofs_idx_local
    idx_val = idx.start if isinstance(idx, slice) else idx[0]
    actuated_indices.append(idx_val)
    
    # Better logic for infinite/continuous joints
    raw_low = all_lowers[idx_val]
    raw_high = all_uppers[idx_val]
    
    # If limit is effectively infinite, use -pi to pi
    low = raw_low if raw_low > -7.0 else torch.tensor(-3.1415, device=gs.device)
    high = raw_high if raw_high < 7.0 else torch.tensor(3.1415, device=gs.device)
    
    low_list.append(low)
    high_list.append(high)
    
    # Determine Init Position
    mode = joint_init_config.get(name, 'mid')
    if mode == 'min':
        init_joint_list.append(low + 0.05)
    elif mode == 'max':
        init_joint_list.append(high - 0.05)
    else: # mid
        init_joint_list.append((low + high) / 2.0)

# Final tensors
j_init = torch.stack(init_joint_list).to(gs.device)
j_low = torch.stack(low_list).to(gs.device)
j_high = torch.stack(high_list).to(gs.device)

# Set Gains
robot.set_dofs_kp([400.0] * len(actuated_indices), actuated_indices) 
robot.set_dofs_kv([20.0] * len(actuated_indices), actuated_indices)

# --- 4. INITIALIZE SIMULATION STATE ---
# Force legal state before first step
robot.set_pos(home_pos.repeat(n_envs, 1))
robot.set_quat(home_quat.repeat(n_envs, 1))
robot.set_dofs_position(j_init.repeat(n_envs, 1), actuated_indices)
scene.step()

# Print all detected links to see what Genesis kept
print("Detected links in Genesis:")
for i, link in enumerate(robot.links):
    print(f"Index {i}: {link.name}")

head_link = robot.get_link('part_9_1')
head_idx = head_link.idx - robot.link_start

print(f"DEBUG: Global link index: {head_link.idx}, Robot link start: {robot.link_start}, Calculated local index: {head_idx}")

# --- 5. TRAINING HELPERS ---
def get_single_obs():
    d_pos = torch.nan_to_num(robot.get_dofs_position()[:, actuated_indices])
    d_vel = torch.clamp(torch.nan_to_num(robot.get_dofs_velocity()[:, actuated_indices]), -20, 20)
    b_vel = torch.clamp(torch.nan_to_num(robot.get_links_vel()[:, 0, :3]), -20, 20)
    head_pos = torch.nan_to_num(robot.get_links_pos()[:, head_idx, :]) 
    return torch.cat([d_pos, d_vel, b_vel, head_pos], dim=-1)

sample_obs = get_single_obs()
single_obs_dim = sample_obs.shape[-1] 
action_dim = len(actuated_indices)
history_len = 50

actor = GroupActor(single_obs_dim, action_dim, history_len).to(gs.device)
optimizer = optim.Adam(actor.parameters(), lr=3e-4)

def reset_history(envs_idx=None):
    init_obs = get_single_obs().repeat(1, history_len)
    return init_obs if envs_idx is None else init_obs[envs_idx]

def check_termination():
    head_height = robot.get_links_pos()[:, head_idx, 2]
    return head_height < 0.05 

# --- 5.5 DIAGNOSTIC SUMMARY & USER CONFIRMATION ---
def print_model_diagnostics():
    print("\n" + "="*60)
    print("      GENESIS ROBOT & NEURAL NETWORK DIAGNOSTICS")
    print("="*60)
    
    # 1. Robot Structural Info
    print(f"\n[ROBOT STRUCTURE]")
    print(f"  - Total Links: {robot.n_links}")
    print(f"  - Total DOFs:  {robot.n_dofs}")
    
    # CORRECTED METHOD NAME: get_links_inertial_mass()
    link_masses = robot.get_links_inertial_mass() 
    
    print(f"  - Link Names:")
    for i, link in enumerate(robot.links):
        m = link_masses[i].item()
        # Highlight the link being used as the 'head'
        head_tag = " <--- [HEAD TRACKING]" if i == head_idx else ""
        print(f"    {i:2}: {link.name:15} (Mass: {m:.3f}kg){head_tag}")

    # 2. Actuation Info
    print(f"\n[ACTUATION]")
    print(f"  - Target Actuated Joints: {list(target_names)}")
    print(f"  - Actuated Indices (Local): {actuated_indices}")
    print(f"  - Joint Limits & Init (Effective):")
    for i, name in enumerate(target_names):
        print(f"    {name:10}: Low={j_low[i]:.3f}, High={j_high[i]:.3f}, Init={j_init[i]:.3f}")
        if j_low[i] >= j_high[i]:
            print(f"      ⚠️ WARNING: Joint '{name}' is locked (Low >= High)!")

    # 3. Physical Properties
    total_mass = torch.sum(link_masses).item()
    print(f"\n[PHYSICAL PROPERTIES]")
    print(f"  - Total Robot Mass: {total_mass:.4f} kg")
    print(f"  - Spawn Height:     {home_pos[2].item()} m")

    # 4. Neural Network Info
    print(f"\n[POLICY NETWORK]")
    print(f"  - Input Dimension:  {actor.input_dim}")
    print(f"    (History: {history_len} steps x Single Obs: {single_obs_dim})")
    print(f"  - Output Dimension: {action_dim}")
    
    # Check for NaNs in NN
    test_obs = torch.zeros((1, actor.input_dim), device=gs.device)
    with torch.no_grad():
        test_dist = actor(test_obs)
        test_mu = test_dist.mean
    print(f"  - Initialization Check: {'PASSED' if not torch.isnan(test_mu).any() else 'FAILED (NaN detected)'}")

    print("\n" + "="*60)
    confirm = input("Configurations loaded. Proceed with training? (y/n): ")
    if confirm.lower() != 'y':
        print("Training cancelled by user.")
        exit()

print_model_diagnostics()

# --- 6. TRAINING LOOP ---
for epoch in range(2000):
    log_probs, rewards, entropies = [], [], []
    
    # GLOBAL RESET
    robot.set_pos(home_pos.repeat(n_envs, 1))
    robot.set_quat(home_quat.repeat(n_envs, 1))
    robot.zero_all_dofs_velocity()
    robot.set_dofs_position(j_init.repeat(n_envs, 1), actuated_indices)
    obs_history = reset_history()
    
    for t in range(200):
        # 1. Action Selection
        dist = actor(obs_history)
        actions = dist.rsample()
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # 2. Step Simulation
        # Scale actions to joint limits
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
            robot.set_dofs_position(j_init.repeat(len(envs_to_reset), 1), actuated_indices, envs_idx=envs_to_reset)
            
            fresh_hists = reset_history(envs_to_reset)
            obs_history = obs_history.clone() 
            obs_history[envs_to_reset] = fresh_hists

        # 4. Update History Buffer
        new_obs = get_single_obs()
        obs_history = torch.cat([obs_history[:, single_obs_dim:], new_obs], dim=-1)

        # 5. Track rewards
        reward = torch.where(terminated, -1.0, 1.0)
        log_probs.append(log_prob)
        rewards.append(reward)
        entropies.append(entropy)

    # --- 7. UPDATE BRAIN ---
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

    if epoch % 2 == 0:
        avg_survival = rewards.sum(dim=0).mean().item()
        print(f"Epoch {epoch} | Avg Survival Time: {avg_survival:.1f} steps")