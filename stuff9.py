import genesis as gs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# --- 1. STOCHASTIC ACTOR (History-Aware) ---
class GroupActor(nn.Module):
    def __init__(self, single_obs_dim, action_dim, history_len=3):
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
sim_options = gs.options.SimOptions(dt=0.01, substeps=1)
scene = gs.Scene(sim_options=sim_options, show_viewer=True)
scene.add_entity(gs.morphs.Plane())

n_envs = 16 
robot = scene.add_entity(
    gs.morphs.URDF(
        file=r'C:\Users\h\Desktop\GenesisPhysics\onshape\urdf_output\robot.urdf',
        fixed=False
    )
)
scene.build(
    n_envs=n_envs,
    env_spacing=(1.0, 1.0)
    )

# --- 3. INDEXING AND GAINS ---
target_names = ['revolute_1', 'revolute_2']
actuated_indices = []
for name in target_names:
    idx = robot.get_joint(name).dofs_idx_local
    actuated_indices.append(idx.start if isinstance(idx, slice) else idx[0])

# Fast Actuators
robot.set_dofs_kp([400.0] * len(actuated_indices), actuated_indices) 
robot.set_dofs_kv([20.0] * len(actuated_indices), actuated_indices)

all_lowers, all_uppers = robot.get_dofs_limit()
j_low, j_high = all_lowers[actuated_indices], all_uppers[actuated_indices]
j_mid = (j_low + j_high) / 2.0

# --- 4. TRAINING HELPERS ---
single_obs_dim = 7
action_dim = len(actuated_indices)
history_len = 3

actor = GroupActor(single_obs_dim, action_dim, history_len).to(gs.device)
optimizer = optim.Adam(actor.parameters(), lr=3e-4)

def get_single_obs():
    d_pos = robot.get_dofs_position()[:, actuated_indices]
    d_vel = torch.clamp(robot.get_dofs_velocity()[:, actuated_indices], -20, 20)
    b_vel = torch.clamp(robot.get_links_vel()[:, 0, :], -20, 20)
    return torch.cat([d_pos, d_vel, b_vel], dim=-1)

def reset_history(envs_idx=None):
    """Resets history for either all envs or specific indices."""
    init_obs = get_single_obs().repeat(1, history_len)
    return init_obs if envs_idx is None else init_obs[envs_idx]

# --- PLACEHOLDER FAILURE FUNCTION (DISABLED) ---
def check_termination():
    """Returns a boolean tensor where True means 'Reset'. Currently all False."""
    return torch.zeros(n_envs, dtype=torch.bool, device=gs.device)


# --- 1. DEFINE YOUR STARTING POSE ONCE ---
# Position: (x=0, y=0, z=0.5)
home_pos = torch.tensor([0.0, 0.0, 0.1], device=gs.device)
# Identity quaternion [W=1, X=0, Y=0, Z=0] means no rotation
home_quat = torch.tensor([1.0, 0.5, 0.0, 0.0], device=gs.device)

# --- 5. TRAINING LOOP ---
for epoch in range(2000):
    log_probs, rewards, entropies = [], [], []
    
    # A. GLOBAL RESET: Force all robots to exact Home Pose at start of epoch
    robot.set_pos(home_pos.repeat(n_envs, 1))
    robot.set_quat(home_quat.repeat(n_envs, 1))
    robot.zero_all_dofs_velocity()
    robot.set_dofs_position(j_mid.repeat(n_envs, 1), actuated_indices)
    obs_history = reset_history()
    
    for t in range(200):
        dist = actor(obs_history)
        actions = dist.rsample()
        
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        target_pos = j_low + (torch.tanh(actions) + 1.0) * 0.5 * (j_high - j_low)
        robot.control_dofs_position(target_pos, actuated_indices)
        
        scene.step()

        # B. INDIVIDUAL RESET: Force failed robots back to exact Home Pose
        terminated = check_termination()
        envs_to_reset = torch.where(terminated)[0]

        if len(envs_to_reset) > 0:
            # Re-apply exact same position and orientation only for these indices
            robot.set_pos(home_pos.repeat(len(envs_to_reset), 1), envs_idx=envs_to_reset)
            robot.set_quat(home_quat.repeat(len(envs_to_reset), 1), envs_idx=envs_to_reset)
            
            robot.zero_all_dofs_velocity(envs_idx=envs_to_reset)
            robot.set_dofs_position(j_mid.repeat(len(envs_to_reset), 1), actuated_indices, envs_idx=envs_to_reset)
            
            obs_history[envs_to_reset] = reset_history(envs_to_reset)

        # Reward: Forward velocity
        reward = robot.get_links_vel()[:, 0, 0]
        
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
        print(f"Epoch {epoch} | Avg Reward: {baseline.mean().item():.3f}")