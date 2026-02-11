import genesis as gs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# --- 1. STOCHASTIC ACTOR (Modified for History) ---
class GroupActor(nn.Module):
    def __init__(self, single_obs_dim, action_dim, history_len=3):
        super().__init__()
        # Input dimension is now 7 * 3 = 21
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
        # obs_history shape: [n_envs, 21]
        x = self.net(obs_history)
        mu = torch.tanh(self.mu(x)) 
        std = torch.exp(self.log_std)
        return Normal(mu, std)

# --- 2. SETUP GENESIS ---
gs.init(backend=gs.gpu)
sim_options = gs.options.SimOptions(dt=0.01, substeps=10)
scene = gs.Scene(sim_options=sim_options, show_viewer=True)
scene.add_entity(gs.morphs.Plane())

n_envs = 16 
robot = scene.add_entity(
    gs.morphs.URDF(
        file=r'C:\Users\h\Desktop\GenesisPhysics\onshape\urdf_output\robot.urdf', 
        pos=(0,0,0.5), 
        fixed=False
    )
)
scene.build(n_envs=n_envs)

# --- 3. INDEXING ---
target_names = ['revolute_1', 'revolute_2']
actuated_indices = []
for name in target_names:
    idx = robot.get_joint(name).dofs_idx_local
    actuated_indices.append(idx.start if isinstance(idx, slice) else idx[0])

# kp: stiffness (snap to target), kv: damping (prevent jitter)
robot.set_dofs_kp([400.0] * len(actuated_indices), actuated_indices) 
robot.set_dofs_kv([20.0] * len(actuated_indices), actuated_indices)

all_lowers, all_uppers = robot.get_dofs_limit()
j_low, j_high = all_lowers[actuated_indices], all_uppers[actuated_indices]
j_mid = (j_low + j_high) / 2.0

# --- 4. TRAINING SETUP (Modified) ---
single_obs_dim = 7
action_dim = len(actuated_indices)
history_len = 3

actor = GroupActor(single_obs_dim, action_dim, history_len).to(gs.device)
optimizer = optim.Adam(actor.parameters(), lr=3e-4)
entropy_coeff = 0.02

def get_single_obs():
    d_pos = robot.get_dofs_position()[:, actuated_indices]
    d_vel = torch.clamp(robot.get_dofs_velocity()[:, actuated_indices], -10, 10)
    b_vel = torch.clamp(robot.get_links_vel()[:, 0, :], -10, 10)
    return torch.cat([d_pos, d_vel, b_vel], dim=-1)

# Helper to initialize or reset history buffer
def reset_history():
    # Fill history with current state repeated 3 times
    init_obs = get_single_obs()
    return init_obs.repeat(1, history_len)

# --- 5. TRAINING LOOP ---
for epoch in range(2000):
    log_probs, rewards, entropies = [], [], []
    
    # Initialize history at start of epoch
    obs_history = reset_history()
    
    for t in range(200):
        # Actor takes the stacked history [n_envs, 21]
        dist = actor(obs_history)
        actions = dist.rsample()
        
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        target_pos = j_low + (torch.tanh(actions) + 1.0) * 0.5 * (j_high - j_low)
        robot.control_dofs_position(target_pos, actuated_indices)
        
        try:
            scene.step()
        except Exception:
            # Handle simulation explosion and reset history
            robot.set_pos(torch.tensor([0.0, 0.0, 0.5], device=gs.device).repeat(n_envs, 1))
            robot.zero_all_dofs_velocity()
            robot.set_dofs_position(j_mid.repeat(n_envs, 1), actuated_indices)
            obs_history = reset_history()
            continue

        # Update History Buffer (Shift left and add newest)
        new_obs = get_single_obs()
        # [old_0, old_1, old_2] -> [old_1, old_2, new_obs]
        obs_history = torch.cat([obs_history[:, single_obs_dim:], new_obs], dim=-1)

        base_h = robot.get_links_pos()[:, 0, 2]
        reward = robot.get_links_vel()[:, 0, 0] + 0.1 * (base_h > 0.2).float()
        
        log_probs.append(log_prob)
        rewards.append(reward)
        entropies.append(entropy)

    # --- 6. UPDATE BRAIN ---
    log_probs = torch.stack(log_probs)
    rewards = torch.stack(rewards)
    entropies = torch.stack(entropies)

    baseline = rewards.mean(dim=1, keepdim=True) 
    advantage = (rewards - baseline) / (rewards.std() + 1e-6)
    loss = -(log_probs * advantage.detach()).mean() - (entropy_coeff * entropies).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Swarm Vel: {baseline.mean().item():.3f}")
        
        # Teleport and Reset
        reset_pos = torch.zeros((n_envs, 3), device=gs.device)
        reset_pos[:, 2] = 0.5
        robot.set_pos(reset_pos)
        robot.zero_all_dofs_velocity()
        robot.set_dofs_position(j_mid.repeat(n_envs, 1), actuated_indices)
        # CRITICAL: Reset history buffer after teleporting
        obs_history = reset_history()