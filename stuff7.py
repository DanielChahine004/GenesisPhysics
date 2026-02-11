import genesis as gs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# --- 1. STOCHASTIC ACTOR ---
class GroupActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
        )
        self.mu = nn.Linear(128, action_dim)
        with torch.no_grad():
            self.mu.weight.fill_(0.0)
            self.mu.bias.fill_(0.0)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0)

    def forward(self, obs):
        x = self.net(obs)
        mu = torch.tanh(self.mu(x)) 
        std = torch.exp(self.log_std)
        return Normal(mu, std)

# --- 2. SETUP GENESIS ---
gs.init(backend=gs.gpu)

sim_options = gs.options.SimOptions(
    dt=0.01,
    substeps=10, 
)

scene = gs.Scene(
    sim_options=sim_options,
    show_viewer=True
    )
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

# --- 3. INDEXING AND GAINS ---
target_names = ['revolute_1', 'revolute_2']
actuated_indices = []
for name in target_names:
    idx = robot.get_joint(name).dofs_idx_local
    actuated_indices.append(idx.start if isinstance(idx, slice) else idx[0])

all_lowers, all_uppers = robot.get_dofs_limit()
j_low, j_high = all_lowers[actuated_indices], all_uppers[actuated_indices]
j_mid = (j_low + j_high) / 2.0

# Pre-set robot to legal pose
robot.set_dofs_position(j_mid.repeat(n_envs, 1), actuated_indices)
robot.set_dofs_kp([40.0] * len(actuated_indices), actuated_indices)
robot.set_dofs_kv([5.0] * len(actuated_indices), actuated_indices)

# --- 4. TRAINING SETUP ---
obs_dim, action_dim = 7, len(actuated_indices)
actor = GroupActor(obs_dim, action_dim).to(gs.device)
optimizer = optim.Adam(actor.parameters(), lr=3e-4)
entropy_coeff = 0.02

def get_obs():
    d_pos = robot.get_dofs_position()[:, actuated_indices]
    d_vel = torch.clamp(robot.get_dofs_velocity()[:, actuated_indices], -10, 10)
    b_vel = torch.clamp(robot.get_links_vel()[:, 0, :], -10, 10)
    return torch.cat([d_pos, d_vel, b_vel], dim=-1)

# --- 5. TRAINING LOOP ---
for epoch in range(2000):
    log_probs, rewards, entropies = [], [], []
    
    for t in range(200):
        obs = get_obs()
        dist = actor(obs)
        actions = dist.rsample()
        
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        target_pos = j_low + (torch.tanh(actions) + 1.0) * 0.5 * (j_high - j_low)
        robot.control_dofs_position(target_pos, actuated_indices)
        
        try:
            scene.step()
        except Exception:
            # Handle simulation explosion safely
            robot.set_pos(torch.tensor([0.0, 0.0, 0.5], device=gs.device).repeat(n_envs, 1))
            robot.zero_all_dofs_velocity() # Safest way to kill all momentum
            robot.set_dofs_position(j_mid.repeat(n_envs, 1), actuated_indices)
            continue

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
        
        # KEY FIX: Teleport robots and RESET VELOCITY
        reset_pos = torch.zeros((n_envs, 3), device=gs.device)
        reset_pos[:, 2] = 0.5 # Set Z-height to 0.5
        
        robot.set_pos(reset_pos)
        robot.zero_all_dofs_velocity() # Safest and correct Genesis API method
        robot.set_dofs_position(j_mid.repeat(n_envs, 1), actuated_indices)