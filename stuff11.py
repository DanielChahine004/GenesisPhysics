import genesis as gs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import webbrowser
import subprocess
import time
from datetime import datetime
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

# Simulation parameters
N_ENVS = 16
ENV_SPACING = (1.0, 1.0)
DT = 0.02
SUBSTEPS = 10

# Robot configuration
URDF_PATH = r'C:\Users\h\Desktop\GenesisPhysics\onshape\urdf_output\robot.urdf'
HOME_POS = [0.0, 0.0, 0.05]
HOME_QUAT = [1.0, 0.0, 0.0, 0.0]
HEAD_LINK_NAME = 'part_9_1'

# Per-joint initialization config (options: 'min', 'max', 'mid')
JOINT_INIT_CONFIG = {
    'revolute_1': 'max',
    'revolute_2': 'max',
    'revolute_3': 'mid',
    'revolute_4': 'mid',
}

# PD controller gains
KP_GAIN = 400.0
KV_GAIN = 20.0

# Training hyperparameters
HISTORY_LEN = 50
HIDDEN_DIM = 128
LEARNING_RATE = 3e-4
ENTROPY_COEFF = 0.02
GRAD_CLIP_NORM = 1.0
N_EPOCHS = 2000
STEPS_PER_EPOCH = 200

# Observation clipping bounds
VEL_CLIP = 20.0

# Termination criteria
HEAD_HEIGHT_THRESHOLD = 0.05

# Reward shaping
HEIGHT_REWARD_SCALE = 2.0  # Multiplier for height reward
TERMINATION_PENALTY = -10.0  # Penalty for falling
SURVIVAL_BONUS = 0.1  # Small bonus for each timestep alive

# TensorBoard configuration
TENSORBOARD_LOG_DIR = 'runs'
TENSORBOARD_PORT = 6006


# ============================================================================
# NEURAL NETWORK
# ============================================================================

class GroupActor(nn.Module):
    """Stochastic policy network with observation history."""

    def __init__(self, single_obs_dim, action_dim, history_len=HISTORY_LEN):
        super().__init__()
        self.input_dim = single_obs_dim * history_len

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, HIDDEN_DIM), nn.ELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ELU(),
        )

        self.mu = nn.Linear(HIDDEN_DIM, action_dim)
        with torch.no_grad():
            self.mu.weight.fill_(0.0)
            self.mu.bias.fill_(0.0)

        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0)

    def forward(self, obs_history):
        x = self.net(obs_history)
        mu = torch.tanh(self.mu(x))
        std = torch.exp(self.log_std)
        return Normal(mu, std)


# ============================================================================
# GENESIS SCENE INITIALIZATION
# ============================================================================

def initialize_scene():
    """Initialize Genesis simulation with robot and plane."""
    gs.init(backend=gs.gpu)

    sim_options = gs.options.SimOptions(dt=DT, substeps=SUBSTEPS)
    scene = gs.Scene(sim_options=sim_options, show_viewer=True)
    scene.add_entity(gs.morphs.Plane())

    robot = scene.add_entity(gs.morphs.URDF(file=URDF_PATH, fixed=False))
    scene.build(n_envs=N_ENVS, env_spacing=ENV_SPACING)

    return scene, robot


# ============================================================================
# JOINT CONFIGURATION
# ============================================================================

def setup_joint_configuration(robot):
    """Configure actuated joints with limits and initial positions."""
    target_names = list(JOINT_INIT_CONFIG.keys())
    actuated_indices = []
    init_joint_list = []
    low_list = []
    high_list = []

    all_lowers, all_uppers = robot.get_dofs_limit()

    for name in target_names:
        idx = robot.get_joint(name).dofs_idx_local
        idx_val = idx.start if isinstance(idx, slice) else idx[0]
        actuated_indices.append(idx_val)

        # Handle infinite/continuous joints
        raw_low = all_lowers[idx_val]
        raw_high = all_uppers[idx_val]

        low = raw_low if raw_low > -7.0 else torch.tensor(-3.1415, device=gs.device)
        high = raw_high if raw_high < 7.0 else torch.tensor(3.1415, device=gs.device)

        low_list.append(low)
        high_list.append(high)

        # Determine initial position
        mode = JOINT_INIT_CONFIG.get(name, 'mid')
        if mode == 'min':
            init_joint_list.append(low + 0.05)
        elif mode == 'max':
            init_joint_list.append(high - 0.05)
        else:  # mid
            init_joint_list.append((low + high) / 2.0)

    j_init = torch.stack(init_joint_list).to(gs.device)
    j_low = torch.stack(low_list).to(gs.device)
    j_high = torch.stack(high_list).to(gs.device)

    # Set PD gains
    robot.set_dofs_kp([KP_GAIN] * len(actuated_indices), actuated_indices)
    robot.set_dofs_kv([KV_GAIN] * len(actuated_indices), actuated_indices)

    return target_names, actuated_indices, j_init, j_low, j_high


def initialize_robot_state(robot, scene, home_pos, home_quat, j_init, actuated_indices):
    """Set robot to initial state before first simulation step."""
    robot.set_pos(home_pos.repeat(N_ENVS, 1))
    robot.set_quat(home_quat.repeat(N_ENVS, 1))
    robot.set_dofs_position(j_init.repeat(N_ENVS, 1), actuated_indices)
    scene.step()


def get_head_link_index(robot):
    """Find and return the local index of the head tracking link."""
    print("Detected links in Genesis:")
    for i, link in enumerate(robot.links):
        print(f"Index {i}: {link.name}")

    head_link = robot.get_link(HEAD_LINK_NAME)
    head_idx = head_link.idx - robot.link_start

    print(f"DEBUG: Global link index: {head_link.idx}, Robot link start: {robot.link_start}, Calculated local index: {head_idx}")
    return head_idx


# ============================================================================
# OBSERVATION & TERMINATION
# ============================================================================

def create_observation_function(robot, actuated_indices, head_idx):
    """Create observation collection function with closure over robot state."""
    def get_single_obs():
        d_pos = torch.nan_to_num(robot.get_dofs_position()[:, actuated_indices])
        d_vel = torch.clamp(torch.nan_to_num(robot.get_dofs_velocity()[:, actuated_indices]), -VEL_CLIP, VEL_CLIP)
        b_vel = torch.clamp(torch.nan_to_num(robot.get_links_vel()[:, 0, :3]), -VEL_CLIP, VEL_CLIP)
        head_pos = torch.nan_to_num(robot.get_links_pos()[:, head_idx, :])
        return torch.cat([d_pos, d_vel, b_vel, head_pos], dim=-1)

    return get_single_obs


def create_termination_check(robot, head_idx):
    """Create termination checking function."""
    def check_termination():
        head_height = robot.get_links_pos()[:, head_idx, 2]
        return head_height < HEAD_HEIGHT_THRESHOLD

    return check_termination


def create_history_reset(get_single_obs):
    """Create history buffer reset function."""
    def reset_history(envs_idx=None):
        init_obs = get_single_obs().repeat(1, HISTORY_LEN)
        return init_obs if envs_idx is None else init_obs[envs_idx]

    return reset_history


def create_reward_function(robot, head_idx):
    """Create reward computation function that encourages height."""
    def compute_reward(terminated):
        head_height = robot.get_links_pos()[:, head_idx, 2]

        # Height-based reward (higher = better)
        height_reward = HEIGHT_REWARD_SCALE * head_height

        # Survival bonus for staying alive
        survival_reward = SURVIVAL_BONUS

        # Combine rewards
        reward = height_reward + survival_reward

        # Apply termination penalty
        reward = torch.where(terminated, torch.tensor(TERMINATION_PENALTY, device=gs.device), reward)

        return reward

    return compute_reward


# ============================================================================
# ENVIRONMENT RESET
# ============================================================================

def reset_environments(robot, envs_to_reset, home_pos, home_quat, j_init, actuated_indices, obs_history, reset_history):
    """Reset specified environments to initial state."""
    if len(envs_to_reset) == 0:
        return obs_history

    robot.set_pos(home_pos.repeat(len(envs_to_reset), 1), envs_idx=envs_to_reset)
    robot.set_quat(home_quat.repeat(len(envs_to_reset), 1), envs_idx=envs_to_reset)
    robot.zero_all_dofs_velocity(envs_idx=envs_to_reset)
    robot.set_dofs_position(j_init.repeat(len(envs_to_reset), 1), actuated_indices, envs_idx=envs_to_reset)

    fresh_hists = reset_history(envs_to_reset)
    obs_history = obs_history.clone()
    obs_history[envs_to_reset] = fresh_hists

    return obs_history


def global_reset(robot, home_pos, home_quat, j_init, actuated_indices, reset_history):
    """Reset all environments globally."""
    robot.set_pos(home_pos.repeat(N_ENVS, 1))
    robot.set_quat(home_quat.repeat(N_ENVS, 1))
    robot.zero_all_dofs_velocity()
    robot.set_dofs_position(j_init.repeat(N_ENVS, 1), actuated_indices)
    return reset_history()


# ============================================================================
# TENSORBOARD UTILITIES
# ============================================================================

def setup_tensorboard():
    """Setup TensorBoard logging and launch server."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("⚠️ TensorBoard not available. Install with: pip install tensorboard")
        return None, None

    # Create unique run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(TENSORBOARD_LOG_DIR) / f'run_{timestamp}'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create SummaryWriter
    writer = SummaryWriter(log_dir=str(log_dir))

    # Launch TensorBoard server in background
    try:
        tb_process = subprocess.Popen(
            ['tensorboard', f'--logdir={TENSORBOARD_LOG_DIR}', f'--port={TENSORBOARD_PORT}'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )

        # Give TensorBoard a moment to start
        time.sleep(2)

        # Open browser
        url = f'http://localhost:{TENSORBOARD_PORT}'
        webbrowser.open(url)
        print(f"✓ TensorBoard launched at {url}")

        return writer, tb_process
    except FileNotFoundError:
        print("⚠️ TensorBoard command not found. Install with: pip install tensorboard")
        return writer, None


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def print_model_diagnostics(robot, head_idx, target_names, actuated_indices, j_low, j_high, j_init, home_pos, actor, single_obs_dim, action_dim):
    """Print comprehensive diagnostic information and request user confirmation."""
    print("\n" + "="*60)
    print("      GENESIS ROBOT & NEURAL NETWORK DIAGNOSTICS")
    print("="*60)

    # Robot structure
    print(f"\n[ROBOT STRUCTURE]")
    print(f"  - Total Links: {robot.n_links}")
    print(f"  - Total DOFs:  {robot.n_dofs}")

    link_masses = robot.get_links_inertial_mass()

    print(f"  - Link Names:")
    for i, link in enumerate(robot.links):
        m = link_masses[i].item()
        head_tag = " <--- [HEAD TRACKING]" if i == head_idx else ""
        print(f"    {i:2}: {link.name:15} (Mass: {m:.3f}kg){head_tag}")

    # Actuation
    print(f"\n[ACTUATION]")
    print(f"  - Target Actuated Joints: {list(target_names)}")
    print(f"  - Actuated Indices (Local): {actuated_indices}")
    print(f"  - Joint Limits & Init (Effective):")
    for i, name in enumerate(target_names):
        print(f"    {name:10}: Low={j_low[i]:.3f}, High={j_high[i]:.3f}, Init={j_init[i]:.3f}")
        if j_low[i] >= j_high[i]:
            print(f"      ⚠️ WARNING: Joint '{name}' is locked (Low >= High)!")

    # Physical properties
    total_mass = torch.sum(link_masses).item()
    print(f"\n[PHYSICAL PROPERTIES]")
    print(f"  - Total Robot Mass: {total_mass:.4f} kg")
    print(f"  - Spawn Height:     {home_pos[2].item()} m")

    # Reward configuration
    print(f"\n[REWARD SHAPING]")
    print(f"  - Height Reward Scale:  {HEIGHT_REWARD_SCALE}")
    print(f"  - Survival Bonus:       {SURVIVAL_BONUS}")
    print(f"  - Termination Penalty:  {TERMINATION_PENALTY}")
    print(f"  - Head Height Threshold: {HEAD_HEIGHT_THRESHOLD} m")

    # Neural network
    print(f"\n[POLICY NETWORK]")
    print(f"  - Input Dimension:  {actor.input_dim}")
    print(f"    (History: {HISTORY_LEN} steps x Single Obs: {single_obs_dim})")
    print(f"  - Output Dimension: {action_dim}")

    test_obs = torch.zeros((1, actor.input_dim), device=gs.device)
    with torch.no_grad():
        test_dist = actor(test_obs)
        test_mu = test_dist.mean
    print(f"  - Initialization Check: {'PASSED' if not torch.isnan(test_mu).any() else 'FAILED (NaN detected)'}")

    print("\n" + "="*60)

    # Ask about TensorBoard
    use_tensorboard = input("Enable TensorBoard logging and open in browser? (y/n): ").lower() == 'y'

    print("="*60)
    confirm = input("Proceed with training? (y/n): ")
    if confirm.lower() != 'y':
        print("Training cancelled by user.")
        exit()

    return use_tensorboard


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def compute_policy_loss(log_probs, rewards, entropies):
    """Compute REINFORCE loss with baseline and entropy regularization."""
    log_probs = torch.stack(log_probs)
    rewards = torch.stack(rewards)
    entropies = torch.stack(entropies)

    baseline = rewards.mean(dim=1, keepdim=True)
    advantage = (rewards - baseline) / (rewards.std() + 1e-6)
    loss = -(log_probs * advantage.detach()).mean() - (ENTROPY_COEFF * entropies).mean()

    return loss


def step_simulation(robot, scene, actor, obs_history, j_low, j_high, actuated_indices, get_single_obs, single_obs_dim):
    """Execute one simulation step: action selection, physics, observation update."""
    # Action selection
    dist = actor(obs_history)
    actions = dist.rsample()
    log_prob = dist.log_prob(actions).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1)

    # Apply actions
    target_pos = j_low + (torch.tanh(actions) + 1.0) * 0.5 * (j_high - j_low)
    robot.control_dofs_position(target_pos, actuated_indices)
    scene.step()

    # Update observation history
    new_obs = get_single_obs()
    obs_history = torch.cat([obs_history[:, single_obs_dim:], new_obs], dim=-1)

    return obs_history, log_prob, entropy


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Initialize scene and robot
    scene, robot = initialize_scene()

    # Configure joints
    target_names, actuated_indices, j_init, j_low, j_high = setup_joint_configuration(robot)

    # Setup tensors
    home_pos = torch.tensor(HOME_POS, device=gs.device)
    home_quat = torch.tensor(HOME_QUAT, device=gs.device)

    # Initialize robot state
    initialize_robot_state(robot, scene, home_pos, home_quat, j_init, actuated_indices)

    # Get head link index
    head_idx = get_head_link_index(robot)

    # Create helper functions
    get_single_obs = create_observation_function(robot, actuated_indices, head_idx)
    check_termination = create_termination_check(robot, head_idx)
    reset_history = create_history_reset(get_single_obs)
    compute_reward = create_reward_function(robot, head_idx)

    # Setup policy network
    sample_obs = get_single_obs()
    single_obs_dim = sample_obs.shape[-1]
    action_dim = len(actuated_indices)

    actor = GroupActor(single_obs_dim, action_dim, HISTORY_LEN).to(gs.device)
    optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)

    # Print diagnostics and get user confirmation
    use_tensorboard = print_model_diagnostics(robot, head_idx, target_names, actuated_indices,
                                             j_low, j_high, j_init, home_pos, actor, single_obs_dim, action_dim)

    # Setup TensorBoard if requested
    writer, tb_process = None, None
    if use_tensorboard:
        writer, tb_process = setup_tensorboard()

    # Training loop
    for epoch in range(N_EPOCHS):
        log_probs, rewards, entropies = [], [], []
        head_heights = []  # Track head heights for TensorBoard

        # Global reset
        obs_history = global_reset(robot, home_pos, home_quat, j_init, actuated_indices, reset_history)

        # Rollout
        for _ in range(STEPS_PER_EPOCH):
            # Step simulation
            obs_history, log_prob, entropy = step_simulation(
                robot, scene, actor, obs_history, j_low, j_high, actuated_indices,
                get_single_obs, single_obs_dim
            )

            # Check termination and reset failed environments
            terminated = check_termination()
            envs_to_reset = torch.where(terminated)[0]
            obs_history = reset_environments(
                robot, envs_to_reset, home_pos, home_quat, j_init,
                actuated_indices, obs_history, reset_history
            )

            # Track metrics
            reward = compute_reward(terminated)
            head_height = robot.get_links_pos()[:, head_idx, 2]

            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            head_heights.append(head_height)

        # Update policy
        loss = compute_policy_loss(log_probs, rewards, entropies)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        # Logging
        if epoch % 2 == 0:
            rewards_tensor = torch.stack(rewards)
            avg_reward = rewards_tensor.mean().item()
            total_reward = rewards_tensor.sum(dim=0).mean().item()

            # Count survival steps (non-terminated steps)
            survival_steps = (rewards_tensor > TERMINATION_PENALTY).sum(dim=0).float().mean().item()

            print(f"Epoch {epoch} | Avg Reward: {avg_reward:+.3f} | Total Reward: {total_reward:+.1f} | Avg Survival: {survival_steps:.1f} steps")

            # TensorBoard logging
            if writer is not None:
                # Training metrics
                writer.add_scalar('Training/AvgReward', avg_reward, epoch)
                writer.add_scalar('Training/TotalReward', total_reward, epoch)
                writer.add_scalar('Training/SurvivalSteps', survival_steps, epoch)
                writer.add_scalar('Training/Loss', loss.item(), epoch)

                # Head height statistics
                heights_tensor = torch.stack(head_heights)
                avg_head_height = heights_tensor.mean().item()
                max_head_height = heights_tensor.max().item()
                writer.add_scalar('Robot/AvgHeadHeight', avg_head_height, epoch)
                writer.add_scalar('Robot/MaxHeadHeight', max_head_height, epoch)

                # Policy statistics
                writer.add_scalar('Policy/LogStdMean', actor.log_std.mean().item(), epoch)
                writer.add_scalar('Policy/EntropyMean', torch.stack(entropies).mean().item(), epoch)

    # Cleanup TensorBoard
    if writer is not None:
        writer.close()
        print("\n✓ TensorBoard logs saved")

    if tb_process is not None:
        print("✓ TensorBoard server still running in background")
        print(f"  Access at: http://localhost:{TENSORBOARD_PORT}")
        print("  (Close terminal to stop server)")


if __name__ == "__main__":
    main()