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
N_ENVS = 1024
ENV_SPACING = (0.5, 0.5)
DT = 0.02
SUBSTEPS = 4  # Reduced from 10 for speed (test stability)
SHOW_VIEWER = True

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
HISTORY_LEN = 5
HIDDEN_DIM = 128
LEARNING_RATE = 3e-4
N_EPOCHS = 2000
STEPS_PER_EPOCH = 200

# PPO-specific hyperparameters
GAMMA = 0.99  # Discount factor for returns
GAE_LAMBDA = 0.95  # Lambda for Generalized Advantage Estimation
PPO_EPSILON = 0.2  # Clipping parameter for PPO objective
VALUE_COEFF = 0.5  # Coefficient for value loss
ENTROPY_COEFF = 0.01  # Coefficient for entropy bonus (reduced for PPO)
GRAD_CLIP_NORM = 1.0  # Gradient clipping
N_PPO_EPOCHS = 4  # Number of PPO update epochs per rollout
MINI_BATCH_SIZE = 2048  # Mini-batch size for PPO updates (N_ENVS * STEPS / k)

# Observation clipping bounds
VEL_CLIP = 20.0
ANG_VEL_CLIP = 10.0  # Clip angular velocity to prevent extreme values

# Termination criteria
HEAD_HEIGHT_THRESHOLD = 0.2

# Reward shaping
HEIGHT_REWARD_SCALE = 2.0  # Multiplier for height reward
UPRIGHTNESS_REWARD_SCALE = 1.0  # Multiplier for staying upright (low tilt)
SMOOTHNESS_REWARD_SCALE = 0.5  # Multiplier for smooth joint movements
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


class Critic(nn.Module):
    """Value function network for PPO."""

    def __init__(self, single_obs_dim, history_len=HISTORY_LEN):
        super().__init__()
        self.input_dim = single_obs_dim * history_len

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, HIDDEN_DIM), nn.ELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ELU(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, obs_history):
        return self.net(obs_history).squeeze(-1)


# ============================================================================
# GENESIS SCENE INITIALIZATION
# ============================================================================

def initialize_scene():
    """Initialize Genesis simulation with robot and plane."""
    gs.init(backend=gs.gpu)

    sim_options = gs.options.SimOptions(dt=DT, substeps=SUBSTEPS)

    # Configure viewer for maximum performance
    if SHOW_VIEWER:
        viewer_options = gs.options.ViewerOptions(
            max_FPS=None,          # Cap FPS to reduce rendering overhead
            camera_pos=(2.0, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            res=(900, 600),     # Lower resolution = faster
        )
        vis_options = gs.options.VisOptions(
            shadow=False,
            rendered_envs_idx=[1] * min(4, N_ENVS) + [0] * (N_ENVS - min(4, N_ENVS)),
            )  
        scene = gs.Scene(
            sim_options=sim_options,
            viewer_options=viewer_options,
            show_viewer=True,
            vis_options=vis_options,
        )
    else:
        scene = gs.Scene(sim_options=sim_options, show_viewer=False)

    plane = scene.add_entity(gs.morphs.Plane())

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=URDF_PATH,
            fixed=False,
            collision=True
        )
    )

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

def create_observation_function(robot, actuated_indices, head_idx, action_dim):
    """Create observation collection function with closure over robot state."""
    def get_single_obs(previous_actions=None):
        # All joint positions and velocities (not just actuated)
        all_dof_pos = torch.nan_to_num(robot.get_dofs_position())
        all_dof_vel = torch.clamp(torch.nan_to_num(robot.get_dofs_velocity()), -VEL_CLIP, VEL_CLIP)

        # Base orientation (quaternion: w, x, y, z)
        base_quat = torch.nan_to_num(robot.get_quat())

        # Base linear and angular velocity
        base_lin_vel = torch.clamp(torch.nan_to_num(robot.get_vel()), -VEL_CLIP, VEL_CLIP)
        base_ang_vel = torch.clamp(torch.nan_to_num(robot.get_ang()), -ANG_VEL_CLIP, ANG_VEL_CLIP)

        # Head position
        head_pos = torch.nan_to_num(robot.get_links_pos()[:, head_idx, :])

        # Previous actions (zero if not provided)
        if previous_actions is None:
            previous_actions = torch.zeros((robot.get_pos().shape[0], action_dim), device=gs.device)

        return torch.cat([
            all_dof_pos,
            all_dof_vel,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            head_pos,
            previous_actions
        ], dim=-1)

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
        # Use zero previous actions for reset
        init_obs = get_single_obs(previous_actions=None).repeat(1, HISTORY_LEN)
        return init_obs if envs_idx is None else init_obs[envs_idx]

    return reset_history


def quaternion_to_uprightness(quat):
    """
    Compute uprightness metric from quaternion.
    Returns value in [0, 1] where 1 = perfectly upright, 0 = upside down.

    Args:
        quat: (N, 4) quaternion in [w, x, y, z] format
    """
    # Extract quaternion components
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Compute the Z-component of the robot's up vector in world frame
    # This is the (2,2) element of the rotation matrix
    # R[2,2] = 1 - 2(x^2 + y^2)
    z_up = 1.0 - 2.0 * (x**2 + y**2)

    # z_up ranges from -1 (upside down) to 1 (upright)
    # Map to [0, 1] range
    uprightness = (z_up + 1.0) / 2.0

    return uprightness


def create_reward_function(robot, head_idx, actuated_indices):
    """Create reward computation function that encourages height, uprightness, and smoothness."""
    # Track previous joint positions for smoothness reward
    prev_joint_pos = None

    def compute_reward(terminated):
        nonlocal prev_joint_pos

        head_height = robot.get_links_pos()[:, head_idx, 2]

        # 1. Height-based reward (higher = better)
        height_reward = HEIGHT_REWARD_SCALE * head_height

        # 2. Uprightness reward (minimize tilt)
        base_quat = robot.get_quat()
        uprightness = quaternion_to_uprightness(base_quat)
        uprightness_reward = UPRIGHTNESS_REWARD_SCALE * uprightness

        # 3. Smoothness reward (penalize rapid joint movements)
        current_joint_pos = robot.get_dofs_position()[:, actuated_indices]

        if prev_joint_pos is not None:
            # Compute L2 norm of joint position changes
            joint_delta = current_joint_pos - prev_joint_pos
            joint_velocity_penalty = torch.mean(joint_delta**2, dim=-1)
            smoothness_reward = -SMOOTHNESS_REWARD_SCALE * joint_velocity_penalty
        else:
            smoothness_reward = torch.zeros(current_joint_pos.shape[0], device=gs.device)

        # Update previous joint positions (detach to avoid unnecessary grad tracking)
        prev_joint_pos = current_joint_pos.detach()

        # 4. Survival bonus
        survival_reward = SURVIVAL_BONUS

        # Combine all rewards
        reward = height_reward + uprightness_reward + smoothness_reward + survival_reward

        # Apply termination penalty
        reward = torch.where(terminated, torch.tensor(TERMINATION_PENALTY, device=gs.device), reward)

        # Return reward and metrics to avoid recomputing
        return reward, head_height, uprightness

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
    print(f"  - Height Reward Scale:      {HEIGHT_REWARD_SCALE}")
    print(f"  - Uprightness Reward Scale: {UPRIGHTNESS_REWARD_SCALE}")
    print(f"  - Smoothness Reward Scale:  {SMOOTHNESS_REWARD_SCALE}")
    print(f"  - Survival Bonus:           {SURVIVAL_BONUS}")
    print(f"  - Termination Penalty:      {TERMINATION_PENALTY}")
    print(f"  - Head Height Threshold:    {HEAD_HEIGHT_THRESHOLD} m")

    # Neural network
    print(f"\n[POLICY NETWORK]")
    print(f"  - Single Observation Breakdown:")
    print(f"    • All DOF Positions: {robot.n_dofs}")
    print(f"    • All DOF Velocities: {robot.n_dofs}")
    print(f"    • Base Orientation (Quat WXYZ): 4")
    print(f"    • Base Linear Velocity (XYZ): 3")
    print(f"    • Base Angular Velocity (XYZ): 3")
    print(f"    • Head Position (XYZ): 3")
    print(f"    • Previous Actions: {action_dim}")
    print(f"    • Total Single Obs Dim: {single_obs_dim}")
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

def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: (T, N) tensor of rewards
        values: (T+1, N) tensor of value estimates
        dones: (T, N) tensor of done flags
        gamma: discount factor
        lam: GAE lambda parameter

    Returns:
        advantages: (T, N) tensor of advantages
        returns: (T, N) tensor of discounted returns
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N, device=rewards.device)

    # Compute GAE backwards through time
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    # Returns = advantages + values
    returns = advantages + values[:-1]

    return advantages, returns


def compute_ppo_loss(actor, critic, obs_history, actions, old_log_probs, advantages, returns):
    """
    Compute PPO clipped surrogate loss.

    Args:
        actor: Policy network
        critic: Value network
        obs_history: Observation history
        actions: Actions taken
        old_log_probs: Log probabilities from old policy
        advantages: GAE advantages
        returns: Discounted returns

    Returns:
        total_loss: Combined actor + critic loss
        policy_loss: Actor loss component
        value_loss: Critic loss component
        entropy: Policy entropy
    """
    # Compute current policy distribution
    dist = actor(obs_history)
    log_probs = dist.log_prob(actions).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1).mean()

    # Compute value predictions
    values = critic(obs_history)

    # PPO clipped surrogate objective
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - PPO_EPSILON, 1 + PPO_EPSILON)

    advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    policy_loss = -torch.min(
        ratio * advantages_normalized,
        clipped_ratio * advantages_normalized
    ).mean()

    # Value loss (MSE)
    value_loss = 0.5 * ((values - returns) ** 2).mean()

    # Total loss
    total_loss = policy_loss + VALUE_COEFF * value_loss - ENTROPY_COEFF * entropy

    return total_loss, policy_loss, value_loss, entropy


def step_simulation(robot, scene, actor, obs_history, j_low, j_high, actuated_indices, get_single_obs, single_obs_dim, previous_actions):
    """Execute one simulation step: action selection, physics, observation update."""
    # Action selection
    dist = actor(obs_history)
    actions = dist.rsample()
    log_prob = dist.log_prob(actions).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1)

    # Apply actions (map from [-1,1] to [j_low, j_high])
    target_pos = j_low + (torch.tanh(actions) + 1.0) * 0.5 * (j_high - j_low)
    robot.control_dofs_position(target_pos, actuated_indices)
    scene.step()

    # Update observation history (include previous actions)
    new_obs = get_single_obs(previous_actions=actions)
    obs_history = torch.cat([obs_history[:, single_obs_dim:], new_obs], dim=-1)

    return obs_history, log_prob, entropy, actions


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

    # Determine action dimension
    action_dim = len(actuated_indices)

    # Create helper functions
    get_single_obs = create_observation_function(robot, actuated_indices, head_idx, action_dim)
    check_termination = create_termination_check(robot, head_idx)
    reset_history = create_history_reset(get_single_obs)
    compute_reward = create_reward_function(robot, head_idx, actuated_indices)

    # Setup policy and value networks
    sample_obs = get_single_obs()
    single_obs_dim = sample_obs.shape[-1]

    actor = GroupActor(single_obs_dim, action_dim, HISTORY_LEN).to(gs.device)
    critic = Critic(single_obs_dim, HISTORY_LEN).to(gs.device)

    # Separate optimizers for actor and critic
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    # Print diagnostics and get user confirmation
    use_tensorboard = print_model_diagnostics(robot, head_idx, target_names, actuated_indices,
                                             j_low, j_high, j_init, home_pos, actor, single_obs_dim, action_dim)

    # Setup TensorBoard if requested
    writer, tb_process = None, None
    if use_tensorboard:
        writer, tb_process = setup_tensorboard()

    # Training loop
    for epoch in range(N_EPOCHS):
        # Storage for rollout data
        all_obs_history = []
        all_actions = []
        all_log_probs = []
        all_rewards = []
        all_values = []
        all_dones = []
        head_heights = []
        uprightness_values = []

        # Global reset
        obs_history = global_reset(robot, home_pos, home_quat, j_init, actuated_indices, reset_history)
        previous_actions = torch.zeros((N_ENVS, action_dim), device=gs.device)

        # Rollout collection
        for step in range(STEPS_PER_EPOCH):
            # Store current observation (no clone needed - we'll detach later)
            all_obs_history.append(obs_history)

            # Get action and value
            with torch.no_grad():
                dist = actor(obs_history)
                actions = dist.rsample()
                log_probs = dist.log_prob(actions).sum(dim=-1)
                values = critic(obs_history)

            # Apply actions
            target_pos = j_low + (torch.tanh(actions) + 1.0) * 0.5 * (j_high - j_low)
            robot.control_dofs_position(target_pos, actuated_indices)
            scene.step()

            # Get new observation
            new_obs = get_single_obs(previous_actions=actions)
            obs_history = torch.cat([obs_history[:, single_obs_dim:], new_obs], dim=-1)

            # Check termination
            terminated = check_termination()
            dones = terminated.float()

            # Reset terminated environments
            envs_to_reset = torch.where(terminated)[0]
            if len(envs_to_reset) > 0:
                obs_history = reset_environments(
                    robot, envs_to_reset, home_pos, home_quat, j_init,
                    actuated_indices, obs_history, reset_history
                )
                actions[envs_to_reset] = 0.0

            # Compute rewards (also returns metrics to avoid recomputation)
            rewards, head_height, uprightness = compute_reward(terminated)

            # Store rollout data
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_rewards.append(rewards)
            all_values.append(values)
            all_dones.append(dones)
            head_heights.append(head_height)
            uprightness_values.append(uprightness)

            # Update previous actions
            previous_actions = actions

        # Get final value for bootstrapping
        with torch.no_grad():
            final_value = critic(obs_history)
            all_values.append(final_value)

        # Convert to tensors
        obs_tensor = torch.stack(all_obs_history)  # (T, N, obs_dim)
        actions_tensor = torch.stack(all_actions)  # (T, N, action_dim)
        old_log_probs_tensor = torch.stack(all_log_probs)  # (T, N)
        rewards_tensor = torch.stack(all_rewards)  # (T, N)
        values_tensor = torch.stack(all_values)  # (T+1, N)
        dones_tensor = torch.stack(all_dones)  # (T, N)

        # Compute GAE
        advantages, returns = compute_gae(rewards_tensor, values_tensor, dones_tensor)

        # Flatten for mini-batch training
        T, N = rewards_tensor.shape
        obs_flat = obs_tensor.reshape(T * N, -1)
        actions_flat = actions_tensor.reshape(T * N, -1)
        old_log_probs_flat = old_log_probs_tensor.reshape(T * N)
        advantages_flat = advantages.reshape(T * N)
        returns_flat = returns.reshape(T * N)

        # PPO update epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for ppo_epoch in range(N_PPO_EPOCHS):
            # Create random mini-batches
            indices = torch.randperm(T * N, device=gs.device)

            for start_idx in range(0, T * N, MINI_BATCH_SIZE):
                end_idx = min(start_idx + MINI_BATCH_SIZE, T * N)
                batch_indices = indices[start_idx:end_idx]

                # Get mini-batch
                obs_batch = obs_flat[batch_indices]
                actions_batch = actions_flat[batch_indices]
                old_log_probs_batch = old_log_probs_flat[batch_indices]
                advantages_batch = advantages_flat[batch_indices]
                returns_batch = returns_flat[batch_indices]

                # Compute PPO loss
                loss, policy_loss, value_loss, entropy = compute_ppo_loss(
                    actor, critic, obs_batch, actions_batch,
                    old_log_probs_batch, advantages_batch, returns_batch
                )

                # Update networks
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP_NORM)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP_NORM)
                actor_optimizer.step()
                critic_optimizer.step()

                # Track losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # Logging
        if epoch % 2 == 0:
            avg_reward = rewards_tensor.mean().item()
            total_reward = rewards_tensor.sum(dim=0).mean().item()
            avg_policy_loss = total_policy_loss / n_updates
            avg_value_loss = total_value_loss / n_updates
            avg_entropy = total_entropy / n_updates

            # Count survival steps (non-terminated steps)
            survival_steps = (rewards_tensor > TERMINATION_PENALTY).sum(dim=0).float().mean().item()

            print(f"Epoch {epoch} | Reward: {avg_reward:+.3f} | Total: {total_reward:+.1f} | Survival: {survival_steps:.1f} | "
                  f"PL: {avg_policy_loss:.3f} | VL: {avg_value_loss:.3f}")

            # TensorBoard logging
            if writer is not None:
                # Training metrics
                writer.add_scalar('Training/AvgReward', avg_reward, epoch)
                writer.add_scalar('Training/TotalReward', total_reward, epoch)
                writer.add_scalar('Training/SurvivalSteps', survival_steps, epoch)
                writer.add_scalar('Training/PolicyLoss', avg_policy_loss, epoch)
                writer.add_scalar('Training/ValueLoss', avg_value_loss, epoch)
                writer.add_scalar('Training/Entropy', avg_entropy, epoch)

                # Advantage statistics
                writer.add_scalar('Training/AdvantagesMean', advantages.mean().item(), epoch)
                writer.add_scalar('Training/AdvantagesStd', advantages.std().item(), epoch)

                # Head height statistics
                heights_tensor = torch.stack(head_heights)
                avg_head_height = heights_tensor.mean().item()
                max_head_height = heights_tensor.max().item()
                writer.add_scalar('Robot/AvgHeadHeight', avg_head_height, epoch)
                writer.add_scalar('Robot/MaxHeadHeight', max_head_height, epoch)

                # Uprightness statistics
                uprightness_tensor = torch.stack(uprightness_values)
                avg_uprightness = uprightness_tensor.mean().item()
                writer.add_scalar('Robot/AvgUprightness', avg_uprightness, epoch)

                # Policy statistics
                writer.add_scalar('Policy/LogStdMean', actor.log_std.mean().item(), epoch)

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