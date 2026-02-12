import genesis as gs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from datetime import datetime
import json
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

N_ENVS = 2048 // 2 // 2 
ENV_SPACING = (0.0, 0.0) 
DT = 0.05
SUBSTEPS = 5
SHOW_VIEWER = True

# Robot configuration
URDF_PATH = r'C:\Users\h\Desktop\GenesisPhysics\onshape\urdf_output\robot.urdf'
HOME_POS = [0.0, 0.0, 0.15] 
HOME_QUAT = [1.0, 0.0, 0.0, 0.0]
CHASSIS_LINK_NAME = 'part_1_1' 

# Task Parameters
SPACE_SIZE = 10.0
TARGET_REACH_THRESHOLD = 0.4
MIN_TARGET_DISTANCE = 2.0

# Physics Parameters
GROUND_FRICTION = 2.0  # Friction coefficient for ground (0.5=slippery, 1.0=normal, 2.0=high grip)
WHEEL_RADIUS = 0.04  # Approximate wheel radius in meters (adjust based on your robot) 

# Training hyperparameters
HISTORY_LEN = 10
HIDDEN_DIM = 128
LEARNING_RATE = 3e-4
N_EPOCHS = 2000
STEPS_PER_EPOCH = 400

# PPO
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
VALUE_COEFF = 1.0  # Increased from 0.5 - value function is important
ENTROPY_COEFF = 0.1  # Increased from 0.02 - more exploration
GRAD_CLIP_NORM = 1.0
N_PPO_EPOCHS = 5  # Increased from 4 - better data utilization
MINI_BATCH_SIZE = 16384  # Decreased from 8192 - more updates per epoch

# Reward scaling (these will be dynamically adjustable via reward_config.json)
DIST_REWARD_SCALE = 200.0
REACH_BONUS = 1000.0
SMOOTHNESS_COEFF = 0.01  # Penalty for jerky actions
FLIP_PENALTY = 10.0  # More significant penalty for flipping
VELOCITY_REWARD_SCALE = 3  # Reward for moving (higher velocity = higher reward)
TIME_PENALTY = 1  # Small penalty per timestep to encourage faster completion
APPROACH_VELOCITY_SCALE = 50.0  # Reward for velocity component toward target

# Config file path for live tuning
REWARD_CONFIG_FILE = "reward_config.json"

# Curriculum learning
INITIAL_SPACE_SIZE = 5.0  # Start smaller
SPACE_SIZE_GROWTH = 0.01  # Growth per epoch
INITIAL_MIN_TARGET_DIST = 1.0  # Start closer
TARGET_DIST_GROWTH = 0.002  # Growth per epoch

# ============================================================================
# OBSERVATION NORMALIZATION
# ============================================================================

class RunningMeanStd:
    """Tracks running mean and std of observations for normalization."""
    def __init__(self, shape, epsilon=1e-4):
        self.mean = torch.zeros(shape, device=gs.device)
        self.var = torch.ones(shape, device=gs.device)
        self.count = epsilon

    def update(self, x):
        """Update statistics from a batch of observations."""
        batch_mean = x.mean(0)
        batch_var = x.var(0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        """Normalize observations using running statistics."""
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)

# ============================================================================
# NEURAL NETWORK
# ============================================================================

class GroupActor(nn.Module):
    def __init__(self, single_obs_dim, action_dim, history_len=HISTORY_LEN):
        super().__init__()
        self.input_dim = single_obs_dim * history_len
        # Deeper network with skip connections
        self.fc1 = nn.Linear(self.input_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.mu = nn.Linear(HIDDEN_DIM, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)  # Start with more exploration

    def forward(self, obs_history):
        x = torch.nn.functional.elu(self.fc1(obs_history))
        x_skip = x
        x = torch.nn.functional.elu(self.fc2(x))
        x = x + x_skip  # Skip connection
        x = torch.nn.functional.elu(self.fc3(x))
        mu = torch.tanh(self.mu(x))
        std = torch.exp(self.log_std).clamp(0.01, 1.0)  # Clamp std for stability
        return Normal(mu, std)

class Critic(nn.Module):
    def __init__(self, single_obs_dim, history_len=HISTORY_LEN):
        super().__init__()
        self.input_dim = single_obs_dim * history_len
        # Deeper network matching actor architecture
        self.fc1 = nn.Linear(self.input_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, obs_history):
        x = torch.nn.functional.elu(self.fc1(obs_history))
        x_skip = x
        x = torch.nn.functional.elu(self.fc2(x))
        x = x + x_skip  # Skip connection
        x = torch.nn.functional.elu(self.fc3(x))
        return self.value_head(x).squeeze(-1)

def get_random_targets(num, current_targets=None, space_size=SPACE_SIZE, min_dist=MIN_TARGET_DISTANCE):
    """Generate random targets within the given space size."""
    new_targets = (torch.rand((num, 2), device=gs.device) - 0.5) * space_size
    if current_targets is not None:
        dist = torch.norm(new_targets - current_targets, dim=-1)
        too_close = dist < min_dist
        if too_close.any():
            new_targets[too_close] = (torch.rand((too_close.sum(), 2), device=gs.device) - 0.5) * space_size
    return new_targets

# ============================================================================
# LIVE CONFIG TUNING
# ============================================================================

def save_reward_config(config):
    """Save reward configuration to JSON file."""
    with open(REWARD_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 Saved reward config to {REWARD_CONFIG_FILE}")

def load_reward_config():
    """Load reward configuration from JSON file if it exists."""
    if os.path.exists(REWARD_CONFIG_FILE):
        try:
            with open(REWARD_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️  Error reading {REWARD_CONFIG_FILE}, using defaults")
            return None
    return None

def get_config_mtime():
    """Get modification time of config file."""
    if os.path.exists(REWARD_CONFIG_FILE):
        return os.path.getmtime(REWARD_CONFIG_FILE)
    return 0

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def initialize_scene():
    gs.init(backend=gs.gpu, performance_mode=False)
    sim_options = gs.options.SimOptions(dt=DT, substeps=SUBSTEPS)

    if SHOW_VIEWER:
        viewer_options = gs.options.ViewerOptions(
            max_FPS=60, camera_pos=(3.0, 3.0, 3.0), camera_lookat=(0.0, 0.0, 0.0), res=(1280, 720)
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

    # Ground with default friction (Rough surface provides high friction)
    # Note: Genesis surfaces don't expose friction parameter directly in current API
    ground_surface = gs.surfaces.Rough()
    scene.add_entity(gs.morphs.Plane(), surface=ground_surface)

    target_vis = scene.add_entity(gs.morphs.Sphere(radius=0.2, collision=False), surface=gs.surfaces.Default(color=(1, 0, 0)))
    robot = scene.add_entity(gs.morphs.URDF(file=URDF_PATH, fixed=False, collision=True))
    
    scene.build(n_envs=N_ENVS, env_spacing=ENV_SPACING)
    return scene, robot, target_vis

def setup_robot(robot):
    target_names = ['revolute_1', 'revolute_2', 'revolute_3', 'revolute_4']
    actuated_indices = [robot.get_joint(name).dofs_idx_local[0] for name in target_names]
    # Skid steer: No stiffness (KP=0), only damping/motor strength (KV)
    robot.set_dofs_kp([0.0] * 4, actuated_indices)
    robot.set_dofs_kv([0.5] * 4, actuated_indices)
    return actuated_indices

def main():
    scene, robot, target_vis = initialize_scene()
    actuated_indices = setup_robot(robot)
    action_dim = len(actuated_indices)
    chassis_idx = robot.get_link(CHASSIS_LINK_NAME).idx - robot.link_start

    # Initialization
    home_pos = torch.tensor(HOME_POS, device=gs.device)
    home_quat = torch.tensor(HOME_QUAT, device=gs.device)

    # Live-tunable reward weights
    reward_weights = {
        'dist_reward': DIST_REWARD_SCALE,
        'reach_bonus': REACH_BONUS,
        'smoothness_coeff': SMOOTHNESS_COEFF,
        'flip_penalty': FLIP_PENALTY,
        'velocity_reward': VELOCITY_REWARD_SCALE,
        'time_penalty': TIME_PENALTY,
        'approach_velocity': APPROACH_VELOCITY_SCALE,
    }

    # Save initial config file for user to edit
    save_reward_config(reward_weights)
    last_config_mtime = get_config_mtime()
    print(f"\n🎮 Live tuning enabled! Edit '{REWARD_CONFIG_FILE}' to adjust rewards in real-time.\n")

    def get_obs(prev_actions, current_targets):
        all_pos = robot.get_dofs_position()
        all_vel = robot.get_dofs_velocity()
        base_quat = robot.get_quat()
        base_lin_vel = robot.get_vel()  # [N_ENVS, 3] - linear velocity only
        base_ang_vel = robot.get_ang()  # [N_ENVS, 3] - angular velocity
        car_pos = robot.get_links_pos()[:, chassis_idx, :2]
        rel_target = current_targets - car_pos

        # Enhanced observations: body-frame information
        # Extract yaw from quaternion (rotation around Z axis)
        qw, qx, qy, qz = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        # Rotation matrix from world to body frame (2D)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        # Transform target to body frame (forward = x, left = y)
        rel_target_body_x = cos_yaw * rel_target[:, 0] + sin_yaw * rel_target[:, 1]
        rel_target_body_y = -sin_yaw * rel_target[:, 0] + cos_yaw * rel_target[:, 1]
        rel_target_body = torch.stack([rel_target_body_x, rel_target_body_y], dim=-1)

        # Distance and bearing to target
        dist_to_target = torch.norm(rel_target, dim=-1, keepdim=True)
        bearing_to_target = torch.atan2(rel_target[:, 1], rel_target[:, 0]).unsqueeze(-1) - yaw.unsqueeze(-1)
        # Normalize bearing to [-pi, pi]
        bearing_to_target = torch.atan2(torch.sin(bearing_to_target), torch.cos(bearing_to_target))

        # Body-frame velocities (forward/lateral/yaw_rate)
        vel_body_x = cos_yaw * base_lin_vel[:, 0] + sin_yaw * base_lin_vel[:, 1]
        vel_body_y = -sin_yaw * base_lin_vel[:, 0] + cos_yaw * base_lin_vel[:, 1]
        yaw_rate = base_ang_vel[:, 2]  # Angular velocity around Z

        # Heading encoding (sin/cos for continuity across -pi/pi boundary)
        heading_sin = torch.sin(yaw).unsqueeze(-1)
        heading_cos = torch.cos(yaw).unsqueeze(-1)

        return torch.cat([
            all_pos,              # 4: wheel positions
            all_vel,              # 4: wheel velocities
            heading_sin,          # 1: sin(yaw) - continuous heading representation
            heading_cos,          # 1: cos(yaw)
            vel_body_x.unsqueeze(-1),  # 1: forward velocity (body frame)
            vel_body_y.unsqueeze(-1),  # 1: lateral velocity (body frame)
            yaw_rate.unsqueeze(-1),    # 1: yaw rate
            rel_target_body,      # 2: target position in body frame (forward, left)
            dist_to_target,       # 1: distance to target
            bearing_to_target,    # 1: bearing to target (body frame)
            prev_actions          # 4: previous actions
        ], dim=-1)  # Total: 21 dims

    def reset_environments(env_indices):
        """Reset specific environments to initial state."""
        if len(env_indices) == 0:
            return

        num_reset = len(env_indices)
        reset_pos = home_pos.repeat(num_reset, 1)
        reset_quat = home_quat.repeat(num_reset, 1)
        robot.set_pos(reset_pos, env_indices)
        robot.set_quat(reset_quat, env_indices)  # Reset orientation
        robot.zero_all_dofs_velocity(env_indices)

    # Initialize targets with curriculum space size
    current_space_size = INITIAL_SPACE_SIZE
    current_min_dist = INITIAL_MIN_TARGET_DIST
    targets = get_random_targets(N_ENVS, space_size=current_space_size, min_dist=current_min_dist)

    # Brain
    sample_obs = get_obs(torch.zeros((N_ENVS, action_dim), device=gs.device), targets)
    single_obs_dim = sample_obs.shape[-1]
    actor = GroupActor(single_obs_dim, action_dim).to(gs.device)
    critic = Critic(single_obs_dim).to(gs.device)
    actor_opt = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_opt = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    # Observation normalization
    obs_normalizer = RunningMeanStd(single_obs_dim)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f"runs/skid_steer_{datetime.now().strftime('%H%M%S')}")

    for epoch in range(N_EPOCHS):
        obs_hist_list, act_list, logp_list, rew_list, val_list, done_list = [], [], [], [], [], []

        # Curriculum learning: gradually increase task difficulty
        current_space_size = min(SPACE_SIZE, INITIAL_SPACE_SIZE + epoch * SPACE_SIZE_GROWTH)
        current_min_dist = min(MIN_TARGET_DISTANCE, INITIAL_MIN_TARGET_DIST + epoch * TARGET_DIST_GROWTH)

        # Reset all environments at start of epoch
        robot.set_pos(home_pos.repeat(N_ENVS, 1))
        robot.zero_all_dofs_velocity()
        targets = get_random_targets(N_ENVS, space_size=current_space_size, min_dist=current_min_dist)

        current_obs = get_obs(torch.zeros((N_ENVS, action_dim), device=gs.device), targets)
        obs_normalizer.update(current_obs)  # Update normalization stats
        normalized_obs = obs_normalizer.normalize(current_obs)
        obs_history = normalized_obs.repeat(1, HISTORY_LEN)

        # Episode tracking
        successes = torch.zeros(N_ENVS, device=gs.device)
        episode_lengths = torch.zeros(N_ENVS, device=gs.device)
        prev_actions = torch.zeros((N_ENVS, action_dim), device=gs.device)
        prev_car_pos = robot.get_links_pos()[:, chassis_idx, :2]

        # Slip tracking
        slip_ratios = []

        for step in range(STEPS_PER_EPOCH):
            episode_lengths += 1
            obs_hist_list.append(obs_history.clone())

            with torch.no_grad():
                dist = actor(obs_history)
                actions = dist.sample()
                log_probs = dist.log_prob(actions).sum(-1)
                values = critic(obs_history)

            # Action: Skid Steer Velocity Control
            target_velocities = actions * 100.0 # Neural net -1..1 -> -20..20 rad/s
            robot.control_dofs_velocity(target_velocities, actuated_indices)

            # Visualize targets
            vis_pos = torch.cat([targets, torch.full((N_ENVS, 1), 0.2, device=gs.device)], dim=-1)
            target_vis.set_pos(vis_pos)

            # !!! CORE PHYSICS STEP !!!
            scene.step()

            # Get new observations and normalize
            new_single_obs = get_obs(actions, targets)
            obs_normalizer.update(new_single_obs)
            normalized_new_obs = obs_normalizer.normalize(new_single_obs)
            obs_history = torch.cat([obs_history[:, single_obs_dim:], normalized_new_obs], dim=-1)

            # ===== REWARD COMPUTATION =====
            car_pos = robot.get_links_pos()[:, chassis_idx, :2]
            dist_to_target = torch.norm(targets - car_pos, dim=-1)

            # Check for config file updates every 50 steps
            if step % 50 == 0:
                current_mtime = get_config_mtime()
                if current_mtime > last_config_mtime:
                    new_config = load_reward_config()
                    if new_config:
                        changes = []
                        for key in reward_weights:
                            if key in new_config and new_config[key] != reward_weights[key]:
                                old_val = reward_weights[key]
                                reward_weights[key] = new_config[key]
                                changes.append(f"{key}: {old_val:.2f} → {new_config[key]:.2f}")
                        if changes:
                            print(f"\n🔄 Reward weights updated: {', '.join(changes)}\n")
                        last_config_mtime = current_mtime

            # Progress-based reward (change in distance) - rewards getting closer
            prev_dist = torch.norm(targets - prev_car_pos, dim=-1)
            progress = prev_dist - dist_to_target
            reward = reward_weights['dist_reward'] * progress

            # Get velocity for reward calculations
            base_lin_vel = robot.get_vel()  # [N_ENVS, 3] - linear velocity only
            xy_velocity = torch.norm(base_lin_vel[:, :2], dim=-1)  # Speed in XY plane

            # Approach velocity reward: reward velocity component toward target
            # This directly rewards moving quickly toward the target
            to_target = targets - car_pos  # Vector to target
            to_target_norm = to_target / (torch.norm(to_target, dim=-1, keepdim=True) + 1e-6)  # Normalized direction
            velocity_toward_target = (base_lin_vel[:, :2] * to_target_norm).sum(dim=-1)  # Dot product
            reward += reward_weights['approach_velocity'] * torch.clamp(velocity_toward_target, min=0)  # Only reward positive (approaching)

            # General velocity reward: encourage movement (faster = better)
            reward += reward_weights['velocity_reward'] * xy_velocity

            # Time penalty: encourage faster completion
            reward -= reward_weights['time_penalty']

            # Wheel slip detection: compare expected vs actual velocity
            wheel_velocities = robot.get_dofs_velocity()[:, actuated_indices]  # [N_ENVS, 4] wheel angular velocities
            avg_wheel_speed = wheel_velocities.abs().mean(dim=-1)  # Average wheel speed (rad/s)
            expected_linear_vel = avg_wheel_speed * WHEEL_RADIUS  # Expected chassis velocity from wheels
            slip_ratio = torch.abs(expected_linear_vel - xy_velocity) / (expected_linear_vel + 1e-6)  # Slip ratio
            slip_ratios.append(slip_ratio.mean().item())

            # Smoothness penalty (penalize jerky actions)
            action_diff = actions - prev_actions
            smoothness_penalty = reward_weights['smoothness_coeff'] * (action_diff ** 2).sum(-1)
            reward -= smoothness_penalty

            # Target reached bonus
            reached = dist_to_target < TARGET_REACH_THRESHOLD
            reward[reached] += reward_weights['reach_bonus']

            # Flip detection: check if robot's local Z-axis is pointing down
            # Transform the local up vector [0, 0, 1] by the quaternion to get world up direction
            base_quat = robot.get_quat()  # [N_ENVS, 4] - [w, x, y, z]
            x, y = base_quat[:, 1], base_quat[:, 2]

            # Rotate local z-axis [0, 0, 1] by quaternion to get world z-component
            # Using quaternion rotation formula: v' = q * v * q^(-1)
            # For unit quaternion, the z-component of rotated [0,0,1] is:
            up_z = 1 - 2*(x*x + y*y)  # This gives the world z-component of the robot's up direction

            # If up_z < 0, the robot is upside down (up vector points down)
            # Use a threshold to allow some tilt: up_z < -0.1 means tilted past ~95 degrees
            flipped = up_z < -0.1  # Only reset if significantly upside down

            reward[flipped] -= reward_weights['flip_penalty']

            # Handle flipped robots: reset robot position AND target
            if flipped.any():
                idx_flipped = torch.where(flipped)[0]

                # Reset robot position for flipped robots
                reset_environments(idx_flipped)

                # New targets for flipped environments
                targets[idx_flipped] = get_random_targets(len(idx_flipped), space_size=current_space_size, min_dist=current_min_dist)

                # Reset observation history for flipped environments
                zero_actions = torch.zeros((N_ENVS, action_dim), device=gs.device)
                zero_actions[idx_flipped] = 0
                all_reset_obs = get_obs(zero_actions, targets)
                normalized_all_obs = obs_normalizer.normalize(all_reset_obs)
                obs_history[idx_flipped] = normalized_all_obs[idx_flipped].repeat(1, HISTORY_LEN)

                # Reset tracking variables
                episode_lengths[idx_flipped] = 0
                prev_actions[idx_flipped] = 0

            # Handle reached targets: reset ONLY target, NOT robot position
            if reached.any():
                idx_reached = torch.where(reached)[0]

                # Track successes
                successes[idx_reached] += 1

                # Reset ONLY targets (robot keeps its position and continues)
                targets[idx_reached] = get_random_targets(len(idx_reached), space_size=current_space_size, min_dist=current_min_dist)

                # Update observation for new targets (but robot state remains the same)
                # No need to reset observation history - just update with new target positions
                current_actions = actions  # Use current actions, not zero
                all_obs = get_obs(current_actions, targets)
                normalized_all_obs = obs_normalizer.normalize(all_obs)
                # Only update the most recent observation frame (last single_obs_dim features)
                obs_history[idx_reached, -single_obs_dim:] = normalized_all_obs[idx_reached]

                # Don't reset episode_lengths or prev_actions - robot continues its trajectory

            # Need reset for done flag (includes both cases)
            need_reset = flipped | reached

            # Update tracking variables
            prev_car_pos = car_pos.clone()
            prev_actions = actions.clone()

            # Store rollout data
            act_list.append(actions)
            logp_list.append(log_probs)
            rew_list.append(reward)
            val_list.append(values)
            done_list.append(need_reset.float())

        # Final value for GAE
        with torch.no_grad():
            val_list.append(critic(obs_history))

        # --- PPO UPDATE ---
        obs_tensor = torch.stack(obs_hist_list)
        actions_tensor = torch.stack(act_list)
        old_log_probs_tensor = torch.stack(logp_list)
        rewards_tensor = torch.stack(rew_list)
        values_tensor = torch.stack(val_list)
        dones_tensor = torch.stack(done_list)

        # Advantage estimation
        advantages = torch.zeros_like(rewards_tensor)
        last_gae_lam = 0
        for t in reversed(range(STEPS_PER_EPOCH)):
            delta = rewards_tensor[t] + GAMMA * values_tensor[t+1] * (1 - dones_tensor[t]) - values_tensor[t]
            advantages[t] = last_gae_lam = delta + GAMMA * GAE_LAMBDA * (1 - dones_tensor[t]) * last_gae_lam
        returns = advantages + values_tensor[:-1]

        # Flatten for batching
        obs_f = obs_tensor.view(-1, obs_tensor.shape[-1])
        act_f = actions_tensor.view(-1, action_dim)
        logp_f = old_log_probs_tensor.view(-1)
        adv_f = (advantages.view(-1) - advantages.mean()) / (advantages.std() + 1e-8)
        ret_f = returns.view(-1)

        # Track metrics for logging
        policy_losses = []
        value_losses = []
        entropies = []

        for _ in range(N_PPO_EPOCHS):
            indices = torch.randperm(obs_f.shape[0], device=gs.device)
            for i in range(0, obs_f.shape[0], MINI_BATCH_SIZE):
                idx = indices[i : i + MINI_BATCH_SIZE]

                dist = actor(obs_f[idx])
                new_logp = dist.log_prob(act_f[idx]).sum(-1)
                new_val = critic(obs_f[idx])

                ratio = torch.exp(new_logp - logp_f[idx])
                surr1 = ratio * adv_f[idx]
                surr2 = torch.clamp(ratio, 1.0-PPO_EPSILON, 1.0+PPO_EPSILON) * adv_f[idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((new_val - ret_f[idx])**2).mean()
                entropy = dist.entropy().mean()

                loss = policy_loss + VALUE_COEFF * value_loss - ENTROPY_COEFF * entropy

                actor_opt.zero_grad(); critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP_NORM)
                nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP_NORM)
                actor_opt.step(); critic_opt.step()

                # Track for logging
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        # ===== LOGGING =====
        avg_success = successes.mean().item()
        avg_reward = rewards_tensor.mean().item()
        avg_dist = dist_to_target.mean().item()
        min_dist = dist_to_target.min().item()
        action_std = actor.log_std.exp().mean().item()
        avg_slip = sum(slip_ratios) / len(slip_ratios) if slip_ratios else 0

        print(f"Epoch {epoch:4d} | Success: {avg_success:.2f} | Reward: {avg_reward:+.3f} | "
              f"AvgDist: {avg_dist:.2f} | MinDist: {min_dist:.2f} | "
              f"Space: {current_space_size:.1f} | Slip: {avg_slip:.2f} | ActionStd: {action_std:.3f}")

        # TensorBoard logging
        writer.add_scalar("Task/SuccessRate", avg_success, epoch)
        writer.add_scalar("Task/AvgReward", avg_reward, epoch)
        writer.add_scalar("Task/AvgDistance", avg_dist, epoch)
        writer.add_scalar("Task/MinDistance", min_dist, epoch)
        writer.add_scalar("Task/SpaceSize", current_space_size, epoch)
        writer.add_scalar("Task/MinTargetDist", current_min_dist, epoch)

        writer.add_scalar("Train/PolicyLoss", sum(policy_losses) / len(policy_losses), epoch)
        writer.add_scalar("Train/ValueLoss", sum(value_losses) / len(value_losses), epoch)
        writer.add_scalar("Train/Entropy", sum(entropies) / len(entropies), epoch)
        writer.add_scalar("Train/ActionStd", action_std, epoch)
        writer.add_scalar("Train/ValueMean", values_tensor[:-1].mean().item(), epoch)
        writer.add_scalar("Train/AdvantageStd", advantages.std().item(), epoch)

        # Physics metrics
        writer.add_scalar("Physics/WheelSlipRatio", avg_slip, epoch)
        writer.add_scalar("Physics/AvgVelocity", xy_velocity.mean().item(), epoch)

if __name__ == "__main__":
    main()