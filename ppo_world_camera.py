import genesis as gs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from datetime import datetime
import numpy as np
import cv2

# ============================================================================
# CONFIGURATION
# ============================================================================

N_ENVS = 1          # Camera rendering per-env is expensive — 1 for honest training
DT = 0.05
SUBSTEPS = 4
SHOW_VIEWER = True  # Genesis 3rd-person viewer
SHOW_CAR_CAM = True # OpenCV window showing the car's front camera POV

# Camera
CAM_W = 128
CAM_H = 128
CAM_TOP_DOWN = False        # True = overhead view, False = forward-facing
CAM_TOP_DOWN_HEIGHT = 1.2   # Metres above car for overhead view
CAM_FOV = 80                # FOV in degrees (wider = more scene visible)
CAM_FORWARD_OFFSET = 0.06   # (forward mode) metres in front of chassis centre
CAM_HEIGHT_OFFSET = 0.08    # (forward mode) metres above chassis centre

# Robot
URDF_PATH = r'C:\Users\h\Desktop\GenesisPhysics\onshape\urdf_output\robot.urdf'
HOME_POS = [0.0, 0.0, 0.15]
HOME_QUAT = [1.0, 0.0, 0.0, 0.0]
CHASSIS_LINK_NAME = 'part_1_1'

# Task
SPACE_SIZE = 5.0
TARGET_REACH_THRESHOLD = 0.4
MIN_TARGET_DISTANCE = 1.5

# Network dims
STATE_DIM = 4           # [sin(bearing), cos(bearing), distance, forward_speed]
CNN_LATENT_DIM = 128    # CNN output size
HIDDEN_DIM = 128

# Training
LEARNING_RATE = 3e-4
N_EPOCHS = 2000
STEPS_PER_EPOCH = 1000  # More steps to compensate for N_ENVS=1

# PPO
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
VALUE_COEFF = 0.5
ENTROPY_COEFF = 0.05
GRAD_CLIP_NORM = 0.5
N_PPO_EPOCHS = 4
MINI_BATCH_SIZE = 256   # Smaller batch for N_ENVS=1

# Rewards
DIST_REWARD_SCALE = 200.0
REACH_BONUS = 1000.0
FLIP_PENALTY = 50.0
APPROACH_VELOCITY_SCALE = 30.0
TIME_PENALTY = 0.5


# ============================================================================
# NEURAL NETWORK
# ============================================================================

class CNNEncoder(nn.Module):
    """Encodes a (1, CAM_H, CAM_W) grayscale image to a flat latent vector.

    Architecture (with default 48x64 input):
      (1,48,64) -> conv(16, k=3, s=2, p=1) -> (16,24,32)
               -> conv(32, k=3, s=2, p=1) -> (32,12,16)
               -> conv(32, k=3, s=2, p=1) -> (32, 6, 8)
               -> flatten -> linear -> CNN_LATENT_DIM
    """
    def __init__(self, h=CAM_H, w=CAM_W, latent_dim=CNN_LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
        )
        # Compute flat size dynamically (works for any h, w)
        with torch.no_grad():
            flat_size = self.conv(torch.zeros(1, 1, h, w)).shape[1]
        self.project = nn.Linear(flat_size, latent_dim)

    def forward(self, img):
        # img: (batch, 1, H, W), values in [0, 1]
        return F.elu(self.project(self.conv(img)))


class Actor(nn.Module):
    """CNN image encoder + state hint -> action distribution."""
    def __init__(self, action_dim):
        super().__init__()
        self.encoder = CNNEncoder()
        self.mlp = nn.Sequential(
            nn.Linear(CNN_LATENT_DIM + STATE_DIM, HIDDEN_DIM),
            nn.ELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ELU(),
            nn.Linear(HIDDEN_DIM, action_dim),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, img, state):
        latent = self.encoder(img)
        mu = self.mlp(torch.cat([latent, state], dim=-1))
        std = self.log_std.exp().clamp(0.01, 1.0)
        return Normal(mu, std)


class Critic(nn.Module):
    """CNN image encoder + state hint -> scalar value."""
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.mlp = nn.Sequential(
            nn.Linear(CNN_LATENT_DIM + STATE_DIM, HIDDEN_DIM),
            nn.ELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ELU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, img, state):
        latent = self.encoder(img)
        return self.mlp(torch.cat([latent, state], dim=-1)).squeeze(-1)


# ============================================================================
# HELPERS
# ============================================================================

def get_random_targets(num):
    return (torch.rand((num, 2), device=gs.device) - 0.5) * SPACE_SIZE


def get_image_and_state(robot, chassis_idx, targets, cam):
    """Update camera pose, render image, compute state hint for env 0."""
    car_pos = robot.get_links_pos()[0, chassis_idx, :]   # (3,)
    car_quat = robot.get_quat()[0]                        # (4,) [w,x,y,z]
    base_vel = robot.get_vel()[0]                         # (3,)

    # Forward vector from quaternion (+X body axis in world frame)
    qw, qx, qy, qz = car_quat[0], car_quat[1], car_quat[2], car_quat[3]
    fwd = torch.stack([
        1 - 2*(qy*qy + qz*qz),
        2*(qx*qy + qw*qz),
        2*(qx*qz - qw*qy),
    ])

    if CAM_TOP_DOWN:
        # Overhead camera: directly above car, looking straight down
        # up= car's forward vector so image is oriented with car heading at top
        cam_pos = car_pos + torch.tensor([0.0, 0.0, CAM_TOP_DOWN_HEIGHT], device=gs.device)
        cam_look = car_pos.clone()
        cam.set_pose(pos=cam_pos.cpu().numpy(), lookat=cam_look.cpu().numpy(),
                     up=fwd.cpu().numpy())
    else:
        # Camera mounted at car front, looking to car's RIGHT (90° CW from forward)
        # right = rotate fwd 90° clockwise around Z: [fx, fy] -> [fy, -fx]
        right = torch.stack([fwd[1], -fwd[0], torch.zeros(1, device=gs.device).squeeze()])
        up_offset = torch.tensor([0.0, 0.0, CAM_HEIGHT_OFFSET], device=gs.device)
        cam_pos = car_pos + up_offset + fwd * CAM_FORWARD_OFFSET
        cam_look = cam_pos + right 
        cam.set_pose(pos=cam_pos.cpu().numpy(), lookat=cam_look.cpu().numpy())

    rgb, _, _, _ = cam.render(rgb=True)  # (H, W, 3) uint8, torch.Tensor or numpy
    if isinstance(rgb, torch.Tensor):
        rgb_np = rgb.cpu().numpy()
    else:
        rgb_np = rgb
    gray = rgb_np.mean(axis=2, keepdims=True) / 255.0           # (H, W, 1) float
    img_t = torch.tensor(gray, dtype=torch.float32, device=gs.device).permute(2, 0, 1).unsqueeze(0)  # (1,1,H,W)

    # State hint: [sin(bearing), cos(bearing), distance, forward_speed]
    yaw = torch.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
    rel = targets[0] - car_pos[:2]
    bearing = torch.atan2(torch.sin(torch.atan2(rel[1], rel[0]) - yaw),
                          torch.cos(torch.atan2(rel[1], rel[0]) - yaw))
    dist = rel.norm()
    fwd_speed = (base_vel[:2] * fwd[:2]).sum()
    state_t = torch.stack([bearing.sin(), bearing.cos(), dist, fwd_speed]).unsqueeze(0)  # (1, 4)

    return img_t, state_t, rgb_np  # also return raw RGB for cv2 display


def reset_all(robot, home_pos, home_quat):
    robot.set_pos(home_pos.unsqueeze(0))
    robot.set_quat(home_quat.unsqueeze(0))
    robot.zero_all_dofs_velocity()


# ============================================================================
# MAIN
# ============================================================================

def main():
    gs.init(backend=gs.gpu, performance_mode=False)
    sim_options = gs.options.SimOptions(dt=DT, substeps=SUBSTEPS)

    if SHOW_VIEWER:
        viewer_options = gs.options.ViewerOptions(
            max_FPS=60,
            camera_pos=(3.0, 0.0, 3.5),
            camera_lookat=(0.0, 0.0, 0.0),
            res=(1280, 720),
        )
        scene = gs.Scene(
            sim_options=sim_options,
            viewer_options=viewer_options,
            show_viewer=True,
            vis_options=gs.options.VisOptions(shadow=False),
        )
    else:
        scene = gs.Scene(sim_options=sim_options, show_viewer=False)

    scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Rough())
    target_vis = scene.add_entity(
        gs.morphs.Sphere(radius=0.2, collision=False),
        surface=gs.surfaces.Default(color=(1, 0, 0)),
    )
    robot = scene.add_entity(gs.morphs.URDF(file=URDF_PATH, fixed=False, collision=True))

    # Front-facing offscreen camera — repositioned each step
    cam = scene.add_camera(res=(CAM_W, CAM_H), fov=CAM_FOV, GUI=False)

    scene.build(n_envs=N_ENVS, env_spacing=(0, 0))

    # Robot joints
    joint_names = ['revolute_1', 'revolute_2', 'revolute_3', 'revolute_4']
    actuated_idx = [robot.get_joint(n).dofs_idx_local[0] for n in joint_names]
    robot.set_dofs_kp([0.0] * 4, actuated_idx)
    robot.set_dofs_kv([0.1] * 4, actuated_idx)
    action_dim = len(actuated_idx)
    chassis_idx = robot.get_link(CHASSIS_LINK_NAME).idx - robot.link_start

    home_pos = torch.tensor(HOME_POS, device=gs.device)
    home_quat = torch.tensor(HOME_QUAT, device=gs.device)

    actor = Actor(action_dim).to(gs.device)
    critic = Critic().to(gs.device)
    actor_opt = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_opt = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f"runs/cam_ppo_{datetime.now().strftime('%H%M%S')}")

    print(f"Camera PPO | Image: {CAM_W}x{CAM_H} grayscale | State hint: {STATE_DIM}D | Latent: {CNN_LATENT_DIM}D")
    print(f"Car camera POV: {'OpenCV window' if SHOW_CAR_CAM else 'disabled'}")
    print(f"3rd-person viewer: {'on' if SHOW_VIEWER else 'off'}\n")

    if SHOW_CAR_CAM:
        cv2.namedWindow('Car Camera POV', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Car Camera POV', CAM_W * 4, CAM_H * 4)
        cv2.moveWindow('Car Camera POV', 100, 100)  # pixels from top-left

    targets = get_random_targets(N_ENVS)

    for epoch in range(N_EPOCHS):
        reset_all(robot, home_pos, home_quat)
        targets = get_random_targets(N_ENVS)

        imgs_buf, states_buf, acts_buf, logps_buf = [], [], [], []
        rews_buf, vals_buf, dones_buf = [], [], []

        prev_car_pos = robot.get_links_pos()[0, chassis_idx, :2].clone()
        successes = 0

        for step in range(STEPS_PER_EPOCH):
            img, state, rgb_frame = get_image_and_state(robot, chassis_idx, targets, cam)

            # Show car camera in OpenCV window
            if SHOW_CAR_CAM:
                display = cv2.resize(rgb_frame, (CAM_W * 4, CAM_H * 4), interpolation=cv2.INTER_NEAREST)
                cv2.imshow('Car Camera POV', display[:, :, ::-1])  # RGB -> BGR for cv2
                cv2.waitKey(1)

            with torch.no_grad():
                dist_net = actor(img, state)
                action = dist_net.sample()       # (1, action_dim)
                log_prob = dist_net.log_prob(action).sum(-1)  # (1,)
                value = critic(img, state)       # (1,)

            robot.control_dofs_velocity(action * 30.0, actuated_idx)
            target_vis.set_pos(torch.cat([targets, torch.tensor([[0.2]], device=gs.device)], dim=-1))
            scene.step()

            # Reward
            car_pos = robot.get_links_pos()[0, chassis_idx, :2]
            progress = torch.norm(targets[0] - prev_car_pos) - torch.norm(targets[0] - car_pos)
            base_vel = robot.get_vel()[0]
            to_target = targets[0] - car_pos
            vel_toward = (base_vel[:2] * to_target / (to_target.norm() + 1e-6)).sum()

            reward = DIST_REWARD_SCALE * progress
            reward += APPROACH_VELOCITY_SCALE * vel_toward.clamp(min=0)
            reward -= TIME_PENALTY

            reached = torch.norm(targets[0] - car_pos) < TARGET_REACH_THRESHOLD
            if reached:
                reward += REACH_BONUS
                successes += 1
                targets = get_random_targets(N_ENVS)

            base_quat = robot.get_quat()[0]
            flipped = (1 - 2*(base_quat[1]**2 + base_quat[2]**2)) < -0.1
            if flipped:
                reward -= FLIP_PENALTY
                reset_all(robot, home_pos, home_quat)
                targets = get_random_targets(N_ENVS)

            done = float(reached or flipped)

            imgs_buf.append(img)
            states_buf.append(state)
            acts_buf.append(action)
            logps_buf.append(log_prob)
            rews_buf.append(reward.unsqueeze(0))
            vals_buf.append(value)
            dones_buf.append(torch.tensor([done], device=gs.device))

            prev_car_pos = car_pos.clone()

        # Final bootstrap value
        with torch.no_grad():
            img_f, state_f, _ = get_image_and_state(robot, chassis_idx, targets, cam)
            final_val = critic(img_f, state_f)
        vals_buf.append(final_val)

        # GAE
        rews_t = torch.cat(rews_buf)       # (T,)
        vals_t = torch.cat(vals_buf)       # (T+1,)
        dones_t = torch.cat(dones_buf)     # (T,)
        adv = torch.zeros_like(rews_t)
        last_gae = 0.0
        for t in reversed(range(STEPS_PER_EPOCH)):
            delta = rews_t[t] + GAMMA * vals_t[t+1] * (1 - dones_t[t]) - vals_t[t]
            adv[t] = last_gae = delta + GAMMA * GAE_LAMBDA * (1 - dones_t[t]) * last_gae
        returns = adv + vals_t[:-1]

        # Stack buffers
        imgs_t = torch.cat(imgs_buf)     # (T, 1, H, W)
        states_t = torch.cat(states_buf) # (T, 4)
        acts_t = torch.cat(acts_buf)     # (T, action_dim)
        logps_t = torch.cat(logps_buf)   # (T,)
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO update
        pl_log, vl_log, ent_log = [], [], []
        for _ in range(N_PPO_EPOCHS):
            perm = torch.randperm(STEPS_PER_EPOCH, device=gs.device)
            for i in range(0, STEPS_PER_EPOCH, MINI_BATCH_SIZE):
                idx = perm[i:i+MINI_BATCH_SIZE]

                d = actor(imgs_t[idx], states_t[idx])
                new_logp = d.log_prob(acts_t[idx]).sum(-1)
                new_val = critic(imgs_t[idx], states_t[idx])
                ent = d.entropy().mean()

                ratio = (new_logp - logps_t[idx]).exp()
                s1 = ratio * adv_norm[idx]
                s2 = ratio.clamp(1 - PPO_EPSILON, 1 + PPO_EPSILON) * adv_norm[idx]
                pl = -torch.min(s1, s2).mean()
                vl = 0.5 * ((new_val - returns[idx]) ** 2).mean()

                loss = pl + VALUE_COEFF * vl - ENTROPY_COEFF * ent
                actor_opt.zero_grad(); critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), GRAD_CLIP_NORM)
                actor_opt.step(); critic_opt.step()

                pl_log.append(pl.item()); vl_log.append(vl.item()); ent_log.append(ent.item())

        avg_rew = rews_t.mean().item()
        print(f"Epoch {epoch:4d} | Reward: {avg_rew:+7.2f} | Targets: {successes:3d} | "
              f"PLoss: {np.mean(pl_log):.4f} | Entropy: {np.mean(ent_log):.4f}")

        writer.add_scalar("Train/Reward", avg_rew, epoch)
        writer.add_scalar("Train/Successes", successes, epoch)
        writer.add_scalar("Train/PolicyLoss", np.mean(pl_log), epoch)
        writer.add_scalar("Train/ValueLoss", np.mean(vl_log), epoch)
        writer.add_scalar("Train/Entropy", np.mean(ent_log), epoch)

    if SHOW_CAR_CAM:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
