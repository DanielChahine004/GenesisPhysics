import genesis as gs

gs.init(backend=gs.gpu,)

scene = gs.Scene(show_viewer=True)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

robot = scene.add_entity(
    gs.morphs.URDF(file=r'C:\Users\h\Desktop\GenesisPhysics\onshape\my-robot\robot.urdf',
        fixed=False,
        ),
)
# control_dofs_force
# display(dir(robot.control_dofs_force))

# franka = scene.add_entity(
# gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
# )

# for link in franka.links:
    # print(link.name)


scene.build(n_envs=2, env_spacing=(100.0, 100.0))

# robot.control_dofs_force
# robot.set_dofs_kp([500.0] * robot.n_dofs) # Set the stiffness (Kp) and damping (Kv) for the motors
# robot.set_dofs_kv([20.0] * robot.n_dofs) # High Kp = stiff/strong, High Kv = smooth/less shaky


for i in range(1000):
    # input("Press Enter to step forward...")
    scene.step()
    # hand_link = franka.get_link("hand")
    
    # if i % 100 == 0:
        # print(f"Step {i}: Robot 0 is at {franka.get_pos()[0].cpu().numpy()}")
        # print(f"Step {i}: Hand link is at {hand_link.get_pos()[0].cpu().numpy()}")
