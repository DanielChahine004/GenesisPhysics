import genesis as gs
import numpy as np
import keyboard 

gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())

robot = scene.add_entity(
    gs.morphs.URDF(
        file=r'C:\Users\h\Desktop\GenesisPhysics\onshape\urdf_output\robot.urdf',
    )
)

scene.build()

robot.set_dofs_kp([400.0] * robot.n_dofs)
robot.set_dofs_kv([100.0] * robot.n_dofs)

target_pos = 0.0
increment = 0.005 
track_robot = True

print("--- Control Guide ---")
print("UP/DOWN : Control joint")
print("C       : Toggle Camera Tracking")
print("ESC     : Exit")

# Run one step to ensure the viewer and entities are fully initialized
scene.step()

while True:
    if keyboard.is_pressed('up'):
        target_pos += increment
    elif keyboard.is_pressed('down'):
        target_pos -= increment
    if keyboard.is_pressed('c'):
        track_robot = not track_robot
        print(f"Tracking: {track_robot}")
    if keyboard.is_pressed('esc'):
        break

    robot.control_dofs_position([target_pos], [0])
    
    # --- TRACKING LOGIC ---
    if track_robot:
        # 1. Get current robot position (x, y, z)
        # We convert to numpy for compatibility
        robot_pos = robot.get_pos().cpu().numpy()
        
        # 2. Define where the camera should sit relative to the robot
        # [x_offset, y_offset, z_offset] -> 3 meters back, 3 meters side, 2 meters up
        cam_offset = np.array([3.0, -3.0, 2.0])
        
        # 3. Update the viewer attributes
        # We update BOTH 'pos' and 'lookat' for a follow-cam effect
        try:
            scene.viewer.pos = robot_pos + cam_offset
            scene.viewer.lookat = robot_pos
        except AttributeError:
            # If your version uses 'camera' sub-object
            try:
                scene.viewer.camera.pos = robot_pos + cam_offset
                scene.viewer.camera.lookat = robot_pos
            except:
                pass

    scene.step()