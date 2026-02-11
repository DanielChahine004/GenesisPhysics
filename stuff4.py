import genesis as gs
import numpy as np
import keyboard
import cv2  # You will need to install opencv-python

gs.init(backend=gs.gpu, logging_level='warning')

scene = gs.Scene(show_viewer=True)

plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.URDF(
    file=r'C:\Users\h\Desktop\GenesisPhysics\onshape\urdf_output\robot.urdf', pos=(0, 0, 1.0),
    fixed=True),
    )

# 1. Add a camera to capture the scene for text overlay
cam = scene.add_camera(
    res=(640, 480),
    pos=(3.0, -3.0, 3.0),
    lookat=(0, 0, 1.0),
    fov=30,
    GUI=False # We handle the GUI manually to add text
)

scene.build()

lowers, uppers = robot.get_dofs_limit([0, 1])
lower_limits = lowers.cpu().numpy()
upper_limits = uppers.cpu().numpy()

initial_pos = (lower_limits[0] + upper_limits[0]) / 2
target_pos = initial_pos

robot.set_dofs_position([initial_pos, initial_pos], [0, 1])
robot.set_dofs_kp([500.0, 500.0], [0, 1])
robot.set_dofs_kv([50.0, 50.0], [0, 1])

increment = 0.02

while True:
    if keyboard.is_pressed('up'):
        target_pos += increment
    elif keyboard.is_pressed('down'):
        target_pos -= increment

    target_pos = np.clip(target_pos, lower_limits[0], upper_limits[0])

    if keyboard.is_pressed('esc'):
        break

    robot.control_dofs_position([target_pos, target_pos], [0, 1])
    
    # Get actual position for the display
    actual_pos = robot.get_dofs_position([0])[0].cpu().item()

    # --- SHOW INFO ON SCREEN INSTEAD OF TERMINAL ---
    # This text will appear in the top-left of the 3D viewer window
    scene.viewer.text(
        f"Target: {target_pos:.3f}\nActual: {actual_pos:.3f}\nLimits: {lower_limits[0]:.2f}-{upper_limits[0]:.2f}",
        pos=(20, 20), # Pixels from top-left
        font_size=24,
        color=(1, 1, 1) # White
    )
    
    scene.step()
    
    # 2. Render the camera frame
    rgb, _, _, _ = cam.render()
    # Convert Genesis tensor to a numpy image OpenCV can use
    frame = rgb[0].cpu().numpy().astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 3. Add text overlay using OpenCV
    text = f"Target: {target_pos:.3f} | Actual: {actual_pos:.3f}"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 4. Show the frame in a separate window
    cv2.imshow("Robot Info", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('esc'):
        break