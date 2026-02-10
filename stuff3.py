import genesis as gs
import numpy as np
import keyboard  # <--- New library

gs.init(backend=gs.gpu)
scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())

# Load your robot
robot = scene.add_entity(
    gs.morphs.URDF(
        file=r'C:\Users\h\Desktop\GenesisPhysics\onshape\urdf_output\robot.urdf',
        # fixed=True
    )
)

scene.build()

# 1. Initialize Controller Gains
robot.set_dofs_kp([1000.0] * robot.n_dofs) 
robot.set_dofs_kv([50.0] * robot.n_dofs)  

# 2. Control state
target_pos = 0.0
increment = 0.01 

print("--- Control Guide ---")
print("Hold 'UP arrow'   : Increase position")
print("Hold 'DOWN arrow' : Decrease position")
print("Press 'R'         : Reset")
print("Press 'ESC'       : Exit")

# 3. Simulation Loop
while True:
    # Check keyboard inputs
    if keyboard.is_pressed('up'):
        target_pos += increment
    elif keyboard.is_pressed('down'):
        target_pos -= increment
        
    if keyboard.is_pressed('r'):
        target_pos = 0.0
        
    if keyboard.is_pressed('esc'):
        break

    # Apply command to joint 0
    # We pass target_pos as a list/array and the index [0]
    robot.control_dofs_position([target_pos], [0])
    
    scene.step()