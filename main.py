import serial
import moderngl
import keyboard
import time
import trimesh
import moderngl_window as glw
import numpy as np
from scenes.temporary_scene import load_temp_scene
from pynput import keyboard
from pathlib import Path

USE_SERIAL = False

if USE_SERIAL:
    ser = serial.Serial('/dev/cu.usbmodem1101', 115200, timeout=0.1)


SCENE_FILENAME = "box1.glb"
UBO_BINDING = 3
BOX_CENTER = np.array([0.278, 0.2, 0.2744], dtype=np.float32)
# BOX_CENTER = np.array([-7, -0.3, 5.5], dtype=np.float32)

pressed_keys = set()

def read_data():
    line = ser.readline().decode('UTF-8', errors='ignore').strip()

    if not line:
        return None

    try:
        yaw = float(line.split('y')[1].split('p')[0])
        pitch = float(line.split('p')[1].split('r')[0])
        roll = float(line.split('r')[1])

        return [yaw, pitch, roll]

    except (ValueError, IndexError):
        return None

def get_camera_direction(yaw, pitch, roll):
    # do some math to take the yaw, pitch, roll and compute the camera position on a sphere of given radius
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)

    x = np.sin(yaw) * np.cos(pitch)
    y = -np.cos(yaw) * np.cos(pitch)
    z = np.sin(pitch)

    return np.array([x, y, z], dtype=np.float32)

    return np.array([x, y, z], dtype=np.float32)

def load_mesh(filename, color=(0.8, 0.8, 0.8)):

    # load as scene
    loaded_scene = trimesh.load(
        filename,
        force="scene"
    )

    # combine all objects into one mesh
    mesh = loaded_scene.to_geometry()

    triangles = []

    for i, face in enumerate(mesh.faces):
        positions = mesh.vertices[face]
        normal = mesh.face_normals[i]

        triangles.append({
            "positions": positions.tolist(),
            "normal": normal.tolist(),
            "color": color,
            "emission": (0.0, 0.0, 0.0),
            "reflectivity": 0.0,
            "transparency": 0.0,
            "ior": 1.0
        })

    # print("Triangles created:", len(triangles))

    return triangles

def create_triangle_ubo(scene, ctx, triangle_count):
    # 7 vec4 slots per triangle:
    # 0-2: positions
    # 3: normal
    # 4: color
    # 5: emission
    # 6: reflectivity, transparency, ior, padding

    gpu_data = np.zeros(
        (triangle_count, 7, 4),
        dtype=np.float32
    )

    for i, tri in enumerate(scene):

        positions = np.asarray(
            tri["positions"],
            dtype=np.float32
        )

        normal = np.asarray(
            tri["normal"],
            dtype=np.float32
        )

        color = np.asarray(
            tri["color"],
            dtype=np.float32
        )

        emission = np.asarray(
            tri["emission"],
            dtype=np.float32
        )

        reflectivity = float(tri.get("reflectivity", 0.0))
        transparency = float(tri.get("transparency", 0.0))
        ior = float(tri.get("ior", 1.0))

        gpu_data[i, 0:3, 0:3] = positions
        gpu_data[i, 3, 0:3] = normal
        gpu_data[i, 4, 0:3] = color
        gpu_data[i, 5, 0:3] = emission
        gpu_data[i, 6, 0] = reflectivity
        gpu_data[i, 6, 1] = transparency
        gpu_data[i, 6, 2] = ior

    # upload to uniform buffer object
    triangle_ubo = ctx.buffer(
        gpu_data.tobytes()
    )

    return triangle_ubo

def load_scene(filename, ctx):

    # load blender file with trimesh
    # temporarily, load hardcoded scene from temporary_scene.py
    # scene = load_mesh(filename)
    scene = load_temp_scene()

    # program 
    program = ctx.program(
    vertex_shader=Path("shaders/vertex.glsl").read_text(encoding="utf-8"), 
    fragment_shader=Path("shaders/fragment.glsl").read_text(encoding="utf-8")
    )

    triangle_ubo = create_triangle_ubo(scene, ctx, len(scene))
    # triangle_ubo = ""

    program["TriangleBuffer"].binding = UBO_BINDING

    triangle_ubo.bind_to_uniform_block(UBO_BINDING)

    # draw fullscreen quad for vertex shader
    quad_vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,

        -1.0,  1.0,
         1.0, -1.0,
         1.0,  1.0,
    ], dtype="f4")

    # upload vertex quad
    vbo = ctx.buffer(quad_vertices.tobytes())

    # attach to vertex array 
    vao = ctx.vertex_array(
        program,
        [(vbo, "2f", "in_vert")]
    )

    return program, vao, vbo, triangle_ubo

# handle input if IMU not connected, can use keyboard to move camera around
def move_camera(pos, dt):
    speed = 1.5
    movement = speed * dt

    if 'w' in pressed_keys:
        pos[2] += movement
    if 's' in pressed_keys:
        pos[2] -= movement
    if 'a' in pressed_keys:
        pos[0] += movement
    if 'd' in pressed_keys:
        pos[0] -= movement
    if 'f' in pressed_keys:
        pos[1] += movement
    if 'r' in pressed_keys:
        pos[1] -= movement

    return pos

def on_press(key):
    try:
        pressed_keys.add(key.char.lower())
    except AttributeError:
        pass

def on_release(key):
    try:
        pressed_keys.discard(key.char.lower())
    except AttributeError:
        pass

if __name__ == '__main__':
    # open moderngl-window 
    window_cls = glw.get_local_window_cls('pyglet')
    window = window_cls(
        size=(512, 512), fullscreen=False, title='ModernGL Window',
        resizable=False, vsync=True, gl_version=(3, 3)
    )
    ctx = window.ctx
    glw.activate_context(window, ctx=ctx)


    # load scene
    program, vao, vbo, triangle_ubo = load_scene(SCENE_FILENAME, ctx)

    fb_width, fb_height = window.fbo.size

    ctx.viewport = (0, 0, fb_width, fb_height)

    program["iResolution"].value = (
                float(fb_width),
                float(fb_height),
                1.0
    )

    ctx.disable(moderngl.DEPTH_TEST)

    # initialize keyboard listener (for moving camera around if IMU not connected)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # initialize camera state
    camera_position = BOX_CENTER
    camera_direction = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    program['tri_count'].value = 30

    last_time = time.perf_counter()
    try:
        while not window.is_closing: 
            if USE_SERIAL:
                imu_data = read_data()
                if imu_data is not None:
                    yaw, pitch, roll = imu_data
                    camera_direction = get_camera_direction(yaw, pitch, roll)
                    

            else:        
                # calculate delta time for camera movement
                current_time = time.perf_counter()
                dt = current_time - last_time
                last_time = current_time

                # clamp
                dt = min(dt, 0.05)

                # update camera position
                camera_position = move_camera(camera_position, dt)

            window.clear(0.0, 0.0, 0.0, 1.0)

            # send new camera position and direction with a uniform 
            program['camera_position'].value = tuple(camera_position)
            program['camera_direction'].value = tuple(camera_direction)

            vao.render(mode=moderngl.TRIANGLES)

            window.swap_buffers()
    except KeyboardInterrupt:
        if USE_SERIAL:
            ser.close()
        window.close()
        triangle_ubo.release()
        vao.release()
        vbo.release()