import glfw
import pyrr
import math
import random
import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram
from OpenGL.GL.shaders import compileShader

WIDTH, HEIGHT = 1280, 720

vertex_shader = """
    # version 330 core

    layout(location = 0) in vec3 a_position;     // vertex position attribute
    layout(location = 1) in vec2 a_texture;     // vertex texture attribute
    layout(location = 2) in vec3 a_normal;       // vertex normal attribute (light tracking)
    //layout(location = 3) in mat4 a_instanceMatrix;      // instance matrix

    uniform mat4 projection;     // projection matrix
    uniform mat4 view;       // view matrix
    uniform mat4 model;      // translation and rotation matrix

    out vec2 v_texture;     // texture vertex pass to fragment shader

    void main()
    {
        //vec3 final_position = a_position + a_offset;
        gl_Position = projection * view * model  * vec4(a_position, 1.0);
        v_texture = a_texture;
    }
"""

fragment_shader = """
    # version 330 core

    in vec2 v_texture;      // texture vertex passed from vertex shader

    uniform sampler2D s_texture;        // texture for vertex

    out vec4 out_color;     // output color of all vectices

    void main()
    {
        // map texture using texture vertex
        out_color = texture(s_texture, v_texture);
    }
"""

# 3D obejct file loader class that is capable of loading .obj file and texture file exported from blender
class ObjectLoader:
    buffer = []

    @staticmethod
    # find the data in one line
    def search_data(data_values, coordinates, skip, data_type):
        for d in data_values:
            if d == skip:
                continue
            # vertex, vertex texture, vertex normal
            if data_type == 'float':
                coordinates.append(float(d))
            # face value
            elif data_type == 'int':
                # the value in obj file start at 1
                # python use 0 as the first index, so we subtract 1
                coordinates.append(int(d)-1)

    @staticmethod
    # sorted vertex buffer for use with glDrawArrays function
    def create_sorted_vertex_buffer(indices_data, vertices, textures, normals):
        for i, ind in enumerate(indices_data):
            if i % 3 == 0: # sort the vertex coordinates
                start = ind * 3
                end = start + 3
                ObjectLoader.buffer.extend(vertices[start:end])
            elif i % 3 == 1: # sort the texture coordinates
                start = ind * 2
                end = start + 2
                ObjectLoader.buffer.extend(textures[start:end])
            elif i % 3 == 2: # sort the normal coordinates
                start = ind * 3
                end = start + 3
                ObjectLoader.buffer.extend(normals[start:end])

    @staticmethod
    def load_obj(file):
        vertex_coords = [] # buffer that contains vertex coordinates
        texture_coords = [] # buffer that contains texture coordinates (u and v)
        normal_coords = [] # buffer that contains normal coordinates
        all_indices = [] # buffer that contains all the vertex, texture and normal indices (face)
        indices = [] # buffer that contains vertex indices for indexed drawing

        with open(file, 'r') as f:
            line = f.readline()
            while line:
                values = line.split()
                if values[0] == 'v': # vertex coordinates
                    ObjectLoader.search_data(values, vertex_coords, 'v', 'float')
                elif values[0] == 'vt': # vertex texture coordinates
                    ObjectLoader.search_data(values, texture_coords, 'vt', 'float')
                elif values[0] == 'vn': # vertex normal coordinates
                    ObjectLoader.search_data(values, normal_coords, 'vn', 'float')
                elif values[0] == 'f': # face (each face is a triangle which contains all vertex, texture and normal indices)
                    for value in values[1:]:
                        val = value.split('/')
                        ObjectLoader.search_data(val, all_indices, 'f', 'int')
                        indices.append(int(val[0])-1) # python index starts at 0
                line = f.readline() # read another line of obj file

        # used for glDrawArrays function
        ObjectLoader.create_sorted_vertex_buffer(all_indices, vertex_coords, texture_coords, normal_coords)

        # vertex for drawing
        vertex = ObjectLoader.buffer.copy()
        ObjectLoader.buffer = [] # free up memory

        return np.array(indices, dtype=np.uint32), np.array(vertex, dtype=np.float32)
    
# Camera class that controls the camera's movement using mouse and keyboard
class Camera:
    def __init__(self):
        # camera attributes
        self.camera_position = pyrr.Vector3([0.0, 0.0, 3.0]) # where is camera located
        self.camera_front = pyrr.Vector3([0.0, 0.0, -1.0]) # the direction of where camera looks
        self.camera_up = pyrr.Vector3([0.0, 1.0, 0.0])
        self.camera_right = pyrr.Vector3([1.0, 0.0, 0.0])
        self.mouse_sensitivity = 0.05   # mouse sensitivity
        self.camera_yaw = -90  # camera yaw left and right
        self.camera_pitch = 0  # camera pitch up and down

    def get_view_matrix(self):
        return pyrr.matrix44.create_look_at(self.camera_position, 
                                            self.camera_position + self.camera_front,
                                            self.camera_up)

    def process_mouse_movement(self, x_offset, y_offset, constrain_pitch=True):
        x_offset *= self.mouse_sensitivity
        y_offset *= self.mouse_sensitivity

        self.camera_yaw += x_offset
        self.camera_pitch += y_offset
        
        # give constrain to the camera pitch range
        if constrain_pitch:
            if self.camera_pitch > 45:
                self.camera_pitch = 45
            if self.camera_pitch < -45:
                self.camera_pitch = -45

        # update the camera vector
        self.update_camera_vector()

    # process keyboard press action
    def process_keyboard(self, direction, velocity):
        if direction == "FORWARD":
            self.camera_position += self.camera_front * velocity
        if direction == "BACKWARD":
            self.camera_position -= self.camera_front * velocity
        if direction == "LEFT":
            self.camera_position -= self.camera_right * velocity
        if direction == "RIGHT":
            self.camera_position += self.camera_right * velocity             

    def update_camera_vector(self):
        front = pyrr.Vector3([0.0, 0.0, 0.0])
        front.x = math.cos(math.radians(self.camera_yaw)) * math.cos(math.radians(self.camera_pitch))
        front.y = math.sin(math.radians(self.camera_pitch))
        front.z = math.sin(math.radians(self.camera_yaw)) * math.cos(math.radians(self.camera_pitch))

        # normalise the camera vector
        self.camera_front = pyrr.vector.normalise(front)
        self.camera_right = pyrr.vector.normalise(pyrr.vector3.cross(self.camera_front, pyrr.Vector3([0.0, 1.0, 0.0])))
        self.camera_up = pyrr.vector.normalise(pyrr.vector3.cross(self.camera_right, self.camera_front))

camera = Camera()
last_x, last_y = WIDTH / 2, HEIGHT / 2
mouse_first_enter = True
left, right, forward, backward = False, False, False, False

# this function will be called everytime we move the mouse inside the glfw window
def mouse_look_callback(window, x_pos, y_pos):
    global last_x, last_y

    # if mouse just entered the window, the mouse position will be reset to mouse's x and y position in glfw window
    if mouse_first_enter:
        last_x = x_pos
        last_y = y_pos

    x_offset = x_pos - last_x
    y_offset = last_y - y_pos # mouse y-axis start from top to bottom and OpenGL y-axis starts from bottom to top

    last_x = x_pos
    last_y = y_pos

    camera.process_mouse_movement(x_offset, y_offset)

# this function will be called everytime the mouse is entered the window or leave the window
def mouse_enter_callback(window, entered):
    global mouse_first_enter

    if entered:
        mouse_first_enter = False
    else:
        mouse_first_enter = True

# this function will be called everytime there is a key pressed in the keyboard
def keyboard_input_callback(window, key, scancode, action, mode):
    global left, right, forward, backward

    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
    if key == glfw.KEY_A and action == glfw.PRESS:
        left = True
    elif key == glfw.KEY_A and action == glfw.RELEASE:
        left = False
    if key == glfw.KEY_D and action == glfw.PRESS:
        right = True
    elif key == glfw.KEY_D and action == glfw.RELEASE:
        right = False
    if key == glfw.KEY_W and action == glfw.PRESS:
        forward = True
    elif key == glfw.KEY_W and action == glfw.RELEASE:
        forward = False
    if key == glfw.KEY_S and action == glfw.PRESS:
        backward = True
    elif key == glfw.KEY_S and action == glfw.RELEASE:
        backward = False

    # reset the key status once the key has been released
    # if key in [glfw.KEY_A, glfw.KEY_D, glfw.KEY_W, glfw.KEY_S] and action == glfw.RELEASE:
    #     left, right, forward, backward = False, False, False, False

# enable continuous movement while key is pressed
def do_movement():
    if left:
        camera.process_keyboard("LEFT", 0.05)
    if right:
        camera.process_keyboard("RIGHT", 0.05)
    if forward:
        camera.process_keyboard("FORWARD", 0.05)
    if backward:
        camera.process_keyboard("BACKWARD", 0.05)

# this function will be called everytime when we resize the window
def resize_window(window, width, height):
    glViewport(0, 0, width, height)
    # generate a new projection matrix everytime window is resized
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, width/height, 0.1, 100)
    # pass the matrix to shader
    glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)

# initialize the glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized")

# create the window
window = glfw.create_window(WIDTH, HEIGHT, "Demo", None, None)
# free up the memory allocated by glfw if window can not be created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created")

# change window's position
glfw.set_window_pos(window, 400 ,200)

# resize window using callback function
glfw.set_window_size_callback(window, resize_window)

# move mouse using callback function
glfw.set_cursor_pos_callback(window, mouse_look_callback)
# enter the window using callback function
glfw.set_cursor_enter_callback(window, mouse_enter_callback)

# keyboard press callback function
glfw.set_key_callback(window, keyboard_input_callback)

# before start drawing we need to initialize OpenGL and this is done by creating an OpenGL context
glfw.make_context_current(window)

# background quad
# quad_vertices = [-0.5, -0.5, 0, 0.0, 0.0,
#                   0.5, -0.5, 0, 1.0, 0.0,
#                   0.5,  0.5, 0, 1.0, 1.0,
#                  -0.5,  0.5, 0, 0.0, 1.0]

# quad_indices = [0, 1, 2, 2, 3, 0]

# quad_vertices = np.array(quad_vertices, dtype=np.float32)
# quad_indices = np.array(quad_indices, dtype=np.uint32)

# load 3d object file
earth_indices, earth_vertices = ObjectLoader.load_obj('assets/3dObject/earth/earth.obj')
moon_indices, moon_vertices = ObjectLoader.load_obj('assets/3dObject/moon/moon.obj')
mars_indices, mars_vertices = ObjectLoader.load_obj('assets/3dObject/mars/mars.obj')
asteroids_indices, asteroids_vertices = ObjectLoader.load_obj('assets/3dObject/asteroids/asteroids.obj')

# compile shader program
shader = compileProgram(compileShader(vertex_shader, GL_VERTEX_SHADER), compileShader(fragment_shader, GL_FRAGMENT_SHADER))

# background image
# Quad VAO
# quad_VAO = glGenVertexArrays(1)
# glBindVertexArray(quad_VAO)

# # Quad Vertex Buffer Object
# quad_VBO = glGenBuffers(1)
# glBindBuffer(GL_ARRAY_BUFFER, quad_VBO)
# glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)

# # Quad Element Buffer Object
# quad_EBO = glGenBuffers(1)
# glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quad_EBO)
# glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)

# glEnableVertexAttribArray(0)
# glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, quad_vertices.itemsize * 5, ctypes.c_void_p(0))

# glEnableVertexAttribArray(1)
# glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, quad_vertices.itemsize * 5, ctypes.c_void_p(12))

# vertex array object (VAO)
# for earth and moon
VAO = glGenVertexArrays(4)
# vertex buffer object (VBO)
VBO = glGenBuffers(4)
# element buffer object (EBO)
EBO = glGenBuffers(4)

# earth VAO
# any subsequent vertex attribute calls after glBindVertexArray will be stored inside the VAO
glBindVertexArray(VAO[0])
# earth VBO
glBindBuffer(GL_ARRAY_BUFFER, VBO[0])
glBufferData(GL_ARRAY_BUFFER, earth_vertices.nbytes, earth_vertices, GL_STATIC_DRAW)
# earth EBO
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[0])
glBufferData(GL_ELEMENT_ARRAY_BUFFER, earth_indices.nbytes, earth_indices, GL_STATIC_DRAW)

# put the vertex data to position attribute of the shader program
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, earth_vertices.itemsize * 8, ctypes.c_void_p(0))
# put the texture data to texture attribute of the shader program
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, earth_vertices.itemsize * 8, ctypes.c_void_p(12))
# put the normal data to normal attribute of the shader program
glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, earth_vertices.itemsize * 8, ctypes.c_void_p(20))

# moon VAO
glBindVertexArray(VAO[1])
# moon VBO
glBindBuffer(GL_ARRAY_BUFFER, VBO[1])
glBufferData(GL_ARRAY_BUFFER, moon_vertices.nbytes, moon_vertices, GL_STATIC_DRAW)
# moon EBO
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[1])
glBufferData(GL_ELEMENT_ARRAY_BUFFER, moon_indices.nbytes, moon_indices, GL_STATIC_DRAW)

# put the vertex data to position attribute of the shader program
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, moon_vertices.itemsize * 8, ctypes.c_void_p(0))
# put the texture data to texture attribute of the shader program
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, moon_vertices.itemsize * 8, ctypes.c_void_p(12))
# put the normal data to normal attribute of the shader program
glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, moon_vertices.itemsize * 8, ctypes.c_void_p(20))

# mars VAO
glBindVertexArray(VAO[2])
# mars VBO
glBindBuffer(GL_ARRAY_BUFFER, VBO[2])
glBufferData(GL_ARRAY_BUFFER, mars_vertices.nbytes, mars_vertices, GL_STATIC_DRAW)
# mars EBO
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[2])
glBufferData(GL_ELEMENT_ARRAY_BUFFER, mars_indices.nbytes, mars_indices, GL_STATIC_DRAW)

# put the vertex data to position attribute of the shader program
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, mars_vertices.itemsize * 8, ctypes.c_void_p(0))
# put the texture data to texture attribute of the shader program
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, mars_vertices.itemsize * 8, ctypes.c_void_p(12))
# put the normal data to normal attribute of the shader program
glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, mars_vertices.itemsize * 8, ctypes.c_void_p(20))

# asteroids VAO
glBindVertexArray(VAO[3])
# asteroids VBO
glBindBuffer(GL_ARRAY_BUFFER, VBO[3])
glBufferData(GL_ARRAY_BUFFER, asteroids_vertices.nbytes, asteroids_vertices, GL_STATIC_DRAW)
# asteroids EBO
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[3])
glBufferData(GL_ELEMENT_ARRAY_BUFFER, asteroids_indices.nbytes, asteroids_indices, GL_STATIC_DRAW)

# put the vertex data to position attribute of the shader program
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, asteroids_vertices.itemsize * 8, ctypes.c_void_p(0))
# put the texture data to texture attribute of the shader program
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, asteroids_vertices.itemsize * 8, ctypes.c_void_p(12))
# put the normal data to normal attribute of the shader program
glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, asteroids_vertices.itemsize * 8, ctypes.c_void_p(20))


# create texture object
# for earth and moon
texture = glGenTextures(5)

# earth texture
glBindTexture(GL_TEXTURE_2D, texture[0])
# set the texture wrapping parameters (U and V coordinates)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
# set texture filtering parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
# load image
earth_image = Image.open("assets/3dObject/earth/Diffuse_2K.png")
earth_image = earth_image.transpose(Image.FLIP_TOP_BOTTOM)
earth_img_data = earth_image.convert("RGBA").tobytes()
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, earth_image.width, earth_image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, earth_img_data)

# moon texture
glBindTexture(GL_TEXTURE_2D, texture[1])
# set the texture wrapping parameters (U and V coordinates)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
# set texture filtering parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
# load image
moon_image = Image.open("assets/3dObject/moon/Diffuse_2K.png")
moon_image = moon_image.transpose(Image.FLIP_TOP_BOTTOM)
moon_img_data = moon_image.convert("RGBA").tobytes()
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, moon_image.width, moon_image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, moon_img_data)

# mars texture
glBindTexture(GL_TEXTURE_2D, texture[2])
# set the texture wrapping parameters (U and V coordinates)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
# set texture filtering parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
# load image
mars_image = Image.open("assets/3dObject/mars/mars.png")
mars_image = mars_image.transpose(Image.FLIP_TOP_BOTTOM)
mars_img_data = mars_image.convert("RGBA").tobytes()
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mars_image.width, mars_image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, mars_img_data)

# asteroids texture
glBindTexture(GL_TEXTURE_2D, texture[3])
# set the texture wrapping parameters (U and V coordinates)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
# set texture filtering parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
# load image
asteroids_image = Image.open("assets/3dObject/asteroids/asteroids.png")
asteroids_image = asteroids_image.transpose(Image.FLIP_TOP_BOTTOM)
asteroids_img_data = asteroids_image.convert("RGBA").tobytes()
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, asteroids_image.width, asteroids_image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, asteroids_img_data)

# background texture
# glBindTexture(GL_TEXTURE_2D, texture[4])
# # set the texture wrapping parameters (U and V coordinates)
# glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
# glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
# # set texture filtering parameters
# glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
# glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
# # load image
# bg_image = Image.open("assets/image/universe.jpg")
# bg_image = bg_image.transpose(Image.FLIP_TOP_BOTTOM)
# bg_img_data = bg_image.convert("RGBA").tobytes()
# glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, bg_image.width, bg_image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_img_data)

# use the shader program
glUseProgram(shader)

# window clear color
glClearColor(0, 0, 0, 1)

# enable the depth buffer and transparency
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# projection matrix
projection = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH/HEIGHT, 0.1, 1000)
# tanslation matrix for 3d object's postion
earth_position = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, -8]))
#moon_position = pyrr.matrix44.create_from_translation(pyrr.Vector3([50, 0, -20]))
mars_position = pyrr.matrix44.create_from_translation(pyrr.Vector3([-100, 0, -100]))
mars_scale = pyrr.matrix44.create_from_scale(pyrr.Vector3([5, 5, 5]))
#bg_scale = pyrr.matrix44.create_from_scale(pyrr.Vector3([10000, 10000, 10000]))
#quad_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, -500]))

# view matrix
# eye target([0, 0, 0] always look at the center of the screen) up
#view = pyrr.matrix44.create_look_at(pyrr.Vector3([0, 0, 3]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))

# in order to pass matrix to shader program we need to first locate where we should put the matrix
projection_loc = glGetUniformLocation(shader, "projection")
model_loc = glGetUniformLocation(shader, "model")
view_loc = glGetUniformLocation(shader, "view")

# pass the matrix to the shader program
# we only need to pass the projection and view matrix once
glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)
#glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

#print(moon_position)

# window loop
while not glfw.window_should_close(window):
    # enable interaction with window using mouse and keyboard
    glfw.poll_events()

    do_movement() # enable keyboard camera movement

    # clear buffers to preset color value
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # model = pyrr.matrix44.multiply(bg_scale, quad_pos)

    # glBindVertexArray(quad_VAO)
    # glBindTexture(GL_TEXTURE_2D, texture[4])
    # glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    # glDrawElements(GL_TRIANGLES, len(quad_indices), GL_UNSIGNED_INT, None)

    # the view matrix will be re-calculated in every frame since camera will be moving
    view = camera.get_view_matrix()
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    # rotation matrix
    # it needs to be updated in every iteration
    rotation_x = pyrr.Matrix44.from_x_rotation(0.5 * glfw.get_time())
    rotation_y = pyrr.Matrix44.from_y_rotation(0.3 * glfw.get_time())
    rotation_z = pyrr.Matrix44.from_z_rotation(0.4 * glfw.get_time())
    rotation = pyrr.matrix44.multiply(rotation_x, rotation_y)
    rotation = pyrr.matrix44.multiply(rotation_z, rotation)

    # model matrix combined rotation matrix and translation matrix for earth
    model = pyrr.matrix44.multiply(rotation_y, earth_position)

    # Draw earth
    glBindVertexArray(VAO[0])
    glBindTexture(GL_TEXTURE_2D, texture[0])
    # pass model matrix to shader program
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glDrawArrays(GL_TRIANGLES, 0, len(earth_indices))
    #glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    moon_position = pyrr.matrix44.create_from_translation(
        pyrr.Vector3([30 * math.cos(0.3 * glfw.get_time()) + 20 * math.sin(0.3 * glfw.get_time()), 
                        0, 
                        -20 * math.cos(0.3 * glfw.get_time()) + 30 * math.sin(0.3 * glfw.get_time())]))
    # model matrix combined rotation matrix and translation matrix for moon
    model = pyrr.matrix44.multiply(rotation, moon_position) 

    # Draw moon
    glBindVertexArray(VAO[1])
    glBindTexture(GL_TEXTURE_2D, texture[1])
    # pass model matrix to shader program
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glDrawArrays(GL_TRIANGLES, 0, len(moon_indices))

    # model matrix combined rotation matrix and translation matrix for moon
    model = pyrr.matrix44.multiply(rotation, mars_scale)
    model = pyrr.matrix44.multiply(model, mars_position)

    # Draw mars
    glBindVertexArray(VAO[2])
    glBindTexture(GL_TEXTURE_2D, texture[2])
    # pass model matrix to shader program
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glDrawArrays(GL_TRIANGLES, 0, len(mars_indices))

    #model = asteroids_position

    # Draw asteroids


    # OpenGL use double buffer, swapping between frot and back buffer is necessary
    glfw.swap_buffers(window)

# when window is terminated, glfw library need to be closed and free up the allocated memory
glfw.terminate()