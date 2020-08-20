
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from PIL import Image
import random
from noisegenerator import noisegenerator
import math
from ult import gen_sphere, gen_cube
from math import sin, cos, radians

vertex_src = """
# version 330
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;
uniform mat4 model; // combined translation and rotation
uniform mat4 projection;
out vec4 v_color;
out vec3 v_texture;
uniform mat4 view;
void main()
{
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    v_texture = a_position;
}
"""

fragment_src_p = """
# version 330
in vec3 v_texture;
out vec4 out_color;
uniform float time;
float random(vec2 n) 
{ 
    return fract(sin(dot(n, vec2(13.7654, 3.5514))) * 43758.5453);
}

float noise(vec2 st)
{
    vec2 i = floor(st);
    vec2 f = fract(st);

    float a = random(i);
    float b = random(i + vec2(1., 0.));
    float c = random(i + vec2(0., 1.));
    float d = random(i + vec2(1., 1.));

    vec2 u = f * f * (3. - 2. * f);

    return mix(a, b, u.x) + (c - a)* u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm (vec2 p)
{
    float value = 0.;
    float freq = 1.;
    float amp = .5;
    int octaves = 10;

    for (int i = 0; i < octaves; i++) {
        value += amp * (noise((p - vec2(1.)) * freq));
        freq *= 1.9;
        amp *= .6;
    }

    return value;
}

float pattern(vec2 p)
{
    vec2 offset = vec2(-.5);

    vec2 aPos = vec2(1, 1) * 6.;
    vec2 aScale = vec2(3.);
    float a = fbm(p * aScale + aPos);

    vec2 bPos = vec2(1, 1) * 1.;
    vec2 bScale = vec2(.5);
    float b = fbm((p + a) * bScale + bPos);

    vec2 cPos = vec2(-.6, -.5) + vec2(1, 1) * 2.;
    vec2 cScale = vec2(2.);
    float c = fbm((p + b) * cScale + cPos);

    return c;
}

vec3 palette(float t,vec3 pos) 
{
    if (abs(pos.y)*0.7 + pattern(pos.xy)*0.5 > 0.9){
        vec3 a = vec3(0.6, 0.6, 0.6);
        vec3 b = vec3(.1, .25, .1);
        vec3 c = vec3(1 ,.5, 0.);
        vec3 d = vec3(0., .3, 0);

        return a + b * cos(7.7321 * (c * t + d));    
    }
    if (abs(pos.y)*-0.3 + abs(pos.x)*0.3 + abs(pos.z)*0.3 + pattern(pos.xz)*0.4 - pattern(pos.xy)*pattern(pos.yz)  > 0.3){
        vec3 a = vec3(.35, .4, .1);
        vec3 b = vec3(.15, 0.3, .0);
        vec3 c = vec3(1 ,.5, 0.);
        vec3 d = vec3(0., .3, 0);

        return a + b * cos(7.4321 * (c * t + d));    
    }
    if (abs(pos.y)*0.5 + pattern(pos.xy)*0.7 > 0.4){
        vec3 a = vec3(.01, .35, .01);
        vec3 b = vec3(.1, .25, .0);
        vec3 c = vec3(1 ,.5, 0.);
        vec3 d = vec3(0., .3, 0);

        return a + b * cos(7.8321 * (c * t + d));    
    }
    vec3 a = vec3(.1, .6, .1);
    vec3 b = vec3(.1, .25, .0);
    vec3 c = vec3(1 ,.5, 0.);
    vec3 d = vec3(0., .3, 0);

    return a + b * cos(7.4321 * (c * t + d));
}

void main()
{
    vec2 resolution = vec2(1,1);
    vec2 p = v_texture.xy / resolution.xy;
    p.x *= resolution.x / resolution.y;

    float value = pow(pattern(p), 2.);
    vec3 colour  = palette(value,v_texture);

    out_color = vec4(colour,1);
}
"""


fragment_src_c = """
# version 330
in vec3 v_texture;
out vec4 out_color;
uniform float time;
float random(vec2 n) 
{ 
    return fract(sin(dot(n, vec2(12.8738, 4.43292))) * 43765.324553);
}

float noise(vec2 st)
{
    vec2 i = floor(st);
    vec2 f = fract(st);

    float a = random(i);
    float b = random(i + vec2(1., 0.));
    float c = random(i + vec2(0., 1.));
    float d = random(i + vec2(1., 1.));

    vec2 u = f * f * (3. - 2. * f);

    return mix(a, b, u.x) + (c - a)* u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm (vec2 p)
{
    float value = 0.;
    float freq = 1.;
    float amp = .5;
    int octaves = 10;

    for (int i = 0; i < octaves; i++) {
        value += amp * (noise((p - vec2(1.)) * freq));
        freq *= 1.9;
        amp *= .6;
    }

    return value;
}

float pattern(vec2 p)
{
    vec2 offset = vec2(-.5);

    vec2 aPos = vec2(sin(time*0.05), (time*0.1)) * 6.;
    vec2 aScale = vec2(3.);
    float a = fbm(p * aScale + aPos);

    vec2 bPos = vec2(sin(time * .1), sin(time * .1)) * 1.;
    vec2 bScale = vec2(.5);
    float b = fbm((p + a) * bScale + bPos);

    vec2 cPos = vec2(-.6, -.5) + vec2(sin(-time * .01), sin(time * .1)) * 2.;
    vec2 cScale = vec2(2.);
    float c = fbm((p + b) * cScale + cPos);

    return c;
}

vec3 palette(float t) 
{
    vec3 a = vec3(0.3, 0., 0.3);
    vec3 b = vec3(1., 1., 1.);
    vec3 c = vec3(1 ,1, 1);
    vec3 d = vec3(0.8, 0.8, 0.8);

    return a + b * cos(6.28318 * (c * t + d));
}

void main()
{
    vec2 resolution = vec2(0.69,0.73);
    vec2 p = v_texture.xy / resolution.xy;
    p.x *= resolution.x / resolution.y;

    float value = pow(pattern(p), 2.);
    vec3 colour  = palette(value);

    out_color = vec4(colour,(colour.x+colour.y+colour.z)/5);
}
"""

fragment_src_w = """
# version 330
in vec3 v_texture;
out vec4 out_color;
uniform float time;
float random(vec2 n) 
{ 
    return fract(sin(dot(n, vec2(12.7463, 4.8093))) * 43743.6543);
}

float noise(vec2 st)
{
    vec2 i = floor(st);
    vec2 f = fract(st);

    float a = random(i);
    float b = random(i + vec2(1., 0.));
    float c = random(i + vec2(0., 1.));
    float d = random(i + vec2(1., 1.));

    vec2 u = f * f * (3. - 2. * f);

    return mix(a, b, u.x) + (c - a)* u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm (vec2 p)
{
    float value = 0.;
    float freq = 1.;
    float amp = .5;
    int octaves = 10;

    for (int i = 0; i < octaves; i++) {
        value += amp * (noise((p - vec2(1.)) * freq));
        freq *= 1.9;
        amp *= .6;
    }

    return value;
}

float pattern(vec2 p)
{
    vec2 offset = vec2(-.5);

    vec2 aPos = vec2(sin(time * .05), sin(time * .1)) * 6.;
    vec2 aScale = vec2(3.);
    float a = fbm(p * aScale + aPos);

    vec2 bPos = vec2(sin(time * .1), sin(time * .1)) * 1.;
    vec2 bScale = vec2(.5);
    float b = fbm((p + a) * bScale + bPos);

    vec2 cPos = vec2(-.6, -.5) + vec2(sin(-time * .01), sin(time * .1)) * 2.;
    vec2 cScale = vec2(2.);
    float c = fbm((p + b) * cScale + cPos);

    return c;
}

vec3 palette(float t) 
{
    vec3 a = vec3(0, .2, .6);
    vec3 b = vec3(.0, .25, .4);
    vec3 c = vec3(1. ,1., 1.);
    vec3 d = vec3(0., .1, .2);

    return a + b * cos(6.875318 * (c * t + d));
}

void main()
{
    vec2 resolution = vec2(6,6);
    vec2 p = v_texture.xy / resolution.xy;
    p.x *= resolution.x / resolution.y;

    float value = pow(pattern(p), 2.);
    vec3 colour  = palette(value);

    out_color = vec4(colour,1.25-(colour.x+colour.y+colour.z));
}
"""

fragment_src_b = """
# version 330
in vec3 v_texture;
out vec4 out_color;
uniform float time;
float random(vec2 n) 
{ 
    return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

float noise(vec2 st)
{
    vec2 i = floor(st);
    vec2 f = fract(st);

    float a = random(i);
    float b = random(i + vec2(1., 0.));
    float c = random(i + vec2(0., 1.));
    float d = random(i + vec2(1., 1.));

    vec2 u = f * f * (3. - 2. * f);

    return mix(a, b, u.x) + (c - a)* u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm (vec2 p)
{
    float value = 0.;
    float freq = 1.;
    float amp = .5;
    int octaves = 10;

    for (int i = 0; i < octaves; i++) {
        value += amp * (noise((p - vec2(1.)) * freq));
        freq *= 1.9;
        amp *= .6;
    }

    return value;
}

float pattern(vec2 p)
{
    vec2 offset = vec2(-.5);
    vec2 aPos = vec2(sin(time * .05), sin(time * .1)) * 6.;
    vec2 aScale = vec2(3.);
    float a = fbm(p * aScale + aPos);

    vec2 bPos = vec2(sin(time * .1), sin(time * .1)) * 1.;
    vec2 bScale = vec2(.5);
    float b = fbm((p + a) * bScale + bPos);

    vec2 cPos = vec2(-.6, -.5) + vec2(sin(-time * .01), sin(time * .1)) * 2.;
    vec2 cScale = vec2(2.);
    float c = fbm((p + b) * cScale + cPos);

    return c;
}

vec3 palette(float t, vec3 pos) 
{
    vec3 a = vec3(0, 0, 0);
    if(0.95*random(pos.xy) + 0.1*fbm(pos.xy) + 0.001*pattern(pos.xy) > 0.99){
        a = vec3(1, 1, 1);
    }
    vec3 b = vec3(.1, .05, .4);
    vec3 c = vec3(1. ,1., 1.);
    vec3 d = vec3(0.1, .1, 0.25);

    return a + b * cos(6.28318 * (c * t + d));
}

void main()
{
    vec2 resolution = vec2(1,1);
    vec2 p = v_texture.xy / resolution.xy;
    p.x *= resolution.x / resolution.y;

    float value = pow(pattern(p), 2.);
    vec3 colour  = palette(value,v_texture);

    out_color = vec4(colour,1);
}
"""

fragment_src_m = """
# version 330
in vec3 v_texture;
out vec4 out_color;
uniform float time;
float random(vec2 n) 
{ 
    return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

float noise(vec2 st)
{
    vec2 i = floor(st);
    vec2 f = fract(st);

    float a = random(i);
    float b = random(i + vec2(1., 0.));
    float c = random(i + vec2(0., 1.));
    float d = random(i + vec2(1., 1.));

    vec2 u = f * f * (3. - 2. * f);

    return mix(a, b, u.x) + (c - a)* u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm (vec2 p)
{
    float value = 0.;
    float freq = 1.;
    float amp = .5;
    int octaves = 10;

    for (int i = 0; i < octaves; i++) {
        value += amp * (noise((p - vec2(1.)) * freq));
        freq *= 1.9;
        amp *= .6;
    }

    return value;
}

float pattern(vec2 p)
{
    vec2 offset = vec2(-.5);

    vec2 aPos = vec2(1, 1) * 6.;
    vec2 aScale = vec2(3.);
    float a = fbm(p * aScale + aPos);

    vec2 bPos = vec2(1, 1) * 1.;
    vec2 bScale = vec2(.5);
    float b = fbm((p + a) * bScale + bPos);

    vec2 cPos = vec2(-.6, -.5) + vec2(1, 1) * 2.;
    vec2 cScale = vec2(2.);
    float c = fbm((p + b) * cScale + cPos);

    return c;
}

vec3 palette(float t) 
{
    vec3 a = vec3(0.6, 0.6, 0.6);
    vec3 b = vec3(.2, .2, .2);
    vec3 c = vec3(1 ,1, 1);
    vec3 d = vec3(0.8, 0.8, 0.8);

    return a + b * cos(6.28318 * (c * t + d));
}

void main()
{
    vec2 resolution = vec2(3,3.5);
    vec2 p = v_texture.xy / resolution.xy;
    p.x *= resolution.x / resolution.y;

    float value = pow(pattern(p), 2.);
    vec3 colour  = palette(value);

    out_color = vec4(colour,1);
}
"""



def window_resize(window, width, height):
    glViewport(0, 0, width, height)
    projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, width, 0, height, -1000, 1000)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

if not glfw.init():
    raise Exception("glfw can not be initialized!")

window = glfw.create_window(1280, 720, "Cosc3000 project", None, None)

if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

glfw.set_window_pos(window, 400, 200)
glfw.set_window_size_callback(window, window_resize)
glfw.make_context_current(window)


resoultion = 10
noise = noisegenerator((resoultion,resoultion),["perlin","perlin","ds","ds",],[1/6,1/12,-1/16,1/16])
noise = noisegenerator((resoultion,resoultion),["perlin","ds","perlin","ds",],[1/16,1/12,-1/16,1/16])
noise = noisegenerator((resoultion,resoultion),["perlin","ds","white","bm","ds",],[1/3,-1/9,2,-1/6,1/4])
#noise = noisegenerator((resoultion,resoultion),["perlin","perlin","perlin"],[3/6,1/6,1/8])
noise = noisegenerator((resoultion,resoultion),["ds"],[1/2])
#noise = noisegenerator((resoultion,resoultion),["bm"],[1/4])
#noise = noisegenerator((resoultion,resoultion),["white"],[1/20])
#noise = noisegenerator((resoultion,resoultion),["ls"],[1/4])
indices = []

index = -1
space = np.linspace(-1, 1, num=10)


ver, ind = gen_sphere(resoultion)
ver_shape = ver.shape
for i in range(6):
    if i == 0:
        ver[i,1:ver.shape[1]-1,1:ver.shape[1]-1,2] += noise.layers[i,1:ver.shape[1]-1,1:ver.shape[1]-1]
    if i == 1:
        ver[i,1:ver.shape[1]-1,1:ver.shape[1]-1,2] += -1*noise.layers[i,1:ver.shape[1]-1,1:ver.shape[1]-1]
    if i == 2:
        ver[i,1:ver.shape[1]-1,1:ver.shape[1]-1,0] += noise.layers[i,1:ver.shape[1]-1,1:ver.shape[1]-1]
    if i ==3:
        ver[i,1:ver.shape[1]-1,1:ver.shape[1]-1,0] += -1*noise.layers[i,1:ver.shape[1]-1,1:ver.shape[1]-1]
    if i == 4:
        ver[i,1:ver.shape[1]-1,1:ver.shape[1]-1,1] += noise.layers[i,1:ver.shape[1]-1,1:ver.shape[1]-1]
    if i ==5:
        ver[i,1:ver.shape[1]-1,1:ver.shape[1]-1,1] += -1*noise.layers[i,1:ver.shape[1]-1,1:ver.shape[1]-1]


bg_res = 10
ver_sky, ind_sky = gen_sphere(resoultion)

ver_water, ind_water = gen_sphere(resoultion)
ver_background, ind_background = gen_cube(bg_res)

ver_moon, ind_moon = gen_sphere(resoultion)

for i in range(6):
    if i == 0:
        ver_moon[i,1:ver.shape[1]-1,1:ver.shape[1]-1,2] += noise.layers[i,1:ver.shape[1]-1,1:ver.shape[1]-1]
    if i == 1:
        ver_moon[i,1:ver.shape[1]-1,1:ver.shape[1]-1,2] += -1*noise.layers[i,1:ver.shape[1]-1,1:ver.shape[1]-1]
    if i == 2:
        ver_moon[i,1:ver.shape[1]-1,1:ver.shape[1]-1,0] += noise.layers[i,1:ver.shape[1]-1,1:ver.shape[1]-1]
    if i ==3:
        ver_moon[i,1:ver.shape[1]-1,1:ver.shape[1]-1,0] += -1*noise.layers[i,1:ver.shape[1]-1,1:ver.shape[1]-1]
    if i == 4:
        ver_moon[i,1:ver.shape[1]-1,1:ver.shape[1]-1,1] += noise.layers[i,1:ver.shape[1]-1,1:ver.shape[1]-1]
    if i ==5:
        ver_moon[i,1:ver.shape[1]-1,1:ver.shape[1]-1,1] += -1*noise.layers[i,1:ver.shape[1]-1,1:ver.shape[1]-1]


vertices = np.zeros((resoultion*resoultion*5*6))
vertices[0::5] = ver[:,:,:,0].flatten()
vertices[1::5] = ver[:,:,:,1].flatten()
vertices[2::5] = ver[:,:,:,2].flatten()
vertices = np.array(vertices, dtype=np.float32)
indices = np.array(ind, dtype=np.uint32)

vertices_sky = np.zeros((resoultion*resoultion*5*6))
vertices_sky[0::5] = ver_sky[:,:,:,0].flatten()
vertices_sky[1::5] = ver_sky[:,:,:,1].flatten()
vertices_sky[2::5] = ver_sky[:,:,:,2].flatten()
vertices_sky = np.array(vertices_sky, dtype=np.float32)
indices_sky = np.array(ind_sky, dtype=np.uint32)

vertices_water = np.zeros((resoultion*resoultion*5*6))
vertices_water[0::5] = ver_water[:,:,:,0].flatten()
vertices_water[1::5] = ver_water[:,:,:,1].flatten()
vertices_water[2::5] = ver_water[:,:,:,2].flatten()
vertices_water = np.array(vertices_water, dtype=np.float32)
indices_water = np.array(ind_water, dtype=np.uint32)

vertices_background = np.zeros((bg_res*bg_res*5*6))
vertices_background[0::5] = ver_background[:,:,:,0].flatten()
vertices_background[1::5] = ver_background[:,:,:,1].flatten()
vertices_background[2::5] = ver_background[:,:,:,2].flatten()
vertices_background = np.array(vertices_background, dtype=np.float32)
indices_background = np.array(ind_background, dtype=np.uint32)

vertices_moon = np.zeros((resoultion*resoultion*5*6))
vertices_moon[0::5] = ver_moon[:,:,:,0].flatten()
vertices_moon[1::5] = ver_moon[:,:,:,1].flatten()
vertices_moon[2::5] = ver_moon[:,:,:,2].flatten()
vertices_moon = np.array(vertices_moon, dtype=np.float32)
indices_moon = np.array(ind_moon, dtype=np.uint32)



planet_shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src_p, GL_FRAGMENT_SHADER))
sky_shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src_c, GL_FRAGMENT_SHADER))
water_shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src_w, GL_FRAGMENT_SHADER))
background_shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src_b, GL_FRAGMENT_SHADER))
moon_shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src_m, GL_FRAGMENT_SHADER))

shaders = [planet_shader,sky_shader,water_shader,background_shader,moon_shader]

VBO = glGenBuffers(5)
glBindBuffer(GL_ARRAY_BUFFER, VBO[0])
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
#sky
glBindBuffer(GL_ARRAY_BUFFER, VBO[1])
glBufferData(GL_ARRAY_BUFFER, vertices_sky.nbytes, vertices_sky, GL_STATIC_DRAW)
#sea
glBindBuffer(GL_ARRAY_BUFFER, VBO[2])
glBufferData(GL_ARRAY_BUFFER, vertices_water.nbytes, vertices_water, GL_STATIC_DRAW)
#background
glBindBuffer(GL_ARRAY_BUFFER, VBO[3])
glBufferData(GL_ARRAY_BUFFER, vertices_background.nbytes, vertices_background, GL_STATIC_DRAW)
#moon
glBindBuffer(GL_ARRAY_BUFFER, VBO[4])
glBufferData(GL_ARRAY_BUFFER, vertices_moon.nbytes, vertices_moon, GL_STATIC_DRAW)


EBO = glGenBuffers(5)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[0])
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
#sky bunner
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[1])
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_sky.nbytes, indices_sky, GL_STATIC_DRAW)
#sea
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[2])
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_water.nbytes, indices_water, GL_STATIC_DRAW)
#background
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[3])
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_background.nbytes, indices_background, GL_STATIC_DRAW)
#moon
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[4])
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_moon.nbytes, indices_moon, GL_STATIC_DRAW)
indices_all = [indices,indices_sky,indices_water,indices_background,indices_moon]
glClearColor(0, 0, 0, 0)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# projection = pyrr.matrix44.create_perspective_projection_matrix(45, 1280/720, 0.1, 100)
projection = pyrr.matrix44.create_orthogonal_projection_matrix(0, 1280, 0, 720, -2000, 1000)
translation1 = pyrr.matrix44.create_from_translation(pyrr.Vector3([500, 400, -5]))
translation2 = pyrr.matrix44.create_from_translation(pyrr.Vector3([500, 400, -5]))
translation3 = pyrr.matrix44.create_from_translation(pyrr.Vector3([500, 400, -5]))
translation4 = pyrr.matrix44.create_from_translation(pyrr.Vector3([500, 400, -5]))
translation5 = pyrr.matrix44.create_from_translation(pyrr.Vector3([700, 400, 10]))
scale1 = pyrr.matrix44.create_from_scale(pyrr.Vector3([100, 100, 100]))
scale2 = pyrr.matrix44.create_from_scale(pyrr.Vector3([25, 25, 25]))
scale3 = pyrr.matrix44.create_from_scale(pyrr.Vector3([115, 115, 120]))
scale4 = pyrr.matrix44.create_from_scale(pyrr.Vector3([1200, 1200, 1200]))

scales = [scale1,scale3,scale1,scale4,scale2]
translations = [translation1,translation2,translation3,translation4,translation5]
#shaders = [planet_shader,sky_shader,water_shader,background_shader,moon_shader]
order = [3,0,2,4,1]
timescale = [0,0.6,0.01,0.009,0]


for i in order:
    if i == 3:
        glDisable(GL_DEPTH_TEST)
    glBindBuffer(GL_ARRAY_BUFFER, VBO[i])
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[i])
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 5, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, vertices.itemsize * 5, ctypes.c_void_p(12))
    myUniform_location = glGetUniformLocation(shaders[i], "time")
    glUseProgram(shaders[i])
    glUniform1f(myUniform_location, 1)
    model_loc = glGetUniformLocation(shaders[i], "model")
    proj_loc = glGetUniformLocation(shaders[i], "projection")
    view_loc = glGetUniformLocation(shaders[i], "view")

    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
    if i == 3:
        glEnable(GL_DEPTH_TEST)

zstep = 0
ystep = 0
ydir = -1
zdir = 1

while not glfw.window_should_close(window):
    glfw.poll_events()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    rot_y = [pyrr.Matrix44.from_y_rotation(0.6 * glfw.get_time())]*6
    zstep += zdir
    ystep += 0.4*ydir
    rot_y[3] = pyrr.Matrix44.identity()
    rot_y[5] = pyrr.Matrix44.from_y_rotation(0.4 * glfw.get_time())
    translations[4] = pyrr.matrix44.create_from_translation(pyrr.Vector3([700 + ystep, 400, zstep]))
    if (700 + ystep <= 300):
        ydir = 1
    if (700 + ystep >= 700):
        ydir = -1

    if (zstep >= 500):
        zdir = -1 
    if (zstep <= -500):
        zdir = 1
    jaw = -90
    pitch = 0
    camera_pos = pyrr.Vector3([0.0, 4.0, 3.0])
    front = pyrr.Vector3([0.0, 0.0, 0.0])
    front.x = cos(radians(jaw)) * cos(radians(pitch))
    front.y = sin(radians(pitch))
    front.z = sin(radians(jaw)) * cos(radians(pitch))

    camera_front = pyrr.vector.normalise(front)
    camera_right = pyrr.vector.normalise(pyrr.vector3.cross(camera_front, pyrr.Vector3([0.0, 1.0, 0.0])))
    camera_up = pyrr.vector.normalise(pyrr.vector3.cross(camera_right, camera_front))
    view = pyrr.matrix44.create_look_at(camera_pos, camera_pos + camera_front, camera_up) 

    for i in order:
        if i == 3:
            glDisable(GL_DEPTH_TEST)
        glBindBuffer(GL_ARRAY_BUFFER, VBO[i])
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[i])
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 5, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, vertices.itemsize * 5, ctypes.c_void_p(12))
        myUniform_location = glGetUniformLocation(shaders[i], "time")
        glUseProgram(shaders[i])
        glUniform1f(myUniform_location, glfw.get_time()*timescale[i])
        model_loc = glGetUniformLocation(shaders[i], "model")
        proj_loc = glGetUniformLocation(shaders[i], "projection")
        view_loc = glGetUniformLocation(shaders[i], "view")

        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

        model = pyrr.matrix44.multiply(scales[i], rot_y[i])
        model = pyrr.matrix44.multiply(model, translations[i])

        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

        glDrawElements(GL_TRIANGLES, len(indices_all[i]), GL_UNSIGNED_INT,None)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        if i == 3:
            glEnable(GL_DEPTH_TEST)

    glfw.swap_buffers(window)

glfw.terminate()
