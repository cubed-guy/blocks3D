import numpy as np
import math
from enum import Enum, auto

class perf:
	from time import perf_counter as now
	neighbours = 0
	render = 0
	projection = 0

	@classmethod
	def attr_str(cls):
		return '; '.join(
			f'{attr} = {getattr(cls, attr)*1e3:.0f}ms' for attr in dir(cls)
			if isinstance(getattr(cls, attr), float)
		)

	@classmethod
	def sum(cls):
		return sum(
			getattr(cls, attr) for attr in dir(cls)
			if isinstance(getattr(cls, attr), float)
		)

import pygame
from pygame.locals import *
pygame.font.init()
font  = pygame.font.Font('./Product Sans Regular.ttf', 16)
sfont = pygame.font.Font('./Product Sans Regular.ttf', 16)

c = type('c', (), {'__matmul__': (lambda s, x: (*x.to_bytes(3, 'big'),)), '__sub__': (lambda s, x: (x&255,)*3)})()
bg = c-34
fg = c@0xff9088
green = c@0xa0ffe0

mouse_detector = c-22
block_hover = c--15
face_hover = c--40
default_transparency = c--1
line_colour = c--96

fps = 60

w, h = res = (1280, 720)

PERSP_CONST = 500
DIST_CONST = 1
PAN_FAC = .1


def updateStat(msg = None, update = True):
	rect = (0, h-25, w, 26)
	display.fill(c-0, rect)

	# tsurf = sfont.render(f'{curr_camera.get_pos_array()} {origin_centered = }: {msg!r}',
	msg = (
		f'{n_faces} of {len(faces)} faces rendered; {origin_centered = }; {msg!r} '
		f'pos [{curr_camera.x:.02f} {curr_camera.y:.02f} {curr_camera.z:.02f}] '
		f'facing {curr_camera.pitch:.02f} {curr_camera.yaw:.02f} '
		f'perf {perf.attr_str()} sum = {perf.sum()*1e3:.0f}ms '
		f'({1e3/tick:.01f}fps)'
	)
	tsurf = sfont.render(msg, True, c--1)
	display.blit(tsurf, (5, h-25))

	if update: pygame.display.update(rect)

def resize(size):
	global w, h, res, display
	w, h = res = size
	display = pygame.display.set_mode(res, RESIZABLE)
	updateDisplay()

def updateDisplay():
	display.fill(bg)

	# global screen_faces
	screen_faces = []
	# max_size = 10

	# print(faces)
	perf.projection = perf.now()

	global face_points
	if persp:
		screen_points = curr_camera.get_screen_coords_multi(face_points)
	else:
		screen_points = (
			curr_camera.ortho_mat
			@ (face_points-curr_camera.get_pos_array())
		) * 200

	for face, points in zip(faces, screen_points):

		# we might have to do some face chopping for faces that come close
		closest = min(points, key=lambda x: x[1])
		if closest[1] < DIST_CONST: continue  # let them just disappear
		# print('Cross between', points[1] - points[0], points[2] - points[1])
		# print('points =', points)
		first_edge = points[1, ::2] - points[0, ::2]
		perp = np.array([-first_edge[1], first_edge[0]])
		perp_dot = np.dot(perp, points[2, ::2] - points[1, ::2])
		if perp_dot >= 0: continue
		points = points[..., ::2] + [w//2, h//2]
		# points = [point[::2] + (w//2, h//2) for point in points]
		if (abs(points) > [w, h]).all(): continue

		# if not -max_size/2 < x+w//2 < w+max_size/2: continue
		# if not -max_size/2 < z+h//2 < h+max_size/2: continue

		# we can add shading to the colour here
		screen_faces.append((closest[1], points, face))

	global n_faces
	n_faces = len(screen_faces)

	screen_faces.sort(reverse=True, key=lambda x: x[0])
	perf.projection = perf.now() - perf.projection

	# NOTE: We react to mouse here, in the updateDisplay() function.
	# Coming to think about it it kind of makes sense.

	global selected
	selected_temp = None

	mouse_pos = pygame.mouse.get_pos()

	# if screen_faces: print(len(screen_faces))

	transparent_helper = pygame.Surface(res)
	transparent_helper.fill(c--1)

	perf.render = perf.now()
	for screen_face in screen_faces:
		dist, points, face = screen_face

		pygame.draw.polygon(display, face.block.type.value, points)

		transparent_helper.set_at(mouse_pos, mouse_detector)
		if face is selected:
			pygame.draw.polygon(transparent_helper, face_hover, points)
		elif selected and face.block is selected.block:
			pygame.draw.polygon(transparent_helper, block_hover, points)
		else:
			pygame.draw.polygon(transparent_helper, default_transparency, points)
		if transparent_helper.get_at(mouse_pos) != mouse_detector:
			# we can use `dist` to emulate minecraft style reach
			selected_temp = face

		# print(i, end = '', flush = True)
		pygame.draw.aalines(transparent_helper, line_colour, True, points)
		# print(end = '|', flush = True)
	# if screen_faces: print('\n')

	if selected_temp is None: transparent_helper.set_at(mouse_pos, default_transparency)
	display.blit(transparent_helper, (0, 0), special_flags=BLEND_RGB_MULT)

	perf.render = perf.now() - perf.render

	# NOTE: In this approach, the block gets highlighted 1 frame later
	selected = selected_temp
	# else:
	# 	pygame.draw.polygon(transparent_helper, c--15, selected[1])



	# camera pos hud
	hud_size = 100
	hud_fov = 1
	view_len = 5

	display.fill(c--27, (0, 0, hud_size, hud_size*2))
	display.fill(c--27, (0, 0, hud_size, hud_size*2))

	x, y, z = curr_camera.get_pos_array()

	view_x, view_y = x + view_len*np.sin(curr_camera.yaw), y + view_len*np.cos(curr_camera.yaw)
	display.fill(c-90, (x//hud_fov+hud_size//2, -y//hud_fov+hud_size//2, 4, 4))
	display.fill(c-int(curr_camera.yaw*255//180%1), (view_x//hud_fov+hud_size//2, -view_y//hud_fov+hud_size//2, 2, 2))
	display.fill(c-0, (hud_size//2, hud_size//2, 4, 4))
	display.fill(c@0xff0000, (hud_size//2+5, hud_size//2, 4, 4))
	display.fill(c@0x00ff00, (hud_size//2, hud_size//2-5, 4, 4))

	dist = np.linalg.norm((x, y))
	view_x, view_y = (
		-dist + view_len*np.cos(curr_camera.pitch), z + view_len*np.sin(curr_camera.pitch)
	)
	display.fill(c-90, (-dist//hud_fov + hud_size//2, z//hud_fov + 3*hud_size//2, 4, 4))  # camera
	display.fill(c-int(curr_camera.yaw*255//180%1), (view_x//hud_fov + hud_size//2, view_y//hud_fov + 3*hud_size//2, 2, 2))
	display.fill(c-0, (hud_size//2, 3*hud_size//2, 4, 4))
	display.fill(c@0xff0000, (hud_size//2+5*np.sin(curr_camera.yaw), 3*hud_size//2, 4, 4))
	display.fill(c@0x00ff00, (hud_size//2+5*np.cos(curr_camera.yaw), 3*hud_size//2, 4, 4))

	updateStat(update = False)
	pygame.display.flip()

def toggleFullscreen():
	global pres, res, w, h, display
	res, pres =  pres, res
	w, h = res
	if display.get_flags()&FULLSCREEN: resize(res)
	else: display = pygame.display.set_mode(res, FULLSCREEN); updateDisplay()

class Point:
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z

	def to_array(self):
		return np.array((self.x, self.y, self.z))

class Blocks(Enum):
	STONE = c-80
	BLUE = c@0xff
	GREEN = c@0x8000
	DIRT = c@0x6f5038
	BLACK = c@0x04091b
	WHITE = c@0xfffcee

class Facing(Enum):
	UP = auto()
	DOWN = auto()
	NORTH = auto()
	SOUTH = auto()
	EAST = auto()
	WEST = auto()
	# X = auto()
	# Y = auto()
	# Z = auto()

face_points = []
class Block:
	def __init__(self, x, y, z, type):
		self.x = x
		self.y = y
		self.z = z
		self.type = type
		self.faces = self.generate_faces()

	def generate_faces(self):
		# X+- Y+- Z+-
		return [
			Face(np.array([self.x + .5, self.y, self.z]), Facing.EAST,	self),
			Face(np.array([self.x - .5, self.y, self.z]), Facing.WEST,	self),
			Face(np.array([self.x, self.y + .5, self.z]), Facing.NORTH,	self),
			Face(np.array([self.x, self.y - .5, self.z]), Facing.SOUTH,	self),
			Face(np.array([self.x, self.y, self.z + .5]), Facing.UP,	self),
			Face(np.array([self.x, self.y, self.z - .5]), Facing.DOWN,	self),
		]

	def get_center(self):
		return np.array([self.x, self.y, self.z])

	@classmethod
	def get_faces(cls, blocks):
		out = []
		global face_points
		face_points = []

		# perf.neighbours = perf.now()
		for block in blocks:
			faces = 0b111111  # X+- Y+- Z+-
			for neighbour in blocks:
				diff = neighbour.get_center() - block.get_center()
				if abs(diff).sum() != 1: continue

				if   diff[0] ==  1: faces &= ~0b100000  # X+
				elif diff[0] == -1: faces &= ~0b010000  # X-
				elif diff[1] ==  1: faces &= ~0b001000  # Y+
				elif diff[1] == -1: faces &= ~0b000100  # Y-
				elif diff[2] ==  1: faces &= ~0b000010  # Z+
				elif diff[2] == -1: faces &= ~0b000001  # Z-
				else:
					raise RuntimeError(
						'unexpected difference value, '
						'maybe the blocks are not aligned?'
					)
				if not faces: break
			else:
				if faces & 0b100000: out.append(block.faces[0])
				if faces & 0b010000: out.append(block.faces[1])
				if faces & 0b001000: out.append(block.faces[2])
				if faces & 0b000100: out.append(block.faces[3])
				if faces & 0b000010: out.append(block.faces[4])
				if faces & 0b000001: out.append(block.faces[5])

		# perf.neighbours = perf.now() - perf.neighbours

		face_points.extend(face.points for face in out)
		return out

class Face:
	East_mat = np.array([
		[0,  .5,  .5],
		[0,  .5, -.5],
		[0, -.5, -.5],
		[0, -.5,  .5],
	])
	West_mat = np.array([
		[0, -.5,  .5],
		[0, -.5, -.5],
		[0,  .5, -.5],
		[0,  .5,  .5],
	])
	North_mat = np.array([
		[-.5, 0,  .5],
		[-.5, 0, -.5],
		[ .5, 0, -.5],
		[ .5, 0,  .5],
	])
	South_mat = np.array([
		[ .5, 0,  .5],
		[ .5, 0, -.5],
		[-.5, 0, -.5],
		[-.5, 0,  .5],
	])
	Up_mat = np.array([
		[ .5,  .5, 0],
		[ .5, -.5, 0],
		[-.5, -.5, 0],
		[-.5,  .5, 0],
	])
	Down_mat = np.array([
		[-.5,  .5, 0],
		[-.5, -.5, 0],
		[ .5, -.5, 0],
		[ .5,  .5, 0],
	])

	def __init__(self, pos: np.ndarray, facing: Facing, block: Block):
		self.pos = pos
		self.facing = facing
		self.block = block
		self.points = self.get_points()

	def __repr__(self):
		return f'Face({*self.pos,}, {self.facing}, {self.block.type})'

	def get_points(self):
		if self.facing is Facing.EAST:	return self.pos + self.East_mat
		if self.facing is Facing.WEST:	return self.pos + self.West_mat
		if self.facing is Facing.NORTH:	return self.pos + self.North_mat
		if self.facing is Facing.SOUTH:	return self.pos + self.South_mat
		if self.facing is Facing.UP:	return self.pos + self.Up_mat
		if self.facing is Facing.DOWN:	return self.pos + self.Down_mat
		raise ValueError(f'cannot face in direction {self.facing}')


class Camera:
	def __init__(self, x, y, z, fov):
		self.x = x
		self.y = y
		self.z = z
		self.fov = fov
		self.pitch = 0  # up and down;				along local x
		self.roll = 0	# tilt (0 for turntable);	along local y
		self.yaw = 0	# side to side;				along local z
		self.ortho_mat = np.eye(3)

	def get_pos_array(self):
		return np.array([self.x, self.y, self.z], dtype=np.float64)

	def set_pos(self, pos):
		self.x, self.y, self.z = pos

	def update_ortho_mat(self):
		ortho_mat = np.array([
			[1, 0, 0],
			[0,  np.cos(self.pitch),	np.sin(self.pitch)],
			[0, -np.sin(self.pitch),	np.cos(self.pitch)],
		])
		ortho_mat = ortho_mat @ np.array([
			[np.cos(self.yaw),	-np.sin(self.yaw),	0],
			[np.sin(self.yaw),	 np.cos(self.yaw),	0],
			[0, 0, 1],
		])
		# ortho_mat = ortho_mat @ np.array([
		# 	[np.cos(self.roll), 0, -np.sin(self.roll)],
		# 	[0, 1, 0],
		# 	[np.sin(self.roll), 0,  np.cos(self.roll)],
		# ])
		self.ortho_mat = ortho_mat

	def get_screen_coords(self, coord_arr):
		point_vec = coord_arr-self.get_pos_array()

		ortho_vec = self.ortho_mat @ point_vec

		# # Fish eye style
		# dist = np.linalg.norm(ortho_vec)*np.sign(ortho_vec[1])

		# No distortion
		dist = ortho_vec[1]

		with np.errstate(divide='ignore'):
			xz_fac = 1/(dist*np.tan(self.fov))
			persp_vec = np.array([
				ortho_vec[0]*xz_fac,
				# np.linalg.norm(ortho_vec)*np.sign(ortho_vec[1]),
				ortho_vec[1],
				ortho_vec[2]*xz_fac,
			])

		# don't consider fov yet
		return persp_vec

	def get_screen_coords_multi(self, coord_arrs):
		point_vecs = coord_arrs-self.get_pos_array()

		ortho_vecs = point_vecs @ self.ortho_mat.transpose()
		# ortho_vecs = self.ortho_mat @ point_vec

		# # Fish eye style
		# dist = np.linalg.norm(ortho_vec)*np.sign(ortho_vec[1])

		# No distortion
		dists = ortho_vecs[..., 1]

		with np.errstate(divide='ignore'):
			xz_facs = 1/(dists*np.tan(self.fov))
		persp_vecs = ortho_vecs
		# print(f'{persp_vecs.shape = } {xz_facs.shape = }')
		persp_vecs[..., ::2] *= xz_facs[..., None]

		# don't consider fov yet
		return persp_vecs

# about origin
def rotate_camera_pos(pitch, yaw, camera):
	point_vec = camera.get_pos_array()
	point_vec = np.array([
		[np.cos(camera.yaw),	-np.sin(camera.yaw),	0],
		[np.sin(camera.yaw),	 np.cos(camera.yaw),	0],
		[0, 0, 1],
	]) @ point_vec
	point_vec = np.array([
		[1, 0, 0],
		[0, np.cos(pitch),	-np.sin(pitch)],
		[0, np.sin(pitch),	 np.cos(pitch)],
	]) @ point_vec
	# point_vec = np.array([
	# 	[0, 1, 0],
	# 	[ np.cos(camera.pitch),	0, np.sin(camera.pitch)],
	# 	[-np.sin(camera.pitch),	0, np.cos(camera.pitch)],
	# ]) @ point_vec
	point_vec = np.array([
		[ np.cos(camera.yaw),	np.sin(camera.yaw),	0],
		[-np.sin(camera.yaw),	np.cos(camera.yaw),	0],
		[0, 0, 1],
	]) @ point_vec
	point_vec = np.array([
		[np.cos(yaw),	-np.sin(yaw),	0],
		[np.sin(yaw),	 np.cos(yaw),	0],
		[0, 0, 1],
	]) @ point_vec
	return point_vec

# points = [
# 	Point(0, 0, 0),
# 	Point(50, 0, 0),
# 	Point(0, 50, 0),
# 	Point(0, 0, 200),
# 	Point(-50, 0, 0),
# 	Point(0, -50, 0),
# 	Point(0, 0, -100),
# ]
# points = [
# 	Point(x*50, y*50, z*50, (x*50+100, y*50+100, z*50+100))
# 		for x in range(-2, 3)
# 		for y in range(-2, 3)
# 		for z in range(-2, 3)
# ] + [
# 	Point(5, 0, 0, c@0xff0000),
# 	Point(0, 5, 0, c@0x00ff00),
# 	Point(0, 0, 5, c@0x0000ff),
# 	Point(0, 0, 0, c@0xffffff),
# ]
# points = [
# 	Point(x*10-50, y*10-50,  0+50, (*c@0x0000ff, 10)) for x in range(10) for y in range(10)
# ] + [
# 	Point(x*10-50, 0-50, -z*10+50, (*c@0x00ff00, 10)) for z in range(10) for x in range(10)
# ] + [
# 	Point(0-50, y*10-50, -z*10+50, (*c@0xff0000, 10)) for y in range(10) for z in range(10)
# 	Point(4*x-50, 4*y-50, -4*x/y+50, (*c--1, 10))
# 		for x in range(1, 25)
# 		for y in range(1, 25)
# ]


# points = [
# 	Point(x*10-80, y*10-80,  0+80, (*c@0x000080, 10)) for x in range(16) for y in range(16)
# ] + [
# 	Point(x*10-80, 0-80, -z*10+80, (*c@0x008000, 10)) for z in range(16) for x in range(16)
# ] + [
# 	Point(0-80, y*10-80, -z*10+80, (*c@0x800000, 10)) for y in range(16) for z in range(16)
# ] + [
# 	Point(x*10-80, y*10-80, -10*(15&~((x+y)*0b10001 >> 4))+80, c--1)
# 	for x in range(16)
# 	for y in range(16)
# ] + [
# 	Point(0, 0, 0, c@0xffff00),
# 	Point(-80, -80, 80, c@0xffff00),
# ]

blocks = [
	Block(0, 0, 0, Blocks.DIRT),
	Block(1, 0, 0, Blocks.STONE),
	Block(0, 1, 0, Blocks.GREEN),
	Block(1, 1, 0, Blocks.GREEN),
	Block(0, 0, 1, Blocks.BLUE),
	*[
		Block(x, y, -1, Blocks.WHITE if (x>>1)+(y>>1) & 1 else Blocks.BLACK)
		for x in range(-10, 11) for y in range(-10, 11)
	]
]

faces = Block.get_faces(blocks)

print(len(blocks), 'blocks')
pos = [0, 0]
dragging = False
tick = 1
n_faces = None
selected = None

# curr_camera = Camera(0, -5, 0, .001)

# 307-380ms using _multi()
# 317-390ms without _multi()
curr_camera = Camera(4.234, 0.533, -8.487, .001)
curr_camera.pitch = 0.802
curr_camera.yaw = -1.562
curr_camera.update_ortho_mat()
# 4.234438329921463 0.5336638211072566 -8.487734282078373
# 0.8028514559173922 0 -1.5620696805349243
persp = True
origin_centered = True
next_block = Blocks.DIRT

resize(res)
pres = pygame.display.list_modes()[0]
clock = pygame.time.Clock()
running = True
while running:
	for event in pygame.event.get():
		if event.type == KEYDOWN:
			if   event.key == K_ESCAPE: running = False
			elif event.key == K_F11: toggleFullscreen()
			elif event.key == K_KP5: persp = not persp
			elif event.key == K_o: origin_centered = not origin_centered
			elif event.key in (K_UP, K_DOWN):
				pitch = 5 * np.pi / 180 * (event.key*2 - K_UP - K_DOWN)
				camera_pos = rotate_camera_pos(
					-pitch, 0, curr_camera)
				curr_camera.set_pos(camera_pos)
				curr_camera.pitch += pitch
				curr_camera.update_ortho_mat()
			elif event.key == K_KP_PERIOD:
				# face the origin
				xy_dist = np.linalg.norm((curr_camera.x, curr_camera.y))
				curr_camera.pitch = -math.atan2(curr_camera.z, xy_dist)

				# zero is towards +Y
				curr_camera.yaw = math.atan2(curr_camera.x, curr_camera.y) + np.pi
				curr_camera.update_ortho_mat()

			elif event.key == K_c:
				# 4.234438329921463 0.5336638211072566 -8.487734282078373
				# 0.8028514559173922 0 -1.5620696805349243
				print(
					curr_camera.x,
					curr_camera.y,
					curr_camera.z,
					curr_camera.pitch,
					curr_camera.roll,
					curr_camera.yaw,
				)

		elif event.type == VIDEORESIZE:
			if not display.get_flags()&FULLSCREEN: resize(event.size)
		elif event.type == QUIT: running = False
		elif event.type == MOUSEBUTTONDOWN:
			if event.button in (4, 5):
				delta = event.button*2-9
				curr_camera.fov += delta/10000
			elif event.button == 1:
				dragging = True
		elif event.type == MOUSEBUTTONUP:
			if event.button == 1:
				dragging = False
		elif event.type == MOUSEMOTION:
			if dragging:
				if pygame.key.get_pressed()[K_LSHIFT]:
					if pygame.key.get_pressed()[K_LCTRL]:
						curr_camera.y -= event.rel[0]*PAN_FAC
					else:
						curr_camera.x -= event.rel[0]*PAN_FAC
					curr_camera.z -= event.rel[1]*PAN_FAC
				elif pygame.key.get_pressed()[K_LCTRL]:
					curr_camera.x -= event.rel[0]*PAN_FAC
					curr_camera.y -= event.rel[1]*PAN_FAC
				else:
					yaw = event.rel[0] * np.pi / 720
					pitch = event.rel[1] * np.pi / 720
					if origin_centered:
						camera_pos = rotate_camera_pos(
							pitch, -yaw, curr_camera)
						curr_camera.set_pos(camera_pos)
						curr_camera.yaw += yaw
						curr_camera.pitch += pitch
					else:
						curr_camera.yaw -= yaw
						curr_camera.pitch -= pitch
				curr_camera.update_ortho_mat()

	updateDisplay()
	tick = clock.tick(fps)
