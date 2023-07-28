import numpy as np
import math
from enum import Enum, auto

import pygame
from pygame.locals import *
pygame.font.init()
font  = pygame.font.Font('./Product Sans Regular.ttf', 16)
sfont = pygame.font.Font('./Product Sans Regular.ttf', 16)

c = type('c', (), {'__matmul__': (lambda s, x: (*x.to_bytes(3, 'big'),)), '__sub__': (lambda s, x: (x&255,)*3)})()
bg = c-34
fg = c@0xff9088
green = c@0xa0ffe0

fps = 60

w, h = res = (1280, 720)

PERSP_CONST = 500
DIST_CONST = .002
PAN_FAC = .1

def updateStat(msg = None, update = True):
	rect = (0, h-25, w, 26)
	display.fill(c-0, rect)

	# tsurf = sfont.render(f'{curr_camera.get_pos_array()} {origin_centered = }: {msg!r}',
	msg = (
		f'{n_faces} faces rendered; {origin_centered = }; {msg!r} '
		f'pos [{curr_camera.x:.02f} {curr_camera.y:.02f} {curr_camera.z:.02f}] '
		f'facing {curr_camera.pitch:.02f} {curr_camera.yaw:.02f} '
	)
	if tick > 1e3: msg = f'{msg} ({tick/1e3} seconds to render)'
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

	transparent_helper = pygame.Surface(res)
	transparent_helper.fill(c--1)

	global screen_faces
	screen_faces = []
	# max_size = 10

	# print(faces)
	for face in faces:
		if persp:
			x, face_dist, z = curr_camera.get_screen_coords(face.pos)
			points = [curr_camera.get_screen_coords(point) for point in face.points]
		else:
			x, face_dist, z = (
				  curr_camera.ortho_mat
				@ (face.pos-curr_camera.get_pos_array())
			)
			points = [
				(  curr_camera.ortho_mat
				@ (point-curr_camera.get_pos_array())) * 200
				for point in face.points
			]


		# we might have to do some face chopping for faces that come close
		closest = min(points, key=lambda x: x[1])
		if closest[1] < DIST_CONST: continue  # let them just disappear
		face_norm = np.cross(points[1] - points[0], points[2] - points[1])
		if face_norm[1] < 0: continue
		points = [point[::2] + (w//2, h//2) for point in points]

		# if not -max_size/2 < x+w//2 < w+max_size/2: continue
		# if not -max_size/2 < z+h//2 < h+max_size/2: continue

		# we can add shading to the colour here
		screen_faces.append((face_dist, points, face.col))

	screen_faces.sort(reverse=True, key=lambda x: x[0])
	global n_faces
	n_faces = len(screen_faces)

	for face in screen_faces:
		dist, points, col = face

		pygame.draw.polygon(display, col, points)

		pygame.draw.polygon(transparent_helper, c--1, points)
		pygame.draw.aalines(transparent_helper, c--50, True, points)

	display.blit(transparent_helper, (0, 0), special_flags=BLEND_RGB_MULT)

	# camera pos hud
	hud_size = 100
	display.fill(c--27, (0, 0, hud_size, hud_size*2))
	display.fill(c--27, (0, 0, hud_size, hud_size*2))

	x, y, z = curr_camera.get_pos_array()

	view_x, view_y = x + 25*np.sin(curr_camera.yaw), y + 25*np.cos(curr_camera.yaw)
	display.fill(c-90, (x//5+hud_size//2, -y//5+hud_size//2, 4, 4))
	display.fill(c-int(curr_camera.yaw*255//180%1), (view_x//5+hud_size//2, -view_y//5+hud_size//2, 2, 2))
	display.fill(c-0, (hud_size//2, hud_size//2, 4, 4))
	display.fill(c@0xff0000, (hud_size//2+5, hud_size//2, 4, 4))
	display.fill(c@0x00ff00, (hud_size//2, hud_size//2-5, 4, 4))

	dist = np.linalg.norm((x, y))
	view_x, view_y = (
		-dist + 25*np.cos(curr_camera.pitch), z + 25*np.sin(curr_camera.pitch)
	)
	display.fill(c-90, (-dist//5 + hud_size//2, z//5 + 3*hud_size//2, 4, 4))  # camera
	display.fill(c-int(curr_camera.yaw*255//180%1), (view_x//5 + hud_size//2, view_y//5 + 3*hud_size//2, 2, 2))
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
			Face(np.array([self.x + .5, self.y, self.z]), Facing.EAST, self.type.value),
			Face(np.array([self.x - .5, self.y, self.z]), Facing.WEST, self.type.value),
			Face(np.array([self.x, self.y + .5, self.z]), Facing.NORTH, self.type.value),
			Face(np.array([self.x, self.y - .5, self.z]), Facing.SOUTH, self.type.value),
			Face(np.array([self.x, self.y, self.z + .5]), Facing.UP, self.type.value),
			Face(np.array([self.x, self.y, self.z - .5]), Facing.DOWN, self.type.value),
		]

	def get_center(self):
		return np.array([self.x, self.y, self.z])

	@classmethod
	def get_faces(cls, blocks):
		out = []

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

	def __init__(self, pos: np.ndarray, facing: Facing, col: tuple[3]):
		self.pos = pos
		self.facing = facing
		self.col = col
		self.points = self.get_points()

	def __repr__(self):
		return f'Face({*self.pos,}, {self.facing}, {Blocks(self.col)})'

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
			persp_vec = np.array([
				ortho_vec[0]/(dist*np.tan(self.fov)),
				# np.linalg.norm(ortho_vec)*np.sign(ortho_vec[1]),
				ortho_vec[1],
				ortho_vec[2]/(dist*np.tan(self.fov)),
			])

		# don't consider fov yet
		return persp_vec

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
	Block(0, 0, 1, Blocks.BLUE),
]

faces = Block.get_faces(blocks)

print(len(blocks))
pos = [0, 0]
dragging = False
tick = 0
n_faces = None

curr_camera = Camera(0, -5, 0, .001)
persp = True
origin_centered = True

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
				curr_camera.pitch = -math.atan(curr_camera.z / xy_dist)

				# zero is towards +Y
				curr_camera.yaw = math.atan(curr_camera.x / curr_camera.y)
				# if curr_camera.y > 0: curr_camera.yaw += np.pi/2
				curr_camera.update_ortho_mat()

			elif event.key == K_c:
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
