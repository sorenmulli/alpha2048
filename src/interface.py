"""
Gui til 2047½
"""
from os import chdir
from os.path import dirname, realpath
chdir(realpath(dirname(__file__)))

from kivy.app import App
from kivy.core.window import Window
Window.fullscreen = True

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout

from kivy.graphics import Color, Rectangle
from kivy.uix.label import Label

from kivy.clock import Clock

import algorithms as alg
from game import Game2048
from utilities import Agent

class Tile(Label):
	'''
	Hentet fra https://stackoverflow.com/questions/42820798/how-to-add-background-colour-to-a-label-in-kivy
	'''
	def __init__(self, tile = 0, **kwargs):
		super().__init__(**kwargs)
		self.tile = tile
		self.tile_font = '35sp'
		self.pad_frac = 0.08
		
	def decide_color(self, tile):
		alpha = .85
		rgbs = [[0.934, 0.895, 0.844],
			[0.926, 0.887, 0.766],
			[0.98, 0.703, 0.426],
			[0.996, 0.57, 0.328],
			[0.996, 0.441, 0.312],
			[0.922, 0.855, 0.383],
			[0.922, 0.848, 0.297],
			[0.926, 0.832, 0.199],
			[0.926, 0.82, 0.0],
			[0.93, 0.809, 0.0],
			[0.934, 0.895, 0.844],
			[0.926, 0.887, 0.766],
			[0.98, 0.703, 0.426],
			[0.996, 0.57, 0.328],
			[0.996, 0.441, 0.312],
			[0.922, 0.855, 0.383],
			[0.922, 0.848, 0.297],
			[0.926, 0.832, 0.199],
			[0.926, 0.82, 0.0],
			[0.93, 0.809, 0.0]]	
				
		if tile > 11:
			rgb = [0.38, 0.85, 0.57]
		else:
			rgb = rgbs[tile]
			
		rgb.append(alpha)
		return Color(*rgb)

		
	def on_size(self, *args):
		if self.tile:
			#Hvis den er større end 0, skal den opløftes i toerpotens
				self.text = str(2**self.tile)
		else:
				return None

		self.font_size = self.tile_font
	
		padded_size = (self.size[0] - self.pad_frac * self.size[0], self.size[1] - self.pad_frac* self.size[1])	
	
		self.canvas.before.clear()

		with self.canvas.before:
			self.decide_color(self.tile)
			Rectangle(pos=self.pos, size=padded_size)


class Board(GridLayout):

	def __init__(self, game, **kwargs):

		super().__init__(**kwargs)
			
		self.cols = game.n

		for tile in game.board.reshape(game.n**2):
					
			self.add_widget(Tile(tile = tile))


class Root(BoxLayout):

	def __init__(self, game,  **kwargs):
		super().__init__(**kwargs)
		self.orientation = 'vertical'
		self.header_font_size = '55sp'
	
		self.game = game
		self.direction_symbols = ['<', '/\\', '>', 'V', '']	
		self.add_widgets(direction = 4)
	
	def add_widgets(self, direction):
		self.header = BoxLayout(orientation = 'horizontal', size_hint_y =.2)


		self.score_header = Label(text = 'Score: {0}'.format(self.game.score), font_size = self.header_font_size)
		self.direction_header = Label(text = self.direction_symbols[direction], font_size = self.header_font_size)
		
		self.header.add_widget(self.score_header)
		self.header.add_widget(self.direction_header)

		
		self.board = Board(self.game)
		
		self.add_widget(self.header)
		self.add_widget(self.board)
	
	def update_board(self, game, direction):
		self.remove_widget(self.board)
		self.remove_widget(self.header)
		
		self.game = game
		self.add_widgets(direction)
				
class Alpha2048(App):
	def __init__(self, agent = None, timestep = .5,  **kwargs):
		super().__init__(**kwargs)
		self.agent = agent
		self.timestep = timestep
		
		self.game = Game2048()
		
	def build(self):
		self.root = Root(game = self.game)
		
		self.main_loop = Clock.schedule_interval(self.move_and_update, self.timestep)
	
		return self.root
	
	def restart_game(self, dt):
		self.game = Game2048()			
		self.main_loop()
		
	def move_and_update(self, dt):
		direction = self.agent.make_a_move(self.game)

		change = self.game.play(direction)

		self.root.update_board(self.game, direction)

		if change == 2:
			self.main_loop.cancel()
			Clock.schedule_once(self.restart_game, self.timestep * 8)
		

if __name__ == "__main__":
	agent = Agent("../results/0120-170133_t0_model.pt")
	# agent.from_saved_model("saved_models/model_0108-1408.ptautosave-tur300", hot_one = False)	
	#agent.from_algorithm(alg.pure_montecarlo)
	Alpha2048(agent=agent, timestep=.2).run()
