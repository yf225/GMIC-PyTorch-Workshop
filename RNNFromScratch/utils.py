import torch
import matplotlib.pyplot as plt

def string_to_numbers(string):
	numbers = []
	for c in string:
		assert 'a' <= c <= 'z' or 'A' <= c <= 'Z', '"{}"" is not alphabetical'.format(string)
		if 'A' <= c <= 'Z':
			numbers.append(ord(c) - ord('A'))
		else:
			numbers.append(ord(c) - ord('a') + 26)
	numbers.append(52)
	return numbers

def numbers_to_string(numbers):
	assert len(numbers) > 1, 'input is empty'
	chars = []
	for i in numbers:
		if i < 26:
			chars.append(chr(ord('A') + i))
		elif i < 52:
			chars.append(chr(ord('a') + i - 26))
		else:
			chars.append('<EoS>')
	return ''.join(chars)

def to_one_hot(tokens, volcab_size):
	return torch.eye(volcab_size)[tokens]

class SmoothPlotter(object):
    def __init__(self, interval, color='b'):
        self.x = 0
        self.interval_total_y = 0
        self.interval = interval
        self.color = color
        self.fig = plt.gcf()
        self.ax = plt.gca()
        self.xs = []
        self.ys = []
        
    def update(self, y):
        self.x += 1
        self.interval_total_y += y
        if self.x % self.interval == 0:
            self.plot()
            
    def plot(self):
        num_y = self.x % self.interval
        if num_y == 0:
            num_y = self.interval
        self.xs.append(self.x)
        self.ys.append(self.interval_total_y / num_y)
        self.interval_total_y = 0
        self.ax.plot(self.xs, self.ys, self.color)
        self.fig.canvas.draw()
        
    def finish(self):
        self.plot()
        
    