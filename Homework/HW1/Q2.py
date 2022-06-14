'''
I used Desmos to prove to myself that p1(x) == p2(x)
'''

def p1(x):
	return ((x-2) ** 9)
	
def p2(x):
	return ((x ** 9) - (18 * (x ** 9)) + (144 * (x ** 7)) - (672 * (x ** 6)) + (2016 * (x ** 5)) - (4032 * (x ** 4)) + (5376 * (x ** 3)) - (4608 * (x ** 2)) + (2304 * x) - 512)
