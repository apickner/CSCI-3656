import numpy as np

def f1(x):
	return ((1 - np.cos(x)) / (np.sin(x) ** 2))
	
def f2(x):
	return (1 / (1 + np.cos(x)))
	
points = []
for k in range(13):
	points.append(10 ** (-k))

print("{:<9}\t|\t{:<21}\t|\t{:<21}".format("x", "f1(x)", "f2(x)"))
print("-----------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------")
for point in points:
	print("{:<9}\t|\t{:<21}\t|\t{:<21}".format(point, f1(point), f2(point)))
	print("-----------------------------------------------------------------------------")