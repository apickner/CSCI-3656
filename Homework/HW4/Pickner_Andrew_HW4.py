import matplotlib.pyplot as plt
import numpy as np

############################################################################
# MATRIX READING

'''
Reads the contents of a csv or comma delimited txt file and turns that into a numpy ndarray
'''
def read_matrix(file_name):
	return np.loadtxt(file_name,delimiter=',')

############################################################################
# REPORT GENERATION

'''
Generates a report of matrices given a matrix, and 
'''

def matrix_report_header():
	header = '''
MATRIX REPORT\n\n
Andrew Pickner\n\n
I worked alone, & all sources I used will be cited at the bottom of this report.\n
--------------------------------------------------------------------------------- 
	'''
	return header

def matrix_report_footer():
	header = '''
---------------------------------------------------------------------------------
SOURCES USED:\n\n
numpy docs were incredibly valuable for obvious reasons. Practically used numpy for the entire assignment.\n
Khan academy has an incredible library, and I was able to make use of some of the lin alg videos.\n
Sauer's textbook\n
--------------------------------------------------------------------------------- 
	'''
	return header

def generate_matrix_report(matrices,  matrix_report_file):
	f = open(matrix_report_file, "w")
	
	f.write(matrix_report_header())
	
	mat_number = 1
	for matrix in matrices:
		num_rows, num_cols = get_size(matrix)
		n_nonzeros = num_nonzeros(matrix)
		is_sym = is_symmetric(matrix)
		is_diag = is_diagonal(matrix)
		is_orth = is_orthogonal(matrix)
		rank = get_rank(matrix)
		max_val = get_max_value(matrix)
		min_val = get_min_value(matrix)
		condition_number = get_condition_number(matrix)
		solve_systems = solve_systems_okay(matrix)
		
		plot_nonzeros(matrix, mat_number)
		plot_value_sizes(matrix, mat_number)
		            
		mat_report = '''
	------------------------------------------------\n
		Matrix #{}:\n
	------------------------------------------------\n
	{:<30} {} rows, {} cols\n
	{:<30} {}\n
	{:<30} {}\n
	{:<30} {}\n
	{:<30} {}\n
	{:<30} {}\n
	{:<30} {}\n
	{:<30} {}\n
	{:<30} {}\n
	{:<30} {}\n
		'''.format(mat_number, "size", num_rows, num_cols, "number nonzeros  ", n_nonzeros, "is symmetric?", is_sym, "is diagonal?", is_diag, "is orthogonal?", is_orth, "rank", rank, "smallest sing val", min_val, "largest sing val", max_val, "condition number", condition_number, "solve systems okay?", solve_systems )
		f.write(mat_report)
		mat_number  += 1
	f.write(matrix_report_footer())
	f.close()

############################################################################
# MATRIX INFORMATION

def get_size(matrix):	
	return matrix.shape[0], matrix.shape[1]

def num_nonzeros(matrix):
	return np.count_nonzero(matrix)
	
  
	
def is_diagonal(matrix):  
	ones_matrix = np.ones(matrix.shape, dtype=np.uint8)
	np.fill_diagonal(ones_matrix, 0)
	return np.count_nonzero(np.multiply(ones_matrix, matrix)) == 0

def is_orthogonal(matrix):
	if matrix.shape[0] == matrix.shape[1]:
		return np.allclose(np.matmul(matrix, matrix.T), np.identity(matrix.shape[0], dtype=type(matrix[0][0])), atol=1e-05)
	return False


def get_rank(matrix):
	if is_symmetric(matrix):
		return np.linalg.matrix_rank(matrix, tol=None, hermitian=True)
	return np.linalg.matrix_rank(matrix, tol=None, hermitian=False)

def get_max_value(matrix):
	return matrix.max()
	
def get_min_value(matrix):
	return matrix.min()
	
def get_condition_number(matrix):
	return np.linalg.cond(matrix)

def solve_systems_okay(matrix):
	bs = []
	for i in range(5):
		bs.append(np.random.rand(matrix.shape[0],1))
		try:
			np.linalg.solve(matrix, bs[i])
		except:
			return False
	return True	

def plot_nonzeros(matrix, mat_num):
	fig2, ax2 = plt.subplots()
	fig2.suptitle('Nonzero elements')
	plt.xlabel('cols')
	plt.ylabel('rows')
	plt.spy(matrix, markersize=1, precision=0)
	plt.savefig('/Users/AndrewMacbook/Downloads/nzz{}'.format(mat_num))

def plot_value_sizes(data, mat_num):
#	def heatmap(data, row_labels, col_labels, ax=None,
#				cbar_kw={}, cbarlabel="", **kwargs):
	"""
	Create a heatmap from a numpy array and two lists of labels.

	Parameters
	----------
	data
		A 2D numpy array of shape (N, M).
	row_labels
		A list or array of length N with the labels for the rows.
	col_labels
		A list or array of length M with the labels for the columns.
	ax
		A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
		not provided, use current axes or create a new one.  Optional.
	cbar_kw
		A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
	cbarlabel
		The label for the colorbar.  Optional.
	**kwargs
		All other arguments are forwarded to `imshow`.
	"""
	fig1, ax1 = plt.subplots()

	# Plot the heatmap
	im = ax1.imshow(data)

	# Create colorbar
	cbar = ax1.figure.colorbar(im, ax=ax1)

	# We want to show all ticks...
	ax1.set_xticks(np.arange(data.shape[1]))
	ax1.set_yticks(np.arange(data.shape[0]))


	ax1.grid(which="minor", color="w", linestyle='-', linewidth=3)
#	ax1.tick_params(which="minor", bottom=False, left=False)
	
	fig1.suptitle('Size of matrix elements')
	plt.xlabel('cols')
	plt.ylabel('rows')
	fig1.tight_layout()
	plt.savefig('/Users/AndrewMacbook/Downloads/mag{}'.format(mat_num))
	
############################################################################

def main():
	# directory in which our matrix files are located, and where we will store our matrix report
	directory = "/Users/AndrewMacbook/Downloads/"
	
	# name of the matrix report file
	matrix_report_file = "{}Matrix_Report.txt".format(directory)
	
	num_matrices = 5
	matrices = []
	
	for i in range(num_matrices):
		file_name = "{}mat{}.txt".format(directory, i+1)
		matrices.append(read_matrix(file_name))
	
	generate_matrix_report(matrices, matrix_report_file)

############################################################################

if __name__ == "__main__":
	main()