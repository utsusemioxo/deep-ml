import numpy as np

def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	# Your code here
	n = len(x)
	J = [[0.0] * n for _ in range(n)]

	x = np.asarray(x, dtype=float)
	exp_x = np.exp(x - np.max(x))
	s = exp_x / np.sum(exp_x)

	for i in range(n):
		for j in range(n):
			if j==i:
				J[i][j] = s[i] * (1 - s[i])
			else:
				J[i][j] = -s[i] * s[j]

	return J			