import numpy as np
dim = 2

vals = []
for i in range(1000):
	vec = np.random.random(int(dim)) * 2 - 1
	
	vec = vec/np.linalg.norm(vec)
	max_val = max(vec)
	min_val = min(vec)
	
	if max_val > 0 and min_val > 0:
		final_val = max_val
	elif max_val < 0 and min_val < 0:
		final_val = abs(min_val)
	elif abs(max_val) > abs(min_val):
		final_val = max_val
	else:
		final_val = abs(min_val)
	ret_val = np.arccos(final_val)/np.pi
	vals.append(ret_val)
	
print(np.mean(vals))