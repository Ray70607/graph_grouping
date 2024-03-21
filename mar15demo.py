#!/usr/bin/env python3


result=['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B','B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']

def calculate_f(result):
	array_length = len(result)
	f = [0] * array_length
	
	for i in range(array_length):
		for j in range(i):
			d=0
			if result[i]==result[j]:
				d=1
			else:
				d=-1
			f[i]+=d/(i-j)**2
		for j in range(i+1,array_length):
			d=0
			if result[i]==result[j]:
				d=1
			else:
				d=-1
			d*=-1
			f[i]+=d/(i-j)**2
			
	return f


def get_short_array_difference(f):
	result = []
	array_length = len(f)
	for i in range(array_length - 1):
		result.append(f[i] - f[i + 1])
	return result


def max_idx(arr):
	if not arr:
			return None  # Return None if the array is empty
	
	max_val = arr[0]  # Initialize max_val to the first element
	max_index = 0  # Initialize max_index to 0
	
	for i in range(1, len(arr)):
			if arr[i] > max_val:
				max_val = arr[i]
				max_index = i
				
	return max_index

while(1):
	f=calculate_f(result)
	f2=get_short_array_difference(f)
	index = max_idx(f2)
	
	print(result)
	print(f)
	print(f2)
	print("Index of element with the largest  value:", index)
	
	#f2[index]=0
	if(f2[index]<2):
		break
	else:
		result[index],result[index+1]=result[index+1],result[index]
		
print(result)