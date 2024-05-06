

import random
import networkx as nx
import numpy
from collections import deque
import matplotlib.pyplot as plt
nodenum=9
edgenum=12
node_type_num=3
test_ratio=1
def visualize_adjacency_matrix(adj_matrix):
	# Create a graph from the adjacency matrix
	G = nx.from_numpy_array(adj_matrix)
	
	# Draw the graph
	pos = nx.spring_layout(G)  # positions for all nodes
	nx.draw(G, pos, with_labels=True, cmap=plt.cm.Blues, node_color='skyblue', node_size=50)
	
	# Display the graph
	plt.show()
	
import numpy as np
from collections import deque
	
def is_fully_interconnected(adj_matrix):
	n = len(adj_matrix)
	visited = np.zeros(n, dtype=bool)
	
	# Function to perform BFS
	def bfs(start):
		visited[start] = True
		queue = deque([start])
		while queue:
			node = queue.popleft()
			for neighbor in np.where(adj_matrix[node] == 1)[0]:
				if not visited[neighbor]:
					visited[neighbor] = True
					queue.append(neighbor)
					
	# Start BFS from each node
	for i in range(n):
		visited.fill(False)
		bfs(i)
		if not all(visited):
			return False
		
	return True

# Test the function


#print(is_fully_interconnected(adj_matrix))



def is_int(x):
	return type(x) == int


def generate_random_graph(nodes, edges):
	if edges > nodes * (nodes - 1) / 2:
		raise ValueError("Too many edges for the given number of nodes")
	G = nx.Graph()
	G.add_nodes_from(range(nodes))
	edge_count = 0
	while edge_count < edges:
		node1 = random.randint(0, nodes - 1)
		node2 = random.randint(0, nodes - 1)
		if node1 != node2 and not G.has_edge(node1, node2):
			G.add_edge(node1, node2)
			edge_count += 1
	return G
firstStep={}
def bfs_with_distances(graph):
	distances = {}
	global firstStep
	for node in graph.nodes():
		distances[node] = {}
		firstStep[node] = {}
		
		#firstStep={}
		queue = [(node, 0)]
		queuefirststep=[-1]
		visited = set()
		while queue:
			current_node, distance = queue.pop(0)
			firstS=queuefirststep.pop(0)
			if current_node not in visited:
				visited.add(current_node)
				distances[node][current_node] = distance
				neighbors = list(graph.neighbors(current_node))
				if distance==1:
					firstS=current_node
				firstStep[node][current_node] = firstS
				if(distance!=1):
					steps=[]
					for n1 in graph.neighbors(current_node):
						for n2 in graph.neighbors(node):
							#dijiesita
							dist=nx.shortest_path_length(graph,n1,n2)
							if(dist==distance-2):
								#print("this is a equal way",node,current_node,n1,n2)
								steps.append((n2,n1))
					if(len(steps)>1):
						firstStep[node][current_node] =steps
								
				for neighbor in neighbors:
					if neighbor not in visited:
						queue.append((neighbor, distance + 1))
						queuefirststep.append(firstS)
	return distances



def calcf(adjacency_matrix,distances,firstStep,particals):
	f={}
	for i in range (nodenum):
		f[i]={}
		for j in range (nodenum):
			f[i][j]=0
		for j in range (nodenum):
			if(i==j):
				continue
			fcon=-1#/(node_type_num-1)
			if test_ratio:
				fcon/=(node_type_num-1)
			if(particals[i]==particals[j]):
				fcon=1
			f_act=fcon/distances[i][j]**2
			if(is_int(firstStep[i][j])):
				f[i][int(firstStep[i][j])]+=f_act
			else:
				for k in firstStep[i][j]:
					f[i][int(k[0])]+=f_act/len(firstStep[i][j])
#					print("this is a force split",i,j,int(k[0]),f_act/len(firstStep[i][j]))
	return f


def find_closest_to_zero_and_recurse(arr):
	# Check if the array is empty
	if not arr:
		#print("Array is empty. Stopping function.")
		return
	
	# Find the value closest to zero and its index
	closest_value = 100000
	closest_index = None
	for i in range(len(arr)):
		for j in range(len(arr[0])):
			if abs(arr[i][j]) < abs(closest_value):
				closest_value = arr[i][j]
				closest_index = (i, j)
				
	if closest_value == 100000:
		print("nothing left")
		return
	
	#print("Closest value to zero:", closest_value)
	print("Index of closest value:", closest_index)
	print(particals[closest_index[0]],particals[closest_index[1]])
	
	# Delete the row and column containing the closest value
	row_idx, col_idx = closest_index
	for i in range(len(arr)):
		for j in range(len(arr[0])):
			if(i==row_idx or j==row_idx or i==col_idx or j==col_idx):
				arr[i][j]=100000
				
				
	# Recurse with the updated array
	find_closest_to_zero_and_recurse(arr)
	
	
particals=[]

def initialize_particals(A,B):
	B=int(B/A)
	result = []
	for i in range(B):
		for j in range(A):
			result.append(j+1)
	return result

def main():
	global particals
	particals=initialize_particals(node_type_num,nodenum)
	
	nodes = nodenum
	edges = edgenum
	graph = generate_random_graph(nodes, edges)
	adjacency_matrix = nx.to_numpy_array(graph)
	while(not is_fully_interconnected(adjacency_matrix)):
		graph = generate_random_graph(nodes, edges)
		adjacency_matrix = nx.to_numpy_array(graph)
		#print("graph generation failure!")
		
	print("\ngraph generation success!\nGenerated Graph (Adjacency Matrix):")
	
	
	adjacency_matrix=numpy.matrix(adjacency_matrix)
	graph=nx.from_numpy_array(adjacency_matrix)
	
	print(adjacency_matrix.tolist())
#	visualize_adjacency_matrix(adjacency_matrix)
	distances = bfs_with_distances(graph)
	'''
	print("\nDistances from each node to all other nodes:")
	for node, distances_from_node in distances.items():
		print(f"From node {node}: {distances_from_node}")
	print()
	for node, distances_from_node in firstStep.items():
		print(f"From node FS {node}: {distances_from_node}")
		'''
	
	lasttension=[]
	for i in range(nodenum):
		lasttension.append([])
		for j in range(nodenum):
			lasttension[i].append(100000)
	while 1:
		fcur=calcf(adjacency_matrix,distances,firstStep,particals)
		#for node, distances_from_node in fcur.items():
			#print(f"From node Force {node}: {distances_from_node}")
		print()
		inforce=[]
		for i in range(nodenum):
			inforce.append(0)
			for j in range(nodenum):
				inforce[i]+=fcur[i][j];
		print(inforce)
		edgelist=graph.edges()
		maxtension=-10000
		maxtindex=0
		for edge in edgelist:
			start_node, end_node =edge
			fstart=inforce[start_node]-2*fcur[start_node][end_node]
			fend=inforce[end_node]-2*fcur[end_node][start_node]
			t=fstart+fend
			print(f"Edge: {start_node} -> {end_node} tension: {t}")
			lasttension[start_node][end_node]=t
			lasttension[end_node][start_node]=t
			if(t>maxtension):
				maxtension=t
				maxtindex=(start_node, end_node)
		if maxtension>2:
			particals[maxtindex[0]],particals[maxtindex[1]]=particals[maxtindex[1]],particals[maxtindex[0]]
		else:
			break
		print('======================================')
		print(particals)
	print('======================================')
	print(particals)
	if node_type_num==2:
		find_closest_to_zero_and_recurse(lasttension)
#		print(lasttension)
	else:
		group_points(lasttension,node_type_num)
	#print(lasttension)
	#visualize_graph_with_labels_and_colors(lasttension,particals)
	
def check_unique_element(arr, elem1, elem2):
	count = 0
	for i in arr:
		if i == elem1 or i == elem2:
			count += 1
	return 1 if count == 1 else 0
	
def group_points(arr,n):
	if not arr:
		return
	grouped=[]
	closest_value = 100000
	closest_index = None
	for i in range(len(arr)):
		for j in range(len(arr[0])):
			if abs(arr[i][j]) < abs(closest_value):
				closest_value = arr[i][j]
				closest_index = (i, j)
				
	if closest_value == 100000:
		print("nothing left")
		return
	grouped.append(closest_index[0])
	grouped.append(closest_index[1])
	while(len(grouped)<n):
		closest_value = 100000
		closest_index = None
		for i in range(len(arr)):
			for j in range(len(arr[0])):
				if check_unique_element(grouped, i, j):
					if abs(arr[i][j]) < abs(closest_value):
						closest_value = arr[i][j]
						closest_index = (i, j)
		if closest_value == 100000:
#			print(grouped)
			print("nothing left")
#			return
			break
		if closest_index[0] in grouped:
			grouped.append(closest_index[1])
		else:
			grouped.append(closest_index[0])
	
	print('Index of closest value:',grouped)
	for i in grouped:
		print(particals[i],end=' ')
	print()
	
	for i in range(len(arr)):
		for j in range(len(arr[0])):
			if i in grouped:
				arr[i][j]=100000
			elif j in grouped:
				arr[i][j]=100000
	group_points(arr,n)
				
				
	# Recurse with the updated array
	find_closest_to_zero_and_recurse(arr)


	
	
if __name__ == "__main__":
	main()
	