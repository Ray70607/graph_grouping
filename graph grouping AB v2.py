import random
import networkx as nx
import numpy
nodenum=6
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
            fcon=-1
            if(particals[i]==particals[j]):
                fcon=1
            f_act=fcon/distances[i][j]**2
            f[i][int(firstStep[i][j])]+=f_act
    return f
    
    
    
    


def main():
    particals=[0,0,0,1,1,1]
    nodes = nodenum
    edges = 12
    graph = generate_random_graph(nodes, edges)
    adjacency_matrix = nx.to_numpy_array(graph)
    print("Generated Graph (Adjacency Matrix):")
    
    adjacency_matrix=[
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 0]
    ]
    adjacency_matrix=numpy.matrix(adjacency_matrix)
    graph=nx.from_numpy_array(adjacency_matrix)
    
    print(adjacency_matrix)
    
    distances = bfs_with_distances(graph)
    print("\nDistances from each node to all other nodes:")
    for node, distances_from_node in distances.items():
        print(f"From node {node}: {distances_from_node}")
    print()
    for node, distances_from_node in firstStep.items():
        print(f"From node FS {node}: {distances_from_node}")
    while 1:
    
        fcur=calcf(adjacency_matrix,distances,firstStep,particals)
        for node, distances_from_node in fcur.items():
            print(f"From node Force {node}: {distances_from_node}")
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
            if(t>maxtension):
                maxtension=t
                maxtindex=(start_node, end_node)
        if maxtension>2:
            particals[maxtindex[0]],particals[maxtindex[1]]=particals[maxtindex[1]],particals[maxtindex[0]]
        else:
            break
        print('======================================')
        print(particals)
    
    '''
    fcur=calcf(adjacency_matrix,distances,firstStep,particals)
    for node, distances_from_node in fcur.items():
        print(f"From node Force {node}: {distances_from_node}")
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
        if(t>maxtension):
            maxtension=t
            maxtindex=(start_node, end_node)
    if maxtension>=2:
        swap(particals[maxtindex[0]],particals[maxtindex[1]])
    '''
    
    
    
    
    
if __name__ == "__main__":
    main()
    