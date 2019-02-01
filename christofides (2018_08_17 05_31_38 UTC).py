import numpy as np
import matplotlib.pyplot as plt

def tsp(data):
    ordata=data
    G = build_graph(data)
    MSTree = minimum_spanning_tree(G)
    odd_vertexes = find_odd_vertexes(MSTree)
    #print("Odd vertexes in MSTree: ", odd_vertexes)

    # add minimum weight matching edges to MST
    minimum_weight_matching(MSTree, G, odd_vertexes)
    #print("Minimum weight matching: ", MSTree)

    # find an eulerian tour
    eulerian_tour = find_eulerian_tour(MSTree, G)

    current = eulerian_tour[0]
    path = [current]
    visited = [False] * len(eulerian_tour)

    length = 0

    for v in eulerian_tour[1:]:
        if not visited[v]:
            path.append(v)
            visited[v] = True

            length += G[current][v]
            current = v

    path.append(path[0])
    
    N = 10
    data = np.random.random((N, 4))
    labels = ['point{0}'.format(i) for i in range(N)]
    
    plt.subplots_adjust(bottom = 0.1)
 

    xval= [i[0] for i in ordata]
    yval = [i[1] for i in ordata]
    plt.scatter(xval, yval)
    
    for i in range(len(path)-1):
        plt.plot([xval[path[i]], xval[path[i+1]]], [yval[path[i]], yval[path[i+1]]], 'ro-')
    

    print("Result path: ", path)
    print("Result length of the path: ", length)

    return length, path


def get_length(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2)


def build_graph(data):
    graph = {}
    for this in range(len(data)):
        for another_point in range(len(data)):
            if this != another_point:
                if this not in graph:
                    graph[this] = {}

                graph[this][another_point] = get_length(data[this][0], data[this][1], data[another_point][0],
                                                        data[another_point][1])

    return graph


class UnionFind:
    def __init__(self):
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        return iter(self.parents)

    def union(self, *objects):
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r], r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest


def minimum_spanning_tree(G):
    tree = []
    subtrees = UnionFind()
    for W, u, v in sorted((G[u][v], u, v) for u in G for v in G[u]):
        if subtrees[u] != subtrees[v]:
            tree.append((u, v, W))
            subtrees.union(u, v)

    return tree


def find_odd_vertexes(MST):
    tmp_g = {}
    vertexes = []
    for edge in MST:
        if edge[0] not in tmp_g:
            tmp_g[edge[0]] = 0

        if edge[1] not in tmp_g:
            tmp_g[edge[1]] = 0

        tmp_g[edge[0]] += 1
        tmp_g[edge[1]] += 1

    for vertex in tmp_g:
        if tmp_g[vertex] % 2 == 1:
            vertexes.append(vertex)

    return vertexes


def minimum_weight_matching(MST, G, odd_vert):
    import random
    random.shuffle(odd_vert)

    while odd_vert:
        v = odd_vert.pop()
        length = float("inf")
        u = 1
        closest = 0
        for u in odd_vert:
            if v != u and G[v][u] < length:
                length = G[v][u]
                closest = u

        MST.append((v, closest, length))
        odd_vert.remove(closest)


def find_eulerian_tour(MatchedMSTree, G):
    # find neigbours
    neighbours = {}
    for edge in MatchedMSTree:
        if edge[0] not in neighbours:
            neighbours[edge[0]] = []

        if edge[1] not in neighbours:
            neighbours[edge[1]] = []

        neighbours[edge[0]].append(edge[1])
        neighbours[edge[1]].append(edge[0])

    # print("Neighbours: ", neighbours)

    # finds the hamiltonian circuit
    start_vertex = MatchedMSTree[0][0]
    EP = [neighbours[start_vertex][0]]

    while len(MatchedMSTree) > 0:
        for i, v in enumerate(EP):
            if len(neighbours[v]) > 0:
                break

        while len(neighbours[v]) > 0:
            w = neighbours[v][0]

            remove_edge_from_matchedMST(MatchedMSTree, v, w)

            del neighbours[v][(neighbours[v].index(w))]
            del neighbours[w][(neighbours[w].index(v))]

            i += 1
            EP.insert(i, w)

            v = w

    return EP


def remove_edge_from_matchedMST(MatchedMST, v1, v2):

    for i, item in enumerate(MatchedMST):
        if (item[0] == v2 and item[1] == v1) or (item[0] == v1 and item[1] == v2):
            del MatchedMST[i]

    return MatchedMST



#tsp([[0, 0],[3, 0],[6, 0],[0, 3],[3, 3],[6, 3],[0, 6],[3, 6],[6, 6]])
    
def generate_tsp_data(tsp_size):
    '''
    first initalize the tsp
    for tsp size5
        [[1,0,x,y],
         [2,0,x,y],
         [3,0,x,y],
         [4,0,x,y],
         [5,0,x,y]]
    the first column represents the cities
    the second column represents the path where is going to
    the third/fourth column just represent the coordinates
    '''
    # initalize the tsp
    complete_tsp = []
    cities = np.arange(1,tsp_size+1).reshape(tsp_size,1)
    paths = np.zeros((tsp_size,1))
    coords = np.random.uniform(size=(tsp_size,2))*100
    
    tsp1 = np.hstack((cities,paths))
    tsp1 = np.hstack((tsp1,coords))
    # take random paths
    
    #available_paths = np.arange(1,tsp_size+1).tolist()
    #startcit = 1;
    length, path = tsp(coords.tolist())
    for i in range(len(path)-1):
        tsp1[path[i]][1] = path[i+1]
    print(tsp1)
#    adjmatr = [[0 for i in range(tsp_size)] for j in range(tsp_size)]
#    #print(adjmatr)
#    #print(coords[1][0])
#    for i in range(tsp_size): 
#        for j in range(tsp_size):
#            adjmatr[i][j] = distance(coords[i],coords[j])
#    #print(adjmatr)
#    g1 = Graph(tsp_size)
#    g1.graph = adjmatr
#   
#    arr1= g1.primMST()
#    lst = [1]
#    
#    print(arr1)
    
    return tsp1

generate_tsp_data(100)
##print(data)
#tsp(data)
