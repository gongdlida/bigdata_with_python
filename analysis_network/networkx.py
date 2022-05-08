import networkx as nx
g = nx.Graph()
g.add_edge('a', 'b', weight=0.1)
g.add_edge('b', 'c', weight=1.5)
g.add_edge('a', 'c', weight=1.0)
g.add_edge('c', 'd', weight=2.2)
 
print(nx.shortest_path(g, "b", "d"))
print(nx.shortest_path(g, "b", "d", weight="weight"))


graph_2 = nx.Graph()
graph_2.add_node(1)
graph_2.add_nodes_from([2, 3])
graph_2.remove_node(2)
graph_2.add_node("HYCU")
print(graph_2.nodes())

# 노드/엣지 추가&삭제
import math
graph_2.add_node(math.cos)
print(graph_2.nodes())

graph_2.add_edge(1,2)
print(graph_2.nodes())
graph_2.add_edges_from([(1, 3), (1, 4)])
print(graph_2.nodes())
graph_2.remove_edge(1,2)
print(graph_2.nodes())

print(graph_2.edges())

print(graph_2.number_of_nodes()) # len(graph_2.nodes()), len(g) 18. 6
print(graph_2.number_of_edges()) # len(graph_2.edges())

# 노드/엣지 속성
graph_2.add_node(5, time="10am") 
print(graph_2.nodes()[5])

print(graph_2.nodes()[5]['time'])

graph_2.add_edge(5, 2, weight=4.0)
print(graph_2.edges())

print(graph_2.edges(data=True))

print(graph_2[5])

print(graph_2[5][2])

print(graph_2[5][2]['weight'])


# 유방향 네트워크 그래프

dg = nx.DiGraph()
dg.add_weighted_edges_from([(1, 4, 0.5), (3, 1, 0.75)]) 
dg.out_degree(1, weight='weight')
dg.degree(1, weight='weight')

for node in dg.nodes():
  print(node, dg.degree(node, weight="weight"))

for n1, n2, attr in dg.edges(data=True):
  print(n1, n2, attr['weight'])

# 그래프 생성기
  # small famous graphs
  petersen = nx.petersen_graph()
  
  tutte = nx.tutte_graph()
  maze = nx.sedgewick_maze_graph()
  tet = nx.tetrahedral_graph()

  # classic graphs
  K_5 = nx.complete_graph(5)
  K_3_5 = nx.complete_bipartite_graph(3, 5) 
  barbell = nx.barbell_graph(10, 10)
  lollipop = nx.lollipop_graph(10, 20)

  # random graphs
  er = nx.erdos_renyi_graph(100, 0.15)
  ws = nx.watts_strogatz_graph(30, 3, 0.1) 
  ba = nx.barabasi_albert_graph(100, 5)
  red = nx.random_lobster(100, 0.9, 0.9)