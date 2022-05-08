characters = ["R2-D2",
                "CHEWBACCA",
                "C-3PO",
                "LUKE",
                "DARTH VADER",
                "CAMIE",
                "BIGGS",
                "LEIA",
                "BERU",
                "OWEN",
                "OBI-WAN",
                "MOTTI",
                "TARKIN",
                "HAN",
                "DODONNA",
                "GOLD LEADER",
                "WEDGE",
                "RED LEADER",
                "RED TEN"]


edges = [("CHEWBACCA", "R2-D2"),
        ("C-3PO", "R2-D2"),
        ("BERU", "R2-D2"),
        ("LUKE", "R2-D2"),
        ("OWEN", "R2-D2"),
        ("OBI-WAN", "R2-D2"),
        ("LEIA", "R2-D2"),
        ("BIGGS", "R2-D2"),
        ("HAN", "R2-D2"),
        ("CHEWBACCA", "OBI-WAN"),
        ("C-3PO", "CHEWBACCA"),
        ("CHEWBACCA", "LUKE"),
        ("CHEWBACCA", "HAN"),
        ("CHEWBACCA", "LEIA"),
        ("CAMIE", "LUKE"),
        ("BIGGS", "CAMIE"),
        ("BIGGS", "LUKE"),
        ("DARTH VADER", "LEIA"),
        ("BERU", "LUKE"),
        ("BERU", "OWEN"),
        ("BERU", "C-3PO"),
        ("LUKE", "OWEN"),
        ("C-3PO", "LUKE"),
        ("C-3PO", "OWEN"),
        ("C-3PO", "LEIA"),
        ("LEIA", "LUKE"),
        ("BERU", "LEIA"),
        ("LUKE", "OBI-WAN"),
        ("C-3PO", "OBI-WAN"),
        ("LEIA", "OBI-WAN"),
        ("MOTTI", "TARKIN"),
        ("DARTH VADER", "MOTTI"),
        ("DARTH VADER", "TARKIN"),
        ("HAN", "OBI-WAN"),
        ("HAN", "LUKE"),
        ("C-3PO", "HAN"),
        ("LEIA", "MOTTI"),
        ("LEIA", "TARKIN"),
        ("HAN", "LEIA"),
        ("DARTH VADER", "OBI-WAN"),
        ("DODONNA", "GOLD LEADER"),
        ("DODONNA", "WEDGE"),
        ("DODONNA", "LUKE"),
        ("GOLD LEADER", "WEDGE"),
        ("GOLD LEADER", "LUKE"),
        ("LUKE", "WEDGE"),
        ("BIGGS", "LEIA"),
        ("LEIA", "RED LEADER"),
        ("LUKE", "RED LEADER"),
        ("BIGGS", "RED LEADER"),
        ("BIGGS", "C-3PO"),
        ("C-3PO", "RED LEADER"),
        ("RED LEADER", "WEDGE"),
        ("GOLD LEADER", "RED LEADER"),
        ("BIGGS", "WEDGE"),
        ("RED LEADER", "RED TEN"),
        ("BIGGS", "GOLD LEADER"),
        ("LUKE", "RED TEN")]

G_starWars = nx.Graph()


G_starWars.add_nodes_from(characters)
G_starWars.add_edges_from(edges)

# 네트워크 그래프 시각화
import matplotlib.pyplot as plt

nx.draw(G_starWars, with_labels=True)
#nx.draw_spring(G_starWars, with_labels=True)
plt.show()
nx.draw_random(G_starWars, with_labels=True)
plt.show()
nx.draw_circular(G_starWars, with_labels=True)
plt.show()
nx.draw_kamada_kawai(G_starWars, with_labels=True) 
plt.show()
nx.draw_spectral(G_starWars, with_labels=True)
plt.show()
nx.draw_shell(G_starWars, with_labels=True)
plt.show()


pos = nx.spring_layout(G_starWars)
nx.draw_networkx(G_starWars, pos=pos, with_labels=True) 
plt.show()
pos = nx.shell_layout(G_starWars)
nx.draw_networkx(G_starWars, pos=pos, with_labels=True) 
plt.show()

# networkx 패키지는 규모가 작고 단순한 정적인 시각화만이 가능함 
# pyvis, gephi, graphviz 등과 같은 전문 시각화 패키지 이용 권장 
# 사례: pyvis를 이용한 대화형 시각화

from pyvis.network import Network 
nt = Network('500px', '500px')
nt.from_nx(G_starWars)
nt.show("G_starWars.html")

## 네트워크 분석
  # 전체 데이터에서 LUKE 데이터 추출
print(G_starWars.degree()["LUKE"])
  # 스타워즈 케릭터 비율
print(G_starWars.degree())
  # 스타워즈 케릭터 비율 오름차순으로 정렬
print(sorted(G_starWars.degree(), key=lambda x: x[1], reverse=True))

## 중심성 얻기
nx.degree_centrality(G_starWars)
nx.closeness_centrality(G_starWars)
nx.harmonic_centrality(G_starWars)
nx.betweenness_centrality(G_starWars)
nx.eigenvector_centrality(G_starWars)
cliques = list(nx.algorithms.clique.find_cliques(G_starWars)) # only max cliques
for c in cliques:
    print(c)
  
c_graph = nx.algorithms.clique.make_max_clique_graph(G_starWars)
nx.relabel_nodes(c_graph, {node: "c_" + node for node in c_graph.nodes()}, copy=False)
print(c_graph.nodes())

nx.draw(c_graph, with_labels = True)
plt.show()

node_labels = {}
for i, node in enumerate(c_graph.nodes()):
  node_labels[node] = cliques[i]
nx.draw(c_graph, labels=node_labels, with_labels=True) 
plt.show()