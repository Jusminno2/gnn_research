import os
from pathlib import Path
import folium
import osmnx as ox

# 対象地域検索クエリ (愛知県名古屋市中村区)
query = "Shimogyo Ward, Kyoto,Japan"

# 各種出力ファイルパス
outdir_path = Path(query.replace(",", "_"))
os.makedirs(outdir_path, exist_ok=True)

# 道路グラフネットワーク取得
graphml_outfile = outdir_path / "road_network.graphml"
if not os.path.isfile(graphml_outfile):
    G = ox.graph_from_place(query, network_type="drive")
    ox.save_graphml(G, filepath=graphml_outfile)
else:
    G = ox.load_graphml(graphml_outfile)

# 道路グラフネットワークの各ノード・エッジ取得
nodes, edges = ox.graph_to_gdfs(G)

# GeoDataFrame.explore()を使用してインタラクティブマップを作成
fmap = edges.explore()

# 最短経路探索
start_point = (34.987973236768724, 135.74245697782106)
end_point = (35.00349553421575, 135.7674524424336)

start_node = ox.distance.nearest_nodes(G, X=start_point[1], Y=start_point[0])
end_node = ox.distance.nearest_nodes(G, X=end_point[1], Y=end_point[0])

shortest_path = ox.shortest_path(G, start_node, end_node)

# 最短経路探索結果の可視化 (foliumの代替としてmatplotlibを使用)
ox.plot_graph_route(G, shortest_path, route_color="red", route_linewidth=4, node_size=5, bgcolor='white', save=True, filepath=outdir_path / "shortest_path_road_network.png")
