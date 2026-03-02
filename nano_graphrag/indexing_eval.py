import os
import pickle
import pandas as pd
import numpy as np
import igraph as ig
from typing import Dict, List, Optional, Tuple
import warnings
import argparse
from dotenv import load_dotenv

try:
    from neo4j import GraphDatabase
    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False

load_dotenv()

def _make_path_idable(path: str) -> str:
    """nano_graphrag の Neo4j namespace と同一の変換。"""
    return path.replace(".", "_").replace("/", "__").replace("-", "_").replace(":", "_").replace("\\", "__")


def analyze_graph(g: ig.Graph) -> Dict[str, float]:
    """
    Analyze a single graph object and calculate related metrics.

    :param g: igraph.Graph object
    :return: A dictionary containing the graph's related metrics
    """
    # Calculate various metrics
    num_nodes = g.vcount()
    num_edges = g.ecount()
    average_degree = sum(g.degree()) / num_nodes if num_nodes > 0 else 0
    density = g.density()
    components = g.components()
    num_components = len(components)
    largest_component_size = components.giant().vcount()
    average_clustering_coefficient = g.transitivity_avglocal_undirected()
    diameter = g.diameter() if g.is_connected() else float('inf')  # If graph is not connected, diameter is infinity

    # Calculate average connected component size (excluding isolated single nodes)
    component_sizes = [len(component) for component in components if len(component) > 1]
    if component_sizes:  # If there are non-isolated connected components
        average_component_size = sum(component_sizes) / len(component_sizes)
        median_component_size = np.median(component_sizes)  # Median of connected components
        num_components_excluding_isolated = len(component_sizes)  # Number of connected components excluding isolated entities
        num_components_above_average = sum(1 for size in component_sizes if size > average_component_size)  # Number of components above average
        num_nodes_excluding_isolated = sum(component_sizes)  # Number of entities excluding isolated ones

        # Calculate trimmed mean (excluding one highest and one lowest value)
        component_sizes_sorted = sorted(component_sizes)
        trimmed_mean_component_size = sum(component_sizes_sorted[1:-1]) / (len(component_sizes_sorted) - 2) if len(component_sizes_sorted) > 2 else average_component_size

        # Calculate geometric mean
        geometric_mean_component_size = np.exp(np.mean(np.log(component_sizes))) if len(component_sizes) > 0 else 0

        # Calculate harmonic mean
        harmonic_mean_component_size = len(component_sizes) / sum(1.0 / size for size in component_sizes) if len(component_sizes) > 0 else 0

    else:  # If all connected components are isolated single nodes
        average_component_size = 0
        median_component_size = 0
        num_components_excluding_isolated = 0
        num_components_above_average = 0
        num_nodes_excluding_isolated = 0
        trimmed_mean_component_size = 0
        geometric_mean_component_size = 0
        harmonic_mean_component_size = 0

    degrees = g.degree(mode="all")  # Use appropriate mode for directed graphs ("in", "out" or "all")

    num_isolated_nodes = sum(1 for d in degrees if d == 0)
    num_nodes_excluding_isolated = sum(1 for d in degrees if d > 0)

    num_nodes_degree_above_1 = sum(1 for d in degrees if d > 1)
    num_nodes_degree_above_2 = sum(1 for d in degrees if d > 2)
    num_nodes_degree_above_3 = sum(1 for d in degrees if d > 3)

    # Return results
    return {
        "num_nodes": float(num_nodes),
        "num_edges": float(num_edges),
        "average_degree": float(average_degree),
        "density": float(density),
        "num_components": float(num_components),
        "largest_component_size": float(largest_component_size),
        "average_clustering_coefficient": float(average_clustering_coefficient),
        "diameter": float(diameter),
        "average_component_size": float(average_component_size),
        "median_component_size": float(median_component_size),
        "trimmed_mean_component_size": float(trimmed_mean_component_size),
        "geometric_mean_component_size": float(geometric_mean_component_size),
        "harmonic_mean_component_size": float(harmonic_mean_component_size),
        "num_components_excluding_isolated": float(num_components_excluding_isolated),
        "num_components_above_average": float(num_components_above_average),
        "num_nodes_excluding_isolated": float(num_nodes_excluding_isolated),
        "num_isolated_nodes": float(num_isolated_nodes),
        "num_nodes_degree_above_1": float(num_nodes_degree_above_1),
        "num_nodes_degree_above_2": float(num_nodes_degree_above_2),
        "num_nodes_degree_above_3": float(num_nodes_degree_above_3)
    }


def load_graph_from_parquet(entities_path: str, relationships_path: str) -> ig.Graph:
    """
    Load graph data from entities.parquet and relationships.parquet files and convert to igraph.Graph object.

    :param entities_path: Path to entities.parquet file
    :param relationships_path: Path to relationships.parquet file
    :return: igraph.Graph object
    """
    # Read entities.parquet file
    entities_df = pd.read_parquet(entities_path)
    
    # Read relationships.parquet file
    relationships_df = pd.read_parquet(relationships_path)
    
    # Create igraph graph object
    g = ig.Graph()

    # Add nodes
    for _, row in entities_df.iterrows():
        entity_id = row['id']  # Use 'id' column as unique identifier for nodes
        g.add_vertex(name=entity_id)  # Use entity's unique identifier as node name

    # Ensure all edge sources and targets are in the graph
    for _, row in relationships_df.iterrows():
        source_id = row['source']  # Use 'source' column as edge source
        target_id = row['target']  # Use 'target' column as edge target

        # Check if source and target are in the graph, add if not
        if source_id not in g.vs['name']:
            g.add_vertex(name=source_id)
        if target_id not in g.vs['name']:
            g.add_vertex(name=target_id)

        # Get edge weight, default to 1 if not present
        weight = row.get('weight', 1)
        g.add_edge(source_id, target_id, weight=weight)  # Add edge with weight as edge attribute

    return g


def load_graph_from_pickle(pickle_path: str) -> ig.Graph:
    """
    Load graph data from pickle file.

    :param pickle_path: Path to pickle file
    :return: igraph.Graph object
    """
    with open(pickle_path, 'rb') as f:
        g = pickle.load(f)
    return g


def load_graph_from_picklez(picklez_path: str) -> ig.Graph:
    """
    Load graph data from picklez file.

    :param picklez_path: Path to picklez file
    :return: igraph.Graph object
    """
    g = ig.Graph.Read_Picklez(picklez_path)
    return g


def load_graph_from_graphml(graphml_path: str) -> ig.Graph:
    """
    Load graph data from GraphML file.

    :param graphml_path: Path to GraphML file
    :return: igraph.Graph object
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = ig.Graph.Read_GraphML(graphml_path)
    return g


def load_graph_from_neo4j(
    neo4j_uri: str,
    neo4j_auth: Tuple[str, str],
    working_dir: str,
) -> ig.Graph:
    """
    Neo4j に保存された nano_graphrag のグラフを読み込み igraph.Graph にする。
    Qdrant はベクトル用のためグラフ構造は Neo4j のみ参照する。

    :param neo4j_uri: Neo4j の URI (例: neo4j://localhost:7687)
    :param neo4j_auth: (user, password) のタプル
    :param working_dir: nano_graphrag の working_dir（namespace の算出に使用）
    :return: igraph.Graph object
    """
    if not _NEO4J_AVAILABLE:
        raise ImportError("Neo4j から読み込むには neo4j パッケージが必要です: pip install neo4j")

    namespace = f"{_make_path_idable(working_dir)}__chunk_entity_relation"
    driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)

    try:
        with driver.session() as session:
            # 全ノード id 取得
            r = session.run(f"MATCH (n:`{namespace}`) RETURN n.id AS id")
            node_ids = [record["id"] for record in r if record["id"] is not None]

            # 全エッジ取得
            r = session.run(
                f"MATCH (s:`{namespace}`)-[r]->(t:`{namespace}`) RETURN s.id AS source, t.id AS target"
            )
            edges = [(record["source"], record["target"]) for record in r if record["source"] and record["target"]]
    finally:
        driver.close()

    g = ig.Graph(directed=True)
    if not node_ids and not edges:
        return g

    # エッジに出てくるがノード一覧に無い id があれば追加
    all_ids = set(node_ids) | {s for s, _ in edges} | {t for _, t in edges}
    id_list = list(all_ids)
    id_to_idx = {nid: i for i, nid in enumerate(id_list)}
    g.add_vertices(len(id_list))
    g.vs["name"] = id_list

    for src, tgt in edges:
        if src in id_to_idx and tgt in id_to_idx:
            g.add_edge(id_to_idx[src], id_to_idx[tgt])

    return g


def process_graphs_neo4j(
    neo4j_uri: str,
    neo4j_auth: Tuple[str, str],
    working_dirs: List[str],
) -> List[Dict]:
    """
    Neo4j に保存された nano_graphrag のグラフを複数 working_dir 分処理する。

    :param neo4j_uri: Neo4j の URI
    :param neo4j_auth: (user, password)
    :param working_dirs: working_dir のリスト（1件でも可）
    :return: 各グラフのメトリクス辞書のリスト
    """
    results = []
    for working_dir in working_dirs:
        try:
            g = load_graph_from_neo4j(neo4j_uri, neo4j_auth, working_dir)
            result = analyze_graph(g)
            results.append(result)
        except Exception as e:
            print(f"Error loading Neo4j graph for working_dir={working_dir}: {e}")
    return results


def process_graphs_microsoft_graphrag(base_path: str, folder_name: str) -> List[Dict]:
    """
    Process graph data generated by Microsoft GraphRAG.

    :param base_path: Root path containing multiple subdirectories
    :param folder_name: Name of subdirectory containing graph data
    :return: A list containing metric dictionaries for each graph
    """
    results = []

    # Traverse each subdirectory under base_path
    for subdir, dirs, files in os.walk(base_path):
        entities_path = os.path.join(subdir, 'entities.parquet')
        relationships_path = os.path.join(subdir, 'relationships.parquet')
        if os.path.exists(entities_path) and os.path.exists(relationships_path):
            try:
                g = load_graph_from_parquet(entities_path, relationships_path)
                result = analyze_graph(g)
                results.append(result)
            except Exception as e:
                print(f"Error processing {subdir}: {e}")

    return results


def process_graphs_lightrag_fastgraphrag(base_path: str, folder_name: str) -> List[Dict]:
    """
    Process graph data generated by LightRAG and Fast-GraphRAG.

    :param base_path: Root path containing multiple subdirectories
    :param folder_name: Name of subdirectory containing graph data
    :return: A list containing metric dictionaries for each graph
    """
    results = []

    # Traverse each subdirectory under base_path
    for subdir, dirs, files in os.walk(base_path):
        # For LightRAG: look for graph_chunk_entity_relation.graphml files
        # For Fast-GraphRAG: look for graph_igraph_data.pklz files
        lightrag_path = os.path.join(subdir, 'graph_chunk_entity_relation.graphml')
        fastgraphrag_path = os.path.join(subdir, 'graph_igraph_data.pklz')
        
        if os.path.exists(lightrag_path):
            try:
                # Load graph from GraphML file (LightRAG)
                g = ig.Graph.Read_GraphML(lightrag_path)
                result = analyze_graph(g)
                results.append(result)
            except Exception as e:
                print(f"Error loading LightRAG graph from {lightrag_path}: {e}")
        elif os.path.exists(fastgraphrag_path):
            try:
                # Load graph from pickle file (Fast-GraphRAG)
                g = load_graph_from_picklez(fastgraphrag_path)
                result = analyze_graph(g)
                results.append(result)
            except Exception as e:
                print(f"Error loading Fast-GraphRAG graph from {fastgraphrag_path}: {e}")

    return results


def process_graphs_hipporag2(base_path: str, folder_name: str) -> List[Dict]:
    """
    Process graph data generated by HippoRAG2.

    :param base_path: Root path containing multiple subdirectories
    :param folder_name: Name of subdirectory containing graph data
    :return: A list containing metric dictionaries for each graph
    """
    results = []

    # Traverse each subdirectory under base_path
    for subdir, dirs, files in os.walk(base_path):
        target_folder = os.path.join(subdir, folder_name)
        if os.path.exists(target_folder):
            graph_path = os.path.join(target_folder, 'graph.pickle')
            if os.path.exists(graph_path):
                try:
                    g = load_graph_from_pickle(graph_path)
                    result = analyze_graph(g)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {subdir}: {e}")

    return results


def process_graphs_graphml(base_path: str, pattern: str = "*.graphml") -> List[Dict]:
    """
    Process graph data in GraphML format.

    :param base_path: Root path containing graph files
    :param pattern: File matching pattern
    :return: A list containing metric dictionaries for each graph
    """
    results = []

    # Traverse each subdirectory under base_path
    for subdir, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.graphml'):
                graph_path = os.path.join(subdir, file)
                try:
                    g = load_graph_from_graphml(graph_path)
                    result = analyze_graph(g)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {graph_path}: {e}")

    return results


def calculate_average(results: List[Dict]) -> Dict[str, float]:
    """
    Calculate average metrics for all graphs.

    :param results: A list containing metric dictionaries for each graph
    :return: A dictionary containing average values for all metrics
    """
    if not results:
        return {}

    # Initialize dictionary to store averages
    avg_results = {key: 0.0 for key in results[0].keys()}

    # Accumulate metrics from all graphs
    for result in results:
        for key, value in result.items():
            avg_results[key] += value

    # Calculate averages
    num_graphs = len(results)
    for key in avg_results:
        avg_results[key] /= num_graphs

    return avg_results


def calculate_indexing_metrics(
    framework: str,
    base_path: str,
    folder_name: Optional[str] = None,
    *,
    neo4j_uri: Optional[str] = None,
    neo4j_auth: Optional[Tuple[str, str]] = None,
    working_dirs: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate indexing graph metrics for specified framework.

    :param framework: Framework name ('microsoft_graphrag', 'lightrag', 'fast_graphrag', 'hipporag2', 'graphml', 'nano_graphrag')
    :param base_path: Root path containing graph data (nano_graphrag の場合は未使用)
    :param folder_name: Subdirectory name (required for some frameworks)
    :param neo4j_uri: Neo4j URI (nano_graphrag 時必須)
    :param neo4j_auth: (user, password) (nano_graphrag 時必須)
    :param working_dirs: nano_graphrag の working_dir のリスト (nano_graphrag 時。1件でも可)
    :return: Average metrics dictionary
    """
    if framework == 'microsoft_graphrag':
        results = process_graphs_microsoft_graphrag(base_path, folder_name or "")
    elif framework in ['lightrag', 'fast_graphrag']:
        results = process_graphs_lightrag_fastgraphrag(base_path, folder_name or "")
    elif framework == 'hipporag2':
        if not folder_name:
            raise ValueError("HippoRAG2 requires folder_name parameter")
        results = process_graphs_hipporag2(base_path, folder_name)
    elif framework == 'graphml':
        results = process_graphs_graphml(base_path)
    elif framework == 'nano_graphrag':
        if not neo4j_uri or not neo4j_auth:
            raise ValueError("nano_graphrag では neo4j_uri と neo4j_auth が必須です")
        if not working_dirs:
            if folder_name:
                working_dirs = [folder_name]
            else:
                raise ValueError("nano_graphrag では working_dirs または folder_name (working_dir 1件) を指定してください")
        results = process_graphs_neo4j(neo4j_uri, neo4j_auth, working_dirs)
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    if not results:
        msg = f"Warning: No graph data found for {framework}"
        if framework != "nano_graphrag":
            msg += f" in {base_path}"
        print(msg)
        return {}

    return calculate_average(results)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Calculate indexing graph metrics for different GraphRAG frameworks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--framework', 
        type=str, 
        required=True,
        choices=['nano_graphrag'],
        help='Framework to analyze (nano_graphrag = Neo4j に保存したグラフ)'
    )
    
    parser.add_argument(
        '--base_path', 
        type=str, 
        default=None,
        help='Root path containing graph data (nano_graphrag の場合は不要)'
    )
    
    parser.add_argument(
        '--folder_name', 
        type=str, 
        default=None,
        help='Subdirectory name (hipporag2 で必須). nano_graphrag の場合は working_dir として1件指定可'
    )
    
    parser.add_argument(
        '--working_dir',
        type=str,
        default=None,
        help='nano_graphrag 用: 対象の working_dir パス（1件）。--folder_name の代わりに指定可'
    )
    
    parser.add_argument(
        '--neo4j_uri',
        type=str,
        default=None,
        help='nano_graphrag 用: Neo4j URI (未指定時は環境変数 NEO4J_URI)'
    )
    
    parser.add_argument(
        '--neo4j_user',
        type=str,
        default=None,
        help='nano_graphrag 用: Neo4j ユーザー (未指定時は環境変数 NEO4J_USER または neo4j)'
    )
    
    parser.add_argument(
        '--neo4j_password',
        type=str,
        default=None,
        help='nano_graphrag 用: Neo4j パスワード (未指定時は環境変数 NEO4J_PASSWORD)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Output file path (optional, prints to stdout if not specified)'
    )
    
    return parser.parse_args()


def main():
    """
    Main function for command line usage.
    """
    args = parse_args()
    
    if args.framework != "nano_graphrag" and not args.base_path:
        print("Error: --base_path is required for frameworks other than nano_graphrag")
        return 1

    try:
        print(f"Calculating indexing graph metrics for {args.framework}...")
        if args.framework == "nano_graphrag":
            neo4j_uri = args.neo4j_uri or os.environ.get("NEO4J_URI")
            neo4j_user = args.neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
            neo4j_password = args.neo4j_password or os.environ.get("NEO4J_PASSWORD", "neo4j")
            working_dirs = [args.working_dir] if args.working_dir else ([args.folder_name] if args.folder_name else None)
            print(f"Neo4j URI: {neo4j_uri}")
            print(f"Working dir(s): {working_dirs}")
        else:
            neo4j_uri = working_dirs = None
            print(f"Base path: {args.base_path}")
            if args.folder_name:
                print(f"Folder name: {args.folder_name}")
        print()
        
        if args.framework == "nano_graphrag":
            if not neo4j_uri or not neo4j_user or not neo4j_password:
                print("Error: Neo4j 接続には NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD の指定か環境変数が必要です")
                return 1
            if not working_dirs:
                print("Error: nano_graphrag では --working_dir または --folder_name を指定してください")
                return 1
            metrics = calculate_indexing_metrics(
                framework=args.framework,
                base_path=args.base_path or "",
                folder_name=args.folder_name,
                neo4j_uri=neo4j_uri,
                neo4j_auth=(neo4j_user, neo4j_password),
                working_dirs=working_dirs,
            )
        else:
            metrics = calculate_indexing_metrics(
                framework=args.framework,
                base_path=args.base_path,
                folder_name=args.folder_name
            )
        
        if metrics:
            output_lines = [f"Average metrics for {args.framework}:"]
            for key, value in metrics.items():
                output_lines.append(f"  {key}: {value:.4f}")
            
            output_text = "\n".join(output_lines)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output_text)
                print(f"Results saved to {args.output}")
            else:
                print(output_text)
        else:
            print(f"No graph data found for {args.framework} in {args.base_path}")
            
    except Exception as e:
        print(f"Error calculating metrics for {args.framework}: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())