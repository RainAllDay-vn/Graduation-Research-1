from app.knowledge_graph_builder import KnowledgeGraphBuilder
from app.data_loader.contract import DataLoaderContract

def rebuild_knowledge_graph(data_loader: DataLoaderContract):
    builder = KnowledgeGraphBuilder(data_loader=data_loader)
    builder.build_knowledge_graph()
