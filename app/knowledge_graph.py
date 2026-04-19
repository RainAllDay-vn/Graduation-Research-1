import os
from typing import Optional

from neo4j import GraphDatabase

from app.data_loader.contract import DataLoaderContract

class KnowledgeGraph:
    def __init__(self,
        data_loader: DataLoaderContract,
        uri: Optional[str] = None,
        database_name: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ) -> None:
        self.data_loader = data_loader
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.database_name = database_name or os.environ.get("NEO4J_DATABASE", "neo4j")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password-to-kg")

        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.node_labels: Optional[list[str]] = None
        self.relation_labels: Optional[list[str]] = None
        try:
            self.driver.verify_connectivity()
        except Exception as e:
            self.driver.close()
            raise RuntimeError(f"Neo4j connection could not be established. Error: {e}") from e

    def close(self):
        """Close the Neo4j database driver connection."""
        self.driver.close()

    def get_node_labels(self) -> list[str]:
        if self.node_labels is None:
            query = "CALL db.labels()"
            with self.driver.session() as session:
                result = session.run(query)
                self.node_labels = [record[0] for record in result]
        return self.node_labels

    def get_relation_labels(self) -> list[str]:
        if self.relation_labels is None:
            query = "CALL db.relationshipTypes()"
            with self.driver.session() as session:
                result = session.run(query)
                self.relation_labels = [record[0] for record in result]
        return self.relation_labels

    def execute_query(self, query: str):
        with self.driver.session() as session:
            result = session.run(query)
            return [record for record in result]
