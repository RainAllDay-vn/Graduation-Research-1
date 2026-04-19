import os
from typing import Optional, Iterator

from neo4j import GraphDatabase
from app.data_loader.contract import DataLoaderContract

class KnowledgeGraphBuilder:
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
        try:
            self.driver.verify_connectivity()
        except Exception as e:
            self.driver.close()
            raise RuntimeError(f"Neo4j connection could not be established. Error: {e}") from e

    def build_knowledge_graph(self) -> None:
        self._clear_database()
        self._create_constraints()
        self._insert_entities()
        self._insert_concepts()
        self._insert_entity_isa_concept_relations()
        self._insert_entity_to_entity_relations()
        self._insert_concept_to_concept_relations()

    def _clear_database(self) -> None:
        print("Dropping database...")
        self.driver.execute_query(
            f"DROP DATABASE {self.database_name} IF EXISTS",
            database_v_= "system"
        )
        print("Creating database...")
        self.driver.execute_query(
            f"CREATE DATABASE {self.database_name}",
            database_v_= "system"
        )

    def _create_constraints(self) -> None:
        print("Creating constraints...")
        query = '''
        CREATE CONSTRAINT id_unique IF NOT EXISTS
        FOR (b:BASE) REQUIRE b.id IS UNIQUE;
        '''
        with self.driver.session(database=self.database_name) as session:
            session.run(query)

    def _insert_concepts(self) -> None:
        concepts = self.data_loader.load_concepts()
        data = ({
            'id': c.id,
            'name': c.name,
            'labels': c.labels
        } for c in concepts)
        query = '''
        UNWIND $batch as item
        CALL apoc.create.node(
            item.labels + ['BASE', 'CONCEPT'], 
            {id: item.id, name: item.name}
        ) YIELD node
        RETURN count(*)
        '''

        self._insert_in_batches(query, data)

    def _insert_entities(self) -> None:
        entities = self.data_loader.load_entities()
        data = ({'id': e.id, 'name': e.name, 'labels': e.labels} for e in entities)
        query = '''
        UNWIND $batch as item
        CALL apoc.create.node(
            item.labels + ['BASE', 'ENTITY'], 
            {id: item.id, name: item.name}
        ) YIELD node
        RETURN count(*)
        '''

        self._insert_in_batches(query, data)

    def _insert_entity_isa_concept_relations(self) -> None:
        relations = self.data_loader.load_entity_isa_concept_relations()
        data = ({
            'source_id': r.source_id, 
            'target_id': r.target_id,
            'label': r.label
        } for r in relations)
        query = '''
        UNWIND $batch as item
        MATCH (a:BASE {id: item.source_id})
        MATCH (b:BASE {id: item.target_id})
        CALL apoc.create.relationship(a, item.label, {}, b) YIELD rel
        WITH a, [label IN labels(b) WHERE label <> 'CONCEPT'] AS filteredLabels
        CALL apoc.create.addLabels(a, filteredLabels) YIELD node
        RETURN count(*)
        '''

        self._insert_in_batches(query, data)

    def _insert_entity_to_entity_relations(self) -> None:
        relations = self.data_loader.load_entity_to_entity_relations()
        data = ({
            'source_id': r.source_id, 
            'target_id': r.target_id,
            'label': r.label,
            'name': r.name
        } for r in relations)
        query = '''
        UNWIND $batch as item
        MATCH (a:BASE {id: item.source_id})
        MATCH (b:BASE {id: item.target_id})
        CALL apoc.create.relationship(a, item.label, {name: item.name}, b) YIELD rel
        RETURN count(*)
        '''
        self._insert_in_batches(query, data)

    def _insert_concept_to_concept_relations(self) -> None:
        pass

    def _insert_in_batches(self,
        query: str,
        data: Iterator,
        batch_size: int = 100_000
    ) -> None:
        batch = []
        count = 0
        try:
            while True:
                batch.append(next(data))
                count += 1
                if count >= batch_size:
                    with self.driver.session(database=self.database_name) as session:
                        session.run(query, batch=batch)
                    batch = []
        except StopIteration:
            if count > 0:
                with self.driver.session(database=self.database_name) as session:
                    session.run(query, batch=batch)
