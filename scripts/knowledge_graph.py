import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv
from scripts import utils
import importlib

from models import Concept, Entity, Relation

load_dotenv()

class KnowledgeGraph:
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password-to-kg")
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        try:
            self.driver.verify_connectivity()
        except Exception as e:
            self.driver.close()
            raise RuntimeError(f"Neo4j connection could not be established. Error: {e}")
            
    def close(self):
        """Close the Neo4j database driver connection."""
        self.driver.close()

    def load_data(self, dataset_path):
        """Loads all data sequentially into the Neo4j database."""
        if not os.path.isabs(dataset_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_path = os.path.abspath(os.path.join(base_dir, '..', dataset_path))
        
        if os.path.isfile(dataset_path):
            dataset_name = os.path.basename(os.path.dirname(dataset_path))
        else:
            dataset_name = os.path.basename(dataset_path)

        if os.path.isdir(dataset_path):
            dataset_path = os.path.join(dataset_path, 'kb.json')

        with open(dataset_path, encoding='utf-8') as f:
            dataset = json.load(f)

        self._clear_database()
        self._create_constraints()

        loader = self._get_loader(dataset_name)
        loader.load(self.driver, dataset)
        
        print(f"Knowledge graph loading complete using '{dataset_name}' loader.")

    def _get_loader(self, dataset_name):
        module_path = f"scripts.dataset_specific.{dataset_name}.data_loader"
        try:
            return importlib.import_module(module_path)
        except ImportError as e:
            raise ValueError(f"No data loader found for dataset '{dataset_name}' (expected at {module_path}). Error: {e}")

    def get_statistic(self):
        """Returns statistics of the knowledge graph."""
        stats = {}
        with self.driver.session() as session:
            node_result = session.run("MATCH (n) RETURN count(n) AS totalNodes")
            record = node_result.single()
            stats['total_nodes'] = record["totalNodes"] if record else 0
            
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) AS totalRelationships")
            record = rel_result.single()
            stats['total_relationships'] = record["totalRelationships"] if record else 0
            
            concept_result = session.run("MATCH (n:Concept) RETURN count(n) AS totalConcepts")
            record = concept_result.single()
            stats['total_concepts'] = record["totalConcepts"] if record else 0
            
            entity_result = session.run("MATCH (n:Entity) RETURN count(n) AS totalEntities")
            record = entity_result.single()
            stats['total_entities'] = record["totalEntities"] if record else 0
            
            is_a_result = session.run("MATCH ()-[r:IS_A]->() RETURN count(r) AS totalIsA")
            record = is_a_result.single()
            stats['total_is_a_relationships'] = record["totalIsA"] if record else 0
            
            # Additional analysis logic from data_loader.py
            analysis_query = """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[:IS_A]->(c:Concept)
                WITH e, count(c) as rel_count
                RETURN min(rel_count) as min_rels, max(rel_count) as max_rels, avg(rel_count) as avg_rels
            """
            analysis_result = session.run(analysis_query)
            record = analysis_result.single()
            if record:
                stats['entity_concept_is_a_min'] = record['min_rels'] if record['min_rels'] is not None else 0
                stats['entity_concept_is_a_max'] = record['max_rels'] if record['max_rels'] is not None else 0
                stats['entity_concept_is_a_avg'] = record['avg_rels'] if record['avg_rels'] is not None else 0.0
            else:
                stats['entity_concept_is_a_min'] = 0
                stats['entity_concept_is_a_max'] = 0
                stats['entity_concept_is_a_avg'] = 0.0
            
        return stats
    
    def get_random_entity(self) -> Entity:
        query = """
        MATCH (e:Entity)
        RETURN e
        ORDER BY RAND()
        LIMIT 1
        """

        with self.driver.session() as session:
            result = session.run(query).single()
            if result is None:
                raise LookupError('There isnt any Entity in the database.')
            entity = result['e']

            return Entity(
                id=entity['id'],
                name=entity['name'],
                attributes=[],
                relationships=[])
    
    def get_random_parent_concept(self, entity: Entity) -> Concept:
        query = """
        MATCH (e:Entity {id: $entity_id})-[:IS_A]->(c:Concept)
        RETURN c
        ORDER BY RAND()
        LIMIT 1
        """

        with self.driver.session() as session:
            result = session.run(query, entity_id=entity.id).single()
            if result is None:
                raise LookupError(f"Entity '{entity.id}' does not have any parent Concept.")
            concept = result['c']

            return Concept(
                id=concept['id'],
                name=concept['name'])
        
    def get_random_relation(self, entity: Entity) -> Relation:
        query = """
        MATCH p = (e:Entity {id: $entity_id})-[]->(target:Entity)
        RETURN p
        ORDER BY RAND()
        LIMIT 1
        """

        with self.driver.session() as session:
            result = session.run(query, entity_id=entity.id).single()
            if result is None:
                raise LookupError(f"Entity '{entity.id}' does not have any relation with other entity.")
            path = result['p']
            label = path.relationships[0].type
            end_node = path.end_node
            target = Entity(
                id=end_node['id'],
                name=end_node['name'],
                attributes=[],
                relationships=[]
            )

            return Relation(
                name=label,
                target=target,
                qualifiers={}
            )

    def _clear_database(self):
        print("Clearing database...")
        cleanup_query = """
        CALL apoc.periodic.iterate(
        "MATCH (n) RETURN n",
        "DETACH DELETE n",
        {batchSize: 10000, parallel: false}
        )
        """
        with self.driver.session() as session:
            session.run(cleanup_query)

    def _create_constraints(self):
        print("Creating constraints...")
        query = '''
        CREATE CONSTRAINT id_unique IF NOT EXISTS
        FOR (b:Base) REQUIRE b.id IS UNIQUE;
        '''
        with self.driver.session() as session:
            session.run(query)


if __name__ == "__main__":
    knowledge_graph = KnowledgeGraph()
    knowledge_graph.load_data('dataset/kqa-pro')
    print(knowledge_graph.get_statistic())