import os
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv
import importlib
import json

from models import Concept, Entity, Relation, DataLoader

CYPHER_PARSER_REGEX = re.compile(
    r"(?i)"  # Case-insensitive
    r"(?P<match_type>OPTIONAL\s+MATCH|MATCH)\s+(?P<match_clause>.+?)"
    r"(?:\s+WHERE\s+(?P<where_clause>.+?))?"
    r"\s+RETURN\s+(?P<return_clause>.+?)"
    r"(?:\s+ORDER\s+BY\s+(?P<order_clause>.+?))?"
    r"(?:\s+LIMIT\s+(?P<limit_clause>\d+))?"
    r"\s*;?$",  # Optional semicolon and trailing whitespace
    re.DOTALL | re.MULTILINE
)

load_dotenv()

class KnowledgeGraph:
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password-to-kg")
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        try:
            self.driver.verify_connectivity()
            self._setup_metadata_cache()
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
        
        dataset_name = os.path.basename(dataset_path)

        module_path = f"scripts.dataset_specific.{dataset_name}.data_loader"
        try:
            loader: DataLoader = importlib.import_module(module_path).get_loader(self.driver, dataset_path)
            loader.load()
            self._setup_metadata_cache()
        except ImportError as e:
            raise ValueError(f"No data loader found for dataset '{dataset_name}' (expected at {module_path}). Error: {e}")
        
        print(f"Knowledge graph loading complete using '{dataset_name}' loader.")

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
    
    def _setup_metadata_cache(self):
        """Initializes the metadata cache node and installs the APOC trigger."""
        setup_query = """
        MERGE (m:GraphMetadata {id: 'singleton'})
        ON CREATE SET m.is_stale = true
        """
        # APOC trigger to mark cache as stale when data changes
        trigger_query = """
        CALL apoc.trigger.install(
            'neo4j',
            'mark_metadata_stale',
            "UNWIND $createdNodes + $deletedNodes + $createdRelationships + $deletedRelationships AS x
             MATCH (m:GraphMetadata {id: 'singleton'})
             SET m.is_stale = true",
            {phase: 'afterAsync'}
        )
        """
        with self.driver.session() as session:
            session.run(setup_query)
            try:
                session.run(trigger_query)
            except Exception as e:
                # Trigger might already exist or APOC might be restricted
                print(f"Warning: Could not install APOC trigger: {e}")

    def _ensure_fresh_metadata(self):
        """Checks if metadata is stale and recomputes if necessary."""
        with self.driver.session() as session:
            result = session.run("MATCH (m:GraphMetadata {id: 'singleton'}) RETURN m.is_stale AS is_stale").single()
            if not result or result['is_stale']:
                print("Metadata cache is stale. Recomputing...")
                self._recompute_and_cache_metadata()

    def _recompute_and_cache_metadata(self):
        """Performs full scan and updates the cache node."""
        node_labels = self._fetch_node_labels()
        rel_labels = self._fetch_relation_labels()
        entity_mappings = self._fetch_entity_concept_mappings()
        concept_rels = self._fetch_relation_labels_between_concepts()

        # Serialize dicts for Neo4j storage
        # Note: JSON keys must be strings, so we join tuple keys
        serialized_concept_rels = {f"{k[0]}|{k[1]}": v for k, v in concept_rels.items()}

        query = """
        MATCH (m:GraphMetadata {id: 'singleton'})
        SET m.node_labels = $node_labels,
            m.relation_labels = $rel_labels,
            m.entity_concept_mappings = $entity_mappings,
            m.concept_relation_labels = $concept_rels,
            m.is_stale = false,
            m.last_updated = datetime()
        """
        with self.driver.session() as session:
            session.run(query, 
                node_labels=node_labels,
                rel_labels=rel_labels,
                entity_mappings=json.dumps(entity_mappings),
                concept_rels=json.dumps(serialized_concept_rels)
            )

    def _fetch_node_labels(self) -> list[str]:
        query = "CALL db.labels()"
        with self.driver.session() as session:
            result = session.run(query)
            return [record[0] for record in result]

    def _fetch_relation_labels(self) -> list[str]:
        query = "CALL db.relationshipTypes()"
        with self.driver.session() as session:
            result = session.run(query)
            return [record[0] for record in result]

    def _fetch_entity_concept_mappings(self) -> dict[str, list[str]]:
        query = """
        MATCH (e:Entity)-[:IS_A]->(c:Concept)
        RETURN e.name AS entity, c.name AS concept
        """
        mappings = {}
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                entity = record['entity']
                concept = record['concept']
                if entity not in mappings:
                    mappings[entity] = []
                mappings[entity].append(concept)
        return mappings

    def _fetch_relation_labels_between_concepts(self) -> dict[tuple[str, str], list[str]]:
        query = """
        MATCH (c1:Concept)-[r]-(c2:Concept)
        WHERE type(r) <> 'IS_A'
        RETURN DISTINCT c1.name AS c1, c2.name AS c2, type(r) AS rel
        """
        mappings = {}
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                key = (record['c1'], record['c2'])
                if key not in mappings:
                    mappings[key] = []
                mappings[key].append(record['rel'])
        return mappings

    def get_node_labels_list(self) -> list[str]:
        self._ensure_fresh_metadata()
        with self.driver.session() as session:
            return session.run("MATCH (m:GraphMetadata {id: 'singleton'}) RETURN m.node_labels").single()[0]

    def get_relation_labels_list(self) -> list[str]:
        self._ensure_fresh_metadata()
        with self.driver.session() as session:
            return session.run("MATCH (m:GraphMetadata {id: 'singleton'}) RETURN m.relation_labels").single()[0]

    def get_entity_concept_mappings(self) -> dict[str, list[str]]:
        self._ensure_fresh_metadata()
        with self.driver.session() as session:
            data = session.run("MATCH (m:GraphMetadata {id: 'singleton'}) RETURN m.entity_concept_mappings").single()[0]
            return json.loads(data)

    def get_relation_labels_between_concepts(self) -> dict[tuple[str, str], list[str]]:
        self._ensure_fresh_metadata()
        with self.driver.session() as session:
            data = session.run("MATCH (m:GraphMetadata {id: 'singleton'}) RETURN m.concept_relation_labels").single()[0]
            raw_dict = json.loads(data)
            # Reconstruct tuple keys
            return {tuple(k.split('|')): v for k, v in raw_dict.items()}
    
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

if __name__ == "__main__":
    knowledge_graph = KnowledgeGraph()
    knowledge_graph.load_data('dataset/kqa-pro')
    print(knowledge_graph.get_statistic())