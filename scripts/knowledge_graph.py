import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

from models import Entity, Concept

load_dotenv()

class KnowledgeGraph:
    def __init__(self, uri=None, user=None, password=None, dataset_path='../dataset/kb.json'):
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password-to-kg")
        self.dataset_path = dataset_path
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        try:
            self.driver.verify_connectivity()
        except Exception as e:
            self.driver.close()
            raise RuntimeError(f"Neo4j connection could not be established. Error: {e}")
            
    def close(self):
        """Close the Neo4j database driver connection."""
        self.driver.close()

    def load_data(self):
        """Loads all data sequentially into the Neo4j database."""
        with open(self.dataset_path, encoding='utf-8') as f:
            dataset = json.load(f)
        self._clear_database()
        self._create_constraints()
        self._insert_concepts(dataset)
        self._insert_concept_relations(dataset)
        self._insert_entities(dataset)
        self._insert_entity_concept_relations(dataset)
        self._insert_entity_relations(dataset)
        print("Knowledge graph loading complete.")

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

    def _insert_concepts(self, dataset):
        print("Inserting concepts...")
        concepts = dataset.get('concepts', {})
        data = [{'id': k, 'name': v['name']} for k, v in concepts.items()]
        query = """
        UNWIND $batch as item
        MERGE (c:Concept:Base {id: item.id})
        SET c.name = item.name
        """
        with self.driver.session() as session:
            session.run(query, batch=data)

    def _insert_concept_relations(self, dataset):
        print("Inserting concept IS_A relations...")
        concepts = dataset.get('concepts', {})
        data = []
        for k, v in concepts.items():
            for parent_id in v.get('instanceOf', []):
                data.append({'child_id': k, 'parent_id': parent_id})
        
        query = """
        UNWIND $batch as item
        MATCH (child:Base {id: item.child_id})
        MATCH (parent:Base {id: item.parent_id})
        MERGE (child)-[:IS_A]->(parent)
        """
        with self.driver.session() as session:
            session.run(query, batch=data)

    def _insert_entities(self, dataset):
        print("Inserting entities...")
        entities = dataset.get('entities', {})
        data = [{'id': k, 'name': v['name'], 'attributes': str(v.get('attributes', {}))} for k, v in entities.items()]
        
        query = """
        UNWIND $batch as item
        MERGE (e:Entity:Base {id: item.id})
        SET e.name = item.name, e.attributes = item.attributes
        """
        with self.driver.session() as session:
            session.run(query, batch=data)

    def _insert_entity_concept_relations(self, dataset):
        print("Inserting entity IS_A relations...")
        entities = dataset.get('entities', {})
        data = []
        for k, v in entities.items():
            for parent_id in v.get('instanceOf', []):
                data.append({'child_id': k, 'parent_id': parent_id})
                
        query = """
        UNWIND $batch as item
        MATCH (child:Base {id: item.child_id})
        MATCH (parent:Base {id: item.parent_id})
        MERGE (child)-[:IS_A]->(parent)
        """
        with self.driver.session() as session:
            session.run(query, batch=data)

    def _insert_entity_relations(self, dataset):
        print("Inserting entity-to-entity relationships...")
        entities = dataset.get('entities', {})
        unique_relations = {}

        for k, v in entities.items():
            for relation in v.get('relations', []):
                name = relation['predicate']
                if relation['direction'] == 'forward':
                    c_id, p_id = k, relation['object']
                else:
                    c_id, p_id = relation['object'], k
                
                triple_key = (c_id, p_id, name)
                new_qualifier = str(relation.get('qualifiers', {}))

                if triple_key in unique_relations:
                    existing_item = unique_relations[triple_key]
                    if new_qualifier not in existing_item['qualifiers']:
                        existing_item['qualifiers'] += f" | {new_qualifier}"
                else:
                    unique_relations[triple_key] = {
                        'name': name,
                        'child_id': c_id,
                        'parent_id': p_id,
                        'qualifiers': new_qualifier
                    }

        data = list(unique_relations.values())
        
        query = """
        UNWIND $batch AS item
        MATCH (child:Base {id: item.child_id})
        MATCH (parent:Base {id: item.parent_id})
        CALL apoc.merge.relationship(
        child, 
        item.name, 
        {}, 
        {qualifiers: item.qualifiers}, 
        parent
        ) YIELD rel
        RETURN count(rel) AS total
        """
        
        BATCH_SIZE = 10000
        with self.driver.session() as session:
            for i in range(0, len(data), BATCH_SIZE):
                batch = data[i:i+BATCH_SIZE]
                session.run(query, batch=batch)