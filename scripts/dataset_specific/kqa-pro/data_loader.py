import os
import json
from neo4j import Driver
from scripts import utils

def load(driver: Driver, dataset_path: str):
    """Loads KQA-Pro data sequentially into the Neo4j database."""
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' is not a directory.")
    
    dataset_path = os.path.join(dataset_path, 'kb.json')
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
        
    _clear_database(driver)
    _create_constraints(driver)
    _insert_concepts(driver, dataset)
    _insert_concept_relations(driver, dataset)
    _insert_entities(driver, dataset)
    _insert_entity_concept_relations(driver, dataset)
    _insert_entity_relations(driver, dataset)

def _clear_database(driver: Driver):
    print("Clearing database...")
    cleanup_query = """
    CALL apoc.periodic.iterate(
    "MATCH (n) RETURN n",
    "DETACH DELETE n",
    {batchSize: 2000, parallel: false}
    )
    """
    with driver.session() as session:
        session.run(cleanup_query)

def _create_constraints(driver: Driver):
    print("Creating constraints...")
    query = '''
    CREATE CONSTRAINT id_unique IF NOT EXISTS
    FOR (b:Base) REQUIRE b.id IS UNIQUE;
    '''
    with driver.session() as session:
        session.run(query)

def _insert_concepts(driver: Driver, dataset: dict):
    print("Inserting concepts...")
    concepts = dataset.get('concepts', {})
    data = [{'id': k, 'name': v['name']} for k, v in concepts.items()]
    query = """
    UNWIND $batch as item
    MERGE (c:Concept:Base {id: item.id})
    SET c.name = item.name
    """
    with driver.session() as session:
        session.run(query, batch=data)

def _insert_concept_relations(driver: Driver, dataset: dict):
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
    with driver.session() as session:
        session.run(query, batch=data)

def _insert_entities(driver: Driver, dataset: dict):
    print("Inserting entities...")
    entities = dataset.get('entities', {})
    data = [{'id': k, 'name': v['name'], 'attributes': str(v.get('attributes', {}))} for k, v in entities.items()]
    
    query = """
    UNWIND $batch as item
    MERGE (e:Entity:Base {id: item.id})
    SET e.name = item.name, e.attributes = item.attributes
    """
    with driver.session() as session:
        session.run(query, batch=data)

def _insert_entity_concept_relations(driver: Driver, dataset: dict):
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
    with driver.session() as session:
        session.run(query, batch=data)

def _insert_entity_relations(driver: Driver, dataset: dict):
    print("Inserting entity-to-entity relationships...")
    entities = dataset.get('entities', {})
    data = []

    for k, v in entities.items():
        for relation in v.get('relations', []):
            name = relation['predicate']
            name = utils.to_screaming_snake_case(name)
            if relation['direction'] == 'forward':
                c_id, p_id = k, relation['object']
            else:
                c_id, p_id = relation['object'], k
            qualifiers = str(relation.get('qualifiers', {}))

            data.append({
                'name': name,
                'child_id': c_id,
                'parent_id': p_id,
                'qualifiers': qualifiers
            })
    
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
    
    BATCH_SIZE = 2000
    with driver.session() as session:
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i+BATCH_SIZE]
            session.run(query, batch=batch)
