# %%
from neo4j import GraphDatabase, Result
import matplotlib.pyplot as plt
import networkx as nx
import json

# %%
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password-to-kg"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# %%
try:
    driver.verify_connectivity()
            
except Exception as e:
    print(f"--- CONNECTION FAILED ---")
    print(f"Error: {e}")
    driver.close()
    raise RuntimeError("Neo4j connection could not be established. Please check your Docker container.")

# %%
with driver.session() as session:
    session.run('MATCH (n) DETACH DELETE n')

# %%
with driver.session() as session:
    query = '''
        CREATE CONSTRAINT concept_id_unique IF NOT EXISTS
        FOR (c:Concept) REQUIRE c.id IS UNIQUE;
        '''
    session.run(query)
    
    query = '''
        CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
        FOR (e:Entity) REQUIRE e.id IS UNIQUE;
        '''
    session.run(query)

# %%
with open('./dataset/kb.json', encoding='utf-8') as f:
    dataset: dict = json.load(f)

print("Loading dataset from './dataset/kb.json'...")
with open('./dataset/kb.json', encoding='utf-8') as f:
    dataset: dict = json.load(f)
print("Dataset loaded successfully.")

# %%
print("Preparing concept data for insertion...")
concepts: dict = dataset['concepts']
data = []
for (k,v) in concepts.items():
    data.append({'id': k, 'name': v['name']})
print(f"Prepared {len(data)} concepts for insertion.")

query = """
    UNWIND $batch as item
    MERGE (c:Concept {id: item.id})
    SET c.name = item.name
    """

print("Inserting concepts into Neo4j database...")
with driver.session() as session:
    session.run(query, batch=data)
    
    print("Counting total nodes in database...")
    result: Result = session.run('MATCH (n) RETURN count(n) AS totalNodes')
    record = result.single()
    
    if record:
        count = record["totalNodes"]
        print(f"Total nodes in database: {count}")
    else:
        print("No nodes found in database.")

# %%
data = []
for (k,v) in concepts.items():
    for parent_id in v['instanceOf']:
        data.append({'child_id': k, 'parent_id': parent_id})

query = """
    UNWIND $batch as item
    MATCH (child:Concept {id: item.child_id})
    MATCH (parent:Concept {id: item.parent_id})
    MERGE (child)-[:IS_A]->(parent)
    """

with driver.session() as session:
    session.run(query, batch=data)

    result: Result = session.run('MATCH ()-[r:IS_A]->() RETURN count(r) AS totalRelation')
    record = result.single()
    
    if record:
        count = record["totalRelation"]
        print(f"Total IS_A relationships in database: {count}")
    else:
        print("No IS_A relationships found in database.")

# %%
query = """
    MATCH p = (:Concept)-[:IS_A]->(:Concept)
    RETURN p
    LIMIT 5
    """

with driver.session() as session:
    result = session.run(query)
    
    G = nx.MultiDiGraph()
    
    for record in result:
        path = record['p']
        for node in path.nodes:
            G.add_node(node.element_id, label=node.get("name", "Unknown"))
        for rel in path.relationships:
            G.add_edge(rel.start_node.element_id, rel.end_node.element_id, type=rel.type)

    plt.figure(figsize=(12, 10))

    node_labels = nx.get_node_attributes(G, 'label')
    pos = nx.spring_layout(G, k=1.5, iterations=50)

    nx.draw(
        G, 
        pos, 
        labels=node_labels,
        with_labels=True, 
        node_color="#1cace4", 
        node_size=3000,
        font_size=9, 
        font_weight='bold',
        edge_color='#A0A0A0',
        arrowsize=20,
        connectionstyle='arc3, rad = 0.1'
    )

    plt.title("Knowledge Graph - Concept Relationships", fontsize=15)
    plt.show()

# %%
print("Preparing entity data for insertion...")
entities = dataset['entities']

data = []
for (k,v) in entities.items():
    data.append({'id': k, 'name': v['name']})
print(f"Prepared {len(data)} entities for insertion.")

query = """
    UNWIND $batch as item
    MERGE (e:Entity {id: item.id})
    SET e.name = item.name
    """

print("Inserting entities into Neo4j database...")
with driver.session() as session:
    session.run(query, batch=data)
    
    print("Counting total nodes in database...")
    result: Result = session.run('MATCH (n) RETURN count(n) AS totalNodes')
    record = result.single()
    
    if record:
        count = record["totalNodes"]
        print(f"Total nodes in database: {count}")
    else:
        print("No nodes found in database.")

# %%


# %%
driver.close()