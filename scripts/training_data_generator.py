# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv (3.14.2)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Training Data Generator
# This notebook generates a synthetic dataset for fine-tuning text-to-Cypher models based on 
# a specific knowledge graph schema.
#
# **Core Architecture**:
# 1. **Cypher Query Generator**: Creates sample Cypher queries targeting specific patterns.
# 2. **Natural Language Query Generator**: Uses an LLM (litellm) to generate English equivalents.
# 3. **Dataset Merger**: Combines paired examples into a JSONL format and writes to disk.

# %% [markdown]
# # 1. Initial set up

# %%
import os
import json
import random
from typing import List, Dict, Any, Literal
import litellm
from neo4j import GraphDatabase, Record
from neo4j.exceptions import ServiceUnavailable

from models import Node

# %% [markdown]
# ## Configuration & Mock Data
# Define the configuration for data generation and some mock data to test the queries.

# %%
CONFIG = {
    # Output path for the training data
    "output_file": "dataset/training_data.jsonl",
    
    # LLM configuration (uses LiteLLM). Ensure relevant API keys are set.
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "base_url": "",
    "api_key": "",
    
    # Neo4j Database Configuration
    "neo4j_uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    "neo4j_user": os.environ.get("NEO4J_USER", "neo4j"),
    "neo4j_password": os.environ.get("NEO4J_PASSWORD", "password"),
    
    # Target number of queries to generate per node
    "number_of_queries": 10,
    # Number of natural language variation questions to generate per Cypher query
    "variations_per_query": 2
}

# %% [markdown]
# ## Neo4j Database Connection
# Set up the Neo4j driver and verify the connection to the database.

# %%
class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver: Any = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print(f"Failed to create the driver: {e}")
        
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
            
    def verify_connectivity(self):
        if self.__driver is None:
            print("Driver not initialized.")
            return False
        try:
            self.__driver.verify_connectivity()
            print("Successfully connected to Neo4j database.")
            return True
        except ServiceUnavailable as e:
            print(f"Failed to connect to Neo4j Database. Error: {e}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False

    def query(self, query, parameters=None):
        if self.__driver is None:
            print("Driver not initialized.")
            return None
        try:
            with self.__driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            print(f"Query execution failed. Error: {e}")
            return None

# Initialize and test connection
neo4j_conn = Neo4jConnection(
    uri=CONFIG.get("neo4j_uri", "bolt://localhost:7687"),
    user=CONFIG.get("neo4j_user", "neo4j"),
    pwd=CONFIG.get("neo4j_password", "password")
)

if neo4j_conn.verify_connectivity():
    print("Querying database statistics...")
    nodes_res = neo4j_conn.query("MATCH (n) RETURN count(n) as node_count")
    rels_res = neo4j_conn.query("MATCH ()-[r]->() RETURN count(r) as rel_count")
    
    node_count = nodes_res[0]["node_count"] if nodes_res else 0
    rel_count = rels_res[0]["rel_count"] if rels_res else 0
    
    print(f"Database Statistics: {node_count} nodes, {rel_count} relations.")

# %% [markdown]
# ## Explore Relationship Statistics
# Before generating training data, let's analyze the distribution of `IS_A` relationships between `Entity` nodes and `Concept` nodes. This will give us insights into the connectivity of our knowledge graph by showing the minimum, maximum, and average number of relationships, as well as highlighting the most connected entities and concepts.

# %%
print("Analyzing IS_A relationships between Entities and Concepts...")
query = """
MATCH (e:Entity)
OPTIONAL MATCH (e)-[r:IS_A]->(c:Concept)
RETURN e.name AS EntityName, count(r) AS TotalRelationships
ORDER BY TotalRelationships DESC
"""
results = neo4j_conn.query(query)

if results:
    counts = [row['TotalRelationships'] for row in results]
    min_rels = min(counts)
    max_rels = max(counts)
    avg_rels = sum(counts) / len(counts) if counts else 0
    
    print("Relationship Statistics (All Types):")
    print(f"- min: {min_rels}")
    print(f"- max: {max_rels}")
    print(f"- average: {avg_rels:.2f}")
    
    print("\nTop 5 Entities by Relationship Count:")
    for row in results[:5]:
        print(f"  {row['EntityName']}: {row['TotalRelationships']}")
else:
    print("No data available for relationship statistics.")

query = """
MATCH (c:Concept)
OPTIONAL MATCH (e:Entity)-[r:IS_A]->(c)
RETURN c.name AS ConceptName, count(r) AS TotalRelationships
ORDER BY TotalRelationships DESC
"""
results = neo4j_conn.query(query)

if results:
    counts = [row['TotalRelationships'] for row in results]
    min_rels = min(counts)
    max_rels = max(counts)
    avg_rels = sum(counts) / len(counts) if counts else 0
    
    print("Concept Relationship Statistics (Incoming IS_A):")
    print(f"- min: {min_rels}")
    print(f"- max: {max_rels}")
    print(f"- average: {avg_rels:.2f}")
    
    print("Top 5 Concepts by Incoming Relationship Count:")
    for row in results[:5]:
        print(f"  {row['ConceptName']}: {row['TotalRelationships']}")
else:
    print("No data available for Concept incoming relationship statistics.")

# %% [markdown]
# ## 2. Creating Cypher Generator
# The `QueryGenerator` class is responsible for programmatically generating training pairs of Cypher queries.
# Currently, it initializes with a database connection and provides functionality to randomly select a 
# root `Entity` node from the Neo4j knowledge graph. This root node will serve as the starting point 
# for building complex graph traversals.

# %%
class QueryGenerator():
    def __init__(self, neo4j_conn: Neo4jConnection):
        self.neo4j_conn = neo4j_conn

    def generate_queries(self, include_raw = False) -> dict[str, str]:
        result = {}
        self.add_random_root_entity()
        if include_raw:
            result['raw'] = str(self.root_entity)
        result['query'] = ''
        return result

    def add_random_root_entity(self):
        query = """
        MATCH (e:Entity)
        RETURN e {.*, labels: labels(e)} AS e
        ORDER BY rand()
        LIMIT 1
        """

        result = self.neo4j_conn.query(query)
        if result is None:
            raise LookupError("There isn't any entity in the knowledge graph")
        row = result[0]["e"]
        self.root_entity = Node.from_database_node(row)
