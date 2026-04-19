import re

from app.knowledge_graph import KnowledgeGraph

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

def validate_query(knowledge_graph: KnowledgeGraph, response: str):
    """
    Checks if the response follows the ontology constraints.
    Checks node labels and relationship labels existence.
    """

    cypher_pattern = re.compile(r"<cypher>(.*?)</cypher>", re.DOTALL | re.IGNORECASE)
    query_match = cypher_pattern.search(response.strip())
    if not query_match:
        return "No <cypher> tags found"

    query = query_match.group(1).strip()
    match = CYPHER_PARSER_REGEX.search(query)
    if not match:
        return (
            "Regex structure mismatch. Ensure the query follows the canonical structure: "
            "MATCH ... [WHERE ...] RETURN ... [ORDER BY ...] [LIMIT ...]"
        )

    match_clause = match.group('match_clause')

    # 1. Check Node Labels
    # Pattern to find labels like :Entity or :Concept inside nodes ( ... )
    node_labels_in_query = re.findall(r":(\w+)(?=[^\]]*\))", match_clause)
    node_labels = knowledge_graph.get_node_labels()
    for label in node_labels_in_query:
        if label not in node_labels:
            return (
                f"Unknown node label: :{label}. "
                f"Valid labels are: {node_labels}"
            )

    # 2. Check Relationship Types
    # Pattern to find relationship types like :BORN_IN inside [ ... ]
    relation_labels_in_query = re.findall(r"\[:(\w+)\]", match_clause)
    relation_labels = knowledge_graph.get_relation_labels()
    for rel_type in relation_labels_in_query:
        if rel_type not in relation_labels:
            return (
                f"Unknown relationship type: :{rel_type}. "
                f"Valid relationship types are: {relation_labels}"
            )

    return "OK"
