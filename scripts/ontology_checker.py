from knowledge_graph import KnowledgeGraph
import re
from typing import Set, Tuple

class OntologyChecker:

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.CYPHER_PARSER_REGEX = re.compile(
            r"(?i)"  # Case-insensitive
            r"(?P<match_type>OPTIONAL\s+MATCH|MATCH)\s+(?P<match_clause>.+?)"
            r"(?:\s+WHERE\s+(?P<where_clause>.+?))?"
            r"\s+RETURN\s+(?P<return_clause>.+?)"
            r"(?:\s+ORDER\s+BY\s+(?P<order_clause>.+?))?"
            r"(?:\s+LIMIT\s+(?P<limit_clause>\d+))?"
            r"\s*;?$",  # Optional semicolon and trailing whitespace
            re.DOTALL | re.MULTILINE
        )
        self.knowledge_graph = knowledge_graph
        self._load_schema()

    def _load_schema(self):
        self.node_labels = self.knowledge_graph.get_node_labels_list()
        self.relation_labels = self.knowledge_graph.get_relation_labels_list()
        self.entity_concept_mappings = self.knowledge_graph.get_entity_concept_mappings()
        self.relation_labels_between_concepts = self.knowledge_graph.get_relation_labels_between_concepts()

    def _normalize_relation(self, rel: str) -> str:
        """
        Normalizes relationship labels for consistent lookup.
        Converts to lowercase and handles specific mappings like 'is_a' -> 'isa'.
        """
        rel = rel.lower().replace(' ', '_')
        if rel == 'is_a':
            return 'isa'
        return rel

    def check_validity(self, response: str) -> str:
        """
        Checks if a Cypher query in the response follows the ontology constraints.
        Checks node labels, relationship types, and concept-to-concept relationship validity.
        """
        if response is None:
            return "Response is None"

        cypher_pattern = re.compile(r"<cypher>(.*?)</cypher>", re.DOTALL | re.IGNORECASE)
        query_match = cypher_pattern.search(response.strip())
        if not query_match:
            return "No <cypher> tags found"
        
        query = query_match.group(1).strip()
        match = self.CYPHER_PARSER_REGEX.search(query)
        if not match:
            return "Regex structure mismatch (canonical order ignored)"

        match_clause = match.group('match_clause')

        # 1. Check Node Labels
        # Pattern to find labels like :Entity or :Concept inside nodes ( ... )
        node_labels_in_query = re.findall(r":(\w+)(?=[^\]]*\))", match_clause)
        for label in node_labels_in_query:
            if label not in self.node_labels:
                return f"Unknown node label: :{label}"

        # 2. Check Relationship Types
        # Pattern to find relationship types like :BORN_IN inside [ ... ]
        rel_types_in_query = re.findall(r"\[:(\w+)\]", match_clause)
        for rel_type in rel_types_in_query:
            if rel_type not in self.relation_labels:
                return f"Unknown relationship type: :{rel_type}"

        # 3. Check Concept-level Relationship Validity
        # Pattern to find triples: (n1)-[:R]->(n2)
        triple_pattern = re.compile(r"\((?P<n1>[^)]+)\)-\[:(?P<rel>\w+)\]->\((?P<n2>[^)]+)\)")
        for triple in triple_pattern.finditer(match_clause):
            n1_str = triple.group('n1')
            rel = triple.group('rel')
            n2_str = triple.group('n2')

            if rel == 'IS_A':
                continue # IS_A is always valid for schema traversal

            src_concepts = self._get_node_concepts(n1_str)
            tgt_concepts = self._get_node_concepts(n2_str)

            if not src_concepts or not tgt_concepts:
                # If we can't determine concepts (e.g. no name property), we might skip or warn
                # For KQA Pro, we usually have names for entities/concepts in the query
                continue

            # Check if relationship is valid between ANY pair of concepts
            is_valid = False
            for c1 in src_concepts:
                for c2 in tgt_concepts:
                    allowed_rels = self.relation_labels_between_concepts.get((c1, c2), [])
                    if rel in allowed_rels:
                        is_valid = True
                        break
                if is_valid: break
            
            if not is_valid:
                return f"Invalid relationship :{rel} between concepts {src_concepts} and {tgt_concepts}"

        return 'OK'

    def _get_node_concepts(self, node_str: str) -> list[str]:
        """Helper to extract concepts from a node string in Cypher."""
        label_match = re.search(r":(\w+)", node_str)
        name_match = re.search(r"name\s*:\s*['\"]([^'\"]+)['\"]", node_str)
        
        label = label_match.group(1) if label_match else None
        name = name_match.group(1) if name_match else None
        
        if label == 'Concept' and name:
            return [name]
        if label == 'Entity' and name:
            return self.entity_concept_mappings.get(name, [])
        return []
