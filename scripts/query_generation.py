from abc import ABC, abstractmethod

from models import Concept, Entity
from knowledge_graph import KnowledgeGraph

database = KnowledgeGraph()

class Hint(ABC):
    @staticmethod
    def generate_hint(entity: Entity, number_of_hops: int) -> 'Hint':
        if number_of_hops < 1:
            return NameHint(entity)
        else:
            return ConceptAndRelationHint(entity, number_of_hops-1)
        
        
    @abstractmethod
    def get_cypher_query(self, index: int) -> str:
        pass

class NameHint(Hint):
    def __init__(self, entity: Entity):
        self.root_entity = entity

    def get_cypher_query(self, index: int):
        return f'(e{index} {{name: "{self.root_entity.name}"}})'
    
    def __str__(self):
        return f'Name: {{name: {self.root_entity.name}}}'

class ConceptAndRelationHint(Hint):
    """the (<C>) that <P> <E> (<QK> is <QV>)"""

    def __init__(self, entity: Entity, number_of_hops: int):
        self.answer_entity = entity
        self.parent_concept = database.get_random_parent_concept(entity)
        self.predicate = database.get_random_relation(entity)
        self.entity = self.predicate.target
        self.hint = Hint.generate_hint(self.entity, number_of_hops) # type: ignore

    def get_cypher_query(self, index:int):
        return f'(e{index})-[:{self.predicate.name}]->{self.hint.get_cypher_query(index+1)}'
    
    def __str__(self):
        result = f'ConceptAndRelation: {{<C>: {self.parent_concept.name}, <P>: {self.predicate.name}, <E>: {self.entity}}}\n'
        result += str(self.hint)
        return result

class Query(ABC):
    def __init__(self, number_of_hops):
        self.target_entity = database.get_random_entity()
        self.number_of_hops = number_of_hops
        self.hint: Hint
        
    def generate_hint(self):
        self.hint = Hint.generate_hint(self.target_entity, self.number_of_hops)

    def get_cypher_query(self):
        return f'MATCH {self.hint.get_cypher_query(0)}'
    
    def __str__(self):
        result = f'target_entity: {self.target_entity}\n'
        if self.hint is not None:
            result += str(self.hint)
        return result

class QueryName(Query):
    """What/Who is <E>?"""
        
    def __init__(self, number_of_hops):
        super().__init__(number_of_hops)
        self.generate_hint()        

if __name__ == "__main__":
    query = QueryName(2)
    print(query)
    print(query.get_cypher_query())