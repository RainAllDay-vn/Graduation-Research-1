from abc import ABC, abstractmethod

from scripts.models import Entity
from scripts.knowledge_graph import KnowledgeGraph

database = KnowledgeGraph()

class Hint(ABC):
    @staticmethod
    def generate_hint(entity: Entity, number_of_hops: int) -> 'Hint':
        if number_of_hops < 1:
            return NameHint(entity)
        else:
            return EntityRelationHint(entity, number_of_hops-1)
        
        
    @abstractmethod
    def get_cypher_query(self, index: int) -> str:
        pass

class NameHint(Hint):
    def __init__(self, entity: Entity):
        self.root_entity = entity

    def get_cypher_query(self, index: int):
        return f'MATCH (e{index} {{name: "{self.root_entity.name}"}})\n'
    
    def __str__(self):
        return f'Name: {{name: {self.root_entity.name}}}'

class EntityRelationHint(Hint):
    """the (<E1>) that <P> <E2> (<QK> is <QV>)"""

    def __init__(self, entity: Entity, number_of_hops: int):
        self.answer_entity = entity
        self.predicate = database.get_random_relation(entity)
        self.hint_entity = self.predicate.target
        self.hint = Hint.generate_hint(self.hint_entity, number_of_hops) # type: ignore

    def get_cypher_query(self, index:int):
        return self.hint.get_cypher_query(index+1) + f'MATCH (e{index})-[:{self.predicate.name}]->(e{index+1})\n'
    
    def __str__(self):
        result = f'ConceptAndRelation: {{<E1>: {self.answer_entity.name}, <P>: {self.predicate.name}, <E2>: {self.hint_entity.name}}}\n'
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
        return f'{self.hint.get_cypher_query(0)}' + 'RETURN e0.name'
    
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
    query = QueryName(3)
    print(query)
    print(query.get_cypher_query())