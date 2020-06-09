from elasticsearch import Elasticsearch

class ElasticService:
    '''
    Service class for implementing Elasticsearch
    operations like inserting new records and reading
    all records.

    '''
    es = None
    def __init__(self):
        if self.es is None:
            self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    
    def insert_record(self, insert_doc, index_name="user_symptoms"):
        if not self.es:
            self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
        
        self.es.index(index=index_name, body=insert_doc)
        return True

# Test code to run the insert operations
# if __name__ == "__main__":
#     elasticService = ElasticService()
#     print(elasticService.es)
#     elasticService.insert_record({"username":"Rahul", "filepath":"random.wav","message":"I had a bad day at college today. I am also experiencing mild fever.","symptoms":"mild fever","timestamp":"20/03/2020 22:17:00"})
