'''
Simple python script that uses Elasticsearch
python package to generate a connection to the 
locally running ES cluster and sets up the
user_symptoms index.

The user_symptoms index is structured as follows:
{
    "username": <type: text>,
    "filepath": <type: text>,
    "message":  <type: text>,
    "symptoms": <type: text>,
    "timestamp":<type: text>
}
'''
from elasticsearch import Elasticsearch

def create_index(es_object, index_name='user_symptoms'):
    created = False
    # index settings
    settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties" : {
                "username": {
                    "type": "text"
                },
                "filepath": {
                    "type": "text"
                },
                "message": {
                    "type": "text"
                },
                "symptoms": {
                    "type": "text"
                },
                "timestamp": {
                    "type": "text"
                }
            }
        }
    }
    print(es_object)

    try:
        if not es_object.indices.exists(index_name):
            # Ignore 400 means to ignore "Index Already Exist" error.
            print(es_object.indices.create(index=index_name, ignore=400, body=settings))
            print('Created Index')
            created = True
    except Exception as ex:
        print(str(ex))
    finally:
        return created

if __name__ == '__main__':
    # create instance of elasticsearch
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

    if es.ping():
        print('IN main(): Connected to ES cluster.')
    else:
        print('IN main(): Could not connect!')

    # create sentiment index in ES
    create_index(es)
