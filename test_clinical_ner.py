'''
Python script to test the Bi-LSTM+CRF
clinical NER model to tag medical text
and identify the symptoms, medicines and
tests taken.

This example script first prints the tagged
response from the model which is essentially
a dictionary, and then parses the dictionary
to identify the symptoms.
'''
from tagging import tag_message

res = tag_message(text_msg="I was having a fever today so I took a Crocin.")
print(res)
symptoms = []
for entity in res['entities']:
    if entity['type'] == "problem":
        symptoms.append(entity['text'])
print("Symptoms identified:")
print(symptoms)