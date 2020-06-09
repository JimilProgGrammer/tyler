"""
Python script that handles connecting to the locally
hosted MongoDB database on port 27017.
"""
from __future__ import division
from pymongo import MongoClient


class MongoCon(object):
    __db = None
    __conn = None
    db_uri = "mongodb://localhost:27017"

    @classmethod
    def get_connection(cls):
        if cls.__db is None or cls.__conn is None:
            cls.__conn = MongoClient(cls.db_uri)
            cls.__db = cls.__conn.tyler
        return cls.__db
