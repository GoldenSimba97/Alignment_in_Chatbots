import unittest
import unittest.mock

from programy.config.bot.redisstorage import BotConversationsRedisStorageConfiguration
from programy.dialog.storage.redis import ConversationRedisStorage
from programytest.aiml_tests.client import TestClient
from programy.dialog.dialog import Conversation

class MockRedisStorage(object):

    def __init__(self, config):
        self.__config = config
        self._properties = None

    def delete(self, key):
        pass

    def save(self, h_key,s_key, clientid, properties):
        self._properties = properties

    def is_member(self, s_key, clientid):
        return True

    def get(self, h_key):
        return self._properties

    def remove(self, s_key, clientid):
        self._properties = None


class MockRedisFactory(object):

    @staticmethod
    def connect(config):
        return MockRedisStorage(config)


class ConversationRedisStorageTests(unittest.TestCase):

    def test_persistence(self):

        storage_config = BotConversationsRedisStorageConfiguration("redis")
        redis = ConversationRedisStorage(storage_config, factory=MockRedisFactory())
        self.assertIsNotNone(redis)

        client = TestClient()
        client_context = client.create_client_context("testid")
        conversation1 = Conversation(client_context)
        conversation1.set_property("topic", "topic1")

        redis.save_conversation(conversation1, client_context.userid)

        conversation2 = Conversation(client_context)
        redis.load_conversation(conversation2, client_context.userid)
        self.assertIsNotNone(conversation2)
        self.assertIsNotNone(conversation2.properties)

        self.assertEquals(conversation1.properties, conversation2.properties)

