import unittest
import os

from programy.context import ClientContext

from programytest.aiml_tests.client import TestClient


class ThatSraiTestClient(TestClient):

    def __init__(self):
        TestClient.__init__(self)

    def load_configuration(self, arguments):
        super(ThatSraiTestClient, self).load_configuration(arguments)
        self.configuration.client_configuration.configurations[0].configurations[0].files.aiml_files._files=[os.path.dirname(__file__)]


class ThatSraiAIMLTests(unittest.TestCase):

    def setUp(self):
        client = ThatSraiTestClient()
        self._client_context = client.create_client_context("testid")

    def test_that_srai(self):
        """
        response = self._client_context.bot.ask_question(self._client_context, "YES")
        self.assertIsNotNone(response)
        self.assertEqual(response, 'Test pattern: AGREEMENT')

        response = self._client_context.bot.ask_question(self._client_context, "NO")
        self.assertIsNotNone(response)
        self.assertEqual(response, 'Test pattern: DISAGREEMENT')
        """
        response = self._client_context.bot.ask_question(self._client_context, "HI")
        self.assertIsNotNone(response)
        self.assertEqual(response, 'Hello. Do you know explore the website?')

        response = self._client_context.bot.ask_question(self._client_context, "YES")
        self.assertIsNotNone(response)
        self.assertEqual(response, 'The website was created in 2014.')

        """
        response = self._client_context.bot.ask_question(self._client_context, "HI")
        self.assertIsNotNone(response)
        self.assertEqual(response, 'Hello. Do you know explore the website?')

        response = self._client_context.bot.ask_question(self._client_context, "NO")
        self.assertIsNotNone(response)
        self.assertEqual(response, 'Disagreement with that pattern.')
        """