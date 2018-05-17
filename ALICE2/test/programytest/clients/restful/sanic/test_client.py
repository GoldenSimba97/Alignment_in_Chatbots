import unittest
import unittest.mock
import os

from programytest.clients.arguments import MockArgumentParser

if os.name != "nt":
    from programy.clients.restful.sanic.client import SanicRestBotClient

    class MockSanicRestBotClient(SanicRestBotClient):

        def __init__(self, argument_parser=None):
            SanicRestBotClient.__init__(self, "sanic", argument_parser)
            self.aborted = False
            self.answer = None
            self.ask_question_exception = False

        def server_abort(self, message, status_code):
            self.aborted = True
            raise Exception("Pretending to abort!")

        def ask_question(self, userid, question):
            if self.ask_question_exception is True:
                raise Exception("Something bad happened")
            return self.answer

    class SanicBotClientTests(unittest.TestCase):

        def test_rest_client_init(self):
            arguments = MockArgumentParser()
            client = SanicRestBotClient("sanic", arguments)
            self.assertIsNotNone(client)

        def test_verify_api_key_usage_inactive(self):
            arguments = MockArgumentParser()
            client = SanicRestBotClient("sanic", arguments)
            self.assertIsNotNone(client)
            client.configuration.client_configuration._use_api_keys = False
            request = unittest.mock.Mock()
            self.assertEquals((None, None),client.verify_api_key_usage(request))

        def test_get_api_key(self):
            arguments = MockArgumentParser()
            client = SanicRestBotClient("sanic", arguments)

            request = unittest.mock.Mock()
            request.raw_args = {}
            request.raw_args['apikey'] = '11111111'

            self.assertEquals('11111111', client.get_api_key(request))

        def test_verify_api_key_usage_active(self):
            arguments = MockArgumentParser()
            client = SanicRestBotClient("sanic", arguments)
            self.assertIsNotNone(client)
            client.configuration.client_configuration._use_api_keys = True
            client.configuration.client_configuration._api_key_file = os.path.dirname(__file__) + os.sep + ".." + os.sep + ".." + os.sep + "api_keys.txt"
            client.load_api_keys()
            request = unittest.mock.Mock()
            request.raw_args = {}
            request.raw_args['apikey'] = '11111111'
            self.assertEquals((None, None),client.verify_api_key_usage(request))

        def test_verify_api_key_usage_active_no_apikey(self):
            arguments = MockArgumentParser()
            client = MockSanicRestBotClient(arguments)
            client.configuration.client_configuration._use_api_keys = True

            request = unittest.mock.Mock()
            request.raw_args = {}

            response = client.verify_api_key_usage(request)
            self.assertIsNotNone(response)

        def test_verify_api_key_usage_active_invalid_apikey(self):
            arguments = MockArgumentParser()
            client = MockSanicRestBotClient(arguments)
            client.configuration.client_configuration._use_api_keys = True

            request = unittest.mock.Mock()
            request.raw_args = {}
            request.raw_args['apikey'] = 'invalid'

            response = client.verify_api_key_usage(request)
            self.assertIsNotNone(response)

        def test_get_question(self):
            arguments = MockArgumentParser()
            client = SanicRestBotClient("sanic", arguments)
            self.assertIsNotNone(client)

            request = unittest.mock.Mock()
            request.raw_args = {}
            request.raw_args['question'] = 'Hello'

            self.assertEquals("Hello", client.get_question(request))

        def test_get_question_no_question(self):
            arguments = MockArgumentParser()
            client = MockSanicRestBotClient(arguments)
            self.assertIsNotNone(client)

            request = unittest.mock.Mock()
            request.raw_args = {}
            request.raw_args['question'] = None

            self.assertFalse(client.aborted)
            with self.assertRaises(Exception):
                self.assertEquals("Hello", client.get_question(request))
            self.assertTrue(client.aborted)

        def test_get_question_none_question(self):
            arguments = MockArgumentParser()
            client = MockSanicRestBotClient(arguments)
            self.assertIsNotNone(client)

            request = unittest.mock.Mock()
            request.raw_args = {}

            self.assertFalse(client.aborted)
            with self.assertRaises(Exception):
                self.assertEquals("Hello", client.get_question(request))
            self.assertTrue(client.aborted)

        def test_get_userid(self):
            arguments = MockArgumentParser()
            client = SanicRestBotClient("sanic", arguments)
            self.assertIsNotNone(client)

            request = unittest.mock.Mock()
            request.raw_args = {}
            request.raw_args['userid'] = '1234567890'

            self.assertEquals("1234567890", client.get_userid(request))

        def test_get_userid_no_userid(self):
            arguments = MockArgumentParser()
            client = MockSanicRestBotClient(arguments)
            self.assertIsNotNone(client)

            request = unittest.mock.Mock()
            request.raw_args = {}

            self.assertFalse(client.aborted)
            with self.assertRaises(Exception):
                self.assertEquals("1234567890", client.get_userid(request))
            self.assertTrue(client.aborted)

        def test_get_userid_none_userid(self):
            arguments = MockArgumentParser()
            client = MockSanicRestBotClient(arguments)
            self.assertIsNotNone(client)

            request = unittest.mock.Mock()
            request.raw_args = {}
            request.raw_args['userid'] = None

            self.assertFalse(client.aborted)
            with self.assertRaises(Exception):
                self.assertEquals("1234567890", client.get_userid(request))
            self.assertTrue(client.aborted)

        def test_format_success_response(self):
            arguments = MockArgumentParser()
            client = SanicRestBotClient("sanic", arguments)
            self.assertIsNotNone(client)

            response = client.format_success_response("1234567890", "Hello", "Hi")
            self.assertIsNotNone(response)
            self.assertEquals("1234567890", response['userid'])
            self.assertEquals("Hello", response['question'])
            self.assertEquals("Hi", response['answer'])

        def test_format_error_response(self):
            arguments = MockArgumentParser()
            client = SanicRestBotClient("sanic", arguments)
            self.assertIsNotNone(client)

            response = client.format_error_response("1234567890", "Hello", "Something Bad")
            self.assertIsNotNone(response)
            self.assertEquals("1234567890", response['userid'])
            self.assertEquals("Hello", response['question'])
            self.assertEquals("", response['answer'])
            self.assertEquals("Something Bad", response['error'])

        def test_process_request(self):
            arguments = MockArgumentParser()
            client = MockSanicRestBotClient(arguments)
            self.assertIsNotNone(client)
            client.configuration.client_configuration._use_api_keys = False

            request = unittest.mock.Mock()
            request.raw_args = {}
            request.raw_args['question'] = 'Hello'
            request.raw_args['userid'] = '1234567890'

            client.answer = "Hi"

            response, status = client.process_request(request)
            self.assertIsNotNone(response)
            self.assertEquals("1234567890", response['userid'])
            self.assertEquals("Hello", response['question'])
            self.assertEquals("Hi", response['answer'])

        def test_process_request_no_api_key(self):
            arguments = MockArgumentParser()
            client = MockSanicRestBotClient(arguments)
            self.assertIsNotNone(client)
            client.configuration.client_configuration._use_api_keys = True

            request = unittest.mock.Mock()
            request.raw_args = {}
            request.raw_args['question'] = 'Hello'
            request.raw_args['userid'] = '1234567890'

            client.answer = "Hi"

            response, status = client.process_request(request)
            self.assertIsNotNone(response)
            self.assertEquals(status, 401)

        def test_process_request_exception(self):
            arguments = MockArgumentParser()
            client = MockSanicRestBotClient(arguments)
            self.assertIsNotNone(client)
            client.configuration.client_configuration._use_api_keys = False

            request = unittest.mock.Mock()
            request.raw_args = {}
            request.raw_args['question'] = 'Hello'
            request.raw_args['userid'] = '1234567890'

            client.answer = "Hi"
            client.ask_question_exception = True

            response, status = client.process_request(request)
            self.assertIsNotNone(response)
            self.assertEquals(status, 500)
