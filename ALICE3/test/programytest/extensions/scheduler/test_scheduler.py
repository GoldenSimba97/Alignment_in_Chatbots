import unittest

from programy.dialog.dialog import Question
from programy.extensions.scheduler.scheduler import SchedulerExtension

from programytest.aiml_tests.client import TestClient


class SchedulerExtensionClient(TestClient):

    def __init__(self, mock_scheduler=None):
        self._mock_scheduler = mock_scheduler
        TestClient.__init__(self)

    def load_configuration(self, arguments):
        super(SchedulerExtensionClient, self).load_configuration(arguments)

    def load_scheduler(self):
        if self._mock_scheduler  is not None:
            self._scheduler = self._mock_scheduler
        else:
            super(SchedulerExtensionClient, self).load_scheduler()


class SchedulerExtensionTests(unittest.TestCase):

    # REDMIND IN|EVERY X SECS|MINS|HOURS|DAYS|WEEKS MESSAGE ...........

    def test_remind_in_n_seconds(self):
        client = SchedulerExtensionClient()
        client_context = client.create_client_context("testid")
        extension = SchedulerExtension()
        response = extension.execute(client_context, "REMIND IN 10 SECONDS MESSAGE WAKEY WAKEY")
        self.assertEquals("OK", response)

    def test_remind_in_n_minutes(self):
        client = SchedulerExtensionClient()
        client_context = client.create_client_context("testid")
        extension = SchedulerExtension()
        response = extension.execute(client_context, "REMIND IN 10 MINUTES MESSAGE WAKEY WAKEY")
        self.assertEquals("OK", response)

    def test_remind_in_n_hours(self):
        client = SchedulerExtensionClient()
        client_context = client.create_client_context("testid")
        extension = SchedulerExtension()
        response = extension.execute(client_context, "REMIND IN 10 HOURS MESSAGE WAKEY WAKEY")
        self.assertEquals("OK", response)

    def test_remind_in_n_days(self):
        client = SchedulerExtensionClient()
        client_context = client.create_client_context("testid")
        extension = SchedulerExtension()
        response = extension.execute(client_context, "REMIND IN 10 DAYS MESSAGE WAKEY WAKEY")
        self.assertEquals("OK", response)

    def test_remind_in_n_weeks(self):
        client = SchedulerExtensionClient()
        client_context = client.create_client_context("testid")
        extension = SchedulerExtension()
        response = extension.execute(client_context, "REMIND IN 10 WEEKS MESSAGE WAKEY WAKEY")
        self.assertEquals("OK", response)

    def test_remind_every_n_seconds(self):
        client = SchedulerExtensionClient()
        client_context = client.create_client_context("testid")
        extension = SchedulerExtension()
        response = extension.execute(client_context, "REMIND EVERY 10 SECONDS MESSAGE WAKEY WAKEY")
        self.assertEquals("OK", response)

    def test_remind_every_n_minutes(self):
        client = SchedulerExtensionClient()
        client_context = client.create_client_context("testid")
        extension = SchedulerExtension()
        response = extension.execute(client_context, "REMIND EVERY 10 MINUTES MESSAGE WAKEY WAKEY")
        self.assertEquals("OK", response)

    def test_remind_every_n_hours(self):
        client = SchedulerExtensionClient()
        client_context = client.create_client_context("testid")
        extension = SchedulerExtension()
        response = extension.execute(client_context, "REMIND EVERY 10 HOURS MESSAGE WAKEY WAKEY")
        self.assertEquals("OK", response)

    def test_remind_every_n_days(self):
        client = SchedulerExtensionClient()
        client_context = client.create_client_context("testid")
        extension = SchedulerExtension()
        response = extension.execute(client_context, "REMIND EVERY 10 DAYS MESSAGE WAKEY WAKEY")
        self.assertEquals("OK", response)

    def test_remind_every_n_weeks(self):
        client = SchedulerExtensionClient()
        client_context = client.create_client_context("testid")
        extension = SchedulerExtension()
        response = extension.execute(client_context, "REMIND EVERY 10 WEEKS MESSAGE WAKEY WAKEY")
        self.assertEquals("OK", response)

