import unittest.mock

from programytest.parser.base import ParserTestsBaseClass

from programy.parser.pattern.nodes.base import PatternNode
from programy.parser.pattern.nodes.word import PatternWordNode
from programy.parser.pattern.nodes.priority import PatternPriorityWordNode
from programy.parser.pattern.nodes.template import PatternTemplateNode
from programy.parser.pattern.nodes.topic import PatternTopicNode
from programy.parser.pattern.nodes.that import PatternThatNode
from programy.parser.pattern.nodes.oneormore import PatternOneOrMoreWildCardNode
from programy.parser.pattern.nodes.zeroormore import PatternZeroOrMoreWildCardNode
from programy.parser.pattern.nodes.set import PatternSetNode
from programy.parser.pattern.nodes.iset import PatternISetNode
from programy.parser.pattern.nodes.bot import PatternBotNode
from programy.parser.pattern.nodes.regex import PatternRegexNode


class PatternBotNodeTests(ParserTestsBaseClass):

    def test_init(self):

        node = PatternNode()
        self.assertIsNotNone(node)

        self.assertFalse(node.is_root())

        self.assertEquals(0, len(node.priority_words))
        self.assertFalse(node.has_priority_words())
        self.assertFalse(node.is_priority())

        self.assertIsNone(node.arrow)
        self.assertIsNone(node.hash)
        self.assertIsNone(node.underline)
        self.assertIsNone(node.star)

        self.assertFalse(node.has_wildcard())
        self.assertFalse(node.has_zero_or_more())
        self.assertFalse(node.has_0ormore_arrow())
        self.assertFalse(node.has_0ormore_hash())
        self.assertFalse(node.has_one_or_more())
        self.assertFalse(node.has_1ormore_star())
        self.assertFalse(node.has_1ormore_underline())
        self.assertFalse(node.is_wildcard())
        self.assertFalse(node.is_zero_or_more())
        self.assertFalse(node.is_one_or_more())

        self.assertEquals(0, len(node.children))
        self.assertFalse(node.has_children())
        self.assertFalse(node.has_nodes())

        self.assertIsNone(node.topic)
        self.assertFalse(node.is_topic())
        self.assertFalse(node.has_topic())

        self.assertIsNone(node.that)
        self.assertFalse(node.is_that())
        self.assertFalse(node.has_that())

        self.assertIsNone(node.template)
        self.assertFalse(node.is_template())
        self.assertFalse(node.has_template())

        self.assertFalse(node.is_set())

        self.assertFalse(node.is_bot())

        self.assertEquals("P(0)^(0)#(0)C(0)_(0)*(0)To(0)Th(0)Te(0)", node._child_count(verbose=True))
        self.assertEquals("", node._child_count(verbose=False))

        self.assertEquals("NODE [P(0)^(0)#(0)C(0)_(0)*(0)To(0)Th(0)Te(0)]", node.to_string(verbose=True))
        self.assertEquals("NODE", node.to_string(verbose=False))

        self.assertEquals("", node.to_xml(self._client_context))
        self.assertEquals("", node.to_xml(self._client_context))

    def test_get_tabs(self):
        node = PatternNode()
        self.assertIsNotNone(node)

        self._client_context.bot.configuration._tab_parse_output = True

        self.assertEquals("", node.get_tabs(self._client_context, 0))
        self.assertEquals("  ", node.get_tabs(self._client_context, 1))
        self.assertEquals("          ", node.get_tabs(self._client_context, 5))

        self._client_context.bot.configuration._tab_parse_output = False

        self.assertEquals("", node.get_tabs(self._client_context, 0))
        self.assertEquals("", node.get_tabs(self._client_context, 1))
        self.assertEquals("", node.get_tabs(self._client_context, 5))

    def test_equals_ignore_case(self):

        node = PatternNode()
        self.assertIsNotNone(node)

        self.assertTrue(node.equals_ignore_case(self._client_context, "", ""))

        self.assertTrue(node.equals_ignore_case(self._client_context, "test", "test"))
        self.assertTrue(node.equals_ignore_case(self._client_context, "Test", "test"))
        self.assertTrue(node.equals_ignore_case(self._client_context, "test", "Test"))
        self.assertTrue(node.equals_ignore_case(self._client_context, "TEST", "test"))
        self.assertTrue(node.equals_ignore_case(self._client_context, "test", "TEST"))

        self.assertFalse(node.equals_ignore_case(self._client_context, "test", "TESTX"))
        self.assertFalse(node.equals_ignore_case(self._client_context, "testX", "TEST"))

        self.assertFalse(node.equals_ignore_case(self._client_context, None, "TEST"))
        self.assertFalse(node.equals_ignore_case(self._client_context, "testX", None))
        self.assertFalse(node.equals_ignore_case(self._client_context, None, None))

    def test_nodes(self):

        node = PatternNode()
        self.assertIsNotNone(node)
        self.assertFalse(node.has_nodes())

        node = PatternNode()
        node._children.append(PatternNode())
        self.assertTrue(node.has_nodes())

    def test_word_nodes(self):
        node = PatternNode()
        self.assertIsNotNone(node)
        self.assertFalse(node.has_nodes())

        node = PatternNode()
        node.add_child(PatternWordNode("test"))
        self.assertTrue(node.has_nodes())
        self.assertTrue(node.has_children())

    def test_priority_nodes(self):
        node = PatternNode()
        self.assertIsNotNone(node)
        self.assertFalse(node.has_nodes())
        self.assertFalse(node.has_priority_words())

        node = PatternNode()
        node.add_child(PatternPriorityWordNode("pTest"))
        self.assertTrue(node.has_nodes())
        self.assertTrue(node.has_priority_words())

    def test_wildcard(self):

        node = PatternNode()
        self.assertIsNotNone(node)

        self.assertFalse(node.has_wildcard())
        self.assertFalse(node.has_zero_or_more())
        self.assertFalse(node.has_one_or_more())
        self.assertFalse(node.has_nodes())

        node = PatternNode()
        node._0ormore_arrow = PatternZeroOrMoreWildCardNode("^")
        self.assertTrue(node.has_wildcard())
        self.assertTrue(node.has_zero_or_more())
        self.assertFalse(node.has_one_or_more())
        self.assertTrue(node.has_nodes())

        node = PatternNode()
        node._0ormore_hash = PatternZeroOrMoreWildCardNode("#")
        self.assertTrue(node.has_wildcard())
        self.assertTrue(node.has_zero_or_more())
        self.assertFalse(node.has_one_or_more())
        self.assertTrue(node.has_nodes())

        node = PatternNode()
        node._1ormore_star = PatternOneOrMoreWildCardNode("*")
        self.assertTrue(node.has_wildcard())
        self.assertFalse(node.has_zero_or_more())
        self.assertTrue(node.has_one_or_more())
        self.assertTrue(node.has_nodes())

        node = PatternNode()
        node._1ormore_underline = PatternOneOrMoreWildCardNode("_")
        self.assertTrue(node.has_wildcard())
        self.assertFalse(node.has_zero_or_more())
        self.assertTrue(node.has_one_or_more())
        self.assertTrue(node.has_nodes())

    def test_topic_nodes(self):

        node = PatternNode()
        self.assertIsNotNone(node)
        self.assertFalse(node.has_nodes())
        self.assertFalse(node.has_topic())

        node = PatternNode()
        node.add_topic(PatternTopicNode())
        self.assertFalse(node.has_nodes())
        self.assertTrue(node.has_topic())

    def test_that_nodes(self):

        node = PatternNode()
        self.assertIsNotNone(node)
        self.assertFalse(node.has_nodes())
        self.assertFalse(node.has_that())

        node = PatternNode()
        node.add_that(PatternThatNode())
        self.assertFalse(node.has_nodes())
        self.assertTrue(node.has_that())

    def test_template_nodes(self):

        node = PatternNode()
        self.assertIsNotNone(node)
        self.assertFalse(node.has_nodes())
        self.assertFalse(node.has_template())

        node = PatternNode()
        node.add_template(PatternTemplateNode(None))
        self.assertFalse(node.has_nodes())
        self.assertTrue(node.has_template())

    def test_node_exists(self):

        node = PatternNode()
        self.assertIsNotNone(node)

        self.assert_child_node_exists(node, PatternWordNode("word"), PatternWordNode("word"))
        self.assert_child_node_exists(node, PatternPriorityWordNode("priority"), PatternPriorityWordNode("priority"))

        self.assert_child_node_exists(node, PatternOneOrMoreWildCardNode('_'), PatternOneOrMoreWildCardNode('_'))
        self.assert_child_node_exists(node, PatternOneOrMoreWildCardNode('*'), PatternOneOrMoreWildCardNode('*'))

        self.assert_child_node_exists(node, PatternZeroOrMoreWildCardNode('#'), PatternZeroOrMoreWildCardNode('#'))
        self.assert_child_node_exists(node, PatternZeroOrMoreWildCardNode('^'), PatternZeroOrMoreWildCardNode('^'))

        self.assert_child_node_exists(node, PatternSetNode({}, "setname"), PatternSetNode([], "setname"))
        self.assert_child_node_exists(node, PatternBotNode({}, "botname"), PatternBotNode([], "botname"))
        self.assert_child_node_exists(node, PatternISetNode({}, "word1 word2"), PatternISetNode([], "word1 word2"))

        self.assert_child_node_exists(node, PatternRegexNode({"pattern": "^LEGION$"}, None), PatternRegexNode({"pattern": "^LEGION$"},  None))
        self.assert_child_node_exists(node, PatternRegexNode({"template": "LEGION"}, None), PatternRegexNode({"template": "LEGION"},  None))

        topic1 = PatternTopicNode()
        topic2 = PatternTopicNode()
        self.assertIsNone(node._node_exists(topic1))
        node.add_topic(topic1)
        new_node = node._node_exists(topic1)
        self.assertIsNotNone(new_node)
        self.assertEquals(new_node, topic1)
        new_node = node.add_topic(topic2)
        self.assertIsNotNone(new_node)
        self.assertEquals(new_node, topic1)

        that1 = PatternThatNode()
        that2 = PatternThatNode()
        self.assertIsNone(node._node_exists(that1))
        node.add_that(that1)
        new_node = node._node_exists(that1)
        self.assertIsNotNone(new_node)
        self.assertEquals(new_node, that1)
        new_node = node.add_that(that2)
        self.assertIsNotNone(new_node)
        self.assertEquals(new_node, that1)

        template1 = PatternTemplateNode(None)
        template2 = PatternTemplateNode(None)
        self.assertIsNone(node._node_exists(template1))
        node.add_template(template1)
        new_node = node._node_exists(template1)
        self.assertIsNotNone(new_node)
        self.assertEquals(new_node, template1)
        new_node = node.add_template(template2)
        self.assertIsNotNone(new_node)
        self.assertEquals(new_node, template1)

        node.dump("")

        self.assertEqual(node.to_xml(self._client_context), """<priority word="priority"></priority>
<zerormore wildcard="^">
</zerormore>
<zerormore wildcard="#">
</zerormore>
<oneormore wildcard="_">
</oneormore>
<oneormore wildcard="*">
</oneormore>
<topic></topic>
<that></that>
<template></template>
<word word="word"></word>
<set name="SETNAME">
</set><bot property="botname">
</bot><iset words="WORD1 WORD2"></iset>
<regex pattern="^LEGION$"></regex>
<regex template="LEGION"></regex>
""")

    def assert_child_node_exists(self, base_node, first_node, second_node, child_equal=True):
        self.assertIsNone(base_node._node_exists(first_node))
        base_node.add_child(first_node)
        if child_equal is True:
            new_node = base_node._node_exists(first_node)
            self.assertIsNotNone(new_node)
            self.assertEquals(new_node, first_node)
            new_node = base_node._node_exists(second_node)
            self.assertIsNotNone(new_node)
            self.assertEquals(new_node, first_node)

