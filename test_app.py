"""Offline tests: no network calls or paid model requests."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rag import Chunk, Match, SemanticIndex, split_markdown
from service import ChatService, RequestBudget, corpus_fingerprint, prompt_messages
from evaluate import retrieval_scores, summary


class FakeClient:
    def __init__(self):
        self.embeddings = self
        self.inputs = []
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.answer))

    def create(self, model, input, encoding_format):
        self.inputs.extend(input)
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[3, 0] if 'refund' in t.lower()
                                                      else [0, 7]) for i, t in enumerate(input)])

    def answer(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Answer [1]'))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


class CoreTests(unittest.TestCase):
    def test_heading_and_code_preserved(self):
        text = '# Notes\n## Refund\nA policy.\n```py\n# code not a heading\n```\n## Roles\nOwners.'
        chunks = split_markdown(text, 'sample.md')
        self.assertEqual(len(chunks), 2)
        self.assertIn('code not a heading', chunks[0].text)

    def test_incremental_cache_survives_restart_and_prunes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notes = root / 'docs'
            notes.mkdir()
            path = notes / 'sample.md'
            path.write_text('# Notes\n## Refund\nA refund.\n## Roles\nOwners.', encoding='utf-8')
            client = FakeClient()
            index = SemanticIndex(client, notes, root / 'index.sqlite3')
            self.assertEqual(index.refresh()['embedded'], 2)
            restart = SemanticIndex(client, notes, root / 'index.sqlite3')
            self.assertEqual(restart.refresh()['embedded'], 0)
            path.write_text('# Notes\n## Refund\nUpdated refund.', encoding='utf-8')
            self.assertEqual(restart.refresh()['embedded'], 1)
            self.assertEqual(len(restart.vectors), 1)
            self.assertEqual(len(restart.search('refund')), 1)
            self.assertEqual(restart.search('unrelated'), [])

    def test_service_keeps_sources_history_and_tokens(self):
        client = FakeClient()
        hit = Match(Chunk('sample.md', 'Refund', 1, 'Refund within 14 days.'), .9)
        index = SimpleNamespace(search=lambda query: [hit])
        answer = ChatService(client, index).ask('Explain that again', [{'role':'user','content':'refund policy'}])
        self.assertEqual(answer.mode, 'grounded')
        self.assertEqual(answer.sources[0]['label'], 1)
        self.assertIn('refund policy', answer.retrieval_query)
        self.assertEqual(answer.prompt_tokens, 10)
        self.assertFalse(client.calls[0]['store'])
        self.assertIn('untrusted', client.calls[0]['messages'][0]['content'])

    def test_no_sources_is_explicit_and_api_error_is_not_fallback(self):
        client = FakeClient()
        index = SimpleNamespace(search=lambda query: [])
        answer = ChatService(client, index).ask('What is Paris?')
        self.assertEqual(answer.mode, 'general knowledge')
        self.assertIn('No relevant sources', client.calls[0]['messages'][-1]['content'])
        def fail(query):
            raise RuntimeError('Unavailable')
        with self.assertRaises(RuntimeError):
            ChatService(client, SimpleNamespace(search=fail)).ask('Any question')

    def test_question_limits_and_budget(self):
        service = ChatService(FakeClient())
        for question in (' ', 'x' * 2001):
            with self.assertRaises(ValueError):
                service.ask(question)
        budget = RequestBudget(limit=1)
        self.assertTrue(budget.reserve())
        self.assertFalse(budget.reserve())

    def test_eval_denominators_and_errors(self):
        sources = [dict(heading='Notes > Refund')]
        self.assertTrue(retrieval_scores(['Refund'], sources)['top1'])
        self.assertFalse(retrieval_scores(['Refunds'], sources)['top4'])
        values = summary([
            dict(expected_headings=['Refund'], retrieval={'top1':True, 'top4':True}),
            dict(expected_headings=['Refund'], error='Timeout'),
            dict(expected_headings=[], retrieval={'fallback':True})])
        self.assertEqual(values['top1'], {'hits':1, 'total':2})
        self.assertEqual(values['fallback'], {'hits':1, 'total':1})
        self.assertEqual(values['errors'], 1)

    def test_fingerprint_and_dataset(self):
        from rag import CONTEXT_DIR
        before = corpus_fingerprint()
        self.assertEqual(before, corpus_fingerprint())
        headings = {c.heading.split(' > ')[-1] for p in CONTEXT_DIR.rglob('*.md')
                    for c in split_markdown(p.read_text(encoding='utf-8'), p.name)}
        cases = json.loads(Path('eval_cases.json').read_text())['cases']
        self.assertEqual(len({c['id'] for c in cases}), len(cases))
        for case in cases:
            self.assertTrue(set(case['expected_headings']).issubset(headings))


class UITests(unittest.TestCase):
    def test_pages_load_without_api_or_access_code(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ, {'OPENAI_API_KEY':'', 'DEMO_ACCESS_CODE':''}):
            app = AppTest.from_file('app.py', default_timeout=15).run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(app.chat_input[0].disabled)
            for page in ['Knowledge base', 'Evaluations']:
                app.sidebar.radio[0].set_value(page).run()
                self.assertEqual(len(app.exception), 0)

    def test_access_code_unlocks_and_clear_is_session_only(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ, {'OPENAI_API_KEY':'not-a-real-api-key', 'DEMO_ACCESS_CODE':'test-pass'}):
            app = AppTest.from_file('app.py', default_timeout=15).run()
            app.text_input[0].set_value('wrong')
            app.button[0].click().run()
            self.assertTrue(app.chat_input[0].disabled)
            app.text_input[0].set_value('test-pass')
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertFalse(app.chat_input[0].disabled)

    def test_chat_submission_displays_answer_and_sources(self):
        from streamlit.testing.v1 import AppTest
        from service import Answer
        fake = SimpleNamespace(ask=lambda *args: Answer('An owner can restore it [1].',
            [dict(label=1, file='atlas.md', heading='Recovery', text='Within 30 days.', score=.8, part=1)],
            'Restore a project?', 'grounded', .1, 'test-model', 10, 5))
        with patch.dict(os.environ, {'OPENAI_API_KEY':'not-a-real-key', 'DEMO_ACCESS_CODE':'test-pass'}):
            app = AppTest.from_file('app.py', default_timeout=15)
            app.session_state['unlocked'] = True
            app.run()
            with patch('service.ChatService', return_value=fake):
                app.chat_input[0].set_value('Restore a project?').run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.chat_message), 2)
            self.assertTrue(any('An owner can restore' in m.value for m in app.markdown))
            clear = next(b for b in app.button if b.label == 'Clear conversation')
            clear.click().run()
            self.assertEqual(len(app.chat_message), 0)


if __name__ == '__main__':
    unittest.main()
