"""Offline boundary checks. Synthetic responses below are unit fixtures only.

No credential is read, no HTTP request is made, and no fixture enters results.
"""
from pathlib import Path
import copy
import csv
import hashlib
import json
import re
import sys
import tempfile
import unittest

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
from api_experiment import MODELS, parse_success
from harness import load_corpus


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ProviderResponseContracts(unittest.TestCase):
    def setUp(self):
        self.job = {'model': MODELS[0], 'payload': {'max_completion_tokens': 256}}
        self.synthetic = {
            'id': 'SYNTHETIC-UNIT-FIXTURE-NOT-A-PROVIDER-RESPONSE',
            'model': MODELS[0],
            'choices': [{'finish_reason': 'stop', 'message': {'content': 'متن روشن است.'}}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110},
        }

    def parse(self, record):
        return parse_success(json.dumps(record).encode(), self.job)

    def test_accepts_valid_fixture_and_computes_declared_usage_cost(self):
        result = self.parse(self.synthetic)
        self.assertEqual(result['output'], 'متن روشن است.')
        self.assertEqual(result['uncached_cost_usd'], '0.000056')

    def test_rejects_other_model_even_with_good_text(self):
        self.synthetic['model'] = MODELS[1]
        with self.assertRaisesRegex(ValueError, 'another model snapshot'):
            self.parse(self.synthetic)

    def test_rejects_token_limited_completion(self):
        self.synthetic['choices'][0]['finish_reason'] = 'length'
        with self.assertRaisesRegex(ValueError, 'Non-normal finish'):
            self.parse(self.synthetic)

    def test_rejects_empty_text(self):
        self.synthetic['choices'][0]['message']['content'] = ' \n\t'
        with self.assertRaisesRegex(ValueError, 'Empty output'):
            self.parse(self.synthetic)

    def test_rejects_inconsistent_usage(self):
        self.synthetic['usage']['total_tokens'] = 109
        with self.assertRaisesRegex(ValueError, 'Usage inconsistent'):
            self.parse(self.synthetic)

    def test_rejects_completion_exceeding_cap(self):
        self.synthetic['usage'].update(completion_tokens=257, total_tokens=357)
        with self.assertRaisesRegex(ValueError, 'Completion token cap exceeded'):
            self.parse(self.synthetic)


class FrozenStudyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = EXPERIMENTS/'api_study_v1'
        cls.plan = json.loads((cls.directory/'plan.json').read_text())
        manifest_path = Path(cls.plan['provenance']['manifest_path'])
        if not manifest_path.is_file():
            manifest_path = EXPERIMENTS.parent/'corpus_analysis/split_manifest.json'
        cls.manifest = json.loads(manifest_path.read_text())
        cls.source_path = (manifest_path.parent/cls.manifest['dataset_path']).resolve()
        cls.records_path = (manifest_path.parent/cls.manifest['records_csv']).resolve()

    def test_plan_and_executing_scripts_match_frozen_hashes(self):
        self.assertEqual(file_sha(self.directory/'plan.json'),
                         (self.directory/'plan.sha256').read_text().strip())
        self.assertEqual(file_sha(EXPERIMENTS/'api_experiment.py'), self.plan['script_sha256'])
        self.assertEqual(file_sha(EXPERIMENTS/'harness.py'), self.plan['harness_sha256'])

    def test_same_100_distinct_groups_and_queries_in_all_eight_cells(self):
        self.assertEqual(self.plan['query_count'], 100)
        self.assertEqual(len(set(self.plan['query_group_ids'])), 100)
        self.assertEqual(len(self.plan['jobs']), 800)
        expected = set(self.plan['query_ids'])
        for model in self.plan['models']:
            for condition in self.plan['conditions']:
                jobs = [j for j in self.plan['jobs'] if j['model'] == model and j['condition'] == condition]
                self.assertEqual(len(jobs), 100)
                self.assertEqual({j['query_id'] for j in jobs}, expected)

    def test_selected_inputs_are_test_only_and_demonstrations_train_only(self):
        train = set(self.manifest['splits']['train'])
        test = set(self.manifest['splits']['test'])
        for job in self.plan['jobs']:
            self.assertIn(job['query_id'], test)
            self.assertTrue(set(job['retrieved_ids']) <= train)
            self.assertNotIn(job['query_id'], job['retrieved_ids'])

    def test_zero_shot_has_no_hidden_demonstration_and_exact_depths_are_visible(self):
        depths = {'LLM0': 0, 'semantic1': 1, 'semantic4': 4, 'random4': 4}
        for job in self.plan['jobs']:
            expected = depths[job['condition']]
            prompt = job['payload']['messages'][0]['content']
            self.assertEqual(len(job['retrieved_ids']), expected)
            self.assertEqual(len(re.findall(r'^ورودی نمونه [1-4]:', prompt, re.MULTILINE)), expected)
            self.assertEqual(len(re.findall(r'^بازنویسی نمونه [1-4]:', prompt, re.MULTILINE)), expected)
            self.assertTrue(prompt.endswith(job['input']))
            if expected == 0:
                self.assertIn('(بدون مثال)', prompt)
                self.assertNotIn('متن اصلی:', prompt)
                self.assertNotIn('متن بازنویسی‌شده:', prompt)

    def test_models_receive_identical_prompt_for_same_query_condition(self):
        prompts = {}
        for job in self.plan['jobs']:
            key = (job['query_id'], job['condition'])
            prompt = job['payload']['messages'][0]['content']
            if key in prompts:
                self.assertEqual(prompts[key], prompt)
            prompts[key] = prompt
            self.assertEqual(job['payload']['temperature'], 0)
            self.assertEqual(job['payload']['seed'], 0)
            self.assertEqual(job['payload']['max_completion_tokens'], 256)

    def test_bad_raw_dataset_hash_is_rejected_before_api_work(self):
        changed = copy.deepcopy(self.manifest)
        changed['dataset_path'] = str(self.source_path)
        changed['records_csv'] = str(self.records_path)
        changed['dataset_sha256'] = '0'*64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad_manifest.json'
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'Source dataset SHA mismatch'):
                load_corpus(path)

    def test_cross_role_text_leakage_is_rejected(self):
        # Deliberately synthetic three-record corpus; never saved into the study.
        rows = [
            {'stable_id': 'UNIT-A', 'group_id': 'GROUP-A', 'ai': 'متن الف', 'human': 'متن مشترک'},
            {'stable_id': 'UNIT-B', 'group_id': 'GROUP-B', 'ai': 'متن ب', 'human': 'جواب ب'},
            {'stable_id': 'UNIT-C', 'group_id': 'GROUP-C', 'ai': 'متن مشترک', 'human': 'جواب ج'},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, fields in [('source.csv', ['ai', 'human']), ('records.csv', list(rows[0]))]:
                with (root/name).open('w', encoding='utf-8', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows([{k:r[k] for k in fields} for r in rows])
            manifest = {
                'dataset_path': 'source.csv', 'dataset_sha256': file_sha(root/'source.csv'),
                'records_csv': 'records.csv', 'records_sha256': file_sha(root/'records.csv'),
                'splits': {'train': ['UNIT-A'], 'dev': ['UNIT-B'], 'test': ['UNIT-C']},
            }
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'overlap across splits'):
                load_corpus(root/'manifest.json')


if __name__ == '__main__':
    unittest.main()
