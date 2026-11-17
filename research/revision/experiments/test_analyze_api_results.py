"""Boundary tests use toy values in memory, never written as study responses."""
import unittest
from analyze_api_results import normalize,tokens,rouge,numeric_tokens,bootstrap


class AnalysisContracts(unittest.TestCase):
    def test_persian_identity_and_digit_variants(self):
        self.assertEqual(normalize('كِتاب\u200cهاي ۱۲ و ٣'),'کتاب های 12 و 3')
        a=tokens('این دوا را هر هشت ساعت مصرف کنید')
        self.assertEqual(rouge(a,a),{'rouge1_f1':1.0,'rouge2_f1':1.0,'rougeL_f1':1.0})

    def test_negation_can_have_high_lexical_overlap(self):
        a=tokens('این دوا را هر هشت ساعت مصرف کنید')
        b=tokens('این دوا را هر هشت ساعت مصرف نکنید')
        self.assertGreater(rouge(a,b)['rouge1_f1'],.85)
        # This intentionally demonstrates why no semantic-correctness flag is inferred.

    def test_multiset_and_lcs_are_distinct(self):
        scores=rouge(['a','b','a'],['a','a','b'])
        self.assertEqual(scores['rouge1_f1'],1.0)
        self.assertEqual(scores['rouge2_f1'],.5)
        self.assertAlmostEqual(scores['rougeL_f1'],2/3)

    def test_number_words_are_not_silently_adjudicated(self):
        self.assertNotEqual(numeric_tokens('دو سال'),numeric_tokens('۲ سال'))
        self.assertEqual(numeric_tokens('۲ سال و ٣ بار'),numeric_tokens('2 سال و 3 بار'))

    def test_pairing_removes_between_query_variation(self):
        rows=[]
        for q,base in enumerate([1.,7.,16.]):
            for condition,value in [('LLM0',base),('random4',base),('semantic4',base+.25)]:
                rows.append({'model':'toy','query_id':str(q),'condition':condition,'metric':value})
        spec={'models':['toy'],'expected_queries':3,'contrast_metrics':['metric'],
              'bootstrap':{'seed':17,'replicates':1000}}
        contrasts=bootstrap(rows,spec)
        self.assertEqual(len(contrasts),2)
        for c in contrasts:
            self.assertEqual(c['mean_paired_difference'],.25)
            self.assertEqual(c['percentile_95_ci_low'],.25)
            self.assertEqual(c['percentile_95_ci_high'],.25)


if __name__=='__main__':unittest.main()
