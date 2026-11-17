"""Independent boundary cases for Persian tokenization and leakage protection."""
import unittest
from analyze import ascii_tokens, connected_groups, norm, rouge, tokens


class AuditContracts(unittest.TestCase):
    def test_identical_persian_is_not_zero_overlap(self):
        text='این دوا را هر هشت ساعت مصرف کنید'
        self.assertEqual(rouge(tokens(text),tokens(text)),{'r1_f1':1.0,'r2_f1':1.0,'rl_f1':1.0})
        self.assertEqual(ascii_tokens(text),[])
        self.assertEqual(rouge(ascii_tokens(text),ascii_tokens(text))['r1_f1'],0.0)

    def test_unicode_equivalence_does_not_create_new_groups(self):
        a='كِتاب\u200cهاي ۱۲ و ٣'
        b='کتاب های 12 و 3'
        self.assertEqual(norm(a),norm(b))
        self.assertEqual(tokens(a),tokens(b))

    def test_negation_can_keep_high_overlap(self):
        source='این دوا را هر هشت ساعت مصرف کنید'
        negated='این دوا را هر هشت ساعت مصرف نکنید'
        self.assertGreater(rouge(tokens(source),tokens(negated))['r1_f1'],.85)
        self.assertNotEqual(source,negated)
        # Deliberately no assertion that lexical score means semantic preservation.

    def test_cross_role_chains_stay_in_one_component(self):
        rows=[{'ai':'متن الف','human':'متن ب'},
              {'ai':'متن الف','human':'متن ج'},
              {'ai':'متن ب','human':'متن د'},
              {'ai':'متن مستقل','human':'جواب مستقل'}]
        components=sorted(sorted(v)for v in connected_groups(rows).values())
        self.assertEqual(components,[[0,1,2],[3]])

    def test_empty_cells_do_not_connect_unrelated_records(self):
        rows=[{'ai':'الف','human':''},{'ai':'ب','human':''}]
        self.assertEqual(sorted(map(len,connected_groups(rows).values())),[1,1])

if __name__=='__main__':unittest.main()
