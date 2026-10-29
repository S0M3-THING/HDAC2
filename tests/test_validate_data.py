import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from validate_data import validate_data


class InputChecks(unittest.TestCase):
    def check_rows(self, rows, header=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.csv'
            with path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(header or ['mean.x', 'race', 'gender', 'age_at_index', 'ajcc_pathologic_stage'])
                writer.writerows(rows)
            return validate_data(path)

    def valid_rows(self):
        return [[0.5, 'example-race', 'example-gender', 50, stage] for stage in ('A', 'B') for _ in range(5)]

    def test_valid_and_alternate_expression_column(self):
        self.assertEqual(self.check_rows(self.valid_rows()), [])
        self.assertEqual(self.check_rows(self.valid_rows(), ['mean2', 'race', 'gender', 'age_at_index', 'ajcc_pathologic_stage']), [])

    def test_missing_schema_and_empty_data(self):
        self.assertIn('Missing columns:', self.check_rows([], ['other'])[0])
        self.assertIn('No data rows', self.check_rows([]))

    def test_invalid_numbers_and_private_values_not_echoed(self):
        rows = self.valid_rows()
        rows[0][0] = 'sensitive-value'
        rows[1][3] = 'inf'
        issues = self.check_rows(rows)
        self.assertEqual(sum('finite number' in issue for issue in issues), 2)
        self.assertNotIn('sensitive-value', '\n'.join(issues))

    def test_missing_values_and_malformed_row(self):
        rows = self.valid_rows()
        rows[0][1] = ''
        rows[1].append('extra')
        issues = self.check_rows(rows)
        self.assertTrue(any('blank required' in issue for issue in issues))
        self.assertTrue(any('field count' in issue for issue in issues))

    def test_small_class_warning(self):
        self.assertTrue(any('fewer than five' in issue for issue in self.check_rows(self.valid_rows()[:-1])))

    def test_cli_missing_file_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'validate_data.py'), str(Path(directory) / 'absent.csv')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Unable to read CSV', result.stdout)
        self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
