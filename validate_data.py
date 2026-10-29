"""Check input CSV structure without loading the training dependencies."""
import argparse
import csv
import math
from collections import Counter


def validate_data(path):
    """Return diagnostic messages; never include patient values in diagnostics."""
    issues = []
    stages = Counter()
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {'race', 'gender', 'age_at_index', 'ajcc_pathologic_stage'}
        missing = required - set(fields)
        if missing:
            return ['Missing columns: ' + ', '.join(sorted(missing))]
        expression = 'mean.x' if 'mean.x' in fields else 'mean2'
        if expression not in fields:
            return ['Missing gene-expression column: mean.x or mean2']
        if len(set(fields)) != len(fields):
            return ['Duplicate column names']
        rows = 0
        for line, row in enumerate(reader, start=2):
            rows += 1
            if None in row or any(value is None for value in row.values()):
                issues.append(f'Row {line}: field count does not match header')
                continue
            if any(not row[key].strip() for key in required | {expression}):
                issues.append(f'Row {line}: blank required value')
                continue
            for key in (expression, 'age_at_index'):
                try:
                    valid = math.isfinite(float(row[key]))
                except ValueError:
                    valid = False
                if not valid:
                    issues.append(f'Row {line}: {key} must be a finite number')
            stages[row['ajcc_pathologic_stage']] += 1
        if not rows:
            issues.append('No data rows')
        elif stages:
            if len(stages) < 2:
                issues.append('At least two target classes are needed for classification')
            if min(stages.values()) < 5:
                issues.append('A target class has fewer than five rows; review SMOTE and five-fold validation')
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', nargs='?', default='HDAC2.csv')
    args = parser.parse_args()
    try:
        issues = validate_data(args.csv_path)
    except (OSError, UnicodeError, csv.Error) as error:
        print(f'Unable to read CSV ({type(error).__name__}).')
        return 1
    if issues:
        print('\n'.join(issues))
        return 1
    print('Basic input checks passed. This does not validate model performance or guarantee training success.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
