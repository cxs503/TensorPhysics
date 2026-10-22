#!/usr/bin/env python3
"""Direct test runner without pytest dependency"""
import sys
import traceback

sys.path.insert(0, '.')

from tests.test_si2_fracture_generation import (
    TestInternalFaceExtraction,
    TestUniformStressFailure,
    TestLedgerConservation,
    TestEqualVolumeSphereMapping,
    TestSerialization,
)

test_classes = [
    TestInternalFaceExtraction,
    TestUniformStressFailure,
    TestLedgerConservation,
    TestEqualVolumeSphereMapping,
    TestSerialization,
]

failed = []
passed = 0
for test_class in test_classes:
    instance = test_class()
    test_methods = sorted([m for m in dir(instance) if m.startswith('test_')])
    for method_name in test_methods:
        full_name = f'{test_class.__name__}.{method_name}'
        try:
            method = getattr(instance, method_name)
            method()
            print(f'  ✓ {full_name}')
            passed += 1
        except Exception as e:
            print(f'  ✗ {full_name}: {e}')
            traceback.print_exc()
            failed.append(full_name)

print(f'\n{passed} passed, {len(failed)} failed')
sys.exit(1 if failed else 0)
