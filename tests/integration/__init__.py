"""Integration test subpackage marker.

Declared so that ``tests.integration`` is importable as a subpackage.
Without this file, pytest's default ``importmode=prepend`` inserts
``tests/integration`` (not the repo root) into ``sys.path`` and
``from tests._felid_fixture import ...`` fails under ``--collect-only``.
Intentionally empty.
"""
