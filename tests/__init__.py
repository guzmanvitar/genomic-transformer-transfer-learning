"""Test package marker.

Declared so that ``tests`` is an importable package under any pytest
invocation mode (full-suite, single-file, ``--collect-only``, IDE runners).
Intentionally empty: cross-module helpers live in ``tests._felid_fixture``
and sibling ``conftest.py`` files.
"""
