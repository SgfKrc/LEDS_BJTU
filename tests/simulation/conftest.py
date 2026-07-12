"""Keep simulation runner scripts out of default pytest collection.

The files in this package start real backend processes and are executed with
`python -m tests.simulation.<runner>`, not as unit tests.
"""

collect_ignore = [
    "test_degradation.py",
    "test_distributed.py",
    "test_exceptions.py",
    "test_single_node.py",
    "test_stress.py",
]
