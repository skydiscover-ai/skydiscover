"""The specification: what the system must do, and the record of how each question was settled.

cards/ holds the specification cards and how they get filled in; knowledge/ what carries across
runs; proof, run, checkpoint, and paths the rest. Stdlib only, no LLM calls. Each module is also
a command: python3 -m skydiscover.synthesize.spec.<module> --help.
"""
