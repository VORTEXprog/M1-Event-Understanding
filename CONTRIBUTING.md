# Contributing

M1 is an experimental source toolkit. Contributions that improve correctness,
independence, documentation and reproducibility are welcome.

1. Install Python 3.12, an appropriate PyTorch build, and `pip install -e '.[test]'`.
2. Run `python -m pytest` and the examples affected by your change.
3. Add a focused regression test for a concrete defect. Preserve the neural
   algorithms and explicit input/domain checks unless your proposal intentionally
   changes them.
4. Submit a pull request describing the problem, resulting behavior, tests and
   limitations. Distinguish implementation tests from evidence of model quality.

Use small pull requests. Discuss architecture or public-interface changes in an
issue first. Keep the feature provider replaceable and prevent labels or target
annotations from entering inference inputs. Preserve complete event sets;
capacity errors must be explicit rather than silently dropping events or text.

Do not submit credentials, personal information, private prompts, traces, model
weights, tokenizer assets, datasets or generated artifacts. Use small invented
fixtures. Include third-party material only when its redistribution terms are
clear, and retain required notices.

By submitting a contribution, you agree to license your original contribution
under this project's MIT license. This does not relicense third-party material.

