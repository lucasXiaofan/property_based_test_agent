You are improving the baseline tests for one target pandas function folder.

Target function folder: `{target_folder}`

Read only these target-folder inputs:
- `baseline_test.py`
- the pandas documentation markdown file in the target folder, such as
  `pandas.Series.mean.md`

Use the existing `baseline_test.py` as the starting point. Read the documentation markdown in
the same target folder to identify important behavior, edge cases, invalid inputs, parameter
interactions, and non-happy-path cases that the baseline does not cover yet.

Create an improved version of the baseline tests with name `improved_baseline_test.py` in
`{target_folder}`.

Requirements:
- Add new tests that cover high-stakes behavior and non-happy-path cases from the target
  documentation.
- Add concise comments above or inside the newly added tests so the added cases are easy to
  distinguish from the original baseline tests.
- Do not reference tests, documentation, or implementation details outside the target function
  folder.
- Run the improved test file and fix any failures caused by the new tests.
