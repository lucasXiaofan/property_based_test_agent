For automated code check
1. check whether those codes (experiments/oracle_evaluation/calc_properties_coverage.py and experiments/oracle_evaluation/run_api_coverage.py) can evaluate the baseline_test.py and ir_generated_test.py in experiments/oracle_generation/pandas 
2. if so write code that run once will generate a json file that contains the evaluation results of baseline_test.py and ir_generated_test.py in experiments/oracle_generation/pandas, with the date and time, compare which test has higher quality in terms of property coverage and api coverage


For agent evaluation
1. mutant kill, read existing folder in experiments/oracle_generation/pandas, and find corresponding docs in experiments/python_library_bug_analysis/downloaded_docs, utilize the doc generate wrapper mutants, and make sure there is mutant kill evaluation script in experiments/oracle_evaluation, that can run all the mutant warpper to test against the baseline_test.py and ir_generated_test.py, and evaluate their mutant kill rate, and generate a json file that contains the evaluation results of baseline_test.py and ir_generated_test.py in experiments/oracle_generation/pandas, with the date and time, compare which test has higher quality in terms of mutant kill rate
2. go through the baseline_test.py and ir_generated_test.py in experiments/oracle_generation/pandas . utilize the experiments/python_library_bug_analysis/counted_case_docs.json, find if the tested function has a issue report, if so, check whether baseline or ir generated test find the reported bug, and explain the reason, and generate a json file that contains the evaluation results of baseline_test.py and ir_generated_test.py in experiments/oracle_generation/pandas, with the date and time, compare which test has higher quality in terms of bug detection rate


run uv run experiments/oracle_evaluation/run_baseline_vs_ir_enhanced_eval.py
to get the latest evaluation results