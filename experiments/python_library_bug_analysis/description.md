bug should be only relevant to library logic, not performance, environment, or installation issues

utilize the gh api to find the confirmed bugs (confirmed by developer)
---

utilizing gh, try to find more bugs that are not in experiments/python_library_bug_analysis/pandas_bug_finding_results.json

for pandas, make the version to be 3.0.0, and only find the issue or pr that is confirmed by developer is a bug 

current goal is try to find all confirmed bug on github for pandas 3.0.0. generate a json, and report for all of them include the experiments/python_library_bug_analysis/pandas_bug_finding_results.json analyze whether those bugs can be reproduced with or related to the Exceptional oracles and input constraints, whether reading the documentation can help to find the reported and confirmed bug,

write your bug finding json, and analysis markdown in experiments/python_library_bug_analysis

---
on numpy and django, 
first pick the latest stable version of numpy and django, then find the confirmed bugs from the selected version, and analyze them in the same way as pandas, write your bug finding json, and analysis markdown in experiments/python_library_bug_analysis, goal is also whether those bugs can be reproduced with or related to the Exceptional oracles and input constraints, whether reading the documentation can help to find the reported and confirmed bug,