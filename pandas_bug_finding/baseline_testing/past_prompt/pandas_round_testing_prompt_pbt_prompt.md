given 
def round(
        self, decimals: int | dict[IndexLabel, int] | Series = 0, *args, **kwargs
    ) -> DataFrame:
        """
        Round a DataFrame to a variable number of decimal places.

        Parameters
        ----------
        decimals : int, dict, Series
            Number of decimal places to round each column to. If an int is
            given, round each column to the same number of places.
            Otherwise dict and Series round to variable numbers of places.
            Column names should be in the keys if `decimals` is a
            dict-like, or in the index if `decimals` is a Series. Any
            columns not included in `decimals` will be left as is. Elements
            of `decimals` which are not columns of the input will be
            ignored.
        *args
            Additional keywords have no effect but might be accepted for
            compatibility with numpy.
        **kwargs
            Additional keywords have no effect but might be accepted for
            compatibility with numpy.

        Returns
        -------
        DataFrame
            A DataFrame with the affected columns rounded to the specified
            number of decimal places.

        See Also
        --------
        numpy.around : Round a numpy array to the given number of decimals.
        Series.round : Round a Series to the given number of decimals.

        Examples
        --------
        >>> df = pd.DataFrame([(.21, .32), (.01, .67), (.66, .03), (.21, .18)],
        ...                   columns=['dogs', 'cats'])
        >>> df
            dogs  cats
        0  0.21  0.32
        1  0.01  0.67
        2  0.66  0.03
        3  0.21  0.18

        By providing an integer each column is rounded to the same number
        of decimal places

        >>> df.round(1)
            dogs  cats
        0   0.2   0.3
        1   0.0   0.7
        2   0.7   0.0
        3   0.2   0.2

        With a dict, the number of places for specific columns can be
        specified with the column names as key and the number of decimal
        places as value

        >>> df.round({'dogs': 1, 'cats': 0})
            dogs  cats
        0   0.2   0.0
        1   0.0   1.0
        2   0.7   0.0
        3   0.2   0.0

        Using a Series, the number of places for specific columns can be
        specified with the column names as index and the number of
        decimal places as value

        >>> decimals = pd.Series([0, 1], index=['cats', 'dogs'])
        >>> df.round(decimals)
            dogs  cats
        0   0.2   0.0
        1   0.0   1.0
        2   0.7   0.0
        3   0.2   0.0
        """

try to find bugs in pandas round function
utilize context7 mcp to search for pandas 2.2 to generate correct test cases

Once you thoroughly understand the target, look for these high-value property patterns:
74
75 - **Invariants**: ‘len(filter(x)) <= len(x)‘, ‘set(sort(x)) == set(x)‘
76 - **Round-trip properties**: ‘decode(encode(x)) = x‘, ‘parse(format(x)) = x‘
77 - **Inverse operations**: ‘add/remove‘, ‘push/pop‘, ‘create/destroy‘
78 - **Multiple implementations**: fast vs reference, optimized vs simple
79 - **Mathematical properties**: idempotence ‘f(f(x)) = f(x)‘, commutativity ‘f(x,y) = f(y,x)‘
80 - **Confluence**: if the order of function application doesn’t matter (eg in compiler optimization passes)
81 - **Metamorphic properties**: some relationship between ‘f(x)‘ and ‘g(x)‘ holds, even without knowing the correct value for ‘
f(x)‘. For example, ‘sin(π - x) = sin(x)‘ for all x.
82 - **Single entry point**: for libraries with 1-2 entrypoints, test that calling it on valid inputs doesn’t crash (no
specific property!). Common in e.g. parsers.
83
84 If there are no candidate properties in $ARGUMENTS, do not search outside of the specified function, module, or file.
Instead, exit with "No testable properties found in $ARGUMENTS".
85
86 **Only test properties that the code is explicitly claiming to have.** either in the docstring, comments, or how other code
uses it. Do not make up properties that you merely think are true. Proposed properties should be **strongly supported
** by evidence.
96 ### 4. Write tests
97
98 Write focused Hypothesis property-based tests to test the properties you proposed.
99
100 - Use smart Hypothesis strategies - constrain inputs to the domain intelligently
101 - Write strategies that are both:
102 - sound: tests only inputs expected by the code
103 - complete: tests all inputs expected by the code
104 If soundness and completeness are in conflict, prefer writing sound but incomplete properties. Do not chase completeness:
90% is good enough.
105 - Focus on a few high-impact properties, rather than comprehensive codebase coverage.
106
107 A basic Hypothesis test looks like this:
108
109 ‘‘‘python
110 @given(st.floats(allow_nan=False, min_value=0))
111 def test_sqrt_round_trip(x):
112 result = math.sqrt(x)
113 assert math.isclose(result * result, x)
114 ‘‘‘
115
116 A more complete reference is available in the *Hypothesis Quick Reference* section below.
For a comprehensive reference:
268
269 - **Basic tutorial**: https://hypothesis.readthedocs.io/en/latest/quickstart.html
270 - **Strategies reference**: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html
271 - **NumPy strategies**: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#numpy
272 - **Pandas strategies**: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#pandas

write your test script in pandas_bug_finding/baseline_testing/pandas_round_pbt_testing_script.py, and generate natural language docstring on the top of script to summarize the properties your tested, and how to run the script with uv