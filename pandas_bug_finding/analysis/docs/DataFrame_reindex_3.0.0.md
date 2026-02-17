# Documentation for DataFrame.reindex (3.0.0)
Source URL: https://pandas.pydata.org/pandas-docs/version/3.0.0/reference/api/pandas.DataFrame.reindex.html


:::::::::::::::::: {#main-content .bd-main role="main"}
::::::::::::::::: bd-content
:::::::::::: bd-article-container
:::::: {.bd-header-article .d-print-none}
::::: {.header-article-items .header-article__inner}
:::: header-article-items__start
::: header-article-item
- [](../../index.html){.nav-link aria-label="Home"}
- [API reference](../index.html){.nav-link}
- [DataFrame](../frame.html){.nav-link}
- [pandas.DataFrame.reindex]{.ellipsis}
:::
::::
:::::
::::::

::: {#searchbox}
:::

::: {#pandas-dataframe-reindex .section}
# pandas.DataFrame.reindex[\#](#pandas-dataframe-reindex "Link to this heading"){.headerlink}

[[DataFrame.]{.pre}]{.sig-prename .descclassname}[[reindex]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[labels=None]{.pre}]{.n}*, *[[\*]{.pre}]{.n}*, *[[index=None]{.pre}]{.n}*, *[[columns=None]{.pre}]{.n}*, *[[axis=None]{.pre}]{.n}*, *[[method=None]{.pre}]{.n}*, *[[copy=\<no_default\>]{.pre}]{.n}*, *[[level=None]{.pre}]{.n}*, *[[fill_value=nan]{.pre}]{.n}*, *[[limit=None]{.pre}]{.n}*, *[[tolerance=None]{.pre}]{.n}*[)]{.sig-paren}[[[\[source\]]{.pre}]{.viewcode-link}](https://github.com/pandas-dev/pandas/blob/v3.0.0/pandas/core/frame.py#L5843-L6089){.reference .external}[\#](#pandas.DataFrame.reindex "Link to this definition"){.headerlink}

:   Conform DataFrame to new index with optional filling logic.

    Places NA/NaN in locations having no value in the previous index. A
    new object is produced unless the new index is equivalent to the
    current one and [`copy=False`{.docutils .literal
    .notranslate}]{.pre}.

    Parameters[:]{.colon}

    :   

        **labels**[array-like, optional]{.classifier}

        :   New labels / index to conform the axis specified by 'axis'
            to.

        **index**[array-like, optional]{.classifier}

        :   New labels for the index. Preferably an Index object to
            avoid duplicating data.

        **columns**[array-like, optional]{.classifier}

        :   New labels for the columns. Preferably an Index object to
            avoid duplicating data.

        **axis**[int or str, optional]{.classifier}

        :   Axis to target. Can be either the axis name ('index',
            'columns') or number (0, 1).

        **method**[{None, 'backfill'/'bfill', 'pad'/'ffill', 'nearest'}]{.classifier}

        :   Method to use for filling holes in reindexed DataFrame.
            Please note: this is only applicable to DataFrames/Series
            with a monotonically increasing/decreasing index.

            - None (default): don't fill gaps

            - pad / ffill: Propagate last valid observation forward to
              next valid.

            - backfill / bfill: Use next valid observation to fill gap.

            - nearest: Use nearest valid observations to fill gap.

        **copy**[bool, default False]{.classifier}

        :   This keyword is now ignored; changing its value will have no
            impact on the method.

            ::: deprecated
            [Deprecated since version 3.0.0: ]{.versionmodified
            .deprecated}This keyword is ignored and will be removed in
            pandas 4.0. Since pandas 3.0, this method always returns a
            new object using a lazy copy mechanism that defers copies
            until necessary (Copy-on-Write). See the [user guide on
            Copy-on-Write](https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html){.reference
            .external} for more details.
            :::

        **level**[int or name]{.classifier}

        :   Broadcast across a level, matching Index values on the
            passed MultiIndex level.

        **fill_value**[scalar, default np.nan]{.classifier}

        :   Value to use for missing values. Defaults to NaN, but can be
            any "compatible" value.

        **limit**[int, default None]{.classifier}

        :   Maximum number of consecutive elements to forward or
            backward fill.

        **tolerance**[optional]{.classifier}

        :   Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations
            most satisfy the equation [`abs(index[indexer]`{.docutils
            .literal .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`-`{.docutils .literal
            .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`target)`{.docutils .literal
            .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`<=`{.docutils .literal
            .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`tolerance`{.docutils .literal
            .notranslate}]{.pre}.

            Tolerance may be a scalar value, which applies the same
            tolerance to all values, or list-like, which applies
            variable tolerance per element. List-like includes list,
            tuple, array, Series, and must be the same size as the index
            and its dtype must exactly match the index's type.

    Returns[:]{.colon}

    :   

        DataFrame

        :   DataFrame with changed index.

    ::: {.admonition .seealso}
    See also

    [[`DataFrame.set_index`{.xref .py .py-obj .docutils .literal .notranslate}]{.pre}](pandas.DataFrame.set_index.html#pandas.DataFrame.set_index "pandas.DataFrame.set_index"){.reference .internal}

    :   Set row labels.

    [[`DataFrame.reset_index`{.xref .py .py-obj .docutils .literal .notranslate}]{.pre}](pandas.DataFrame.reset_index.html#pandas.DataFrame.reset_index "pandas.DataFrame.reset_index"){.reference .internal}

    :   Remove row labels or move them to new columns.

    [[`DataFrame.reindex_like`{.xref .py .py-obj .docutils .literal .notranslate}]{.pre}](pandas.DataFrame.reindex_like.html#pandas.DataFrame.reindex_like "pandas.DataFrame.reindex_like"){.reference .internal}

    :   Change to same indices as other DataFrame.
    :::

    Examples

    [`DataFrame.reindex`{.docutils .literal .notranslate}]{.pre}
    supports two calling conventions

    - [`(index=index_labels,`{.docutils .literal
      .notranslate}]{.pre}` `{.docutils .literal
      .notranslate}[`columns=column_labels,`{.docutils .literal
      .notranslate}]{.pre}` `{.docutils .literal
      .notranslate}[`...)`{.docutils .literal .notranslate}]{.pre}

    - [`(labels,`{.docutils .literal .notranslate}]{.pre}` `{.docutils
      .literal .notranslate}[`axis={'index',`{.docutils .literal
      .notranslate}]{.pre}` `{.docutils .literal
      .notranslate}[`'columns'},`{.docutils .literal
      .notranslate}]{.pre}` `{.docutils .literal
      .notranslate}[`...)`{.docutils .literal .notranslate}]{.pre}

    We *highly* recommend using keyword arguments to clarify your
    intent.

    Create a DataFrame with some fictional data.

    :::: {.doctest .highlight-default .notranslate}
    ::: highlight
        >>> index = ["Firefox", "Chrome", "Safari", "IE10", "Konqueror"]
        >>> columns = ["http_status", "response_time"]
        >>> df = pd.DataFrame(
        ...     [[200, 0.04], [200, 0.02], [404, 0.07], [404, 0.08], [301, 1.0]],
        ...     columns=columns,
        ...     index=index,
        ... )
        >>> df
                   http_status  response_time
        Firefox            200           0.04
        Chrome             200           0.02
        Safari             404           0.07
        IE10               404           0.08
        Konqueror          301           1.00
    :::
    ::::

    Create a new index and reindex the DataFrame. By default values in
    the new index that do not have corresponding records in the
    DataFrame are assigned [`NaN`{.docutils .literal
    .notranslate}]{.pre}.

    :::: {.doctest .highlight-default .notranslate}
    ::: highlight
        >>> new_index = ["Safari", "Iceweasel", "Comodo Dragon", "IE10", "Chrome"]
        >>> df.reindex(new_index)
                       http_status  response_time
        Safari               404.0           0.07
        Iceweasel              NaN            NaN
        Comodo Dragon          NaN            NaN
        IE10                 404.0           0.08
        Chrome               200.0           0.02
    :::
    ::::

    We can fill in the missing values by passing a value to the keyword
    [`fill_value`{.docutils .literal .notranslate}]{.pre}. Because the
    index is not monotonically increasing or decreasing, we cannot use
    arguments to the keyword [`method`{.docutils .literal
    .notranslate}]{.pre} to fill the [`NaN`{.docutils .literal
    .notranslate}]{.pre} values.

    :::: {.doctest .highlight-default .notranslate}
    ::: highlight
        >>> df.reindex(new_index, fill_value=0)
                       http_status  response_time
        Safari                 404           0.07
        Iceweasel                0           0.00
        Comodo Dragon            0           0.00
        IE10                   404           0.08
        Chrome                 200           0.02
    :::
    ::::

    :::: {.doctest .highlight-default .notranslate}
    ::: highlight
        >>> df.reindex(new_index, fill_value="missing")
                      http_status response_time
        Safari                404          0.07
        Iceweasel         missing       missing
        Comodo Dragon     missing       missing
        IE10                  404          0.08
        Chrome                200          0.02
    :::
    ::::

    We can also reindex the columns.

    :::: {.doctest .highlight-default .notranslate}
    ::: highlight
        >>> df.reindex(columns=["http_status", "user_agent"])
                   http_status  user_agent
        Firefox            200         NaN
        Chrome             200         NaN
        Safari             404         NaN
        IE10               404         NaN
        Konqueror          301         NaN
    :::
    ::::

    Or we can use "axis-style" keyword arguments

    :::: {.doctest .highlight-default .notranslate}
    ::: highlight
        >>> df.reindex(["http_status", "user_agent"], axis="columns")
                   http_status  user_agent
        Firefox            200         NaN
        Chrome             200         NaN
        Safari             404         NaN
        IE10               404         NaN
        Konqueror          301         NaN
    :::
    ::::

    To further illustrate the filling functionality in
    [`reindex`{.docutils .literal .notranslate}]{.pre}, we will create a
    DataFrame with a monotonically increasing index (for example, a
    sequence of dates).

    :::: {.doctest .highlight-default .notranslate}
    ::: highlight
        >>> date_index = pd.date_range("1/1/2010", periods=6, freq="D")
        >>> df2 = pd.DataFrame(
        ...     {"prices": [100, 101, np.nan, 100, 89, 88]}, index=date_index
        ... )
        >>> df2
                    prices
        2010-01-01   100.0
        2010-01-02   101.0
        2010-01-03     NaN
        2010-01-04   100.0
        2010-01-05    89.0
        2010-01-06    88.0
    :::
    ::::

    Suppose we decide to expand the DataFrame to cover a wider date
    range.

    :::: {.doctest .highlight-default .notranslate}
    ::: highlight
        >>> date_index2 = pd.date_range("12/29/2009", periods=10, freq="D")
        >>> df2.reindex(date_index2)
                    prices
        2009-12-29     NaN
        2009-12-30     NaN
        2009-12-31     NaN
        2010-01-01   100.0
        2010-01-02   101.0
        2010-01-03     NaN
        2010-01-04   100.0
        2010-01-05    89.0
        2010-01-06    88.0
        2010-01-07     NaN
    :::
    ::::

    The index entries that did not have a value in the original data
    frame (for example, '2009-12-29') are by default filled with
    [`NaN`{.docutils .literal .notranslate}]{.pre}. If desired, we can
    fill in the missing values using one of several options.

    For example, to back-propagate the last valid value to fill the
    [`NaN`{.docutils .literal .notranslate}]{.pre} values, pass
    [`bfill`{.docutils .literal .notranslate}]{.pre} as an argument to
    the [`method`{.docutils .literal .notranslate}]{.pre} keyword.

    :::: {.doctest .highlight-default .notranslate}
    ::: highlight
        >>> df2.reindex(date_index2, method="bfill")
                    prices
        2009-12-29   100.0
        2009-12-30   100.0
        2009-12-31   100.0
        2010-01-01   100.0
        2010-01-02   101.0
        2010-01-03     NaN
        2010-01-04   100.0
        2010-01-05    89.0
        2010-01-06    88.0
        2010-01-07     NaN
    :::
    ::::

    Please note that the [`NaN`{.docutils .literal .notranslate}]{.pre}
    value present in the original DataFrame (at index value 2010-01-03)
    will not be filled by any of the value propagation schemes. This is
    because filling while reindexing does not look at DataFrame values,
    but only compares the original and desired indexes. If you do want
    to fill in the [`NaN`{.docutils .literal .notranslate}]{.pre} values
    present in the original DataFrame, use the [`fillna()`{.docutils
    .literal .notranslate}]{.pre} method.

    See the [[user guide]{.std
    .std-ref}](../../user_guide/basics.html#basics-reindexing){.reference
    .internal} for more.
:::

::::: prev-next-area
[](pandas.DataFrame.idxmin.html "previous page"){.left-prev}

::: prev-next-info
previous

pandas.DataFrame.idxmin
:::

[](pandas.DataFrame.reindex_like.html "next page"){.right-next}

::: prev-next-info
next

pandas.DataFrame.reindex_like
:::
:::::
::::::::::::

:::::: {#pst-secondary-sidebar .bd-sidebar-secondary .bd-toc}
::::: {.sidebar-secondary-items .sidebar-secondary__inner}
:::: sidebar-secondary-item
::: {#pst-page-navigation-heading-2 .page-toc .tocsection .onthispage}
On this page
:::

- [[`DataFrame.reindex()`{.docutils .literal
  .notranslate}]{.pre}](#pandas.DataFrame.reindex){.reference .internal
  .nav-link}
::::
:::::
::::::
:::::::::::::::::
::::::::::::::::::
