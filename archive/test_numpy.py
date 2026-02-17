"""
Test script for numpy.meshgrid function behavior

This script reproduces various calling patterns of np.meshgrid to explore
its behavior with different parameter combinations, particularly focusing on:
- Variable positional arguments
- Named parameters: indexing, sparse, copy
- Return value types and structures

Based on documentation from Context7 for numpy
"""

import numpy as np


def test_meshgrid_basic():
    """Test basic meshgrid usage with two 1-D arrays"""
    print("=" * 70)
    print("Test 1: Basic meshgrid with two arrays")
    print("=" * 70)

    x = np.array([0, 1, 2, 3])
    y = np.array([0, 1, 2, 3, 4, 5])
    xx, yy = np.meshgrid(x, y)

    print(f"Input x: {x}")
    print(f"Input y: {y}")
    print(f"\nOutput xx shape: {xx.shape}")
    print(f"Output xx:\n{xx}")
    print(f"\nOutput yy shape: {yy.shape}")
    print(f"Output yy:\n{yy}")
    print()


def test_meshgrid_with_parameters():
    """Test meshgrid with indexing, sparse, and copy parameters"""
    print("=" * 70)
    print("Test 2: Meshgrid with indexing='xy', sparse=True, copy=False")
    print("=" * 70)

    x = np.array([0, 1])
    indexing = 'xy'
    sparse = True
    copy = False

    result = np.meshgrid(x, indexing=indexing, sparse=sparse, copy=copy)

    print(f"Input x: {x}")
    print(f"Parameters: indexing='{indexing}', sparse={sparse}, copy={copy}")
    print(f"Result: {result}")
    print(f"Result type: {type(result)}")
    print()


def test_meshgrid_no_positional_args():
    """Test meshgrid with no positional arguments"""
    print("=" * 70)
    print("Test 3: Meshgrid with no positional arguments")
    print("=" * 70)

    indexing = 'xy'
    sparse = True
    copy = False

    result = np.meshgrid(indexing=indexing, sparse=sparse, copy=copy)

    print(f"Parameters only: indexing='{indexing}', sparse={sparse}, copy={copy}")
    print(f"Result: {result}")
    print(f"Result type: {type(result)}")
    print()


def test_meshgrid_two_arrays_with_params():
    """Test meshgrid with two arrays and parameters"""
    print("=" * 70)
    print("Test 4: Meshgrid with two arrays, indexing='xy', sparse=True, copy=False")
    print("=" * 70)

    indexing = 'xy'
    sparse = True
    copy = False

    result = np.meshgrid([1, 2], [3, 4], indexing=indexing, sparse=sparse, copy=copy)

    print(f"Input arrays: [1, 2], [3, 4]")
    print(f"Parameters: indexing='{indexing}', sparse={sparse}, copy={copy}")
    print(f"Result length: {len(result)}")
    print(f"Result[0]:\n{result[0]}")
    print(f"Result[1]:\n{result[1]}")
    print(f"Result type: {type(result)}")
    print()


def test_meshgrid_copy_true():
    """Test meshgrid with copy=True parameter"""
    print("=" * 70)
    print("Test 5: Meshgrid with copy=True")
    print("=" * 70)

    indexing = 'xy'
    sparse = True
    copy = True

    result = np.meshgrid([1, 2], [3, 4], indexing=indexing, sparse=sparse, copy=copy)

    print(f"Input arrays: [1, 2], [3, 4]")
    print(f"Parameters: indexing='{indexing}', sparse={sparse}, copy={copy}")
    print(f"Result: {result}")
    print(f"Result type: {type(result)}")
    print(f"Result[0]:\n{result[0]}")
    print(f"Result[1]:\n{result[1]}")
    print()


def test_meshgrid_sparse_false():
    """Test meshgrid with sparse=False (dense grid)"""
    print("=" * 70)
    print("Test 6: Meshgrid with sparse=False (dense grid)")
    print("=" * 70)

    x = [1, 2, 3]
    y = [4, 5]

    result = np.meshgrid(x, y, indexing='xy', sparse=False, copy=False)

    print(f"Input arrays: {x}, {y}")
    print(f"Parameters: indexing='xy', sparse=False, copy=False")
    print(f"Result[0] shape: {result[0].shape}")
    print(f"Result[0]:\n{result[0]}")
    print(f"Result[1] shape: {result[1].shape}")
    print(f"Result[1]:\n{result[1]}")
    print()


def test_meshgrid_indexing_ij():
    """Test meshgrid with indexing='ij' (matrix indexing)"""
    print("=" * 70)
    print("Test 7: Meshgrid with indexing='ij' (matrix indexing)")
    print("=" * 70)

    x = [1, 2, 3]
    y = [4, 5]

    result_xy = np.meshgrid(x, y, indexing='xy')
    result_ij = np.meshgrid(x, y, indexing='ij')

    print(f"Input arrays: {x}, {y}")
    print(f"\nWith indexing='xy':")
    print(f"Result[0] shape: {result_xy[0].shape}")
    print(f"Result[0]:\n{result_xy[0]}")
    print(f"Result[1] shape: {result_xy[1].shape}")
    print(f"Result[1]:\n{result_xy[1]}")

    print(f"\nWith indexing='ij':")
    print(f"Result[0] shape: {result_ij[0].shape}")
    print(f"Result[0]:\n{result_ij[0]}")
    print(f"Result[1] shape: {result_ij[1].shape}")
    print(f"Result[1]:\n{result_ij[1]}")
    print()


if __name__ == "__main__":
    print(f"\nNumPy version: {np.__version__}\n")

    test_meshgrid_basic()
    test_meshgrid_with_parameters()
    test_meshgrid_no_positional_args()
    test_meshgrid_two_arrays_with_params()
    test_meshgrid_copy_true()
    test_meshgrid_sparse_false()
    test_meshgrid_indexing_ij()

    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)
