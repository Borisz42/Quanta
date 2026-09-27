"""Bubble Sort Algorithm Implementation in Python.

Provides an optimized bubble sort implementation with early-termination flag
that sorts a list of comparable elements in ascending order.
"""

from typing import List, TypeVar

T = TypeVar("T")


def bubble_sort(arr: List[T]) -> List[T]:
    """Sorts a list of elements in ascending order using the bubble sort algorithm.

    Iterates through the list, comparing adjacent elements and swapping them if
    they are in the wrong order. Terminates early if an entire pass completes
    without performing any swaps, indicating the list is already sorted.

    Args:
        arr: Input list of comparable elements.

    Returns:
        The sorted list in ascending order.
    """
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


if __name__ == "__main__":
    test_data = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original array : {test_data}")
    sorted_data = bubble_sort(test_data.copy())
    print(f"Sorted array   : {sorted_data}")
