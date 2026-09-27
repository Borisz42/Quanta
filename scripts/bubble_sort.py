from typing import List, TypeVar

T = TypeVar("T")


def bubble_sort(arr: List[T]) -> List[T]:
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
