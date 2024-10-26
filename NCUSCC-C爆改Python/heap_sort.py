def heapify(arr, n, i):
    largest = i  # 初始化最大值为根节点
    left = 2 * i + 1
    right = 2 * i + 2

    # 如果左子节点存在且大于根节点
    if left < n and arr[left] > arr[largest]:
        largest = left

    # 如果右子节点存在且大于根节点
    if right < n and arr[right] > arr[largest]:
        largest = right

    # 如果最大值不是根节点，则交换
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        # 递归地堆化受影响的子树
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)

    # 构建最大堆
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # 依次取出堆顶元素并调整堆
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # 交换堆顶和最后一个元素
        heapify(arr, i, 0)  # 调整堆

    return arr

