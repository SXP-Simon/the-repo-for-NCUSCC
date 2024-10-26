class FibonacciHeapNode:
    def __init__(self, key):
        self.key = key
        self.degree = 0
        self.parent = None
        self.child = None
        self.mark = False
        self.left = self
        self.right = self

class FibonacciHeap:
    def __init__(self):
        self.min_node = None
        self.num_nodes = 0

    def insert(self, key):
        node = FibonacciHeapNode(key)
        if self.min_node is None:
            self.min_node = node
        else:
            node.left = self.min_node
            node.right = self.min_node.right
            self.min_node.right.left = node
            self.min_node.right = node
            if node.key < self.min_node.key:
                self.min_node = node
        self.num_nodes += 1

    def _link(self, node1, node2):
        node2.left = node1.left
        node2.right = node1
        node1.left.right = node2
        node1.left = node2

    def extract_min(self):
        min_node = self.min_node
        if min_node is not None:
            child = min_node.child
            if child is not None:
                while True:
                    next_child = child.right
                    self._link(self.min_node, child)
                    child.parent = None
                    child = next_child
                    if child == min_node.child:
                        break
            self._remove_from_list(min_node)
            if min_node == min_node.right:
                self.min_node = None
            else:
                self.min_node = min_node.right
                self._consolidate()
            self.num_nodes -= 1
        return min_node.key if min_node else None

    def _remove_from_list(self, node):
        node.left.right = node.right
        node.right.left = node.left

    def _consolidate(self):
        max_degree = int(self.num_nodes ** 0.5) + 1
        degree_table = [None] * max_degree

        nodes = []
        current = self.min_node
        while True:
            nodes.append(current)
            current = current.right
            if current == self.min_node:
                break

        for current in nodes:
            degree = current.degree
            while degree_table[degree] is not None:
                other = degree_table[degree]
                if current.key > other.key:
                    current, other = other, current
                self._link(other, current)
                degree_table[degree] = None
                degree += 1
            degree_table[degree] = current

        self.min_node = None
        for node in degree_table:
            if node is not None:
                if self.min_node is None:
                    self.min_node = node
                else:
                    node.left = self.min_node
                    node.right = self.min_node.right
                    self.min_node.right.left = node
                    self.min_node.right = node
                    if node.key < self.min_node.key:
                        self.min_node = node

def fibonacci_heap_sort(arr):
    heap = FibonacciHeap()
    for item in arr:
        heap.insert(item)
    sorted_arr = []
    while heap.num_nodes > 0:
        sorted_arr.append(heap.extract_min())
    return sorted_arr

