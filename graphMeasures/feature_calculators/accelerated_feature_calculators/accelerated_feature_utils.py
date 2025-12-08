"""
Utils for accelerated features.
"""

def get_edge_order(neighbors, offsets):
    """
    Build (u, v) edge pairs from CSR adjacency data.
    """
    order = []
    for v in range(len(offsets) - 1):
        for ui in range(offsets[v], offsets[v + 1]):
            order.append((v, neighbors[ui]))
    return order


def get_c_array(ctype, lst: list):
    """
    Convert a Python list into a C array of type `ctype`.
    """
    return (ctype * len(lst))(*lst)


def empty_c_array(ctype, size: int):
    """
    Create an empty C array of length `size`.
    """
    return (ctype * size)()


def build_chunks(lst, n):
    """
    Yield chunks of size `n` from list `lst`.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
