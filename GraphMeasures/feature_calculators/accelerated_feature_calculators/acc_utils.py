def get_edge_order(neighbors, offsets):
    order = []
    for v in range(len(offsets) - 1):
        for ui in range(offsets[v], offsets[v + 1]):
            order.append((v, neighbors[ui]))

    return order


def get_c_array(ctype, lst: list):
    return (ctype * len(lst))(*lst)


def empty_c_array(ctype, size: int):
    return (ctype * size)()


def build_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
