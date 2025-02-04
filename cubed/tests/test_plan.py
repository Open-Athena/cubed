import cubed as xp
from cubed.core.plan import traverse_array_keys


def test_traverse_array_keys():
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2))
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2))
    c = a - 1
    d = b.T
    e = xp.add(c, d)

    e.visualize(optimize_graph=False, show_hidden=True)

    dag = e.plan._finalize(optimize_graph=False).dag

    output_arrays_to_keys = {e.name: [(0, 0), (0, 1)]}
    arrays_to_keys = traverse_array_keys(dag, output_arrays_to_keys)

    assert arrays_to_keys[a.name] == [(0, 0), (0, 1)]
    assert arrays_to_keys[b.name] == [(0, 0), (1, 0)]  # transpose
    assert arrays_to_keys[c.name] == [(0, 0), (0, 1)]
    assert arrays_to_keys[d.name] == [(0, 0), (0, 1)]
