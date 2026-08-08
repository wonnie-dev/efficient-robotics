from household_pilot_scene import hollow_mug_mesh


def test_hollow_mug_mesh_is_nonempty_and_indexed() -> None:
    points, counts, indices = hollow_mug_mesh()
    assert len(points) > 400
    assert len(counts) > 400
    assert sum(counts) == len(indices)
    assert min(indices) == 0
    assert max(indices) < len(points)


def test_hollow_mug_mesh_has_open_top_and_handle_extent() -> None:
    points, _counts, _indices = hollow_mug_mesh()
    xs = [point[0] for point in points]
    zs = [point[2] for point in points]
    assert max(xs) > 0.075
    assert min(zs) == 0.0
    assert max(zs) > 0.10
