import gpmp.num as gnp
from gpmp.misc.param import Param, Normalization

def test_basic_construction():
    p = Param(
        values=[0.0, 1.0, -1.0],
        paths=[["covparam", "variance"], ["covparam", "lengthscale"], ["meanparam"]],
        normalizations=["log", "log_inv", "none"],
        names=["sigma", "rho", "mu"],
        bounds=[(0.1, 10.0), (0.01, 2.0), None]
    )
    assert p.dim == 3
    assert gnp.allclose(p.denormalized_values, gnp.asarray([1.0, gnp.exp(-1.0), -1.0]))
    assert p.get_by_name("sigma") == 0.0
    assert p.get_by_name("rho", return_view=True).shape == (1,)

def test_bounds_check():
    p = Param(
        values=[0.0, 0.5],
        paths=[["a"], ["b"]],
        normalizations=["none", "none"],
        names=["x", "y"],
        bounds=[(0.0, 1.0), (0.6, 1.0)]
    )
    assert p.check_bounds() == [True, False]
    p.set_by_name("y", 0.7)
    assert p.check_bounds() == [True, True]

def test_path_get_set():
    p = Param(
        values=[0.1, 0.2, 0.3],
        paths=[["gp", "a"], ["gp", "b"], ["mean"]],
        normalizations=["none"] * 3,
        names=["a", "b", "m"]
    )

    # Check copy semantics (default)
    v = p.get_by_path(["gp"], prefix_match=True)
    assert gnp.allclose(v, gnp.asarray([0.1, 0.2]))
    v[0] = 999.0  # should not affect internal state
    assert gnp.allclose(p.get_by_path(["gp"], prefix_match=True), gnp.asarray([0.1, 0.2]))

    # Now use return_view=True
    v_view = p.get_by_path(["gp"], prefix_match=True, return_view=True)
    v_view[0] = 0.9
    v_view[1] = 1.0
    assert gnp.allclose(p.get_by_path(["gp"], prefix_match=True), gnp.asarray([0.9, 1.0]))

    # Assign values with slicing
    v_view[:] = gnp.array([0.5, 0.5])
    assert gnp.allclose(p.get_by_path(["gp"], prefix_match=True), gnp.asarray([0.5, 0.5]))
    
    # Explicit set_by_path
    p.set_by_path(["gp"], [0.4, 0.5], prefix_match=True)
    assert gnp.allclose(p.get_by_path(["gp"], prefix_match=True), gnp.asarray([0.4, 0.5]))

def test_concat_and_slice():
    p1 = Param(values=[0.0], paths=[["a"]], normalizations=["none"], names=["x"])
    p2 = Param(values=[1.0, 2.0], paths=[["b"], ["c"]], normalizations=["none"] * 2, names=["y", "z"])
    p = p1 + p2
    assert p.dim == 3
    assert p.names == ["x", "y", "z"]
    sliced = p[1:]
    assert sliced.names == ["y", "z"]
    assert gnp.allclose(sliced.values, gnp.asarray([1.0, 2.0]))

def test_set_from_unnormalized():
    p = Param(
        values=[0.0, 0.0],
        paths=[["p", "var"], ["p", "lengthscale"]],
        normalizations=[Normalization.LOG, Normalization.LOG_INV],
        names=["s", "r"]
    )
    p.set_from_unnormalized(s=2.0, r=0.5)
    assert gnp.allclose(p.denormalized_values, gnp.asarray([2.0, 0.5]))
    assert gnp.allclose(gnp.exp(p.values[0]), gnp.asarray(2.0))
    assert gnp.allclose(gnp.exp(-p.values[1]), gnp.asarray(0.5))



def test_tree_queries():
    p = Param(
        values=[0.1, 0.2, 0.3, 0.4],
        paths=[
            ["gp", "variance"],
            ["gp", "lengthscale"],
            ["mean", "bias"],
            ["mean", "slope"]
        ],
        normalizations=["none"] * 4,
        names=["v", "l", "b", "s"]
    )

    # Test get_paths
    all_paths = p.get_paths()
    assert set(map(tuple, all_paths)) == {
        ("gp", "variance"),
        ("gp", "lengthscale"),
        ("mean", "bias"),
        ("mean", "slope")
    }

    # Filtered paths
    mean_paths = p.get_paths(prefix=["mean"])
    assert set(map(tuple, mean_paths)) == {
        ("mean", "bias"), ("mean", "slope")
    }

    # Indices by path prefix
    indices = p.indices_by_path_prefix(["mean"])
    assert indices == [2, 3]

    # Names by path prefix
    names = p.names_by_path_prefix(["mean"])
    assert names == ["b", "s"]

    # Value selection via select_by_path_prefix
    values = p.select_by_path_prefix(["mean"], return_view=True)
    assert gnp.allclose(values, gnp.asarray([0.3, 0.4]))
    
def run_all():
    test_basic_construction()
    test_bounds_check()
    test_path_get_set()
    test_concat_and_slice()
    test_set_from_unnormalized()
    test_tree_queries()
    print("All tests passed.")


if __name__ == "__main__":
    run_all()
