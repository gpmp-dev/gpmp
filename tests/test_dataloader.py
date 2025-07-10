"""
Unit tests for Dataset / DataLoader / scaler utilities (unittest version).
"""

import gpmp.num as gnp
import unittest

from gpmp.dataloader import (
    Dataset,
    DataLoader,
    Normalizer,
    RobustScaler,
    MinMaxScaler,
    ObservationScaler,
    collate_batches,
)


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------
def make_xz(n=20, d=3):
    gnp.set_seed(0)  # Set global random seed
    x = gnp.randn(n, d)  # Sample x ~ N(0,1)
    z = gnp.randn(n, 1)  # Sample z ~ N(0,1)
    return x, z


# ======================================================================
#                           Test cases
# ======================================================================
class TestDataset(unittest.TestCase):

    def test_single_vs_sharded(self):
        x, z = make_xz()
        ds_single = Dataset(x, z)
        ds_sharded = Dataset([x[:10], x[10:]], [z[:10], z[10:]])

        self.assertEqual(len(ds_single), 20)
        self.assertEqual(len(ds_sharded), 20)
        # first / last rows equal
        self.assertTrue(gnp.allclose(ds_single[0][0], ds_sharded[0][0]))
        self.assertTrue(gnp.allclose(ds_single[19][1], ds_sharded[19][1]))

    def test_split_ratios(self):
        x, z = make_xz(30)
        ds = Dataset(x, z)
        tr, va, te = Dataset.split(ds, ratios=(0.5, 0.3, 0.2), seed=42)

        self.assertEqual(len(tr), 15)
        self.assertEqual(len(va), 9)
        self.assertEqual(len(te), 6)

    def test_k_fold_exhaustive(self):
        splits = Dataset.k_fold_indices(12, 4, seed=123)
        self.assertEqual(len(splits), 4)
        val_sets = [set(val.tolist()) for _, val in splits]
        flattened = {i for s in val_sets for i in s}
        self.assertEqual(flattened, set(range(12)))


class TestDataLoader(unittest.TestCase):

    def test_batches_and_len(self):
        x, z = make_xz(25)
        dl = DataLoader(Dataset(x, z), batch_size=8, shuffle=False, drop_last=False)
        batches = list(dl)
        self.assertEqual(len(dl), 4)  # 8 + 8 + 8 + 1
        self.assertEqual(batches[-1][0].shape[0], 1)

    def test_infinite_iterator(self):
        x, z = make_xz(10)
        dl = DataLoader(Dataset(x, z), batch_size=4, infinite=True)
        it = iter(dl)
        count = 0
        for _ in range(13):  # iterate beyond dataset size -> must wrap
            xb, _ = next(it)
            count += xb.shape[0]
        self.assertGreater(count, 10)


class TestScalers(unittest.TestCase):

    def _roundtrip(self, Scaler, **kwargs):
        x, _ = make_xz()
        scaler = Scaler.fit(x, **kwargs) if kwargs else Scaler.fit(x)
        x_std = scaler.transform(x)
        x_back = scaler.inverse_transform(x_std)
        self.assertTrue(gnp.allclose(x_back, x, atol=1e-8))

    def test_normalizer(self):
        self._roundtrip(Normalizer)

    def test_robust_scaler(self):
        self._roundtrip(RobustScaler, q_low=20, q_high=80)

    def test_minmax_scaler(self):
        self._roundtrip(MinMaxScaler)

    def test_observation_scaler(self):
        _, z = make_xz()
        sc = ObservationScaler.fit(z)
        z_std = sc.transform(z)
        z_back = sc.inverse_transform(z_std)
        self.assertTrue(gnp.allclose(z_back, z))

    def test_copy_flag(self):
        x, _ = make_xz()
        scaler = Normalizer.fit(x)

        x_copy = gnp.copy(x)
        x_copy_transformed = scaler.transform(x_copy, copy=True)

        self.assertTrue(gnp.allclose(x, x))  # x unchanged
        self.assertFalse(gnp.allclose(x, x_copy_transformed))  # x_copy_transformed ≠ x


class TestCollate(unittest.TestCase):

    def test_collate(self):
        x, z = make_xz(10)
        ds = Dataset(x, z)
        dl = DataLoader(ds, batch_size=4, shuffle=False)
        big_x, big_z = collate_batches(list(dl))

        self.assertTrue(gnp.allclose(big_x, x))
        self.assertTrue(gnp.allclose(big_z, z))


class TestDatasetStats(unittest.TestCase):

    def test_basic_stats(self):
        x, z = make_xz(50, d=5)
        ds = Dataset(x, z)

        # x statistics
        self.assertTrue(gnp.allclose(ds.x_min(), gnp.min(x, axis=0)))
        self.assertTrue(gnp.allclose(ds.x_max(), gnp.max(x, axis=0)))
        self.assertTrue(gnp.allclose(ds.x_mean(), gnp.mean(x, axis=0)))
        self.assertTrue(gnp.allclose(ds.x_var(), gnp.var(x, axis=0)))
        self.assertTrue(gnp.allclose(ds.x_std(), gnp.std(x, axis=0)))
        self.assertTrue(gnp.allclose(ds.x_median(), gnp.percentile(x, 50, axis=0)))

        # z statistics
        self.assertTrue(gnp.allclose(ds.z_min(), gnp.min(z, axis=0)))
        self.assertTrue(gnp.allclose(ds.z_max(), gnp.max(z, axis=0)))
        self.assertTrue(gnp.allclose(ds.z_mean(), gnp.mean(z, axis=0)))
        self.assertTrue(gnp.allclose(ds.z_var(), gnp.var(z, axis=0)))
        self.assertTrue(gnp.allclose(ds.z_std(), gnp.std(z, axis=0)))
        self.assertTrue(gnp.allclose(ds.z_median(), gnp.percentile(z, 50, axis=0)))

    def test_quantile_argument(self):
        x, z = make_xz(30, d=2)
        ds = Dataset(x, z)

        q = 0.75
        self.assertTrue(gnp.allclose(ds.x_quantile(q), gnp.percentile(x, q * 100, axis=0)))
        self.assertTrue(gnp.allclose(ds.z_quantile(q), gnp.percentile(z, q * 100, axis=0)))

    def test_quantile_value_error(self):
        x, z = make_xz()
        ds = Dataset(x, z)

        with self.assertRaises(ValueError):
            ds.x_quantile(-0.1)

        with self.assertRaises(ValueError):
            ds.z_quantile(1.1)

        
# ----------------------------------------------------------------------
# Run tests
# ----------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
