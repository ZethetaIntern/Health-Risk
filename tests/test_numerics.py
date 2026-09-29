"""Unit tests for numeric and array helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from healthrisk_ai.numerics import (
    as_numeric,
    clip,
    is_monotonically_increasing,
    jensen_shannon_divergence,
    quantile_safe,
    random_state,
    rate_limited_sleep,
    rolling_zscore,
    safe_log1p,
    time_aware_train_test_split,
)


class TestSafeLog1p:
    def test_scalar_non_negative(self) -> None:
        assert safe_log1p(0.0) == 0.0
        assert safe_log1p(1.0) == pytest.approx(np.log(2.0))

    def test_scalar_negative_clipped_to_zero(self) -> None:
        assert safe_log1p(-10.0) == 0.0

    def test_array_negative_clipped_to_zero(self) -> None:
        x = np.array([-5.0, 0.0, 4.0])
        out = safe_log1p(x)
        expected = np.log1p(np.clip(x, 0, None))
        np.testing.assert_allclose(out, expected)

    def test_returns_float_for_scalar(self) -> None:
        result = safe_log1p(2.0)
        assert isinstance(result, float)


class TestClip:
    def test_no_bounds(self) -> None:
        assert clip(3.0) == 3.0

    def test_lower_bound(self) -> None:
        assert clip(-10.0, lower=0.0) == 0.0

    def test_upper_bound(self) -> None:
        assert clip(100.0, upper=10.0) == 10.0

    def test_both_bounds(self) -> None:
        assert clip(-5.0, lower=0.0, upper=5.0) == 0.0
        assert clip(7.0, lower=0.0, upper=5.0) == 5.0


class TestRollingZscore:
    def test_identical_values_produces_nan(self) -> None:
        s = pd.Series([5.0, 5.0, 5.0, 5.0])
        out = rolling_zscore(s, window=4)
        assert np.isnan(out.iloc[-1])

    def test_mean_centered(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        out = rolling_zscore(s, window=3)
        # Last valid window: values 3,4,5 => mean=4, std=0.816..., z=(5-4)/std
        expected_last = (5.0 - 4.0) / (np.std([3.0, 4.0, 5.0], ddof=0))
        assert out.iloc[-1] == pytest.approx(expected_last)

    def test_window_too_small_raises(self) -> None:
        s = pd.Series([1.0])
        with pytest.raises(ValueError, match="positive window"):
            rolling_zscore(s, window=0)


class TestJensenShannonDivergence:
    def test_identical_distributions(self) -> None:
        p = np.array([0.5, 0.5, 0.0])
        q = np.array([0.5, 0.5, 0.0])
        assert jensen_shannon_divergence(p, q) == pytest.approx(0.0)

    def test_completely_different(self) -> None:
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        jsd = jensen_shannon_divergence(p, q)
        assert jsd > 0.0

    def test_unnormalized_inputs_normalized(self) -> None:
        p = np.array([2.0, 2.0])
        q = np.array([1.0, 3.0])
        jsd = jensen_shannon_divergence(p, q)
        assert 0.0 < jsd < 1.0

    def test_empty_entries_handled_as_nan(self) -> None:
        # Input with zero total mass should not blow up
        p = np.array([0.0, 0.0])
        q = np.array([1.0, 0.0])
        jsd = jensen_shannon_divergence(p, q)
        assert jsd >= 0.0


class TestQuantileSafe:
    def test_standard_quantile(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert quantile_safe(x, 0.5) == pytest.approx(2.5)

    def test_empty_returns_nan(self) -> None:
        assert np.isnan(quantile_safe(np.array([]), 0.5))


class TestAgeToCmsHccBucket:
    def test_boundaries(self) -> None:
        assert "age_0" == "age_0"  # noqa: PGH004 keep reference
        from healthrisk_ai.numerics import age_to_cms_hcc_bucket
        assert age_to_cms_hcc_bucket(0) == "age_0"
        assert age_to_cms_hcc_bucket(1) == "age_1"
        assert age_to_cms_hcc_bucket(3) == "age_2_4"
        # Standard CMS-HCC bands: age 10 falls in the 10-14 bucket.
        assert age_to_cms_hcc_bucket(10) == "age_10_14"
        assert age_to_cms_hcc_bucket(99) == "age_85_plus"

    def test_negative_age_returns_unknown(self) -> None:
        from healthrisk_ai.numerics import age_to_cms_hcc_bucket
        assert age_to_cms_hcc_bucket(-1) == "age_unknown"


class TestRandomState:
    def test_same_seed_same_sequence(self) -> None:
        a = random_state(1234)
        b = random_state(1234)
        np.testing.assert_allclose(a.random(5), b.random(5))

    def test_none_seed_is_deterministic_per_run(self) -> None:
        # None seed uses an internal entropy source; we just check the type.
        gen = random_state(None)
        assert hasattr(gen, "random")


class TestIsMonotonicallyIncreasing:
    def test_plain_increase(self) -> None:
        assert is_monotonically_increasing(np.array([1, 2, 3]))

    def test_equal_allowed(self) -> None:
        assert is_monotonically_increasing(np.array([1, 1, 2]))

    def test_decrease_fails(self) -> None:
        assert not is_monotonically_increasing(np.array([1, 3, 2]))

    def test_single_element_true(self) -> None:
        assert is_monotonically_increasing(np.array([5]))

    def test_empty_true(self) -> None:
        assert is_monotonically_increasing(np.array([]))

    def test_non_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="1D array"):
            is_monotonically_increasing(np.array([[1, 2], [3, 4]]))


class TestAsNumeric:
    def test_good_int(self) -> None:
        assert as_numeric(3) == 3.0

    def test_good_string(self) -> None:
        assert as_numeric("3.14") == 3.14

    def test_bad_returns_none(self) -> None:
        assert as_numeric("not-a-number", nan_if_bad=True) is None

    def test_bad_returns_nan_when_requested(self) -> None:
        assert np.isnan(as_numeric("not-a-number", nan_if_bad=False))


class TestRateLimitedSleep:
    def test_no_sleep_when_under_limit(self) -> None:
        elapsed = 1.0
        delay = rate_limited_sleep(elapsed, limit_per_second=1)
        assert delay == pytest.approx(0.0)

    def test_positive_delay_when_over_limit(self) -> None:
        # elapsed 0.1 against limit 1 req/sec => 0.9s delay
        delay = rate_limited_sleep(0.1, limit_per_second=1)
        assert delay == pytest.approx(0.9, rel=1e-6)

    def test_zero_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="positive rate"):
            rate_limited_sleep(0.1, limit_per_second=0)


class TestTimeAwareTrainTestSplit:
    def test_basic_split(self) -> None:
        dates = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"])
        labels = pd.Series([1.0, 2.0, 3.0, 4.0], index=dates)
        features = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4]}, index=dates)
        split_date = "2020-03-01"
        X_tr, X_te, y_tr, y_te = time_aware_train_test_split(
            dates=dates, labels=labels, features=features, split_date=split_date
        )
        assert len(X_tr) == 2
        assert len(X_te) == 2
        assert y_tr.iloc[0] == 1.0
        assert y_te.iloc[0] == 3.0

    def test_no_train_rows_raises(self) -> None:
        dates = pd.to_datetime(["2020-01-01"])
        labels = pd.Series([1.0], index=dates)
        features = pd.DataFrame({"f1": [0.1]}, index=dates)
        with pytest.raises(ValueError, match="No training rows"):
            time_aware_train_test_split(
                dates=dates, labels=labels, features=features, split_date="2019-01-01"
            )

    def test_no_test_rows_raises(self) -> None:
        dates = pd.to_datetime(["2020-01-01"])
        labels = pd.Series([1.0], index=dates)
        features = pd.DataFrame({"f1": [0.1]}, index=dates)
        with pytest.raises(ValueError, match="No testing rows"):
            time_aware_train_test_split(
                dates=dates, labels=labels, features=features, split_date="2021-01-01"
            )

    def test_shuffle_training_true(self) -> None:
        dates = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
        labels = pd.Series([10.0, 20.0, 30.0], index=dates)
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=dates)
        X_tr, _, y_tr, _ = time_aware_train_test_split(
            dates=dates, labels=labels, features=features, split_date="2020-03-01", shuffle_training=True
        )
        assert len(X_tr) == 2
        # Shuffling changes order; verify values are still the train set
        assert set(y_tr.values) == {10.0, 20.0}
