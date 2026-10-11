import pytest
from tensorphysics import FieldDescriptor, FieldExchangeError, FieldStore


def descriptor(shape=(2, 3)):
    return FieldDescriptor(
        name="fluid.velocity", producer="TensorLBM", units="m/s",
        location="cell_center", shape=shape, dtype="float64",
    )


class ArrayLike:
    def __init__(self, shape):
        self.shape = shape


def test_register_publish_and_read_preserves_reference_and_metadata():
    store = FieldStore()
    spec = descriptor()
    data = ArrayLike((2, 3))
    store.register(spec)
    value = store.publish("fluid.velocity", data, producer="TensorLBM", step=4, time=0.2)
    read = store.read("fluid.velocity", current_step=5, max_age_steps=1)
    assert read is value
    assert read.data is data
    assert read.version == 1
    assert read.descriptor.units == "m/s"


def test_identical_registration_is_idempotent_but_conflicts_fail():
    store = FieldStore()
    store.register(descriptor())
    store.register(descriptor())
    with pytest.raises(FieldExchangeError, match="different metadata"):
        store.register(FieldDescriptor("fluid.velocity", "other", "m/s", "cell_center"))


def test_only_declared_producer_can_publish():
    store = FieldStore()
    store.register(descriptor())
    with pytest.raises(FieldExchangeError, match="owned by producer"):
        store.publish("fluid.velocity", object(), producer="TensorDEM", step=0, time=0.0)


def test_shape_mismatch_is_rejected_when_array_exposes_shape():
    store = FieldStore()
    store.register(descriptor())
    with pytest.raises(FieldExchangeError, match="shape mismatch"):
        store.publish("fluid.velocity", ArrayLike((3, 2)), producer="TensorLBM", step=0, time=0.0)


def test_stale_field_and_unpublished_field_are_reported():
    store = FieldStore()
    store.register(descriptor())
    with pytest.raises(FieldExchangeError, match="not been published"):
        store.read("fluid.velocity")
    store.publish("fluid.velocity", ArrayLike((2, 3)), producer="TensorLBM", step=2, time=0.1)
    with pytest.raises(FieldExchangeError, match="stale"):
        store.read("fluid.velocity", current_step=5, max_age_steps=1)


def test_publication_time_must_be_monotonic():
    store = FieldStore()
    store.register(descriptor())
    store.publish("fluid.velocity", ArrayLike((2, 3)), producer="TensorLBM", step=2, time=0.1)
    with pytest.raises(FieldExchangeError, match="move backwards"):
        store.publish("fluid.velocity", ArrayLike((2, 3)), producer="TensorLBM", step=1, time=0.05)



def test_read_at_time_interpolates_scalar_samples_and_keeps_metadata():
    store = FieldStore()
    store.register(FieldDescriptor(
        name="fluid.pressure", producer="TensorLBM", units="Pa",
        location="cell_center",
    ))
    store.publish("fluid.pressure", 10.0, producer="TensorLBM", step=0, time=0.0)
    store.publish("fluid.pressure", 30.0, producer="TensorLBM", step=2, time=2.0)
    value = store.read_at_time("fluid.pressure", 0.5)
    assert value.data == pytest.approx(15.0)
    assert value.time == pytest.approx(0.5)
    assert value.step == 2
    assert value.descriptor.units == "Pa"


def test_read_at_time_returns_exact_sample_without_interpolation():
    store = FieldStore()
    store.register(FieldDescriptor("x", "solver", "m", "node"))
    store.publish("x", 1.0, producer="solver", step=0, time=0.0)
    exact = store.publish("x", 2.0, producer="solver", step=1, time=1.0)
    assert store.read_at_time("x", 1.0) is exact


@pytest.mark.parametrize("requested", [-0.1, 1.1])
def test_read_at_time_forbids_extrapolation(requested):
    store = FieldStore()
    store.register(FieldDescriptor("x", "solver", "m", "node"))
    store.publish("x", 1.0, producer="solver", step=0, time=0.0)
    store.publish("x", 2.0, producer="solver", step=1, time=1.0)
    with pytest.raises(FieldExchangeError, match="cannot extrapolate"):
        store.read_at_time("x", requested)


def test_read_at_time_enforces_maximum_interpolation_gap():
    store = FieldStore()
    store.register(FieldDescriptor("x", "solver", "m", "node"))
    store.publish("x", 1.0, producer="solver", step=0, time=0.0)
    store.publish("x", 2.0, producer="solver", step=10, time=10.0)
    with pytest.raises(FieldExchangeError, match="exceeds max_gap"):
        store.read_at_time("x", 5.0, max_gap=2.0)


def test_read_at_time_reports_non_interpolable_data():
    store = FieldStore()
    store.register(FieldDescriptor("x", "solver", "object", "node"))
    store.publish("x", object(), producer="solver", step=0, time=0.0)
    store.publish("x", object(), producer="solver", step=1, time=1.0)
    with pytest.raises(FieldExchangeError, match="does not support linear interpolation"):
        store.read_at_time("x", 0.5)


def test_clear_values_also_clears_interpolation_history():
    store = FieldStore()
    store.register(FieldDescriptor("x", "solver", "m", "node"))
    store.publish("x", 1.0, producer="solver", step=0, time=0.0)
    store.publish("x", 2.0, producer="solver", step=1, time=1.0)
    store.clear_values()
    with pytest.raises(FieldExchangeError, match="not been published"):
        store.read_at_time("x", 0.5)
