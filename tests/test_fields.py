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
