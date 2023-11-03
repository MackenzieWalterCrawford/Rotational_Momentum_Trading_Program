from RectangularPrism import RectangularPrism
import pytest


class TestRectangularPrism:
    # Tests that RectangularPrism initializes with valid float values
    def test_valid_initialization(self):
        rp = RectangularPrism(1.0, 2.0, 3.0, 4.0)
        assert rp.length == 1.0
        assert rp.height == 2.0
        assert rp.width == 3.0
        assert rp.density == 4.0

    # Tests that RectangularPrism calculates volume and weight with valid float values
    def test_valid_volume_and_weight_calculation(self):
        rp = RectangularPrism(1.0, 2.0, 3.0, 4.0)
        assert rp.volume() == 6.0
        assert rp.weight() == 24.0

    # # Tests that the volume calculation is correct
    # def test_volume_calculation(self):
    #     rp = RectangularPrism(2.0, 3.0, 4.0, 1.0)
    #     assert rp.volume() == 24.0

    # # Tests that the weight calculation is correct
    # def test_weight_calculation(self):
    #     rp = RectangularPrism(2.0, 3.0, 4.0, 1.0)
    #     assert rp.weight() == 24.0

    # Tests that RectangularPrism raises ValueError when initialized with negative values
    def test_negative_initialization(self):
        with pytest.raises(ValueError):
            RectangularPrism(-1.0, -2.0, -3.0, -4.0)

    # # Tests that the class raises TypeError for non-float values
    # def test_invalid_initialization(self):
    #     with pytest.raises(TypeError):
    #         rp = RectangularPrism('a', 3.0, 4.0, 1.0)

    # Tests that RectangularPrism raises TypeError when initialized with non-float values
    def test_non_float_initialization(self):
        with pytest.raises(TypeError):
            RectangularPrism("a", "b", "c", "d")

    # Tests that RectangularPrism calculates volume and weight with zero values
    def test_zero_volume_and_weight_calculation(self):
        rp = RectangularPrism(0.0, 0.0, 0.0, 0.0)
        assert rp.volume() == 0.0
        assert rp.weight() == 0.0

    # Tests that RectangularPrism calculates volume and weight with large and small float values
    def test_large_and_small_float_values(self):
        rp = RectangularPrism(1e-10, 1e10, 1e-10, 1e-10)
        assert rp.volume() == 1e-30
        assert rp.weight() == 1e-20

    #     rp = RectangularPrism(1e10, 1e-10, 1e10, 1e10)
    #     assert rp.volume() == 1e-30
    #     assert rp.weight() == 1e30
