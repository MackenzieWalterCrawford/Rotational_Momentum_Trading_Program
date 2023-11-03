class RectangularPrism:
    def __init__(self, length: float, height: float, width: float, density: float):
        self.length = self.is_valid_number(length)
        self.height = self.is_valid_number(height)
        self.width = self.is_valid_number(width)
        self.density = self.is_valid_number(density)

        self.calculate_volume()
        self.caluculate_weight()

    def is_valid_number(self, value):
        if not isinstance(value, float):
            raise TypeError("Only floats are allowed")
        elif value < 0.0:
            raise ValueError("Attributes of the rectangular prism should be positive")
        else:
            return value

    def caluculate_weight(self):
        self._weight = self.height * self.length * self.width * self.density

    def calculate_volume(self):
        self._volume = self.height * self.length * self.width

    def volume(self):
        return self._volume

    def weight(self):
        return self._weight
