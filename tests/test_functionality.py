from ngsxditto import *
import pytest

@pytest.mark.parametrize("number", [i for i in range(10)])
def test_addition(number):
    assert add_one(number) == number+1

if __name__ == "__main__":
    test_addition(2)
    test_addition(3)