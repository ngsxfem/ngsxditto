# You can write your code in files like these.



# class BaseClass:
#     ...

def add_one(number: int = 0) -> int: # type hinting: I think Johann showed us this some time ago, this may be nice for us?
    """Return the increment of a number.

    :param number: A number you want to increase, defaults to 0.
    :type number: int

    :return: The incremented value.
    :rtype: int
    """
    return number+1
