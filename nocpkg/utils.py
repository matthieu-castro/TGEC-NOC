import re


class NOCError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


def get_instance(instance_list, attrib_name, value):
    """
    Return the instance of a given attribute name with a certain value in a list of instances
    :param instance_list: list of instances
    :param attrib_name: attribute name
    :param value: value of the attribute

    :return: instance
    """
    for instance in instance_list:
        if getattr(instance, attrib_name) == value:
            return instance
    return None
