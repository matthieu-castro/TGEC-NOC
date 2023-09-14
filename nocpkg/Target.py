class Target:
    name = ''
    value = 0.0
    sigma = -1.0
    is_input = False

    def __init__(self, elem):
        """
        Class representing a targeted value
        :param elem: Instance of JSON element from the JSON 'target' field, read in the input JSON file.
        :type elem: <Object>
        """
        self.elem = elem
        self.read_target()

    def __str__(self):
        """
        Return a string describing the Target as "name = value +/- sigma"
        :return: outputs the Target
        :rtype: str
        """

        return f"{self.name:8s} = {self.value:8f} +/- {self.sigma:8f}"

    def __repr__(self):
        """
        Defines the official representation of the Target
        :return: outputs the value of <Target.__str__>
        :rtype: str
        """

        return self.__str__()

    def copy(self):
        """
        Create a copy of the <noc.Parameter> instance

        :return: a new instance of the Parameter object with identical attributes
        :rtype: <noc.Parameter>
        """
        clas = self.__class__
        new_target = clas.__new__(clas)
        new_target.__dict__.update(self.__dict__)
        return new_target

    def read_target(self):
        """
        Extract information relative to a given target from the JSON file
        """
        self.name = self.elem.get('name')
        self.value = float(self.elem.get('value'))
        read_sigma = self.elem.get('sigma')
        if type(read_sigma) == str and read_sigma.strip() == 'input':
            self.sigma = -1.0
            self.is_input = True
        else:
            self.sigma = float(read_sigma)