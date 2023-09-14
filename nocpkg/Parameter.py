from nocpkg.utils import NOCError


class Parameter:
    name = ''
    value = 0.0
    step = 0.0
    rate = 0.0
    bounds = [0.0, 0.0]
    sigma = -1.0
    evol = False
    seismic = False
    is_input = False

    def __init__(self, elem=None):
        """
        Define the Parameter class, from information extracted from the JSON file

        :param elem: Instance of JSON element from the 'parameter' field, read in the input JSON file.
        :type elem: <Object>

        :member name: Name of the parameter.
        :mtype name: str

        :member value: Value of the parameter
        :mtype value: any

        :member step: Step used to vary this parameter in the Levenberg-Marquardt algorithm.
        :mtype step: float

        :member rate: Rate of modification of this parameter between two iterations of the
            Levenberg-Marquardt algorithm.
        :mtype rate: float

        :member bounds: Bound that limit the possible values of the parameter.
        :mtype bounds: list of float

        :member sigma: Standard deviation of this parameter. Defined for symmetry with the Target class.
            It is always set to -1 here
        :mtype sigma: float

        :member evol: True if the parameter control the evolution
        :mtype evol: bool

        :member seismic: True if the parameter control the modes
        :mtype seismic: bool

        :member is_input: This attribute is here to keep a symmetry with the Target class.
            It should always be False.
        :mtype is_input: bool
        """
        if elem is not None:
            self.elem = elem
            self.read_parameter()
            self.__check_boundaries()

    def __str__(self):
        """
        Return a string describing the Parameter as "name = value"
        :return: outputs the Parameter
        :rtype: str
        """

        return f"{self.name:10} = {self.value:8g}"

    def __repr__(self):
        """
        Defines the official representation of the Parameter
        :return: outputs the value of <Parameter.__str__>
        :rtype: str
        """

        return self.__str__()

    def __check_boundaries(self):
        """
        Checks that the given value of the parameter is inside the specified boundaries. If not, raises an error.
        """
        if self.value < self.bounds[0] or self.value > self.bounds[1]:
            raise NOCError(f"Error in NOC: the initial value for {self.name} is out of bounds")

    def copy(self):
        """
        Create a copy of the <noc.Parameter> instance

        :return: a new instance of the Parameter object with identical attributes
        :rtype: <noc.Parameter>
        """
        clas = self.__class__
        new_parameter = clas.__new__(clas)
        new_parameter.__dict__.update(self.__dict__)
        return new_parameter

    def read_parameter(self):
        """
        Extract information relative to a given parameter from the JSON file
        """
        self.name = self.elem.get('name')
        self.value = float(self.elem.get('value'))
        self.step = float(self.elem.get('step'))
        self.rate = float(self.elem.get('rate'))
        self.bounds = [float(self.elem.get('bounds')[0]), float(self.elem.get('bounds')[1])]