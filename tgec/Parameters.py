import os.path
import re

from nocpkg.utils import NOCError


def convert_value(value):
    if value.isdigit():  # if the value is an integer
        return int(value)
    value = re.sub('[dD]', 'e', value)  # replace 'd' ou 'D' of Fortran float by 'e'
    try:
        return float(value)  # try casting to a float
    except ValueError:
        if value.lower() == 't':
            return True
        elif value.lower() == 'f':
            return False
        else:
            return value


class Parameters:

    # Create a regular expression to find each parameter in the .com file
    pattern = r'\b(\w+)\s*=(\s*)([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?|[Tt]|[Ff])\b'

    def __init__(self, name):
        """
        Represents the inputs parameters of the TGEC model

        :param: name of the model
        """

        self.model_name = name
        self.com_file = name + '.com'

        # Parameters are stored in a dictionary
        self.params_dict = {}
        if os.path.exists(self.com_file):  # read the .com file if it exists,
            self.read_params()
        else:
            raise NOCError("No .com file has been found")

        # Initial mass
        self.mass = self.params_dict['GMS']

        # Start age
        if self.params_dict['IZAMS'] == 0:
            self.start = 'pms'
        else:
            self.start = 'ms'

        # Initial composition
        self.y0 = self.params_dict['YINI']
        if self.params_dict['FESURHINI'] == 9:
            self.x0 = (1 - self.y0)/(1 + self.params_dict['ZOXINI'])
            self.z0 = 1 - self.x0 - self.y0
        else:
            self.z0 = 0.0181*(10 ** self.params_dict['FESURHINI'])
            self.x0 = 1 - self.y0 - self.z0
        self.zox0 = self.z0 / self.x0

        # Diffusion
        if self.params_dict['IDIFCC'] == 0:
            self.diffusion = False
        else:
            self.diffusion = True

    def read_params(self):
        """
        Reads the model physics input parameters in the .com file
        """

        print(f"Reading {self.com_file}... ", end='')
        with open(self.com_file, 'r') as f:
            content = f.read()

        self.model_name = content[:8]
        # Look for the matches with the pattern in the file content
        matches = re.findall(self.pattern, content)
        # Create the dictionary from the matches
        for match in matches:
            key = match[0]
            value = match[2]
            self.params_dict[key] = convert_value(value)

        print('Done')

    def update_com(self, age_model):
        """
        Update the .com file with the model parameters
        :param age_model: age of the optimized model
        """
        print(f"Updating {self.com_file}... ", end='')
        with open(self.com_file, 'r+') as f:
            content = f.read()
            matches = re.findall(self.pattern, content)  # List the parameters in .com file
            # update of the model name
            self.model_name = f"e{int(self.params_dict['GMS']*100):03d}" \
                              f"{'p' if self.params_dict['FESURHINI']>=0 else 'm'}" \
                              f"{int(abs(self.params_dict['FESURHINI']*100)):03d}"
            content = self.model_name + content[8:]
            # update of the printed model number corresponding to the optimized model.
            # Needed for the frequencies calculation
            deltat = min(self.params_dict['DZEITM'], self.params_dict['DZEIT'])
            self.params_dict['IPRN'] = round(age_model/(deltat/(60*60*24*365.25)))
            self.params_dict['NZMOD'] = self.params_dict['IPRN'] + 1
            # update the suffix of evolution files
            content = content[:9] + f"00001-{self.params_dict['NZMOD']:05d}" + content[20:]
            # update of the parameters of the model
            for match in matches:
                key = match[0]
                spaces = len(match[1])
                value = match[2]
                # Find if the parameter has been updated
                if convert_value(value) != self.params_dict[key]:
                    # Pattern to find the parameter in the file
                    pattern = re.compile(f"{key:<{len(key)}}={value:>{spaces+len(value)}}")
                    # Text with updated value to substitute
                    new_value = self.params_dict[key]
                    subs = f"{key:<{len(key)}}={new_value:>{spaces+len(str(new_value))}}"
                    # Substitution of the updated value in the .com file
                    content = pattern.sub(subs, content)
            f.seek(0)
            f.truncate()
            f.write(content)

        print('Done')


                    
