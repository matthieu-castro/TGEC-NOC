import os
import subprocess


class RunModel:

    def __init__(self, name):
        self.name = name
        # tgec_path = "/data/Programmes/TGEC"
        #tgec_path = "~/Programmes/TGEC"
        tgec_path = os.environ['TGEC']
        # pulse_path = "/data/Programmes/Pulse/pulse2-1.12.5"
        # pulse_path = "~/Programmes/Pulse/pulse2-1.12.5"
        pulse_path = os.environ['PULSE']
        self.calc_dir = tgec_path + "/exec/"
        self.data_dir = tgec_path + "/Data/"
        self.tgec_exec = tgec_path + "/CodeF90/starexec"
        self.pulse_exec = pulse_path + "/pulse-1.12"
        self.com_file = name + '.com'
        self.tgec_log = name + '_tgec.log'
        self.pulse_log = name + '_pulse.log'

    def run_tgec(self, verbose=False, debug=False, log=False):
        """
        Run the tgec executable. If needed, store the standard output to log file
        :param verbose: if true, print output on standard output (default=false)
        :param debug: if true, run in debug mode (default=false)
        :param log: if true, store output to log file (default=false)
        """
        if log:
            # if verbose:
            #     cmd = f"{self.exec} < {self.com_file} | tee {self.log_file}"
            # else:
            cmd = f"{self.tgec_exec} < {self.com_file} > {self.tgec_log}"
        else:
            cmd = f"{self.tgec_exec} < {self.com_file}"
        os.system(cmd)

        # s = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        # output, err = s.communicate()
        # # return output

    def load_tables(self):
        """
        Create a symbolic link to the tables of opacities and equations of state
        """
        path = ["mhd/tabeos.bin?", "Opal2005/EOS5_0?z?x", "Opal2001/IEOS0?z?x", "paquette/*.dat",
                "opacites/opal09/A09photo", "opacites/opal09/Asp09hz"]
        cmd_base = f"ln -s {self.data_dir}"

        for p in path:
            cmd = cmd_base + p + " ."
            os.system(cmd)

    def run_pulse(self, verbose=False, log=False):
        """
        Run the pulse executable
        :param verbose: if true, print output on standard output (default=false)
        :param log: if true, store output to log file (default=false)
        """
        if log:
            cmd = self.pulse_exec + ' > ' + self.pulse_log
        else:
            cmd = self.pulse_exec

        os.system(cmd)

        # Copy files in the model directory
        cmd = f"mkdir {self.name}/freqs && mv pulse.* {self.pulse_log} {self.name}/freqs && " \
              f"rm model.out_p"

        os.system(cmd)
