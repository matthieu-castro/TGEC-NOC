import os
import subprocess

import tgec.constants


class RunModel:

    def __init__(self, name):
        self.name = name
        # tgec_path = "/data/Programmes/TGEC"
        # tgec_path = "~/Programmes/TGEC"
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

    def run_pulse(self, log=False):
        """
        Run the pulse executable
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

    def run_adipls(self):
        """
        Run the ADIPLS executable
        """

        # Clean former files
        os.system(f"rm -f {self.name}.agsm {self.name}.ssm {self.name}.ef {self.name}-adipls.log")

        # os.system(f"runadipls.pl {name}.amdl {center_name}.adipls > /dev/null")
        self.write_infile()
        template = "TEMPLATE"
        subprocess.run(["cp", f"{self.name}.adipls", template])

        with open(template, "r") as parin, open(self.name + ".par", "w") as parout:
            for line in parin:
                if not line.lstrip().startswith("-1") and any(char.isdigit() for char in line) and "''" not in line:
                    line = line.replace(" '", "'")
                    parout.write("2 '" + self.name + ".amdl'@\n" if line.startswith("2 '") else
                                 "9 '" + self.name + ".alog'@\n" if line.startswith("9 '") else
                                 "10 ''\n" if line.startswith("10 '") else
                                 "11 '" + self.name + ".agsm'@\n" if line.startswith("11 '") else
                                 "15 '" + self.name + ".ssm'@\n" if line.startswith("15 '") else
                                 "16 '" + self.name + ".fsm'@\n" if line.startswith("16 '") else
                                 "4 '" + self.name + ".ef'@\n" if line.startswith("11 '") else line)
                else:
                    parout.write(line)

        # print("-------------------------------------------------\n")
        # print("Running adipls\n")
        # print("-------------------------------------------------\n")
        # print(f"adipls.c.d {self.name}.par")
        subprocess.run(["adipls.c.d", self.name + ".par", "/dev/null"], capture_output=True)
        os.remove(self.name + ".ssm")
        # print(f"rm -f {template}")

        subprocess.run(["rm", "-f", template])

        # Copy files in the model directory
        cmd = (f"mkdir {self.name}/freqs && mv {self.name}.adipls {self.name}.agsm {self.name}.amdl {self.name}.par "
               f"{self.name}/freqs")
        os.system(cmd)

    def write_infile(self):
        """
        Write the adipls input file
        """
        infile = f"{self.name}.adipls"

        # nfmode = int(self.fparams.amde)
        # irotkr = int(self.fparams.rotkr)
        # igm1kr = int(self.fparams.gm1kr)

        if os.path.exists(infile):
            os.remove(infile)

        amde = f"{self.name}.amde"
        agsm = f"{self.name}.agsm"
        ssm = f"{self.name}.ssm"
        log = f"{self.name}-adipls.log"
        miss = f"{self.name}.ssm.miss"
        amdl = f"{self.name}.amdl"
        rkr = f"{self.name}.rkr"
        gkr = f"{self.name}.gkr"

        with open(infile, 'w') as f:
            f.write(f"2  {amdl}   @\n")
            # if self.fparams.amde: f.write(f'4  {amde}    @\n')
            f.write(f'9  {log}   @\n')

            # if nn == 1:
            f.write('10  \'0\'   @\n')
            # else:
            #     f.write(f'10 {miss}   @\n')

            f.write(f'11 {agsm}   @\n')
            # if self.fparams.rotkr: f.write(f'12 {rkr}   @\n')
            # if self.fparams.gm1kr: f.write(f'13 {gkr}   @\n')
            f.write(f'15 {ssm}   @\n')

            f.write('-1 \'\'   @\n')
            f.write('\ncntrd,\n')
            f.write('mod.osc.cst.int.out.dgn     @\n')

            f.write('\nmod:\n')
            f.write('ifind,xmod,imlds,in,irname,nprmod,\n')
            f.write(',,,,,,,   @\n')
            f.write('ntrnct,ntrnsf,imdmod,\n')
            f.write(',,,,,,,,,,,,,,,,,,,,,, @\n')

            f.write('\nosc:\n')
            f.write('el,nsel,els1,dels,dfsig1,dfsig2,nsig1,nsig2,\n')
            # if nn == 1: f.write(f'0,{self.fparams.nsel},{self.fparams.lmin},{self.fparams.dels},,,,,,,,,,,,,,,,,,,,
            # ,,     @\n')
            f.write(f'0,4,0,1,,,,,,,,,,,,,,,,,,,,,,     @\n')
            # else:
            #     f.write(f',0,0,{self.fparams.dels},,,,,,,,,,,,,,,,     @\n')

            f.write('itrsig,sig1,istsig,inomde,itrds,\n')
            # if nn == 1:
            #     f.write(f'1,{self.fparams.sig1},,,,,,,,,,,,,,,   @\n')
            f.write(f'1,4.0,,,,,,,,,,,,,,,   @\n')
            f.write('dfsig,nsig,iscan,sig2,\n')
                # f.write(f'0,{self.fparams.nsig_},{self.fparams.iscan},{self.fparams.sig2},,,,,,,,,,,,,,,,,,,     @\n')
            f.write(f'0,2,250,3000.0,,,,,,,,,,,,,,,,,,,     @\n')
            # else:
            #     f.write('4,,6000,1,10,,,,,,,,   @\n')
            #     f.write('dfsig,nsig,iscan,sig2,\n')
            #     f.write('0,,,,,,,,,,,,,,,,,,,,,,,,     @\n')

            f.write('eltrw1,eltrw2,sgtrw1,sgtrw2,\n')
            f.write(',,,,,,,,,,,,,,,,    @\n')

            f.write('\ncst:\n')

            f.write('cgrav,\n')
            f.write(f'{tgec.constants.ggrav}               @\n')

            f.write('\nint:\n')

            f.write('iplneq,iturpr,icow,alb,\n')
            f.write(',,,,,             @\n')
            f.write('istsbc,fctsbc,ibotbc,fcttbc,\n')
            # f.write(f'{self.fparams.istsbc_},{self.fparams.fscb},,,,,,,,,,,,,,  @\n')
            f.write(f'1,0.0,,,,,,,,,,,,,,  @\n')
            f.write('mdintg,iriche,xfit,fcnorm,eps,epssol,itmax,\n')
            # if self.fparams.remesh_ == 'r':
            #     f.write(f'{self.fparams.mdintg_},{int(self.fparams.iriche)},0.1,,1.0d-9,,15,,,,,,,,,,,  @\n')
            # else:
            #     f.write(f'{self.fparams.mdintg_},{int(self.fparams.iriche)},0.9,,,,,,,,,,,,,,,  @\n')
            f.write(f'1,1,0.9,,,,,,,,,,,,,,,  @\n')
            f.write('fsig,dsigmx,irsevn,\n')
            f.write(',,,,,,,,,,,,,,,,  @\n')

            f.write('\nout:\n')

            f.write('istdpr,nout,nprcen,irsord,iekinr,\n')
            # f.write(f'9,,,{self.fparams.irsord},{self.fparams.iekinr_},,,,,,,,,,,     @\n')
            f.write(f'9,,,20,1,,,,,,,,,,,     @\n')
            f.write('iper,ivarf,kvarf,npvarf,nfmode,\n')
            # f.write(f'1,{self.fparams.ivarf},2,0,{nfmode} @\n')
            f.write(f'1,1,2,0,0 @\n')
            f.write('irotkr,nprtkr,igm1kr,npgmkr,ispcpr,,\n')
            # f.write(f'{irotkr},1,{igm1kr},1,,,,,,,,,,     @\n')
            f.write(f'0,1,0,1,,,,,,,,,,     @\n')
            icaswn = 10 + 10000  # * self.fparams.istsbc_
            f.write('icaswn,sigwn1,sigwn2,frqwn2,frqwn2,iorwn1,iorwn2,frlwn1,frlwn2,ewnmax\n')
            f.write(f'{icaswn},,,,,,,,,,,,,,,,,,,,,,,,,,,,,   @\n')

            f.write('\ndgn:\n')

            f.write(',,,,,,,,,     @\n')
            f.write(',,,,,,,,,,,,    @\n')
