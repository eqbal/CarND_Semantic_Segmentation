import helper
import warnings
from distutils.version import LooseVersion
from fcn import FCN
import project_tests as tests



def run():
    fcn = FCN()
    fcn.run_tests()
    fcn.run()

if __name__ == '__main__':
    run()
