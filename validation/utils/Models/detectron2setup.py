import sys
import os
import distutils.core
import subprocess

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}")
        print(f"Error: {str(e)}")
        raise

def setup_detectron2():

    if not os.path.exists("detectron2"):
        install_package("pyyaml==5.1")
        subprocess.check_call(["git", "clone", "https://github.com/facebookresearch/detectron2"])
    
        dist = distutils.core.run_setup("./detectron2/setup.py")
        
        for requirement in dist.install_requires:
            install_package(requirement.strip("'"))

    sys.path.insert(0, os.path.abspath('./detectron2'))
    
    print("Detectron2 setup completed successfully!")

setup_detectron2()