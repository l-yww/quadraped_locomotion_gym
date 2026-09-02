from setuptools import find_packages
from distutils.core import setup

setup(
    name="wheel_legged_gym",
    version="1.0.0",
    author="Hao Wang",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="wanghao@cowarobot.com",
    description="Isaac Gym environments for Wheel Legged Robots",
    install_requires=[
        "isaacgym",
        "matplotlib",
        "tensorboard",
        "setuptools==59.5.0",
        "numpy==1.23.5",
        "GitPython",
        "onnx",
        "torchsummary",
        "opencv-python",
        "opencv-python-headless",
        "warp-lang",
        "tqdm",
        "GitPython",
        "onnx",
        "torchsummary",
        "IPython"
    ],
)