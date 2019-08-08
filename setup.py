from setuptools import setup, find_packages

setup(
    name='floating_gate_measurements',
    version='0.1',
    description="Experiment code for floating gate measurements",
    url='https://github.com/QNLSydney/floating_gate_measurements',
    packages=find_packages(),
    install_requires=[
        'qcodes>=0.4',
        'numpy',
        'matplotlib'
    ],
    python_requires='>3.6'
)
