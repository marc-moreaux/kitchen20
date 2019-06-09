from distutils.core import setup

setup(
    name='Kitchen20',
    version='0.1',
    packages=['kitchen20', ],
    package_data={'': ['config.ini']},
    include_package_data=True,
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read())
