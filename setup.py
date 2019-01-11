from distutils.core import setup

setup(
   name='mgridoptimizer',
   version='1.1.7',
   description='Model and optimize a microgrid/grid connected DER system',
   author='Brian Joseph',
   author_email='briandjoseph@gmail.com',
   packages=['mgridoptimizer', 'mgridoptimizer.modules', 'mgridoptimizer.data',],
   install_requires=['pandas', 'matplotlib'], #external packages as dependencies
)
