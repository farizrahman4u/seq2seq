from setuptools import setup
from setuptools import find_packages

install_requires = [
    'numpy==1.9.2',
    'Theano==0.7.0',
    'Keras==0.2.0'
]

setup(
      name='Seq2seq',
      version='0.0.1',
      description='Sequence to Sequence Learning with Keras',
      author='Fariz Rahman',
      author_email='farizrahman4u@gmail.com',
      url='https://github.com/farizrahman4u/seq2seq',
      license='GNU GPL v2',
      install_requires=install_requires,
      packages=find_packages()
)