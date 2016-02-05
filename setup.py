from setuptools import setup
from setuptools import find_packages

install_requires = [
    'Theano',
    'Keras==0.3.1'
]

setup(
      name='seq2seq',
      version='0.0.2',
      description='Sequence to Sequence Learning with Keras',
      author='Fariz Rahman',
      author_email='farizrahman4u@gmail.com',
      url='https://github.com/farizrahman4u/seq2seq',
      license='GNU GPL v2',
      install_requires=install_requires,
      packages=find_packages()
)
