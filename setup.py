#!/usr/bin/env python

from setuptools import setup

setup(name       ='ghsolver',
      version    ='0.1',
      description='A python library for Gauss-Helmert-Model based estimation',
      url        ='http://github.com/mshicom/pyGHSolver',
      author     ='Kaihong Hunag',
      author_email='kaihong.huang11@gmail.com',
      license    ='MIT',
      packages   =['ghsolver'],
      install_requires=[ ],
      include_package_data=True,
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)

