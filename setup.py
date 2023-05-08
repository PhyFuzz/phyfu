#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read()

test_requirements = [ ]

setup(
    author="PhyFu",
    author_email='phyfu@proton.me',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    description="A Fuzzer for Modern Physics Simulation Engines",
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='phyfu',
    name='phyfu',
    packages=find_packages(include=['phyfu', 'phyfu.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/PhyFuzz/phyfu',
    version='0.1.0',
    zip_safe=False,
)
