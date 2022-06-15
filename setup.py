from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

VERSION = '0.1.1'
DESCRIPTION = 'Machine Learning scripts that will quicken the modelling and data analysis process'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="mb_scripts",
    version=VERSION,
    author="Sathya Krishnan Suresh",
    author_email="<satyakrishnan.s@pec.edu>",
    description=DESCRIPTION,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "tensorflow"
    ],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'

    keywords=['python', 'mb_scripts','machine learning','data science','data analysis','deep learning'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)