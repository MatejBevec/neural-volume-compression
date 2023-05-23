from setuptools import setup, find_packages

setup(
    name='neural-volume-compression',
    version='0.1.0',
    description='A brief description of your package',
    author='Matej Bevec',
    author_email='matejbevec98@gmail.com',
    url='https://github.com/MatejBevec/neural-volume-compression',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'torch',
        'gdown',
        'matplotlib',
        'pyvista',
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)