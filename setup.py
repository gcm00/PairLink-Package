from setuptools import setup, find_packages

setup(
    name='pairlink',
    version='0.0.1',
    description='Python package to test whether a pair of asset price series is suitable for pair-based investment strategies.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Anthony Gocmen',
    author_email='anthony.gocmen@gmail.com',
    url='https://www.developexx.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.11',
    install_requires=[
        'pandas',
        'numpy',
        'statsmodels'
    ],
    license='MIT'
)