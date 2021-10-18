from setuptools import setup

setup(
    name='feature-vectors',
    version='0.1.1',
    description='Explain the output of a tree based model using importance of features and their semantic relationship',
    long_description='FeatureVectors is a unified conceptual explanation algorithm designed specifically for tabular data. '
                     'This library can be used either to explain a tabular dataset or to explain an existing tree-based '
                     'machine learning model that are trained on tabular datasets.',
    url='https://github.com/amiratag/feature-vectors',
    author='Amirata Ghorbani, Dina Berenbaum',
    author_email='amirataghorbani.tw@gmail.com',
    license='MIT',
    packages=['fvecs'],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7'],
    install_requires=[
        'scikit-learn == 0.23.2',
        'pandas == 1.0.3',
        'numpy == 1.19.4',
        'plotly == 4.14.1'
    ],
    python_requires='>=3.6'
)

