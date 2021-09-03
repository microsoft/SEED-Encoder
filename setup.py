from setuptools import setup

with open('README.md') as f:
    readme = f.read()

setup(
   install_requires=[
        'transformers==2.3.0', 
        'pytrec-eval',
        'faiss-cpu',
        'wget',
        'scikit-learn',
        'pandas',
        'tensorboardX',
        'tqdm',
        'tokenizers',
        'six',
        'requests'
    ],
)
