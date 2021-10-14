from setuptools import setup

with open('README.md') as f:
    readme = f.read()

setup(
    name='SEED-Encoder',
    version='0.1.0',
    description='Less is More: Pre-train a Strong Text Encoder for Dense Retrieval Using a Weak Decoder',
    url='https://github.com/microsoft/SEED-Encoder',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
   license="MIT",
   long_description=readme,
   install_requires=[
        'pytrec-eval',
        'faiss-cpu',
        'wget',
        'scikit-learn',
        'pandas',
        'tensorboardX',
        'tokenizers',
        'tqdm',
        'six',
        'requests'
    ],
)
