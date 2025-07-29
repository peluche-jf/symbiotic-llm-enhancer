from setuptools import setup, find_packages

setup(
    name="symbiotic-llm-enhancer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.0',
        'transformers>=4.5.0',
        'cryptography>=3.4.7',
        'ray>=1.0.0',
        'aiohttp>=3.7.4',
    ],
    author="Peluche & Sol.",
    author_email="peluche.jf@icloud.com",
    description="A symbiotic enhancement system for LLMs",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/peluch-3/symbiotic-llm-enhancer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
