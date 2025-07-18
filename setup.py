from setuptools import setup, find_packages

setup(
    name="gen_dual",
    version="0.1.0",
    author="Luka Lavš",
    author_email="lavslukal@gmail.com", 
    description="An implementation of generalized dual numbers with support for arbitrary percision and vectorization.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="automatic differentiation, generalized dual numbers, numpy, mpmath",
    url="https://github.com/LukaLavs/Automatic-Differentiation",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "mpmath",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
