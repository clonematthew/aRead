import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aRead",
    version="0.34",
    author="Matt Cusack",
    author_email="cusackmt@cardiff.ac.uk",
    description="Package for opening AREPO snapshot files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages()
)