import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gh-tools-gradient", # Replace with your own username
    version="0.0.1",
    author="Ouwen Huang",
    author_email="ouwen.huang@duke.edu",
    description="Gradient Health training tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gradienthealth/gh-tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['awscli>=1.18.223', 'tensorflow>=2.4.1', 'tfa-nightly']
)
