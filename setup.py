import setuptools
from setuptools.command.install import install

VERSION = '0.0.1'

class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(tag, VERSION)
            sys.exit(info)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gh-tools-gradient", # Replace with your own username
    version=VERSION,
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
