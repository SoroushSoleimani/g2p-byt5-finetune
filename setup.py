from setuptools import setup, find_packages

setup(
    name="persiang2p",
    version="1.0.0",
    description="Fast, High-Accuracy Persian Grapheme-to-Phoneme (G2P) Engine for TTS and Speech Processing",
    author="Soroush Soleimani",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Text Processing :: Linguistic",
    ],
)
