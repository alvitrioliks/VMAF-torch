from setuptools import setup

setup(
    name="vmaf_torch",
    version="1.0",
    url="https://github.com/alvitrioliks/VMAF-torch",
    author="Kirill Aistov",
    author_email="kirill.aistov1@huawei.com",
    description="VMAF Reimplementation in PyTorch",
    packages=[
        "vmaf_torch",
    ],
    install_requires=["torch>=2.0.0", "yuvio"],
)
