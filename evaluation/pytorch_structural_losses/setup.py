from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Python interface
setup(
    name='PyTorchStructuralLosses',
    version='0.1.0',
    install_requires=['torch'],
    packages=['StructuralLosses'],
    package_dir={'StructuralLosses': 'StructuralLosses'},
    ext_modules=[
        CUDAExtension(
            name='StructuralLossesBackend',
            include_dirs=['./'],
            sources=[
                'pybind/bind.cpp',
                'src/structural_loss.cpp',
                'src/approxmatch.cu',
                'src/nndistance.cu',
            ],
            extra_compile_args=['-g']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
