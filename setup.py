from setuptools import setup, Extension
import pybind11
import os

tlx_include_dir = os.path.abspath('tlx')
shared_include_dir = os.path.abspath('shared')

ext_original = Extension(
    'lazy_tree_original',
    sources=['src/original/bindings.cpp'],
    include_dirs=[pybind11.get_include(), tlx_include_dir, shared_include_dir],
    language='c++'
)

ext_learned = Extension(
    'lazy_tree_learned',
    sources=['src/learned/bindings.cpp'],
    include_dirs=[pybind11.get_include(), tlx_include_dir, shared_include_dir],
    language='c++'
)

setup(
    name='lazy_tree_benchmark',
    version='2.0',
    description='Unified Lazy Search Tree Benchmarking Suite',
    ext_modules=[ext_original, ext_learned],
)
