from setuptools import setup

setup(
    name='bus_app',
    packages=['bus_app'],
    include_package_data=True,
    install_requires=[
        'flask',
        'pandas'
    ],
)