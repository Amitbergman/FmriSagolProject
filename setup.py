from setuptools import setup, find_packages


def main():
    setup(
        name='sagol',
        packages=find_packages(),
        version='0.1',
        description='fMRI analysis',
        include_package_data=True,
    )


if __name__ == '__main__':
    main()
