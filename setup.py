from setuptools import setup, find_packages

setup(
    name='your_package_name',            # 包名称
    version='0.1.0',                     # 包版本
    author='Your Name',                  # 作者名称
    author_email='your_email@example.com',  # 作者邮箱
    description='A short description of the package',  # 简短描述
    long_description=open('README.md').read(),  # 长描述
    long_description_content_type='text/markdown',  # 长描述的格式
    url='https://github.com/username/your_package',  # 项目的 URL
    packages=find_packages(),            # 包含的包
    classifiers=[                        # 分类器，描述包的性质
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',            # Python 版本要求
    install_requires=[                   # 依赖包
        'numpy',
        'requests',
    ],
)
