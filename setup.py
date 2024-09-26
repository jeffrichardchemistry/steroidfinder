from setuptools import setup, find_packages

with open("README.md", 'r') as fr:
	description = fr.read()

setup(
    name='MSteroid',
    version='1.0.0',
    url='https://github.com/vitor-mendes-iq/MSteroid_Finder',
    license='GNU GPL',
    author='Vitor Mendes',
    author_email='vitor.mendes.ag@gmail.com',
    keywords='Mass-Spectrometry, Similarity-Alogirthm, Machine-Learning',
    description='A web application to study steroids.',
    long_description = description,
    long_description_content_type = "text/markdown",
    packages=['MSteroid'],
    install_requires=['pandas', 'numpy', 'streamlit'],
	classifiers = [
		'Intended Audience :: Developers',
		'Intended Audience :: End Users/Desktop',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: BSD License',
		'Natural Language :: English',
		'Operating System :: Unix',
		'Operating System :: Microsoft :: Windows',
		'Operating System :: MacOS',
		'Topic :: Scientific/Engineering :: Artificial Intelligence',
		'Programming Language :: Python',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.8']
)
