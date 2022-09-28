from setuptools import setup, find_packages


setup(
    name="project_metrics",
    version="1.0",
    packages=find_packages(),
    install_requires=[
    	"numpy",
    	"pandas",
    	"keras",
		"tensorflow",
    	"seaborn",
    	"sklearn",
    	"scikit-plot",
    	"matplotlib",
    ]
)
