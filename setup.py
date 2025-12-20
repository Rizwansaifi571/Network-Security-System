from setuptools import find_packages, setup

def get_requirements():
    req_lst = []
    try:
        with open('requirements.txt') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    req_lst.append(requirement)
    except FileNotFoundError:
        print('requiremnets.txt file not found')

    return req_lst

setup(
    name = "Network-Security-System", 
    version= "0.0.1", 
    author= "Mohd Rizwan", 
    author_email="rizwansaifi2614@gmail.com", 
    packages= find_packages(), 
    install_requires = get_requirements()
)