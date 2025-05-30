import sys

print(sys.path) 
# a list where the Python interpreter will look for modules to import

from pathlib import Path

print(Path().cwd()) 
# current working directory



from pkg.subPkg1 import moduleA

