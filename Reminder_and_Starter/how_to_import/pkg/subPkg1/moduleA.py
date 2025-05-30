# import moduleB 

# 相对导入 不应该直接执行这个py文件，而是应该在其他文件中导入这个模块
from . import moduleB 
from ..subPkg2 import moduleX
