# How to import a module in Python

- `import` 模块搜索的路径是根据 `sys.path` 来做的
  - `sys.path` 是一个列表，可以 `append` 

```python
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()).absolue())
```

- 相对导入
  - 不能直接运行包含相对导入的py文件！
