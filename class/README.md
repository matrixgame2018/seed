### 1.代码结构

```
 {repo_root}
  ├── models	//模型文件夹
  ├── utils		//一些函数包
  |   ├── eval.py		// 求精度
  │   ├── misc.py		// 模型保存，参数初始化，优化函数选择
  │   ├── radam.py
  │   └── ...
  ├── args.py		//参数配置文件
  ├── build_net.py		//搭建模型
  ├── dataset.py		//数据批量加载文件
  ├── preprocess.py		//数据预处理文件，生成坐标标签
  ├── train_2.py		//训练运行文件
  ├── transform.py		//数据增强文件
```


## 2.运行步骤


1. 调整好运行split.py的路径参数之后可以将数据按csv文件的标号分类到data文件夹中。
2. 运行train_2.py。
3. 运行test_2.py文件，记得检查result.csv是否为csv逗号分值文件，不带utf8编码。
