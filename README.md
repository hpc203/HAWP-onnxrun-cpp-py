# HAWP-onnxrun-cpp-py
使用ONNXRuntime部署HAWP线框检测，包含C++和Python两个版本的程序
看到CVPR 2020里有一篇文章《Holistically-Attracted Wireframe Parsing》，
它是检测图片里的直线的，我觉得挺有意思的，就编写了这套程序的。起初打算使用opencv部署的，
可是opencv的dnn模块读取onnx文件出错了，无赖只能使用onnxruntime做部署，
依然是包含C++和Python两种版本的程序。

由于onnx文件25M，无法直接上传到github仓库，因此onnx文件需要从百度云盘下载，
链接：https://pan.baidu.com/s/1nriCghu3dzz4U94pC61_0Q 
提取码：9jcd
