echo "# 5cls_F1-score" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Miracle-QCC/5cls_F1-score.git
git push -u origin main

## 用pandas读取csv文件，然后用神经网络进行5分类,要点：
- 数据有缺失值，所以用训练集的均值进行填充，对测试集和验证集ye是这样；
