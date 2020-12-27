# hw4 16~20題
## 使用方法
1. 請先下載liblinear https://github.com/cjlin1/liblinear
2. 將hw4.py、train.dat以及test.dat移至liblinear/python目錄下
3. 執行hw4.py

## 題目簡介
利用liblinear實作課程中教的Validation觀念與技巧：
1. 不可用train與testing data選擇model
2. 使用validation data挑出model後，再用所有的training data在這個model下train出最好的hypothesis，會比單純validation loss最低的hypothesis要好
3. n-fold Cross validation