from PySide2.QtWidgets import QApplication, QMessageBox,QWidget, QVBoxLayout, QFileDialog, QApplication,QLineEdit
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import Signal, Slot
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use({'figure.figsize':(25,20)})
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import model_selection
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve,auc,roc_auc_score
import scipy
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers
from keras import losses
#from keras import metrics
from keras import optimizers
from keras import backend as K

class Stats:

    def __init__(self):
        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
##导入建立好的ui
        self.ui = QUiLoader().load('可视化.ui')
##导入文件显示的button
        self.ui.import_text.clicked.connect(self.handleCalc_import_text)
        self.ui.layout = QVBoxLayout()
        self.ui.layout.addWidget(self.ui.import_text)
        self.ui.setLayout(self.ui.layout)
##导入数据分析的button
        self.ui.tongji.clicked.connect(self.handleCalc_tongji)
        self.ui.cor.clicked.connect(self.handleCalc_cor)
        self.ui.corr_heat.clicked.connect(self.handleCalc_corr_heat)
##导入不同模型的button
        self.ui.Button_SVM.clicked.connect(self.handleCalc_SVM)
        self.ui.KNN.clicked.connect(self.handleCalc_KNN)
        self.ui.random_forest.clicked.connect(self.handleCalc_random_forest)
        self.ui.logisticregression.clicked.connect(self.handleCalc_logisticregression)
        self.ui.Decision_tree.clicked.connect(self.handleCalc_Decision_tree)
        self.ui.flood_b.clicked.connect(self.handleCalc_flood_b)
        self.ui.processdata.clicked.connect(self.process_data)
#*************************************************************************
#                               函数部分
#*************************************************************************

#############################路径导入的函数##################################
    #def handleCalc_import_text(self):
    @Slot()
    def handleCalc_import_text(self):
        # 生成文件对话框对象
        dialog = QFileDialog()
        # 设置文件过滤器，这里是任何文件，包括目录噢
        dialog.setFileMode(QFileDialog.AnyFile)
        # 设置显示文件的模式，这里是详细模式
        dialog.setViewMode(QFileDialog.Detail)
        if dialog.exec_():
            fileNames = dialog.selectedFiles()
            self.ui.lineEdit.setText(fileNames[0])

#-------------------------------------------------------------------------------------------
##########################################数据预处理##########################################
    def process_data(self):
        ##读入文件数据，并进行预处理
        info = self.ui.lineEdit.text()
        print(info)
        global seed ,I_data,train,test,train_x,train_y,test_x,test_y,s1,s2
        seed = 5
        I_data = pd.read_csv(info)
        train, test = train_test_split(I_data, test_size=0.3, random_state=seed)
        train_x = train[
            ['after_proj', 'podu_1', 'poxiang_1', 'Convergenc', 'Terrain_Ru', 'Topographi',
             'js_2019', 'shuixi1', 'jianzhu201']]
        train_y = train.GRID_CODE
        test_x = test[
            ['after_proj', 'podu_1', 'poxiang_1', 'Convergenc', 'Terrain_Ru', 'Topographi',
             'js_2019', 'shuixi1', 'jianzhu201']]
        test_y = test.GRID_CODE
        # mean = train_x.mean(axis=0)
        # train_x -= mean
        # std = train_x.std(axis=0)
        # train_x /= std
        # test_x -= mean
        # test_x /= std
        Max_train = train_x.max(axis=0)
        Min_train = train_x.min(axis=0)
        Max_test = test_x.max(axis=0)
        Min_test = test_x.min(axis=0)
        train_x = train_x - Min_train
        train_x = train_x / (Max_train - Min_train)
        test_x = test_x - Min_test
        test_x = test_x / (Max_test - Min_test)
        s1 = pd.concat([train_x, train_y], axis=1, join='outer')
        s2 = pd.concat([test_x, test_y], axis=1, join='outer')
#-------------------------------------------------------------------------
############################分析的函数######################################
#-------------------------------------------------------------------------
    def handleCalc_tongji(self):

        print(s1.describe())
        print(s2.describe())

    def handleCalc_cor(self):

        print(s1.corr())
        print(s2.corr())

    def handleCalc_corr_heat(self):

        sns.heatmap(s1.corr(), square=True, annot=True, cmap='YlGnBu')
        sns.heatmap(s1.corr(), square=True, annot=True, cmap='YlGnBu')
        plt.show()

#############################模型函数#######################################

#-------------------------------------------------------------------------
    def handleCalc_SVM(self):

        model = svm.SVC()
        model.fit(train_x, train_y)
        prediction = model.predict(test_x)
        print('The accuracy of the SVM is:', metrics.accuracy_score(prediction, test_y))
        t=metrics.accuracy_score(prediction, test_y)
        self.ui.lineEdit_10.setText(str(t))

    def handleCalc_KNN(self):

        model = KNeighborsClassifier()
        model.fit(train_x, train_y)
        prediction = model.predict(test_x)
        print('The accuracy of the KNN is:', metrics.accuracy_score(prediction, test_y))
        t = metrics.accuracy_score(prediction, test_y)
        self.ui.lineEdit_9.setText(str(t))

    def handleCalc_random_forest(self):

        y = I_data.GRID_CODE
        x=I_data.drop('GRID_CODE',axis=1)
        rfc = RandomForestClassifier()
        rfc = rfc.fit(train_x, train_y)
        result = rfc.score(test_x, test_y)
        print(result)
        t = result
        self.ui.lineEdit_8.setText(str(t))
        print(roc_auc_score(test_y, rfc.predict_proba(test_x)[:, 1]))
        print('各feature的重要性：%s' % rfc.feature_importances_)
        importances = rfc.feature_importances_
        print(np.argsort(importances)[::-1])

    ##绘制重要性图
        std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        print('Feature ranking:')
        for f in range(min(20, train_x.shape[1])):
            print("%2d') %-*s %f" % (f + 1, 30, train_x.columns[indices[f]], importances[indices[f]]))
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(train_x.shape[1]), importances[indices], color='r', yerr=std[indices], align='center')
        plt.xticks(range(train_x.shape[1]), indices)
        plt.xlim([-1, train_x.shape[1]])
        plt.show()

    ##绘制线性图
        predictions_validation = rfc.predict_proba(test_x)[:, 1]
        fpr, tpr, _ = roc_curve(test_y, predictions_validation)
        roc_auc = auc(fpr, tpr)
        plt.title('ROC Validation')
        plt.plot(fpr, tpr, 'b', label='AUC= %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    def handleCalc_logisticregression(self):

        model = LogisticRegression()
        model.fit(train_x, train_y)
        prediction = model.predict(test_x)
        print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(prediction, test_y))
        t = metrics.accuracy_score(prediction, test_y)
        self.ui.lineEdit_12.setText(str(t))

    def handleCalc_Decision_tree(self):

        model = DecisionTreeClassifier()
        model.fit(train_x, train_y)
        prediction = model.predict(test_x)
        print('The accuracy of the Decision Tree is:', metrics.accuracy_score(prediction, test_y))
        t = metrics.accuracy_score(prediction, test_y)
        self.ui.lineEdit_13.setText(str(t))

    def handleCalc_flood_b(self):
        # preparing the data:将各个数据进行标准化
        I_data = pd.read_csv('D:\\big\\shaobinghao\\yanzheng\\新建文件夹\\Export_Output2.csv')
        seed=5
        train, test = train_test_split(I_data, test_size=0.3, random_state=seed)
        train_x = train[
            ['after_proj', 'podu_1', 'poxiang_1',  'Convergenc', 'Terrain_Ru', 'Topographi',
             'js_2019', 'shuixi1', 'jianzhu201']]
        train_y = train.GRID_CODE
        test_x = test[
            ['after_proj', 'podu_1', 'poxiang_1',  'Convergenc', 'Terrain_Ru', 'Topographi',
             'js_2019', 'shuixi1', 'jianzhu201']]
        test_y = test.GRID_CODE
        mean = train_x.mean(axis=0)
        train_x -= mean
        std = train_x.std(axis=0)
        train_x /= std

        test_x -= mean
        test_x /= std
        # Max_train=train_data.max(axis=0)
        # Min_train=train_data.min(axis=0)
        # Max_test=test_data.max(axis=0)
        # Min_test=test_data.min(axis=0)
        # train_data-=Min_train
        # train_data/=(Max_train-Min_train)
        # test_data-=Min_test
        # test_data/=(Max_test-Min_test)
     ##### 构建一个函数，因为之后会多次调用
        def build_module():
            model = models.Sequential()
            model.add(layers.Dense(64, activation='relu',
                                   input_shape=(train_x.shape[1],)))
#            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(32, activation='relu'))
#            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dense(16, activation='relu'))
            model.add(layers.Dense(8, activation='relu'))
            model.add(layers.Dense(4, activation='relu'))
            model.add(layers.Dense(2, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer=optimizers.RMSprop(lr=0.01), loss='binary_crossentropy',
                          metrics=[tf.keras.metrics.binary_accuracy])
            return model

        # some memory clean-up
        k = 10
        num_val_samples = len(train_x) // k
        K.clear_session()
        num_epochs = 50
        all_acc_histories = []
        for i in range(k):
            print('processing fold #', i)
            # prepare the validation data:data from partition # k
            val_data = train_x[i * num_val_samples:(i + 1) * num_val_samples]
            val_targets = train_y[i * num_val_samples:(i + 1) * num_val_samples]

            # prepare the training data:data from all other partitions
            partial_train_data = np.concatenate(
                [train_x[:i * num_val_samples],
                 train_x[(i + 1) * num_val_samples:]],
                axis=0)
            partial_train_targets = np.concatenate(
                [train_y[:i * num_val_samples],
                 train_y[(i + 1) * num_val_samples:]],
                axis=0)
            model = build_module()
            history = model.fit(partial_train_data, partial_train_targets,
                                validation_data=(val_data, val_targets),
                                epochs=num_epochs, batch_size=100, verbose=2)
            acc_history = history.history['binary_accuracy']
            all_acc_histories.append(acc_history)
        average_acc_history = [
            np.mean([x[i] for x in all_acc_histories]) for i in range(num_epochs)
        ]

    #####损失函数绘图
        acc = history.history['binary_accuracy']
        val_acc = history.history['val_binary_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # 对图像进行指数平滑
        def smooth_curve(points, factor=0.9):
            smoothed_points = []
            for point in points:
                if smoothed_points:
                    previous = smoothed_points[-1]
                    smoothed_points.append(previous * factor + point * (1 - factor))
                else:
                    smoothed_points.append(point)
            return smoothed_points

        smooth_acc_history = smooth_curve(average_acc_history[10:])

        plt.plot(range(1, len(smooth_acc_history) + 1), smooth_acc_history)
        plt.xlabel('Epochs')
        plt.ylabel('ACC')
        plt.show()
    #####准确率
        acc = history.history['binary_accuracy']
        val_acc = history.history['val_binary_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and Validation acc')
        plt.xlabel('Epochs')
        plt.ylabel('acc')
        plt.legend()
        plt.show()
        self.ui.lineEdit_9.setText(str(acc))

app = QApplication([])
stats = Stats()
stats.ui.show()
app.exec_()