# mydocumention
from sklearn.datasets import load_iris
iris=load_iris()

#1.数据预处理
#1.1无量纲
#1.1.1标准化（正态标准化）
from sklearn.preprocessing import StandardScaler
StandardScaler().fit_tansform(iris.data)
#1.1.2区间缩放法（返回值为缩放到[0, 1]区间的数据）
from sklearn.preprocessing import MinMaxScaler
MinMaxScaler().fit_transform(iris.data)
#1.2连续值的离散化（根据阈值3，将连续值离散化，即将其二值化，变成0及1）
from sklearn.preprocessing import Binarizer
Binarizer(threshold=3).fit_transform(iris.data)
#1.3 哑编码（独热编码）（将定性数据转化为定量数据）
from sklearn.preprocessing import OneHotEncoder
OneHotEncoder().fit_transform(iris.target.reshape((-1, 1)))
#1.4 缺失值
 #缺失值计算，返回值为计算缺失值后的数据
 #参数missing_value为缺失值的表示形式，默认为NaN
 #参数strategy为缺失值填充方式，默认为mean（均值）
from sklearn.preprocessing import Imputer
Imputer().fit_transform(iris.data)
#1.5数据变换
from sklearn.preprocessing import PolynomialFeatures
PolynomialFeatures().fit_transform(iris.data)

#总结：都是调用对应类，然后使用类中的fit_transform方法，并输入对应的数据作为参数，就可以进行对应的处理


#2.特征选择
#2.1 Filter
#2.1.1 方差选择法
from sklearn.feature_selection import VarianceThreshold#方差阈值
#方差选择法，返回值为特征选择后的数据
#参数threshold为方差的阈值
VarianceThreshold(threshold=3).fit_transform(iris.data)
#2.1.2相关系数法
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
#选择K个最好的特征，返回选择特征后的数据
#第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。
#在此定义为相关系数。
#参数k为选择的特征个数
SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)

#2.1.3卡方检验
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#选择K个最好的特征，返回选择特征后的数据
SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)

#2.1.4信息熵
from sklearn.feature_selection import SelectKBest
from minepy import MINE
#由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)
#选择K个最好的特征，返回特征选择后的数据
SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)

#2.2  Wrapper
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#递归特征消除法，返回特征选择后的数据
#参数estimator为基模型
#参数n_features_to_select为选择的特征个数
RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)

#2.3 Embeded

#2.3.1基于惩罚项的特征选择（正则化=正确的规则化）
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target)

#L1惩罚项降维的原理在于保留多个对目标值具有同等相关性的特征中的一个，所以没选到的特征不代表不重要。
#故可结合L2惩罚项来优化。
#具体操作为：若一个特征在L1中的权值为1，选择在L2中权值差别不大且在L1中权值为0的特征构成同类集合，
#将这一集合中的特征平分L1中的权值，故需要构建一个新的逻辑回归模型：
from sklearn.linear_model import LogisticRegression

class LR(LogisticRegression):
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):
        #权值相近的阈值
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                 fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                 random_state=random_state, solver=solver, max_iter=max_iter,
                 multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        #使用同样的参数创建L2逻辑回归
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                      intercept_scaling=intercept_scaling, class_weight = class_weight,
                                      random_state=random_state, solver=solver, max_iter=max_iter,
                                      multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
    def fit(self, X, y, sample_weight=None):
        #训练L1逻辑回归
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        #训练L2逻辑回归
        self.l2.fit(X, y, sample_weight=sample_weight)
        cntOfRow, cntOfCol = self.coef_.shape
        #权值系数矩阵的行数对应目标值的种类数目
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.coef_[i][j]
                #L1逻辑回归的权值系数不为0
                if coef != 0:
                    idx = [j]
                    #对应在L2逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        #在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
                        if abs(coef1-coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)
                    #计算这一类特征的权值系数均值
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return self

#带L1和L2惩罚项的逻辑回归作为基模型的特征选择
#参数threshold为权值系数之差的阈值
SelectFromModel(LR(threshold=0.5, C=0.1)).fit_transform(iris.data, iris.target)

#2.3.2 基于树模型的特征选择（GBDT）
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)

#3.降维
#3.1主成分分析法
from sklearn.decomposition import PCA
#主成分分析法，返回降维后的数据
#参数n_components为主成分数目
PCA(n_components=2).fit_transform(iris.data)

#3.2线性判别分析法
from sklearn.lda import LDA
#线性判别分析法，返回降维后的数据
#参数n_components为降维后的维数
LDA(n_components=2).fit_transform(iris.data, iris.target)
