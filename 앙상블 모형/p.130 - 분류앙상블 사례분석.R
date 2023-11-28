# 분류 앙상블 사례 분석

# 1) 데이터 읽기
# R 코드 4-1 데이터 읽기

# Importing data
wine = read.csv("winequalityCLASS.csv", header = TRUE)

# Factorize for classifiication
wine$quality = factor(wine$quality)

# 2 ) 배깅 방법의 실행
# R 코드 4-2 배깅 앙상블 실행

### Bagging
install.packages("adabag")
library(rpart)
library(adabag)
set.seed(1234)

my.control = rpart.control(xval = 0, cp = 0, minsplit = 5)
bag.wine = bagging(quality~., data = wine, mfinal = 100, control = my.control)


# R 코드 4-3 변수중요도 수행
# Variable importance

print(bag.wine$importance)
importanceplot(bag.wine)


# R 코드 4-4 의사결정나무의 개수와 오분류율 그리기
# Error vs. number of trees

evol.wine = errorevol(bag.wine, newdata = wine)
plot.errorevol(evol.wine)

# R 코드 4-5 배깅 앙상블의 예측값 산출
# Making predictions
prob.bag.wine = predict.bagging(bag.wine, newdata = wine)$prob
head(prob.bag.wine, 5)
cutoff = 0.3 #cutoff
yhat.bag.wine = ifelse(prob.bag.wine[,2] > cutoff, 1, 0)

# R 코드 4-6 배깅 앙상블의 예측값과 실제 목표변수값 비교
# Evaluation
tab = table(wine$quality, yhat.bag.wine, dnn = c("Observed", "Predicted"))
print(tab)          # confusinon matrix
sum(diag(tab)) / sum(tab)       # accuracy (정확도)
tab[2,2] / sum(tab[2,])         # sensitivity (민감도)
tab[1,1] / sum(tab[1,])         # specificity (특이도)

# 3 ) 부스팅 방법의 실행
# R 코드 4-7 부스팅 앙상블 실행
### Boosting 
library(rpart)
library(adabag)
set.seed(1234)
my.control = rpart.control(xval = 0, cp = 0, maxdepth = 4)
boo.wine = boosting(quality~., data = wine, boos = T, mfinal = 100,
                    control = my.control)

# R 코드 4-8 변수중요도 수행
# Variable importance
print(boo.wine$importance)
importanceplot(boo.wine)

# R 코드 4-9 의사결정나무의 개수와 오분류울 그리기
# Error vs. number of trees
evol.wine = errorevol(boo.wine, newdata = wine)
plot.errorevol(evol.wine)

# R 코드 4-10 부스팅 앙상블의 예측값 산출
# Making predictions
prob.boo.wine = predict.boosting(boo.wine, newdata = wine)$prob
head(prob.boo.wine, 5)
cutoff = 0.5      # cutoff
yhat.boo.wine = ifelse(prob.boo.wine[,2] > cutoff, 1, 0)

# R 코드 4-11 부스팅 앙상블의 에측값과 실제 목표변수값 비교
# Evaluation
tab = table(wine$quality, yhat.boo.wine, dnn = c("Observed", "Predicted"))
print(tab)        # confusion matrix

sum(diag(tab)) / sum(tab)   # accracy 정확도
tab[2,2] / sum(tab[2,])     # sensitvityh 민감도
tab[1,1] / sum(tab[1,])     # specificity 특이도

# 4) 랜덤 포레스트 방법의 실행
# R 코드 4-12 랜덤포레스트 실행
### Random Forest
install.packages("randomForest")
library(randomForest)
set.seed(1234)
rf.wine = randomForest(quality~., data = wine, ntree = 100, mtry = 5, 
                       importance = T, na.action = na.omit)

# R 코드 4-13 변수중요도 수행
# Variable importance
importance(rf.wine, type = 1)
varImpPlot(rf.wine, type = 1)

# R 코드 4-14 의사결정나무의 개수와 오분류율 그리기
# Plot error rates
plot(rf.wine, type = 'l')

# R 코드 4-15 랜덤포레스트의 부분 종속그림 (PDP)
# Partiall dependence plot
partialPlot(rf.wine, pred.data = wine, x.var = 'alcohol', which.class = 1)

# R 코드 4-16 : 랜덤포레스트의 예측값 산출
# Making predictions
prob.rf.wine = predict(rf.wine, newdata = wine, type = "prob")
head(prob.rf.wine, 5)
cutoff = 0.5      # cutoff
yhat.rf.wine = ifelse(prob.rf.wine[,2] > cutoff, 1, 0)

# R 코드 4-17 : 랜덤포레스트 앙상블의 예측값과 실제 목표변수값 비교
# Evaluation
tab = table(wine$quality, yhat.rf.wine, dnn = c ("Observed", "Predicted"))
print(tab)          # confusion matrix
sum(diag(tab)) / sum(tab)       # accuracy 정확도
tab[2,2] / sum(tab[2,])         # sensitvity 민감도
tab[1,1] / sum(tab[1,])         # specificity 특이도