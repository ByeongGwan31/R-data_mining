# 회귀앙상블 사례분석 p.147
# 1) 데이터 읽기
# R 코드 4-18 : 데이터 읽기

# Importing Data
prod = read.csv("productivityREG.csv", header = TRUE)

# Factorizing predictor variables
prod$quarter = factor(prod$quarter)
prod$department = factor(prod$department)
prod$day = factor(prod$day)
prod$team = factor(prod$team)

# R 코드 4-19 랜덤포레스트 실행
### Random Forest
library(randomForest)
set.seed(1234)
rf.prod <- randomForest(productivity~., data = prod, ntree = 100, mtry = 5,
                        importance = T, na.action = na.omit)

# R 코드 4-20  변수중요도 수행
# Variable importance
importance(rf.prod, type = 1)
varImpPlot(rf.prod, type = 1)

# R 코드 4-21 의사결정나무의 개수와 평균제곱오차 그리기
# Plot error rates
plot(rf.prod, type="l")

# R 코드 4-22 랜덤포레스트의 부분종속 그림
# Partial dependence plot
partialPlot(rf.prod, pred.data = prod, x.var = 'incentive')

# 3) 회귀예측 정확도 계산하기
# R 코드 4-23 랜덤포레스트의 예측값 및 산출
# Making predictions
pred.rf.prod = predict(rf.prod, newdata = prod, type = "response")
head(pred.rf.prod, 5)

# Evaluation
# Evaluation
mean((prod$productivity - pred.rf.prod)^2)        # MSE
mean(abs(prod$productivity - pred.rf.prod))       # MAE

# R 코드 4-24 랜덤포레스트 회귀앙상블의 예측값과 실제값의 산점도
# Observed vs. Predited
plot(prod$productivity, pred.rf.prod, xlab = "Observed Values",
     ylab = "Fitted Values")
abline(0,1)