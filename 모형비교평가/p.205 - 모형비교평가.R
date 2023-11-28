# R 코드 및 결과 6-1 의류생산성 데이터 호출 및 데이터 분할

# Importing data
prod = read.csv("productivityREG.csv", header = TRUE)

# Factorizing predictor variables

prod$quarter = factor(prod$quarter)
prod$department = factor(prod$department)
prod$day = factor(prod$day)
prod$team = factor(prod$team)

# Partitioning data into train and test sets
set.seed(1234)
train.index = sample(1:nrow(prod), size = 0.7 * nrow(prod), replace = F)
prod.train = prod[ train.index, ]    # train data
prod.test = prod[-train.index, ]    # test data

# R코드 및 결과6-2  의류생산성 데이터에 휘귀모형 적합 및 예측 결과

fit.reg = lm(productivity~., data = prod.train)
fit.step.reg = step(fit.reg, direction = "both", trace = FALSE)

pred.reg = predict(fit.step.reg, newdata = prod.test, type = "response")
mean((prod.test$productivity - pred.reg) ^2)    # MSE
mean(abs(prod.test$productivity - pred.reg))    # MAE

# R 코드 및 결과 6-3 의류생산성 데이터에 회귀나무모형 적합 및 예측 결과

library(rpart)

my.control = rpart.control(Xval = 10, cp = 0, minsplit = 5)
fit.tree = rpart(productivity~., data = prod.train, method = "anova", control = my.control)
tmp = printcp(fit.tree)

k = which.min(tmp[,"xerror"])
cp.tmp = tmp[k, "CP"]

fit.prun.tree = prune(fit.tree, cp = cp.tmp)

pred.tree = predict(fit.prun.tree, newdata = prod.test, type = "vector")

mean((prod.test$productivity - pred.tree)^2)      # MSE
mean(abs(prod.test$productivity - pred.tree))     # MAE

lib