# R 코드 결과 2-1 의류생산성 데이터 회귀모형 적합 결과
# Importing data
prod = read.csv("productivityREG.csv", header = TRUE)

# Factorizing predictor variables
prod$quarter = factor(prod$quarter)
prod$department = factor(prod$department)
prod$day = factor(prod$day)
prod$team = factor(prod$team)

# Fitting a linear regression model
fit.all = lm(productivity~., data = prod)
fit.step = step(fit.all, direction = "both")
fit.step$anova

# R 코드 결과 2-2 의류생산성데이터 최종 회귀모형
summary(fit.step)

# R 코드 및 결과 2-3 의류생산성 데이터 예측과 평가
# Making predictions
pred.reg = predict(fit.step, newdata = prod, type = "response")
print(pred.reg)

# Evaluation 
mean((prod$productivity - pred.reg)^2)        # MSE 평균치

mean (abs(prod$productivity - pred.reg))      # MAE 절대값을 평균 계산

# Making predictions