# R 코드 결과 2-4 와인품질 데이터 회귀모형 적합 결과
# Importing data
wine = read.csv("winequalityCLASS.csv", header = TRUE)

# Fitting a logistic regression model
fit.all = glm(quality ~., family = binomial, data = wine)
fit.step = step(fit.all, direction = "both")        #stepwise vaiable selection
fit.step$anova


# R 코드 및 결과 2-5 와인품질 데이터 최종 회귀 모형
summary(fit.step)

# R 코드 및 결과 2-6 와인품질 데이터의 예측과 평가
# Making predictions
p = predict(fit.step, newdata = wine, type = "response")    # prediction
cutoff = 0.5  # cutoff
yhat = ifelse(p > cutoff, 1, 0)

# Evaluatuon 
tab = table(wine$quality, yhat, dnn = c("Observed", "Predicted"))
print(tab)          # confusion matrix

sum(diag(tab)) / sum(tab)       # accuracy (예측정확도)

tab[2,2] / sum(tab[2,])         # sensitivty (민감도)

tab[1,1] /sum(tab[1,])          # specificty (특이도)
