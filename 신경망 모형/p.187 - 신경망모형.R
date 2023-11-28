# R 코드 5-1 신경망모형을 이용한 코사인 함수의 적합

install.packages("neuralnet")
library(neuralnet)
set.seed(130)
ind1 = 1:100
ind2 = ind1/100

cos2 = cos(ind2 * 4 * pi)
cdat = data.frame(cbind(ind2, cos2))

cos2.nn = neuralnet(cos2 ~ ind2, data = cdat, hidden = 5, linear.output = T)

plot(cos2.nn)

# R 코드 5-2 신경망모형을 이용한 예측값 산출
cos.pred = predict(cos2.nn, data.frame(ind2))
plot(ind1, cos.pred)

lines(cos2)

# R 코드 5-3 신경망모형을 통한 의류생산성 데이터 분석

library(neuralnet)
install.packages("dummy")
library(dummy)
prod = read.csv("productivityREG.csv", header = TRUE)
prod$quarter = factor(prod$quarter)
prod$department = factor(prod$department)
prod$day = factor(prod$day)
prod$team = factor(prod$team)

# 더미변수 (dummy variables) 생성
dvar = c(1:4)
prod2 = dummy(x = prod[,dvar])
prod2 = prod2[,-c(5, 7, 13, 25)]
prod2 = cbind(prod[,-dvar], prod2)

for(i in 1: ncol(prod2)) if(!is.numeric(prod2[,i])) prod2[,i] = as.numeric(prod2[,i])

# 데이터의 표준화
max1 = apply(prod2, 2, max)
min1 = apply(prod2, 2, min)
sdat = scale(prod2, center = min1, scale = max1 - min1)
sdat = as.data.frame(sdat)
pn = names(sdat)

f = as.formula(paste("productivity~", paste(pn[!pn %in% "productivity"], collapse = " + ")))
set.seed(1234)
fit.nn = neuralnet(f, data = sdat, hidden = c(3,1), linear.output = T)
plot(fit.nn)

# 예측값 계산 및 MSE 산출
pred.nn = predict(fit.nn, sdat)
pred.nn = pred.nn * (max1[7] -min1[7] + min1[7])

# Mean Squared Error (MSE) 평균 제곱 오차
mean((prod2$productivity - pred.nn)^2)

# 관측값 대 예측값 (Observed vs. Fitted) 산점도
plot(prod2$productivity, pred.nn, xlab = "Observed Values", ylab = "Fitted Values")
abline(0, 1)

# R 코드 및 결과 5-4 신경망모형을 통한 와인품질 데이터 분석
library(neuralnet)
wine = read.csv("winequalityCLASS.csv", header = TRUE)

#임계점 정의
cutoff = 0.5

# 데이터의 표준화
max1= apply(wine, 2, max)
min1 = apply(wine, 2, min)
gdat = scale(wine, center = min1, scale = max1 - min1)
gdat = as.data.frame(gdat)

gn = names(gdat)
f = as.formula(paste("quality~", paste(gn[!gn %in% "quality"], collapse = " + ")))
set.seed(1234)
fit.nn = neuralnet(f, data = gdat, hidden = c(2,1), linear.output = F)
plot(fit.nn)

# 예측력 평가
p.nn = predict(fit.nn, gdat)
yhat.nn = ifelse(p.nn > cutoff, 1, 0)

# Confusion matrix
tab = table(gdat$quality, yhat.nn, dnn = c("Observed", "Predicted"))
print(tab)

sum(diag(tab)) / nrow(gdat)       # accuracy 정확도

tab[2,2] / sum(tab[2,])           # sensitivity 민감도

tab[1,1] /sum(tab[1,])            # specificity 특이도