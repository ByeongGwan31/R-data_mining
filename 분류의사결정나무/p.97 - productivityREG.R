# p - 97
# 회귀의사결정 나무 사례 분석
# 1) 데이터 일곡 CART 의사결정나무 실행하기

# R 코드 3-8 : 의사결정나무 실행
# Importing DATA
prod = read.csv("productivityREG.csv", header = TRUE)

# Factorizing predictor variables
prod$quarter = factor(prod$quarter)
prod$department = factor(prod$department)
prod$day = factor(prod$day)
prod$team = factor(prod$team)

### Regression Tree
library(rpart)
set.seed(1234)
my.control = rpart.control(xval = 10, cp = 0.01, minsplit = 30)
tree.prod = rpart(productivity~., data = prod, method = "anova", control = my.control)
print(tree.prod)

# R 코드 3-9 : 의사결정나무의 그래프 출력
# Display tree
library(rpart.plot)
prp(tree.prod, type = 4, extra = 1, digits = 2, box.palette = "Grays")

# 2) 가지치기 수행하기
# R 코드 3-10 : CART의사결정나무 가지치기 단계별 정보
# Pruning with c-s.e.
cps = printcp(tree.prod)

# R 코드 3-11 : 1-s.e. 법칙으로 가지치기 수행
# Pruning with c-s.e.

k = which.min(cps[,"xerror"])
err = cps[k, "xerror"]; se = cps[k, "xstd"]
c = 1 # 1-s.e.
k1 = which(cps[,"xerror"] <= err + c * se) [1]
cp.chosen = cps[k1, "CP"]
tree.pruned.prod = prune(tree.prod, cp = cp.chosen)
print(tree.pruned.prod)

# R 코드 3-12 : 의사결정나무의 그래프 출력
# Display tree
prp(tree.pruned.prod, type = 4, extra = 1, digits = 2, box.palette = "Grays")

# R 코드 3-13 : CART 의사결정나무의 예측값 및 정확도 산출
# Making predictions
pred.tree.prod = predict(tree.pruned.prod, newdata = prod, type = 'vector')
head(pred.tree.prod, 5)

# Evaluation
mean((prod$productivity - pred.tree.prod)^2)    # MSE
mean(abs(prod$productivity - pred.tree.prod))   # MAE

# R코드 3-14 : CART 회귀의사결정나무의 예측값과 실제값의 산점도
# Observed vs. Predicted
plot(prod$productivity, pred.tree.prod, xlab = "Observed Values", ylab = "Fitted Values")
abline(0, 1)