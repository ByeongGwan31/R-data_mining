# p.88 분류 의사결정나무 사례분석
# 1) 데이터 읽고 CART 의사결정 나무 실행하기
# 의사결정 나무 실행
# Importing Data
wine = read.csv("winequalityCLASS.csv")

# Factorize for classification
wine$quality = factor(wine$quality)

### Classification Tree
library(rpart)
set.seed(1234)

my.control = rpart.control(xval = 10, cp = 0, minsplit = 20)
tree.wine = rpart(quality~., data = wine, method = "class", control = my.control)
print(tree.wine)

# 의사결정나무의 그래프 출력
# Display Tree

library(rpart.plot)
prp(tree.wine, type = 4, extra = 1, digits = 2, box.palette = "Grays")


# 2) 가지치기 수행하기
# CART 의사결정나무 가지치기 단계별 정보
# Pruning with c-s.e.
cps = printcp(tree.wine)

# R 코드 3-4 1-s.e. 법칙으로 가지치기 수행
# Pruning with c-s.e.
cps = printcp(tree.wine)
k = which.min(cps[, "xerror"])
err = cps[k, "xerror"]; se = cps[k, "xstd"]
c = 1 # 1-s.e.

k1 = which(cps[, "xerror"] <= err + c * se) [1]
cp.chosen = cps[k1, "CP"]
tree.pruned.wine = prune(tree.wine, cp = cp.chosen)
print(tree.pruned.wine)

# R 코드 3-5 : 의사결정나무의 그래프 출력
# Display tree
prp(tree.pruned.wine, type = 4, extra = 1, digits = 2, box.palette = "Grays")

# 3) 분류정확도 계산하기
# R코드 3-6 : CART 의사결정나무의 예측값 산출
# Making predictions - probablity prediction
prob.tree.wine = predict(tree.pruned.wine, newdata = wine, type = "prob")
head(prob.tree.wine, 5)
cutoff = 0.5 #cutoff
yhat.tree.wine = ifelse(prob.tree.wine[,2] > cutoff, 1, 0)

# R코드 3-7 : CART 의사결정나무의 예측값과 실제 목표변수값 비교
# Evaluation
tab = table(wine$quality, yhat.tree.wine, dnn=c("Observed", "Predicted"))
print(tab)      # confusin matrix

sum(diag(tab)) / sum(tab)   # accuracy [정확도 (387 + 502) / 1194 = 74.46%]

tab[2,2] / sum(tab[2,])     # sensitivity [민감도 (502 / 645) = 77.8%]

tab[1,1] / sum(tab[1,])     # specificity [특이도 (387 / 549) = 70.5%]
