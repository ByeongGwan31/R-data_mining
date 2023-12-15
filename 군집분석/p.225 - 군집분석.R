# R 코드 결과 7-1 거리구하기
ex2 = read.table("ex7-2.txt", header = TRUE)
dist(ex2)

dist(ex2, method = "manhattan")

# R 코드 결과 7-2 단일연결법, 완전연결법, 평균연결법에 의한 응집분석
clustering1 = hclust(dist(ex2, method = "manhattan"), method = "single")
clustering2 = hclust(dist(ex2, method = "manhattan"), method = "complete")
clustering3 = hclust(dist(ex2, method = "manhattan"), method = "average")

par(mfrow = c(1,3))
plot(clustering1)
plot(clustering2)
plot(clustering3)

# R 코드 및 결과 7-3 나무형 그림의 확대 (군집분석)
dendrogram1 = as.dendrogram(clustering1)
plot(dendrogram1[[2]])

# R 코드 및 결과 7-4 DIANA를 이용한 분할분석
library(cluster)
ex2 = read.table("ex7-2.txt", header = T)
dianaclustering = diana(ex2, metric = "manhattan")
plot(dianaclustering)


# R 코드 및 결과 7-5 K-평균 군집분석
ex2 = read.table("ex7-2.txt", header = T)
ex2 = as.matrix(ex2)
aveclustering = hclust(dist(ex2), method = "average")
initialcent = tapply(ex2, list(rep(cutree(aveclustering, 2), ncol(ex2)), col(ex2)), mean)
kmclustering = kmeans(ex2, initialcent, algorithm = "MacQueen")
kmclustering

# R 코드 및 결과 7-6 state.x77 데이터의 기술 통계량
dim(state.x77)

summary(state.x77)

statescale <- data.frame(scale(state.x77, center = TRUE, scale = TRUE))