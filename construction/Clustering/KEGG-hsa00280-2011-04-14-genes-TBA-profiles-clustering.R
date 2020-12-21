require("gplots")

aff.dat <- read.table("KEGG-hsa00280-2011-04-14-genes-TBA-profiles-affinity-hg19-idcorrected-noversion-asEntrez-viaRefSeq-2011-04-11-maximum-ztransf.dat")
aff.mat <- as.matrix(aff.dat)

aff.dist <- dist(aff.mat, method="manhattan")

aff.hclust <- hclust(aff.dist, method = "average")

plot(aff.hclust)

aff.tmat <- t(aff.mat)
aff.tdist <- dist(aff.tmat, method="manhattan")
aff.thclust <- hclust(aff.tdist, method = "average")

plot(aff.thclust)

# ---

heatmap.2(aff.mat, Rowv=TRUE, Colv=TRUE, hclustfun=hclust, distfun=dist, dendrogram="both")

