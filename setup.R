update.packages(checkBuilt=TRUE, ask=FALSE, quiet=TRUE)
install.packages(c("docopt", "BiocManager"), quiet=TRUE)

BiocManager::install(c("graph"), quiet=TRUE)
BiocManager::install(c("RBGL", "ggm", "Rgraphviz"), quiet=TRUE)
BiocManager::install(c("ggm"), quiet=TRUE)
BiocManager::install(c("Rgraphviz"), quiet=TRUE)

# BiocManager::install(c("graph", "RBGL", "ggm", "Rgraphviz"), quiet=TRUE)

install.packages("devtools", dependencies=TRUE, quiet=TRUE)
install.packages(c("MASS", "momentchi2"), dependencies=TRUE, quiet=TRUE)

library(devtools)
install_github("ericstrobl/RCIT")

install.packages(c("pcalg"), dependencies=TRUE)
install.packages(c("kpcalg"), dependencies=TRUE)


