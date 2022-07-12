#!/usr/bin/env Rscript
require(ggplot2)
require(dplyr)
require(plyr)

load_results <- function(cluster) {
    results <- data.frame()
    for (file in list.files(cluster, pattern = "results.csv.*", full.names = TRUE)) {
        csv <- read.csv(file)
        csv$Scale <- factor(csv$Scale)
        csv$MPI <- as.numeric(csv$MPI)
        csv$MBs <- as.numeric(csv$MBs)
        results <- rbind(results, csv)
    }
    print(results$MBs)
    aggregator <- function(df) 
        data.frame(MBs = mean(df$MBs, na.rm = TRUE))

    results <- ddply(results, c("MPI", "Topology", "Algorithm", "Scale"), aggregator)
    return(results)
}

plot_results <- function(results) {
    topologies <- unique(results$Topology)
    for (topology in topologies) {
        df <- subset(results, Topology == topology & Scale %in% c(1, 5, 10))
        plot <- ggplot(df, aes(x = MPI, y = MBs, color = Algorithm, shape = Scale, linetype = Scale)) +
            geom_line() +
            geom_point() +
            scale_x_continuous(trans = "log2") +
            scale_y_continuous(trans = "log2") +
            ggtitle(paste0("Topology: ", topology))
        print(plot)
    }
}

supermuc <- load_results("supermuc/v1/")


pdf("supermuc.pdf")
plot_results(supermuc)
dev.off()

