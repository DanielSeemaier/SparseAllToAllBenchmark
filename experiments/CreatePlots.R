#!/usr/bin/env Rscript
require(ggplot2)
require(dplyr)
require(plyr)

load_results <- function(cluster) {
    results <- data.frame()
    for (file in list.files(cluster, pattern = "results.csv.*", full.names = TRUE)) {
        csv <- read.csv(file)
        csv$Scale <- factor(csv$Scale)
        csv$MPI <- as.integer(csv$MPI)
        csv$MBs <- as.numeric(as.numeric(csv$MBs) / csv$MPI)
        csv <- csv %>% dplyr::filter(Topology != "")
        results <- rbind(results, csv)
    }
    print(results$MBs)
    aggregator <- function(df) 
        data.frame(MBs = mean(df$MBs, na.rm = TRUE),
                   Time = mean(df$Time, na.rm = TRUE))

    results <- ddply(results, c("MPI", "Topology", "Algorithm", "Scale"), aggregator)
    return(results)
}

plot_results <- function(results) {
    topologies <- unique(results$Topology)
    for (topology in topologies) {
        df <- subset(results, Topology == topology & Scale %in% c(1, 5, 10))
        subtitle <- if (topology %in% c("identity", "adjacent_cells_4", "adjacent_cells_8"))
            "Send 4 * 2^<Scale> bytes to each adjacent PE"
        else
            "Send 4 * <Scale> bytes for each cut edge"

        plot <- ggplot(df, aes(x = MPI, y = MBs, color = Algorithm, shape = Scale, linetype = Scale)) +
            theme_bw() +
            geom_line() +
            geom_point() +
            scale_x_continuous(trans = "log2") +
            scale_y_continuous(trans = "log2", breaks = c(1, 4, 16, 64, 256, 1024, 4096)) +
            xlab("Number of MPI processes") +
            ylab("Bandwidth per MPI process [MB/s]") +
            labs(title = paste0("Topology: ", topology),
                 subtitle = subtitle)
        print(plot)
    }
}

supermuc <- load_results("supermuc/v1/")


pdf("supermuc.pdf")
plot_results(supermuc)
dev.off()

