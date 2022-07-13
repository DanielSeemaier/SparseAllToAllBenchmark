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
        csv$Density <- as.numeric(csv$Density)
        csv <- csv %>% dplyr::filter(Topology != "")
        results <- rbind(results, csv)
    }
    aggregator <- function(df) 
        data.frame(MBs = mean(df$MBs, na.rm = TRUE),
                   Time = mean(df$Time, na.rm = TRUE))

    results <- ddply(results, c("MPI", "Topology", "Density", "Algorithm", "Scale"), aggregator)
    return(results)
}

plot_results <- function(results) {
    topologies <- unique(results$Topology)
    for (topology in topologies) {
        df <- subset(results, Topology == topology & Scale %in% c(1, 10, 20))
        topology_desc <- switch(topology, 
                                "identity" = "single message to itself",
                                "adjacent_cells_4" = "message to 4 adjacent PEs in square grid",
                                "adjacent_cells_8" = "message to 8 adjacent PEs in square grid",
                                "rgg2d" = "message for each cut edge",
                                "rgg3d" = "message for each cut edge",
                                "rhg" = "message for each cut edge")

        message_size_desc <- if (topology %in% c("identity", "adjacent_cells_4", "adjacent_cells_8"))
            "4 * 2^<Scale> bytes to each adjacent PE"
        else
            "4 * <Scale> bytes for each cut edge"

        densities <- df %>% 
            dplyr::distinct(MPI, Density) %>%
            dplyr::mutate(Density = sprintf("%.3f", Density))

        plot <- ggplot(df, aes(x = MPI, y = MBs, color = Algorithm, shape = Scale, linetype = Scale)) +
            theme_bw() +
            geom_line() +
            geom_point() +
            scale_x_continuous(trans = "log2", breaks = c(16, 64, 256, 1024, 4096)) +
            scale_y_continuous(trans = "log2", breaks = c(1, 4, 16, 64, 256, 1024, 4096)) +
            xlab("Number of MPI processes") +
            ylab("Bandwidth per MPI process [MB/s]") +
            labs(title = topology, subtitle = paste0("Topology: ", topology_desc, 
                                                                           "\nMessage size: ", message_size_desc)) +
            geom_text(data = densities, mapping = aes(x = MPI, y = 0, label = Density), inherit.aes = FALSE, vjust = -1)
        print(plot)
    }
}

supermuc <- load_results("supermuc/v2.5/")

pdf("supermuc.pdf")
plot_results(supermuc)
dev.off()

