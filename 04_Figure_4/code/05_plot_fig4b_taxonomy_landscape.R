suppressPackageStartupMessages({
  library(tidyverse)
  library(ggtree)
  library(ggtreeExtra)
  library(ggnewscale)
  library(ape)
  library(showtext)
})

## ================================================================
## Fig. 4B: genus-level landscape of three overall modules
## Inner -> outer:
## phylum dots | replication | transcription | translation | plasmid count
## ================================================================

## 0. Input, output, and plotting parameters
infile <- "D:/LAB/togit/loadgithub/Preprocessing/Plasmid_CDE_module_subcategory_landscape_per_plasmid.csv"
outdir <- "Fig4_main/Fig4B_genus_module_landscape"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

min_plasmids       <- 10
presence_min_count <- 1
n_top_phyla        <- 7
FONT               <- 7
show_bar           <- FALSE

## Fixed module order from inner to outer
modules_in_fig <- c("replication", "transcription", "translation")

## Taxonomic tree geometry
level_len <- c(1.5, 1.5, 1.2, 1.2, 1.1, 0.7)
tree_lw   <- 0.7

## Ring geometry
phy_pwidth <- 0.015
phy_offset <- 0.001
phy_size   <- 0.85

ring_pw   <- 0.035
ring_off1 <- 0.06
ring_offg <- 0.13

bar_pw  <- 0.14
bar_off <- 0.055

## Continuous module gradients: 0-1 prevalence
module_high_cols <- c(
  replication   = "#8dd3c8",
  transcription = "#70a4d5",
  translation   = "#e26261"
)

legend_title_by_module <- c(
  replication   = "Replication prevalence (%)",
  transcription = "Transcription prevalence (%)",
  translation   = "Translation prevalence (%)"
)

## Phylum colors
phylum_pal <- c(
  "#cea479", "#a479ce", "#f180ba", "#a2c1b1",
  "#e6ab02", "#7570b3", "#a6761d"
)
other_col <- "#7a7a7a"

## Optional outer plasmid-count bar
plasmid_breaks <- c(-Inf, 100, Inf)
plasmid_labels <- c("≤100", ">100")
plasmid_cols   <- c("≤100" = "#d0d0d0", ">100" = "#8f8f8f")

## Require Arial on Windows
arial_path <- "C:/Windows/Fonts/arial.ttf"
if (!file.exists(arial_path)) stop("Arial font file was not found.")
font_add("Arial", regular = arial_path)
BASE_FAMILY <- "Arial"
showtext_auto()
 
## 1. Read data and identify subcategory count columns
read_any <- function(file) {
  first_line <- readLines(file, n = 1, warn = FALSE)
  if (grepl("\t", first_line)) {
    read.delim(file, check.names = FALSE, stringsAsFactors = FALSE)
  } else {
    read.csv(file, check.names = FALSE, stringsAsFactors = FALSE)
  }
}

df_raw <- read_any(infile)

tax_cols <- c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")
missing_tax <- setdiff(tax_cols, colnames(df_raw))
if (length(missing_tax) > 0) {
  stop("Missing taxonomy columns: ", paste(missing_tax, collapse = ", "))
}

df_raw <- df_raw %>%
  mutate(across(
    all_of(tax_cols),
    ~ replace_na(as.character(.x), "Unknown")
  )) %>%
  mutate(across(
    all_of(tax_cols),
    ~ ifelse(str_trim(.x) == "", "Unknown", str_trim(.x))
  ))

count_cols <- grep(
  "^(replication|transcription|translation)_.+_count$",
  colnames(df_raw),
  value = TRUE
)

if (length(count_cols) == 0) {
  stop("No module subcategory count columns were detected.")
}

module_info <- tibble(count_col = count_cols) %>%
  mutate(
    module = str_extract(
      count_col,
      "^(replication|transcription|translation)"
    )
  )

df_raw <- df_raw %>%
  mutate(across(
    all_of(count_cols),
    ~ suppressWarnings(as.numeric(.x))
  ))

## 2. Collapse subcategories into module-level presence variables
## A plasmid is positive when the summed count of all subcategories
## in that module is >= presence_min_count.
module_prev_cols <- character(0)

for (mod in modules_in_fig) {
  mod_count_cols <- module_info %>%
    filter(module == mod) %>%
    pull(count_col)
  
  if (length(mod_count_cols) == 0) {
    stop("No count columns were detected for module: ", mod)
  }
  
  out_col <- paste0(mod, "_module_present")
  module_prev_cols <- c(module_prev_cols, out_col)
  
  mod_count_matrix <- as.data.frame(df_raw[, mod_count_cols, drop = FALSE])
  mod_count_matrix[] <- lapply(
    mod_count_matrix,
    function(x) replace(x, is.na(x), 0)
  )
  
  df_raw[[out_col]] <- as.integer(
    rowSums(mod_count_matrix, na.rm = TRUE) >= presence_min_count
  )
}

names(module_prev_cols) <- modules_in_fig

## ----------------------------------------------------------------
## 3. Aggregate to genus level
## ----------------------------------------------------------------
## Prevalence is retained as a proportion in [0, 1].
df_genus <- df_raw %>%
  group_by(Kingdom, Phylum, Class, Order, Family, Genus) %>%
  summarise(
    Total_Plasmids = n(),
    across(
      all_of(unname(module_prev_cols)),
      ~ mean(.x, na.rm = TRUE)
    ),
    .groups = "drop"
  ) %>%
  filter(
    Total_Plasmids > min_plasmids,
    !is.na(Genus),
    Genus != "",
    Genus != "Unknown"
  ) %>%
  as.data.frame() %>%
  mutate(Tip_ID = make.unique(Genus))

message(sprintf(
  "Retained %d genera with > %d plasmids.",
  nrow(df_genus), min_plasmids
))

if (nrow(df_genus) < 2) {
  stop("Too few genera remain after filtering.")
}

## ----------------------------------------------------------------
## 4. Build taxonomy tree
## ----------------------------------------------------------------
tax_for_tree <- df_genus %>%
  transmute(
    K   = Kingdom,
    P   = paste(Kingdom, Phylum, sep = "|"),
    C   = paste(P, Class, sep = "|"),
    O   = paste(C, Order, sep = "|"),
    Fa  = paste(O, Family, sep = "|"),
    Tip = Tip_ID
  ) %>%
  mutate(across(everything(), as.factor))

tree <- as.phylo(
  ~ K / P / C / O / Fa / Tip,
  data = tax_for_tree,
  collapse = FALSE
)

tree$edge.length <- rep(1, nrow(tree$edge))
child_depth <- node.depth.edgelength(tree)[tree$edge[, 2]]
depth_index <- pmax(
  1,
  pmin(length(level_len), round(child_depth))
)
tree$edge.length <- level_len[depth_index]

## ----------------------------------------------------------------
## 5. Define phylum display groups
## ----------------------------------------------------------------
top_n_phyla <- df_genus %>%
  count(Phylum, sort = TRUE) %>%
  filter(Phylum != "Unknown") %>%
  pull(Phylum) %>%
  head(n_top_phyla)

all_phylum_colors <- c(
  setNames(phylum_pal[seq_along(top_n_phyla)], top_n_phyla),
  "Other" = other_col
)

df_genus <- df_genus %>%
  mutate(
    Display_Phylum = ifelse(Phylum %in% top_n_phyla, Phylum, "Other"),
    Display_Phylum = factor(
      Display_Phylum,
      levels = c(top_n_phyla, "Other")
    )
  )

## ----------------------------------------------------------------
## 6. Draw Fig. 4B
## ----------------------------------------------------------------
p <- ggtree(
  tree,
  layout = "circular",
  color = "#585858",
  linewidth = tree_lw
)

## 6a. Inner phylum ring: circular points in both plot and legend
p <- p +
  geom_fruit(
    data = df_genus %>% select(Tip_ID, Display_Phylum),
    geom = geom_point,
    mapping = aes(
      x = 1,
      y = Tip_ID,
      color = Display_Phylum
    ),
    shape = 16,
    size = phy_size,
    offset = phy_offset,
    pwidth = phy_pwidth
  ) +
  scale_color_manual(
    values = all_phylum_colors,
    drop = FALSE,
    name = "Phylum",
    guide = guide_legend(
      order = 1,
      override.aes = list(shape = 16, size = 2, linewidth = 0)
    )
  )

## 6b. Three continuous module-prevalence rings
module_legend <- guide_colorbar(
  title = NULL,
  title.position = "top",
  title.hjust = 0,
  label.position = "bottom",
  direction = "horizontal",
  barwidth = unit(10, "mm"),
  barheight = unit(1.5, "mm"),
  ticks = TRUE,
  frame.colour = NA
)

for (i in seq_along(modules_in_fig)) {
  
  mod <- modules_in_fig[i]
  prev_col <- module_prev_cols[[mod]]
  
  fruit_df <- df_genus %>%
    transmute(
      Tip_ID,
      prevalence = pmin(pmax(.data[[prev_col]], 0), 1)
    )
  
  current_offset <- if (i == 1) ring_off1 else ring_offg
  
  p <- p +
    new_scale_fill() +
    geom_fruit(
      data = fruit_df,
      geom = geom_tile,
      mapping = aes(
        x = 1,
        y = Tip_ID,
        fill = prevalence
      ),
      color = NA,             # No white gaps between adjacent module tiles
      linewidth = 0,
      width = 1,
      offset = current_offset,
      pwidth = ring_pw
    ) +
    scale_fill_gradient(
      low = "white",
      high = unname(module_high_cols[[mod]]),
      limits = c(0, 1),
      breaks = c(0, 0.5, 1),
      labels = c("0", "50", "100"),
      name = legend_title_by_module[[mod]],
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0,
        label.position = "bottom",
        direction = "horizontal",
        barwidth = unit(10, "mm"),
        barheight = unit(1.5, "mm"),
        ticks = TRUE,
        frame.colour = NA
      )
    )
}

## 6c. Optional outer genus-level plasmid-count bar
if (show_bar) {
  bardat <- df_genus %>%
    transmute(
      Tip_ID,
      log10_n = log10(Total_Plasmids),
      plasmid_bin = factor(
        cut(
          Total_Plasmids,
          breaks = plasmid_breaks,
          labels = plasmid_labels
        ),
        levels = plasmid_labels
      )
    )
  
  p <- p +
    new_scale_fill() +
    geom_fruit(
      data = bardat,
      geom = geom_bar,
      mapping = aes(
        x = log10_n,
        y = Tip_ID,
        fill = plasmid_bin
      ),
      stat = "identity",
      orientation = "y",
      color = NA,
      offset = bar_off,
      pwidth = bar_pw
    ) +
    scale_fill_manual(
      values = plasmid_cols,
      drop = FALSE,
      name = "Plasmids per genus (n)",
      guide = guide_legend(
        order = 5,
        override.aes = list(linewidth = 0)
      )
    )
}

p <- p +
  theme(
    text = element_text(
      size = FONT,
      family = BASE_FAMILY
    ),
    legend.position = "right",
    legend.box = "vertical",
    legend.title = element_text(
      size = FONT,
      family = BASE_FAMILY,
      face = "plain"
    ),
    legend.text = element_text(
      size = FONT,
      family = BASE_FAMILY
    ),
    legend.key.height = unit(2.2, "mm"),
    legend.key.width = unit(2.2, "mm"),
    legend.spacing.y = unit(0.7, "mm"),
    legend.box.spacing = unit(1.5, "mm"),
    legend.margin = margin(0, 0, 0, 0),
    plot.margin = margin(1, 1, 1, 1, unit = "mm")
  )

## ----------------------------------------------------------------
## 7. Diagnostics and output
## ----------------------------------------------------------------
cat("\n== Ring order, inner to outer ==\n")
cat("1. Phylum dots\n")
cat("2. Replication prevalence\n")
cat("3. Transcription prevalence\n")
cat("4. Translation prevalence\n")
if (show_bar) cat("5. Genus plasmid count\n")

cat("\n== Genus-level plasmid count summary ==\n")
print(summary(df_genus$Total_Plasmids))
print(quantile(
  df_genus$Total_Plasmids,
  probs = c(0.25, 0.50, 0.75, 0.90, 0.95, 1.00)
))

cat("\n== Mean module prevalence across retained genera ==\n")
print(
  df_genus %>%
    summarise(across(
      all_of(unname(module_prev_cols)),
      mean,
      na.rm = TRUE
    )) %>%
    pivot_longer(
      everything(),
      names_to = "module",
      values_to = "mean_prevalence"
    ) %>%
    mutate(mean_prevalence = scales::percent(mean_prevalence, accuracy = 0.1))
)

base <- file.path(
  outdir,
  "Fig4B_genus_replication_transcription_translation_continuous"
)

ggsave(
  paste0(base, ".pdf"),
  p,
  width = 5.2,
  height = 5.8,
  device = cairo_pdf,
  bg = "white"
)

ggsave(
  paste0(base, ".png"),
  p,
  width = 5.2,
  height = 5.8,
  dpi = 600,
  bg = "white"
)

message("Done: ", base, ".pdf / .png")
print(p)
