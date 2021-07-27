setwd('~/GitHub/perfect_squares')
getwd()

new <- read.table('losses/loss_improve_1627081086.log', sep = ',',
                  fill = T, col.names = c('loss', 'improvement'))
old <- read.table('losses/loss_improve_1627081187.log', sep = ',',
                  fill = T, col.names = c('loss', 'improvement'))

# removes invalid rows
new <- subset(new, new[,2] != "")
old <- subset(old, old[,2] != "")

str(new)

true_new = subset(new, new[,2] == " True")
true_old = subset(old, old[,2] == " True")

length(row.names(true_new)) / length(row.names(new))
length(row.names(true_old)) / length(row.names(old))

