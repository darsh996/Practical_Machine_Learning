# mean

x = c(10, 20, 53, 64, 36, 72, 51, 32, 66, 72, 84, 92)

# calculate mean by using formula
calculate.mean = function(x) {
  sum.x = sum(x)
  print(sum.x)
  mean = sum.x / length(x)
  print(mean)
}

# calculate.mean(x)

# calculate mean by using built-in function
mean.x = mean(x)
print(mean.x)
