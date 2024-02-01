x = c(10, 20, 30, 10, 40, 20, 10, 10)

# which
# - function that returns the position(s) of value in the vector
print(which(x == 20))
print(which(x == 10))


calculate.mode = function(x) {
  
  # table 
  # - used to build frequency table
  frequency.table = table(x)
  print(frequency.table)
  
  # find the max frequency
  max.count = max(frequency.table)
  print(max.count)
  
  # find the position of max_count
  max.count.position = which(frequency.table == max.count)
  print(max.count.position)
  
  # find the values inside the table
  mode = names(max.count.position)
  print(mode)
}


calculate.mode(c(10, 20, 30, 10, 40, 20, 10, 10))
calculate.mode(c(10, 20, 30, 10, 40, 20, 20, 10))
calculate.mode(c(10, 20, 30, 10, 20, 30))
