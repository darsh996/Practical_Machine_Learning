names = c("p1", "p2", "p3", "p4")
ages = c(20, 21, 23, 24)
salaries = c(10, 11, 15, 18)

persons = data.frame(
  name = names,
  age = ages,
  salary = salaries
)

print(persons)
str(persons)
print(summary(persons))
