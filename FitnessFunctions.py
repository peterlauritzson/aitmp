

def all_ones(individual):
    fit = 0
    for number in individual:
        if number == 1:
            fit += 1
    return fit


#Beale's function
def beale_function(individual):
    x = individual[0]
    y = individual[1]
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
