import numpy as np
print("www")
def assign_prior_power_law(degrees):
    # Calcola la distribuzione power law normalizzata dei gradi
    power_law_distribution = degrees / np.sum(degrees)
    
    # Assegna le prior proporzionalmente alla distribuzione power law normalizzata
    priors = power_law_distribution / np.sum(power_law_distribution)
    
    return priors

# Esempio di utilizzo
degrees = [10, 20, 5, 15, 8]  # Esempio di gradi dei nodi
priors = assign_prior_power_law(degrees)

# Stampa delle prior assegnate a ciascun nodo
for i, prior in enumerate(priors):
    print(f"Nodo {i+1}: Prior = {prior}")