import json

# Leggi il file JSON
with open('n_nodes_200000.json', 'r') as file:
    data = json.load(file)

# Ordina il dizionario per i valori in ordine decrescente
sorted_data = {k: v/20000 for k, v in sorted(data.items(), key=lambda item: item[1], reverse=True)}

# Scrivi il dizionario ordinato nel nuovo file JSON
with open('n_nodes_20000_.json', 'w') as file:
    json.dump(sorted_data, file, indent=4)