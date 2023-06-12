# SocialNetworkAnalysis
La cartelle è strutturata nel seguente modo. Vi sono 4 cartelle denominate **task1**, **task2**, **task3** e **task4** contenenti codici e risultati per i task della midterm.
Per il final project il task1 è stato analizzato con i file che si trovano nella cartella **task1_final** contenente il codice per individuare il tipo di modello di rete e i rispettivi parametri.
Il file python *SocNetMec.py* contiene la classe richiesta per il task 2. 

La funzione *input_data* del file *final_test.py* è stata cambiata in modo da restituire due oggetti (arms_set, auctions) che servono per la classe SocNetMac. Qual ora la funzione debba restituire il numero di elementi come il file originale vi è un file *final_test2.py* che calcola questi valori fuori la funzione

Per il task1 del final project è presente anche un report *task1.pdf* dove vi è un'analisi dettagliata dei vari esperimenti che hanno portato alla sconclusione che il modello utilizzato per generare la rete net_4 è quello di **Watts-Strogatz** con i seguenti parametri:
- r = 2.71
- k = 1
- q = 4

Per il task 2 l’algoritmo del bandito utilizzato è stato **UCB1**, mentre il formato d’asta scelto è stato **GIDM** (*Generalized Information Diffusion Mechanism* dal paper “Selling Multiple Items via Social Networks”) che è la generalizzazione dell’asta IDM per la vendita di più oggetti. 