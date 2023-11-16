Abgabe AKI - Anwendungen künstlicher Intelligenz
Nicolas Flake - 573871

main.py -> trainiert und testet das Modell
clstm_model.py -> Definition des C-LSTM Modells
dataProcessor.py -> Alle Methoden die zur Aufbereitung der Daten verwendet wurden; Wird im laufenden Programm nur noch benötigt um die vortrainierten Google Vektoren zu laden
1511.08630.pdf -> Das umzusetzende Paper

data -> beinhaltet alle Rohdaten und Modell States
datasets -> beinhaltet die aufbereiteten Daten, wie Torchtext sie verarbeiten kann
results -> beinhaltet die Logs, Modell States und das Loss Diagramm des letzten Durchlaufs
stanfordSentimentTreebank -> Die Daten, so wie sie ursprünglicherweise bereitgestellt wurden

Benötigte Bibliotheken und Ausführhinweise:
torchtext (0.2.1)
torch
matplotlib
numpy
pandas
gensim


Das Programm wird aktuell mittels Cuda auf der Grafikkarte ausgeführt, um die Berechnungszeit einer Epoche von 160s auf 37-40s zu verkürzen.
Dafür muss Cuda und Pytorch mit Cuda Support installiert werden. Ansonsten kann man in main.py die Variable cuda auf "False" setzen.
Dann wird das Programm auf der CPU ausgeführt.


Ergebnis:
Im Paper kann das Label mit einer Genauigkeit von 49,2% vorausgesagt werden.
Nach Ausprobieren von verschiedenen Setups kann ich den Testdatensatz mit einer Genauigkeit von 39,9% vorhersagen.
Woran die Differenz liegt kann ich aktuell nicht sagen.
