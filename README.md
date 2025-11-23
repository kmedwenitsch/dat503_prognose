# DAT503 - KI-basiertes Prognosemodell für den österreichischen Häuserpreisindex (HPI)

Dieses Projekt umfasst ein GPR-basiertes Prognosemodell für den österreichischen Häuserpreisindex (HPI).
Es wurde im Rahmen der LV DAT503 - AI Assisted Engineering des Masterstudiengangs Informationstechnologie an der Ferdinand Porsche FernFH entwickelt.

Es verwendet die historischen Daten des österreichischen HPI, zur Verfügung gestellt von [Statistik Austria](https://www.statistik.at/statistiken/volkswirtschaft-und-oeffentliche-finanzen/preise-und-preisindizes/haeuserpreisindex-und-ooh-pi), als Basis.
In dem aktuellen Prototyp wird eine zuvor manuell bearbeitete csv-Datei mit den historischen Daten eingelesen, vorverarbeitet, ein GPR-Modell erstellt, dieses trainiert, im Anschluss die Prognose der nächsten 4 Quartale durchgeführt und schließlich visualisiert.
