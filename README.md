# DIPLOMSKA NALOGA: Napovedovanje gibanja finančnih trgov z modeli Chronos in TimesFM

## Opis projekta

Ta projekt primerja dva napredna modela za napovedovanje časovnih vrst – Amazon Chronos-Bolt in Google TimesFM – na zgodovinskih podatkih o cenah izbranih delnic. Analiza vključuje oceno uspešnosti modelov na različnih časovnih intervalih (5 minut, 15 minut, 1 ura) ter vizualizacijo in statistično obdelavo rezultatov.

## Struktura projekta

- `notebooks/`: Jupyter notebooki za pridobivanje podatkov, izvajanje napovedi in analizo rezultatov.
- `src/`: Python skripte za obdelavo podatkov in generiranje vizualizacij.
- `data/`: Shranjeni zgodovinski podatki o cenah delnic po intervalih.
- `results/`: Napovedi posameznih modelov za vsako delnico in interval.
- `analysis_results/`: Grafični prikazi in primerjave uspešnosti modelov.
- `notes.md`: Povzetki, ugotovitve in priporočila na podlagi analize.

## Uporabljeni podatki

- Zgodovinski podatki za 11 izbranih delnic (npr. IONQ, NVDA, TSLA, VKTX ...), ki kotirajo ali na NASDAQ in NYSE.
- Podatki so pridobljeni za intervale 5M, 15M in 1H.
- Uporabljena je bila closing cena ('Close').

## Uporabljeni modeli

### Chronos-Bolt
- Transformer model za časovne vrste (Amazon).
- Uporablja 512 preteklih vrednosti (context window) in napove naslednjih 128 točk.
- Napovedi se izvajajo v drsečem oknu (rolling window).

### TimesFM
- Foundation model za časovne vrste (Google).
- Uporablja 2048 (512) preteklih vrednosti in napove naslednjih 128 točk.
- Prav tako uporablja drseče okno.

## Analiza in vizualizacija

- Izračunani so bili ključni metrični kazalniki: MAPE, MAE, RMSE, volatilnost napake, smerna pravilnost-natančnost.
- Rezultati so vizualizirani v obliki histogramov, boxplotov, distribucij, in stolpčnih diagramov.
- Statistični testi (t-test) za preverjanje pomembnosti razlik med modeli.

## Navodila za uporabo


## Ključne ugotovitve

- Model TimesFM je v povprečju natančnejši (nižji MAPE) in bolj stabilen kot Chronos.
- Najboljše rezultate dosegata oba modela na 1-urnem intervalu.
- Največje napake in volatilnost so pri napovedih na 5-minutnem intervalu.
- Smerna točnost je za oba modela podobna in presega naključno napovedovanje.

## Priporočila

- Za splošno uporabo priporočamo TimesFM na 1-urnem intervalu.
- Za visoko-frekvenčno trgovanje so potrebne dodatne izboljšave modelov.
- Razmislite o uporabi ansambla ali prilagodljivih intervalov za boljše rezultate.
---


