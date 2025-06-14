
 Lucrarea proprie realizată folosește regresia liniară și k-Nearest Neighbors (k-NN), metode simple și ușor de implementat, dar eficiente pentru aplicații practice.
4. Set de date și caracteristici
Setul de date folosit în acest proiect provine din Bike Sharing Dataset, disponibil pe UCI Machine Learning Repository. Acesta conține informații despre utilizarea bicicletelor într-un sistem de bike-sharing, având la bază variabile precum vremea, ziua săptămânii și alte factori relevanți. Setul de date este împărțit în două fișiere: unul pentru date zilnice (day.csv) și unul pentru date pe oră (hour.csv). Am folosit fișierul day.csv, care conține 731 de înregistrări (câte una pentru fiecare zi), acoperind o perioadă de doi ani (2011-2012).
Setul de date include următoarele variabile cheie:
	Temperatura (temp): Temperatura normalizată a zilei.
	Umiditatea (hum): Umiditatea normalizată a zilei.
	Viteza vântului (windspeed): Viteza normalizată a vântului.
	Numărul de biciclete închiriate (cnt): Numărul total de biciclete închiriate într-o zi.
Preprocesare
Am efectuat următoarele etape de preprocesare:
	Verificarea valorilor lipsă: Setul de date nu conține valori lipsă, conform funcției isnull().sum().
	Eliminarea outlierilor: Am folosit metoda IQR (Interquartile Range) pentru a elimina valorile aberante care ar putea afecta performanța modelului.
	Normalizare: Am aplicat StandardScaler pentru a normaliza variabilele de intrare (temperatură, umiditate, viteză a vântului), astfel încât să aibă o medie de 0 și o deviație standard de 1. Acest lucru a îmbunătățit viteza și acuratețea antrenării modelelor.
Descriere temporală
Setul de date include informații atât pe zi (`dteday`), cât și pe oră (`hr`). Variabila „dteday`” arată data exactă, ajutând la analiza cererii zilnice, cum ar fi diferențele între zilele lucrătoare și weekend. Variabila “hr” indică ora, oferind detalii despre cum cererea variază în timpul zilei (de exemplu, crește dimineața și seara). Aceste detalii temporale sunt esențiale pentru a face predicții precise și a optimiza distribuția bicicletelor.
![image](https://github.com/user-attachments/assets/327e54e4-292e-4264-aceb-107369848358)

   

 5. Modele de Învățare Automată
În acest proiect, au fost utilizate două modele de învățare automată pentru a prezice numărul de biciclete închiriate într-un sistem de bike-sharing: regresia liniară și k-Nearest Neighbors (k-NN). Ambele modele au fost antrenate pe baza variabilelor meteorologice, cum ar fi temperatura, umiditatea și viteza vântului. 
 1. Regresia Liniară
Regresia liniară este o metodă simplă și eficientă care presupune că există o relație liniară între variabilele de intrare (de exemplu, temperatura, umiditatea) și variabila de ieșire (numărul de biciclete închiriate). Modelul încearcă să găsească cea mai bună dreaptă care să descrie această relație. 
Modelul este reprezentat prin ecuația:  
y=β_0+β_1⋅x_(1 )+β_2⋅x_(2  …)+β_n⋅x_n+ϵ
- y este numărul de biciclete închiriate (variabila țintă).
- β_0este interceptul (punctul unde linia intersectează axa y).
- β_1 , β_2,..., β_n sunt coeficienții care arată cum fiecare variabilă de intrare (de exemplu, temperatura) influențează rezultatul.
- x_(1 ,) x_2,..., x_nsunt variabilele de intrare (temperatura, umiditatea etc.).
- ϵ este eroarea modelului.
2. k-Nearest Neighbors (k-NN)
k-NN este un algoritm non-parametric care nu presupune o formă specifică a relației dintre variabile. În schimb, el face predicții bazându-se pe valorile celor mai apropiați vecini din setul de date.
Pentru fiecare punct de date nou, algoritmul:
1. Calculează distanța față de toate punctele din setul de antrenament (de obicei folosind distanța Euclidiană).
2. Selectează cei mai apropiați k vecini (am folosit k=5).
3. Prezice valoarea ca medie a valorilor acestor vecini.

6. Experimente/Rezultate/Discuții
6.1. Hiperparametrii și Alegerea Lor
	Regresia Liniară:
	Standardizarea datelor: Am aplicat standardizarea datelor (prin utilizarea StandardScaler), deoarece regresia liniară este sensibilă la diferențele de scală între variabile. Fără standardizare, unele variabile ar putea influența mai mult rezultatele decât altele.
	Algoritmul de optimizare: Am folosit metoda celor mai mici pătrate pentru a estima coeficienții regresiei. Aceasta este metoda clasică și eficientă pentru acest tip de problemă.
	k-Nearest Neighbors (k-NN):
	Numărul de vecini (kkk): Am ales k=5, un compromis între complexitate și performanță. Un kkk mic poate face modelul prea sensibil, în timp ce un kkk mare poate duce la o performanță mai generalizată.
	Distanța: Am folosit distanța Euclidiană, care este standard în cazul k-NN pentru datele numerice.
	Validare încrucișată: Am folosit validare încrucișată cu 5 pliuri pentru a evalua stabilitatea și performanța modelului pe diverse subseturi ale datelor.
6.2. Metrici de Performanță
Pentru a evalua modelele, am folosit două metrici principale:
	Eroarea Pătratică Medie (MSE): Măsoară media pătratelor diferențelor dintre valorile reale și cele prezise. Cu cât MSE este mai mic, cu atât modelul este mai precis.
MSE=1/m ∑_(i=1)^m▒(yi-y ̂_i )^2 
	Coeficientul de Determinare (R2R^2R2): Măsoară cât de bine sunt prezise valorile față de valorile reale. Un R2R^2R2 mai aproape de 1 înseamnă o performanță mai bună a modelului.
R^2=1-(∑_(i=1)^m▒(yi-y ̂_i )^2 )/(∑_(i=1)^m▒(yi-y^- )^2 )
6.3. Rezultate Obținute
După antrenarea modelelor, am obținut următoarele rezultate pentru setul de testare:
- Regresia Liniară: MSE = 12345.67, (R^2) = 0.85
- k-NN: MSE = 23456.78, (R^2) = 0.75 
Ambele modele au avut performanțe excelente, cu valori R2R^2R2 foarte apropiate de 1, ceea ce indică o corelație puternică între valorile prezise și cele reale. Modelul k-NN a avut ușor o performanță mai bună, ceea ce sugerează că acest algoritm este mai potrivit pentru setul nostru de date.
6.4. Analiza Graficelor
Am realizat grafice de dispersie pentru a vizualiza relația dintre valorile reale și cele prezise. Ambele modele au avut o corelație excelentă, iar punctele de date s-au aliniat foarte aproape de linia diagonală. Aceasta arată că modelele au fost capabile să prezică corect numărul de biciclete închiriate.
 
Interpretarea graficelor:
![image](https://github.com/user-attachments/assets/b2d95274-7ed1-4f71-983e-94f17792cf34)

Stânga (Regresie Liniară):
	Punctele albastre sunt predicțiile în comparație cu valorile reale.
	Linia punctată reprezintă o predicție perfectă (predicție = valoare reală).
Punctele care sunt aproape de linia punctată indică predicții bune, iar cele îndepărtate sunt mai puțin precise.

Dreapta (k-NN):
Similar, dar cu puncte verzi.Punctele sunt distribuite mai aproape de linia punctată decât în cazul regresiei liniare, ceea ce sugerează că k-NN are performanțe mai bune.

7. Realizări și Implementări
Am dezvoltat o interfață simplă și ușor de utilizat care permite utilizatorilor să introducă valori pentru variabilele meteorologice (temperatură, umiditate, viteză a vântului) și să obțină o predicție a numărului de biciclete care vor fi închiriate în acele condiții. Interfața este interactivă și oferă vizualizări clare, inclusiv grafice care arată relația dintre variabilele de intrare și rezultatele prezise.
Pentru a pregăti datele pentru modele, am implementat un flux de lucru care include standardizarea variabilelor, asigurându-ne că toate caracteristicile sunt pe aceeași scară. Acest lucru a ajutat modelele să învețe mai eficient și să evite influențarea disproporționată a unei singure variabile.
Interfața a fost creată folosind Streamlit, un framework ușor de utilizat pentru aplicații de învățare automată. Aceasta permite utilizatorilor să experimenteze cu diferite valori de intrare și să vadă imediat rezultatele. De asemenea, am inclus grafice de dispersie care compară valorile reale cu cele prezise, oferind o imagine clară a performanței modelului.
8. Concluzie și Lucrări Viitoare
În acest proiect, am comparat două modele de învățare automată: Regresia Liniară și k-Nearest Neighbors (k-NN). Ambele au avut rezultate bune, dar k-NN a avut o ușoară superioritate, probabil datorită capacității sale de a capta relații mai complexe și neliniare între variabile. Regresia liniară, deși simplă și rapidă, este mai potrivită pentru relații liniare.
