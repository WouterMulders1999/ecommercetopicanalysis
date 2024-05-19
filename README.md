# Topic Analysis
Deze repo bevat de codes voor het maken van een Topic Analysis dashboard voor ambtenaren. Er worden verschillende bronnen (zoals vng.nl en binnenlandsbestuur.nl) geraadpleegd voor de laatste nieuwsartikelen rondom de onderwerpen 'techniek en digitaal'. In praktijk worden deze websites gescraped om de nieuwsartikelen op te halen, waarna er  verschillende NLP-technieken, zoals bijvoorbeeld LDA, op worden toegepast.

In het dashboard wordt vervolgens inzichtelijk gemaakt welke onderwerpen er op dit moment spelen. Dit is oorspronkelijk bedoeld geweest voor consultancy partijen binnen de publieke sector om zo hun marketingbeleid te kunnen voeren. In het verleden is dit ook succesvol gebleken, maar momenteel is deze repo ernstig verouderd.

Mochten er vragen zijn over de codes in deze repo, neem dan gerust contact met mij op. Om zelf met de code aan de slag te gaan, kunt u het beste beginnen met "Excelsmaken.py" deze code bevat de scraper voor de diverse websites. In docker_dashboard/app vindt u app.py. Dit bestand bevat de code voor het uitvoeren van de diverse analyses en het opstellen van het dashboard. Ook is het dockerfile waarmee wij deze code op een raspi hebben geïmplementeerd te vinden in de docker_dashboard map.
 
