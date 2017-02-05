# HIERARCHICAL-MULTISCALE-RECURRENT-NEURAL-NETWORKS
L’un des problèmes les plus étudiés en matière de réseaux de neurones est la compréhension
par la machine de la structure hiérarchique et temporelle des données et de réussir d’en tirer une
représentation sur plusieurs niveaux d’abstraction. Ce qui confèrerait au réseau de neurones en
question de pouvoir profiter de la généralisation envers des exemples non vus, du partage de la
connaissance apprise entre les différentes taches ou la découverte de nouveaux facteurs de variation
entremêlés.<br>
<br>
Récemment les CNNs profonds ont prouvées avoir une grande capacité à apprendre la
représentation hiérarchique des données spatiales et les RNNs ont menés vers des bonnes avancées
dans la modélisation de données temporelles. Inversement l’apprentissage d’une représentation à la
fois temporelle et hiérarchique est longtemps resté un défi pour les RNNs.<br>
<br>
Schmidhuber (1992), El Hihi et Bengio (1995) et Koutník et al. (2014) ont proposés une
approche prometteuse pour modéliser de telles représentations appelée les RNNs multi-échelles
(multiscale RNN ou M-RNN). Cette approche est basée sur le fait que l’abstraction dite « haut niveau »
change lentement dans le temps par rapport à l’abstraction dite « bas niveau » dont les
caractéristiques changent très rapidement et sont sensibles au temps local précis. D’où les M-RNN
groupe des couches cachées en multiples modules d’échelle de temps différente. Cette approche
apporte également multiples avantages par rapport au RNNs classique qui sont détaillés dans l’article
(efficacité de calcul obtenue, allocation de ressources flexible, le partage d’information en aval vers les
sous-tâches ou sou-niveaux d’abstraction des données).<br>
<br>
Plusieurs implémentations de RNNs multi-échelles ont été proposées. La plus populaire est
l’approche consistant à considérer les échelles de temps (ou délais) comme des hyper paramètres à
fixer plutôt que de les traiter comme des variables dynamiques qui peuvent être apprises. Mais en
prenant en compte le fait que les données temporelles sont généralement non-stationnaires et que
beaucoup d’entités d’abstraction (comme les mots et les phrases par exemple) ont des tailles variables,
les auteurs de l’article Hierarchical Multiscale Recurrent Neural Networks (Junyoung Chung, Sungjin
Ahn et Yoshua Bengio de l’Université de Montréal) préconisent de construire un modèle de RNN qui
adapte ses échelles de temps de façon dynamique en fonction des entités données en entrée (de tailles
différentes).<br>
<br>
Ils proposent donc un nouveau modèle de RNNs multi-échelles capable d’apprendre la
structure hiérarchique multi-échelle des données temporelles sans avoir d’information spécifique sur
les limites temporelles. Ce modèle, appelé RNN Hiérarchique Multi-échelle (HM-RNN), ne fixe de
taux de mises à jour mais détermine de façon adaptative les délais propres de mises à jour
correspondants aux différents niveaux d’abstraction des couches.
