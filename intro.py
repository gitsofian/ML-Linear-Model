# Importiere Bibliotheken
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

# Lade Datensatz
digits = datasets.load_digits()

# Erstelle Model
clf = svm.SVC(gamma=0.001, C=100.)

# Training
clf.fit(digits.data[0:500], digits.target[0:500])
#clf.score(digits.data[0:500], digits.target[0:500])
#clf.score(digits.data[501:1000], digits.target[501:1000])

# Test
sample = 639                # Beispiel einer falschen Klassifizierung
print("Vorhersage: ")
print(clf.predict(digits.data[sample:sample+1]))
print("Wahrheit: ")
print(digits.target[sample])

# Plotte die Eingabe (8x8 Bild)
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[sample], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()











#_______________________
'''
print(digits.data)
print(digits.target)
print(digits.images[0])

clf = svm.SVC(gamma=0.001, C=100.)

clf.fit(digits.data[:-1], digits.target[:-1])

print(clf.predict(digits.data[-1:]))

#print(digits.data[:-1])'''