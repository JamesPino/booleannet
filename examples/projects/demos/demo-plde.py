import boolean2
import matplotlib.pyplot as plt

text = """
A = True
B = True 
C = False
1: A* = not C 
2: B* = A and B
3: C* = B
"""

model = boolean2.Model(text=text, mode='plde')
model.initialize()
model.iterate(fullt=7, steps=100)

plt.plot(model.data['A'], 'ob-', label='A')
plt.plot(model.data['B'], 'sr-', label='B')
plt.plot(model.data['C'], '^g-', label='C')
plt.legend(loc=0)
plt.show()
