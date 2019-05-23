import timeit
import vcf
listy = []
print(timeit.timeit('for n in range(100): [].append(n)', number=1000000))

