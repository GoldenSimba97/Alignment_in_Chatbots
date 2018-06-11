N = 33+17+17+147
p0 = (33+147)/N
pyes = ((33+17)/(33+17+17+147))*((33+17)/(33+17+17+147))
pno = ((17+147)/(33+17+17+147))*((17+147)/(33+17+17+147))
pe = pyes + pno
k = (p0-pe)/(1-pe)
# print(k)

def kappa(a, b, c, d):
    N = a + b + c + d
    p0 = (a + d)/N
    pyes = ((a + b)/N) * ((a + c)/N)
    pno = ((c + d)/N) * ((b + d)/N)
    pe = pyes + pno
    return (p0-pe)/(1-pe)

# print(kappa(33, 17, 17, 147))

# print(kappa(21, 29, 29, 135))
# print(kappa(24, 49, 25, 96))
# print(kappa(36, 14, 14, 150))
# print(kappa(38, 77, 17, 83))
# print(kappa(35, 15, 15, 149))
# print(kappa(55, 23, 62, 74))

print(kappa(28, 22, 22, 142))
print(kappa(31, 19, 19, 145))
print(kappa(34, 16, 16, 148))
