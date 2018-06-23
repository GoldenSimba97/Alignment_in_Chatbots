# Calculate Cohen's Kappa
def kappa(a, b, c, d):
    N = a + b + c + d
    p0 = (a + d)/N
    pyes = ((a + b)/N) * ((a + c)/N)
    pno = ((c + d)/N) * ((b + d)/N)
    pe = pyes + pno
    return (p0-pe)/(1-pe)

# GloVe v.s. formal and informal words lists
print(kappa(21, 29, 29, 135))

# Linguistic alignment binary conditional probability vs. formal and informal words lists
print(kappa(36, 14, 14, 150))

# Linguistic alignment frequency conditional probability vs. formal and informal words lists
print(kappa(14, 36, 36, 128))
