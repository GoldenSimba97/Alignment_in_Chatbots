N = 33+17+17+147
print(N)
p0 = (33+147)/N
p02 = (26+140)/N
p03 = (39+153)/(39+11+11+153)
p04 = (31+145)/(N)
p05 = (36+90)/N
print(p0, p02, p03, p04, p05)
pyes = ((33+17)/(33+17+17+147))*((33+17)/(33+17+17+147))
pyes5 = ((36+20)/(33+17+17+147))*((36+68)/(33+17+17+147))
# pyes2 = ((26+24)/(26+24+24+140))*((26+24)/(26+24+24+140))
# pyes3 = ((39+11)/(39+11+11+153))*((39+11)/(39+11+11+153))
# pyes2 = ((33+17)*(33+17))/N
print(pyes, pyes5)
pno = ((17+147)/(33+17+17+147))*((17+147)/(33+17+17+147))
pno5 = ((68+90)/(33+17+17+147))*((20+90)/(33+17+17+147))
# pno2 = ((24+140)/(26+24+24+140))*((24+140)/(26+24+24+140))
# pno3 = ((11+153)/(39+11+11+153))*((11+153)/(39+11+11+153))
# pno2 = ((17+147)*(17+147))/N
print(pno, pno5)
pe = pyes + pno
pe5 = pyes5 + pno5
# pe2 = pyes2 + pno2
# pe3 = pyes3 + pno3
# pe2 = (pyes2 + pno2)/N
print(pe, pe5)
k = (p0-pe)/(1-pe)
k2 = (p02-pe)/(1-pe)
k3 = (p03-pe)/(1-pe)
k4 = (p04-pe)/(1-pe)
k5 = (p05-pe5)/(1-pe5)
print(k, k2, k3, k4, k5)
