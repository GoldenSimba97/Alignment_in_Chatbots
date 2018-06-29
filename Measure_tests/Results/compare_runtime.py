glove1 = 15.83999252319336 + 7.853325128555298 + 17.404993295669556 + 12.768200159072876 + 6.380335569381714 + 12.735698461532593 + 6.490077972412109 + 11.304192543029785 + 4.922359228134155 + 2.176537036895752
glove2 = 4.978275299072266 + 8.148227214813232 + 13.085256099700928 + 4.990418195724487 + 6.755056142807007 + 19.843159437179565 + 3.923243522644043 + 6.705487966537476 + 8.348900318145752 + 6.680120944976807
glove3 = 11.245543003082275 + 1.6362583637237549 + 3.885110855102539 + 8.646761655807495 + 19.154739379882812 + 3.3365325927734375 + 6.594624757766724 + 9.896395206451416 + 2.248586416244507 + 29.173794984817505
glove4 = 7.2040135860443115 + 2.250251054763794 + 10.995946407318115 + 6.672685146331787 +  2.179388999938965 + 9.961640357971191 + 2.8126721382141113 + 6.6692328453063965 + 13.115929365158081 + 11.513656616210938
glove5 = 10.944499969482422 + 5.008238077163696 + 8.311928987503052 + 5.028008937835693 + 0.0006248950958251953 + 11.779347896575928 + 8.22848629951477 + 6.5459043979644775 + 4.918145418167114 + 13.16305422782898

# Runtime GloVe
glove = (glove1+glove2+glove3+glove4+glove5)/50
print(glove)

form1 = 0.004648685455322266 + 0.002583026885986328 + 0.0028226375579833984 + 0.002118825912475586 + 0.002332925796508789 + 0.00310516357421875 + 0.0030524730682373047 + 0.0038313865661621094 + 0.0034906864166259766 + 0.0035560131072998047
form2 = 0.003055095672607422 + 0.002477884292602539 + 0.006231784820556641 + 0.08656835556030273 + 0.0040242671966552734 + 0.00475621223449707 + 0.0053501129150390625 + 0.0039272308349609375 + 0.004292726516723633 + 0.0028579235076904297
form3 = 0.0053882598876953125 + 0.0031499862670898438 + 0.004694461822509766 + 0.0028603076934814453 + 0.006261587142944336 + 0.0024979114532470703 + 0.0042934417724609375 + 0.08178329467773438 + 0.0029785633087158203 + 0.0043451786041259766
form4 = 0.007150888442993164 + 0.0033936500549316406 + 0.0025267601013183594 + 0.0037202835083007812 + 0.0031232833862304688 + 0.004418134689331055 + 0.002957582473754883 + 0.003824472427368164 + 0.0031316280364990234 + 0.002247333526611328
form5 = 0.002828359603881836 + 0.002720355987548828 + 0.0019526481628417969 + 0.002534627914428711 + 0.003921031951904297 + 0.0047113895416259766 + 0.003705739974975586 + 0.0020825862884521484 + 0.0028204917907714844 + 0.002077341079711914

# Runtime formal and informal words lists
form = (form1+form2+form3+form4+form5)/50
print(form)

al1 = 0.009807348251342773 + 0.006708621978759766 + 0.008419036865234375 + 0.008610248565673828 + 0.006894111633300781 + 0.01047205924987793 + 0.0118255615234375 + 0.012129068374633789 + 0.012029886245727539 + 0.015437841415405273
al2 = 0.013780832290649414 + 0.012669563293457031 + 0.024452924728393555 + 0.013444662094116211 + 0.017679452896118164 + 0.02795886993408203 + 0.03141975402832031 + 0.019311189651489258 + 0.016525983810424805 + 0.016455650329589844
al3 = 0.028942108154296875 + 0.01731085777282715 + 0.03566908836364746 + 0.017714977264404297 + 0.029824495315551758 + 0.018173694610595703 + 0.036309003829956055 + 0.01943349838256836 + 0.025248050689697266 + 0.05510139465332031
al4 = 0.07501935958862305 + 0.027323007583618164 + 0.02881026268005371 + 0.022832155227661133 + 0.02950596809387207 + 0.041754722595214844 + 0.03574347496032715 + 0.03389167785644531 + 0.02466559410095215 + 0.024426937103271484
al5 = 0.032268524169921875 + 0.025297880172729492 + 0.025423288345336914 + 0.02557849884033203 + 0.040276288986206055 + 0.08309459686279297 + 0.04197812080383301 + 0.027027606964111328 + 0.027694225311279297 + 0.028834104537963867

# Runtime linguistic alignment
al = (al1+al2+al3+al4+al5)/50
print(al)

# Comparing runtime
print(glove/form)
print(glove/al)
print(al/form)