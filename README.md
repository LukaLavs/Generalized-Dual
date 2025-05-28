# NDual: Razširjena podpora za avtomatsko odvajanje z uporabo generaliziranih dualnih števil

Ta repozitorij implementira **razširjene (generalne) dualne številske tipe**, ki omogočajo **avtomatsko računanje odvodov** funkcij do poljubnega reda.

## 📘 Teorija

Dualna števila so razširitev realnih števil ob dodatku nilpotentnega elementa \( \varepsilon \), za katerega velja:

\[
\varepsilon^{n+1} = 0, \quad \varepsilon^k \ne 0 \text{ za } k \leq n
\]

Splošna oblika razširjenega dualnega števila reda \( n \) je:

\[
X = a + b_1 \varepsilon + b_2 \varepsilon^2 + \cdots + b_n \varepsilon^n
\]

kjer:
- \( a \in \mathbb{R} \) je glavna vrednost,
- \( b_k \in \mathbb{R} \) predstavljajo \( k \)-ti odvod funkcije v \( a \), pomnožen z \( \frac{1}{k!} \),
- \( \varepsilon \) je nilpotenten simbol (formalni parameter) z lastnostjo \( \varepsilon^{n+1} = 0 \).

Če funkcijo \( f \colon \mathbb{R} \to \mathbb{R} \) ovrednotimo na \( X \), dobimo:

\[
f(X) = f(a) + f'(a)\varepsilon + \frac{f''(a)}{2!}\varepsilon^2 + \cdots + \frac{f^{(n)}(a)}{n!}\varepsilon^n
\]

To omogoča hkratni izračun funkcije in njenih odvodov z enim samim klicem.

## ✅ Kaj podpira ta knjižnica?

- Sinus, kosinus, tangens, eksponentna, logaritemska funkcija, potence, koreni, inverzne funkcije, ...
- Posebne funkcije: \(\Gamma(x)\), faktoriele, binomski koeficienti, padejoča in naraščajoča faktoriela
- Kombiniranje funkcij za ustvarjanje kompleksnih izrazov
- Stabilna numerika, ki samodejno zazna singularnosti ali nedefiniranosti

## 🧪 Primer

```python
X = NDualNumber(1.0, n=3)
f = lambda X: sin(X) * exp(X)
print(f(X))  # vrne: [f(1), f'(1), f''(1)/2!, f'''(1)/3!]

