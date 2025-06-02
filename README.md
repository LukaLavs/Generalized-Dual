# Posplošene dualne številke

Ta projekt implementira razširjeno aritmetiko posplošenih (trunciranih) dualnih števil za avtomatsko diferenciacijo višjih redov v več dimenzijah.

Več o teoriji: [Audi algebra](https://darioizzo.github.io/audi/theory_algebra.html)

## Funkcionalnosti

### Osnovne funkcije

- **Aregati**: `gsum(list)`, `prod(list)`
- **Osnovna matematika**: `gabs`, `sign`, `exp`, `log(x, base)`, `log2`, `log10`, `gpow`, `sqrt`, `cbrt`
- **Trigonometrične**: `sin`, `cos`, `tan`, `cot`, `sec`, `csc`
- **Inverzne trigonometrične**: `asin`, `acos`, `atan`, `acot`
- **Hiperbolične**: `sinh`, `cosh`, `tanh`, `coth`, `sech`, `csch`
- **Inverzne hiperbolične**: `asinh`, `acosh`, `atanh`, `acoth`, `asech`, `acsch`
- **Posebne**: `erf`

### Napredne funkcije (`advanced.`)

- **Gamma funkcije**: `gamma`, `loggamma`, `factorial`, `falling_factorial`, `rising_factorial`, `comb`, `beta`
- **Posebni integrali**: `Li`, `Ei`, `Si`, `Ci`, `S`, `C`
- **Orodja**: `taylor_val`

### Metode za GDual objekte

Uporabljamo kot `X.method(...)`, kjer je `X` objekt tipa `GDual`:

- `diff()` – parcialni odvod  
- `gradient()` – gradientni vektor  
- `hessian()` – Hessova matrika  
- `univariate_derivatives(var)` – vrne \([f, f', f'', ..., f^{(n)}]\)  
- `taylor_coeffs(order)` – Taylorjevi koeficienti do reda `order` za eno spremenljivko


