import ndual

# Primer 1:
f1 = lambda X: ndual.sin(ndual.pow(ndual.advanced.gamma(X), X / 10)) + ndual.cosh(ndual.advanced.comb(ndual.exp(X), X))
f = lambda X: f1(X) * (ndual.acoth(9*ndual.advanced.falling_factorial(X, X*2.1) * X*ndual.atanh(X/10)))

X = ndual.NDualNumber(2, n=3)
result = f(X)
mathematica_result = 2.0354581145635088 * 10**15 # Za vrednost 3 odvoda v x=2
error_ = abs(mathematica_result - result.derivate(3))
print(f"Napaka je: {error_}, vendar se rešitvi ujemata na prvih 14 cifrah")
print(result.derivate("all"))

############################### 
print("\n")

# Primer 2:
f = lambda X: (
    0.1 * ndual.tan(
        ndual.pow(
            ndual.advanced.gamma(ndual.log1p(X + 1e-4)),  # prepreči singularnost v log1p(0)
            ndual.exp(X / 15)
        )
    )
    + 0.01 * ndual.sinh(
        ndual.advanced.comb(
            ndual.advanced.rising_factorial(X, 2),
            ndual.advanced.falling_factorial(X + 1, 2)
        )
    )
    * ndual.log(
        ndual.sqrt(
            ndual.advanced.factorial(X) / 10 + 1
        ) * ndual.atanh(X / 6 + 1e-5)
    )
)


X = ndual.NDualNumber(1.1, n=4)
result = f(X)
mathematica_result = 556.2586596664949 # 4 odvod v točki x=1.1
error_ = abs(mathematica_result - result.derivate(4))
print(f"Napaka je: {error_}, rešitvi se zopet ujemata na prvih 14 cifrah")
print(result.derivate("all"))

print("\n")
X = ndual.NDualNumber(1.1, n=9)
result = f(X)
mathematica_result = -2.8143203546923037*10**9
error_ = abs(mathematica_result - result.derivate(9))
print(f"Napaka je: {error_}, rešitvi se ujemata v 13-ih cifrah, le da je Mathematica rabila 18 sekund")
print(result.derivate("all"))