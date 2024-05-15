import matplotlib.pyplot as plt

frozen_errs = {
        100: 25.3834,
        200: 26.5535,
        300: 26.827,
        400: 26.9352,
        500: 26.8758,
        600: 26.7711,
        700: 27.2708,
        800: 27.5764,
        900: 27.4393,
        1000: 27.5509}

plt.plot(frozen_errs.keys(), frozen_errs.values(), marker='o', linestyle='-')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Graph of Integers vs Floats')
plt.show()
