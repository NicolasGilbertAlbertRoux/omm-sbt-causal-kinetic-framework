from experiments.run_simulation import run_simulation

def test_reproducibility():
    X1, _, _ = run_simulation()
    X2, _, _ = run_simulation()

    diff = abs(X1.mean() - X2.mean())
    print("Mean difference:", diff)

if __name__ == "__main__":
    test_reproducibility()