class A:
    def test(self):
        print("A")


class B(A):
    def print(self):
        self.test()

    # def test(self):
    #     print("B")


B().print()

a = [1, 2]
b = a
print(b)
a[1] = 3
print(b)
