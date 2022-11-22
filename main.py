from abc import ABC, abstractmethod

class Expr:
    _der: float

    def __init__(self):
        self._der = 0

    def accumulate(self, val):
        self._der += val

    def _clear_grad(self):
        self._der = 0

    @abstractmethod
    def evaluate(self):
        ...

    @abstractmethod
    def backprop(self):
        ...

    def backwards(self):
        self.accumulate(1)
        self.backprop()

    @abstractmethod
    def __str__(self):
        ...

class Const(Expr):
    _val: float
    
    def __init__(self, val, name=None):
        super().__init__()
        self._val = val
        self.name = name

    def __str__(self):
        if self.name:
            return f"{self.name}({self._val})"
        return self._val.__str__()

    def evaluate(self):
        return self._val

    def backprop(self):
        ...

class BinaryOp(Expr):
    l: Expr 
    r: Expr

    def __init__(self, l, r):
        super().__init__()
        self.l = l
        self.r = r

class Plus(BinaryOp):

    def evaluate(self):
        return self.l.evaluate()+self.r.evaluate()

    def backprop(self):
        self.l.accumulate(self._der)
        self.r.accumulate(self._der)

        self.l.backprop()
        self.r.backprop()

    def __str__(self):
        return f"({self.l}+{self.r})"


class Times(BinaryOp):

    def evaluate(self):
        return self.l.evaluate()*self.r.evaluate()

    def backprop(self):
        self.l.accumulate(self._der*self.r.evaluate())
        self.r.accumulate(self._der*self.l.evaluate())

        self.l.backprop()
        self.r.backprop()


    def __str__(self):
        return f"({self.l}*{self.r})"


if __name__ == "__main__":
    a = Const(-12312, "a")
    b = Const(19123, "b")

    c = Plus(a,b)
    d = Times(c,b)
    
    print(f"{d}={d.evaluate()}")
    
    d.backwards()
    print(f"grad(a)={a._der}")
    print(f"grad(b)={b._der}")
    print(f"grad(c)={c._der}")
    print(f"grad(d)={d._der}")
   
    d_before = d.evaluate()
    b._val += 0.001

    print("empirical grad for b=",(d.evaluate() - d_before) / 0.001)
