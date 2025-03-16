/-- Auxiliary definition for `proveFib` extension. -/
def IsFibAux (n a b : ℕ) :=
  fib n = a ∧ fib (n + 1) = b


theorem isFibAux_zero : IsFibAux 0 0 1 :=
  ⟨fib_zero, fib_one⟩


theorem isFibAux_one : IsFibAux 1 1 1 :=
  ⟨fib_one, fib_two⟩


theorem isFibAux_two_mul {n a b n' a' b' : ℕ} (H : IsFibAux n a b)
    (hn : 2 * n = n') (h1 : a * (2 * b - a) = a') (h2 : a * a + b * b = b') :
    IsFibAux n' a' b' :=
      /-
        n a b n' a' b' : Nat
        H : Mathlib.Meta.NormNum.IsFibAux n a b
        hn : Eq (HMul.hMul 2 n) n'
        h1 : Eq (HMul.hMul a (HSub.hSub (HMul.hMul 2 b) a)) a'
        h2 : Eq (HAdd.hAdd (HMul.hMul a a) (HMul.hMul b b)) b'
        ⊢ Eq (Nat.fib n') a'
      -/
  ⟨by rw [← hn, fib_two_mul, H.1, H.2, ← h1],
      /-
        🎉 no goals
      -/
      /-
        n a b n' a' b' : Nat
        H : Mathlib.Meta.NormNum.IsFibAux n a b
        hn : Eq (HMul.hMul 2 n) n'
        h1 : Eq (HMul.hMul a (HSub.hSub (HMul.hMul 2 b) a)) a'
        h2 : Eq (HAdd.hAdd (HMul.hMul a a) (HMul.hMul b b)) b'
        ⊢ Eq (Nat.fib (HAdd.hAdd n' 1)) b'
      -/
   by rw [← hn, fib_two_mul_add_one, H.1, H.2, pow_two, pow_two, add_comm, h2]⟩
      /-
        🎉 no goals
      -/


theorem isFibAux_two_mul_add_one {n a b n' a' b' : ℕ} (H : IsFibAux n a b)
    (hn : 2 * n + 1 = n') (h1 : a * a + b * b = a') (h2 : b * (2 * a + b) = b') :
    IsFibAux n' a' b' :=
      /-
        n a b n' a' b' : Nat
        H : Mathlib.Meta.NormNum.IsFibAux n a b
        hn : Eq (HAdd.hAdd (HMul.hMul 2 n) 1) n'
        h1 : Eq (HAdd.hAdd (HMul.hMul a a) (HMul.hMul b b)) a'
        h2 : Eq (HMul.hMul b (HAdd.hAdd (HMul.hMul 2 a) b)) b'
        ⊢ Eq (Nat.fib n') a'
      -/
  ⟨by rw [← hn, fib_two_mul_add_one, H.1, H.2, pow_two, pow_two, add_comm, h1],
      /-
        🎉 no goals
      -/
      /-
        n a b n' a' b' : Nat
        H : Mathlib.Meta.NormNum.IsFibAux n a b
        hn : Eq (HAdd.hAdd (HMul.hMul 2 n) 1) n'
        h1 : Eq (HAdd.hAdd (HMul.hMul a a) (HMul.hMul b b)) a'
        h2 : Eq (HMul.hMul b (HAdd.hAdd (HMul.hMul 2 a) b)) b'
        ⊢ Eq (Nat.fib (HAdd.hAdd n' 1)) b'
      -/
   by rw [← hn, fib_two_mul_add_two, H.1, H.2, h2]⟩
      /-
        🎉 no goals
      -/


partial def proveNatFibAux (en' : Q(ℕ)) : (ea' eb' : Q(ℕ)) × Q(IsFibAux $en' $ea' $eb') :=
  match en'.natLit! with
  | 0 =>
    show (ea' eb' : Q(ℕ)) × Q(IsFibAux 0 $ea' $eb') from
    ⟨mkRawNatLit 0, mkRawNatLit 1, q(isFibAux_zero)⟩
  | 1 =>
    show (ea' eb' : Q(ℕ)) × Q(IsFibAux 1 $ea' $eb') from
    ⟨mkRawNatLit 1, mkRawNatLit 1, q(isFibAux_one)⟩
  | n' =>
    have en : Q(ℕ) := mkRawNatLit <| n' / 2
    let ⟨ea, eb, H⟩ := proveNatFibAux en
    let a := ea.natLit!
    let b := eb.natLit!
    if n' % 2 == 0 then
      have hn : Q(2 * $en = $en') := (q(Eq.refl $en') : Expr)
      have ea' : Q(ℕ) := mkRawNatLit <| a * (2 * b - a)
      have eb' : Q(ℕ) := mkRawNatLit <| a * a + b * b
      have h1 : Q($ea * (2 * $eb - $ea) = $ea') := (q(Eq.refl $ea') : Expr)
      have h2 : Q($ea * $ea + $eb * $eb = $eb') := (q(Eq.refl $eb') : Expr)
      ⟨ea', eb', q(isFibAux_two_mul $H $hn $h1 $h2)⟩
    else
      have hn : Q(2 * $en + 1 = $en') := (q(Eq.refl $en') : Expr)
      have ea' : Q(ℕ) := mkRawNatLit <| a * a + b * b
      have eb' : Q(ℕ) := mkRawNatLit <| b * (2 * a + b)
      have h1 : Q($ea * $ea + $eb * $eb = $ea') := (q(Eq.refl $ea') : Expr)
      have h2 : Q($eb * (2 * $ea + $eb) = $eb') := (q(Eq.refl $eb') : Expr)
      ⟨ea', eb', q(isFibAux_two_mul_add_one $H $hn $h1 $h2)⟩


theorem isFibAux_two_mul_done {n a b n' a' : ℕ} (H : IsFibAux n a b)
    (hn : 2 * n = n') (h : a * (2 * b - a) = a') : fib n' = a' :=
  (isFibAux_two_mul H hn h rfl).1


theorem isFibAux_two_mul_add_one_done {n a b n' a' : ℕ} (H : IsFibAux n a b)
    (hn : 2 * n + 1 = n') (h : a * a + b * b = a') : fib n' = a' :=
  (isFibAux_two_mul_add_one H hn h rfl).1


/-- Given the natural number literal `ex`, returns `Nat.fib ex` as a natural number literal
and an equality proof. Panics if `ex` isn't a natural number literal. -/
def proveNatFib (en' : Q(ℕ)) : (em : Q(ℕ)) × Q(Nat.fib $en' = $em) :=
  match en'.natLit! with
  | 0 => show (em : Q(ℕ)) × Q(Nat.fib 0 = $em) from ⟨mkRawNatLit 0, q(Nat.fib_zero)⟩
  | 1 => show (em : Q(ℕ)) × Q(Nat.fib 1 = $em) from ⟨mkRawNatLit 1, q(Nat.fib_one)⟩
  | 2 => show (em : Q(ℕ)) × Q(Nat.fib 2 = $em) from ⟨mkRawNatLit 1, q(Nat.fib_two)⟩
  | n' =>
    have en : Q(ℕ) := mkRawNatLit <| n' / 2
    let ⟨ea, eb, H⟩ := proveNatFibAux en
    let a := ea.natLit!
    let b := eb.natLit!
    if n' % 2 == 0 then
      have hn : Q(2 * $en = $en') := (q(Eq.refl $en') : Expr)
      have ea' : Q(ℕ) := mkRawNatLit <| a * (2 * b - a)
      have h1 : Q($ea * (2 * $eb - $ea) = $ea') := (q(Eq.refl $ea') : Expr)
      ⟨ea', q(isFibAux_two_mul_done $H $hn $h1)⟩
    else
      have hn : Q(2 * $en + 1 = $en') := (q(Eq.refl $en') : Expr)
      have ea' : Q(ℕ) := mkRawNatLit <| a * a + b * b
      have h1 : Q($ea * $ea + $eb * $eb = $ea') := (q(Eq.refl $ea') : Expr)
      ⟨ea', q(isFibAux_two_mul_add_one_done $H $hn $h1)⟩


theorem isNat_fib : {x nx z : ℕ} → IsNat x nx → Nat.fib nx = z → IsNat (Nat.fib x) z
  | _, _, _, ⟨rfl⟩, rfl => ⟨rfl⟩


/-- Evaluates the `Nat.fib` function. -/
@[norm_num Nat.fib _]
def evalNatFib : NormNumExt where eval {_ _} e := do
  let .app _ (x : Q(ℕ)) ← Meta.whnfR e | failure
  let sℕ : Q(AddMonoidWithOne ℕ) := q(instAddMonoidWithOneNat)
  let ⟨ex, p⟩ ← deriveNat x sℕ
  let ⟨ey, pf⟩ := proveNatFib ex
  let pf' : Q(IsNat (Nat.fib $x) $ey) := q(isNat_fib $p $pf)
  return .isNat sℕ ey pf'


