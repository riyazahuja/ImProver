/-- Evaluate a polynomial `p` given a ring hom `f` from the scalar ring
  to the target and a value `x` for the variable in the target -/
irreducible_def eval₂ (p : R[X]) : S :=
  p.sum fun e a => f a * x ^ e


theorem eval₂_eq_sum {f : R →+* S} {x : S} : p.eval₂ f x = p.sum fun e a => f a * x ^ e := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    ⊢ Eq (Polynomial.eval₂ f x p) (p.sum fun e a => HMul.hMul (f a) (HPow.hPow x e))
  -/
  rw [eval₂_def]
  /-
    🎉 no goals
  -/


theorem eval₂_congr {R S : Type*} [Semiring R] [Semiring S] {f g : R →+* S} {s t : S}
    {φ ψ : R[X]} : f = g → s = t → φ = ψ → eval₂ f s φ = eval₂ g t ψ := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f g : RingHom R S
    s t : S
    φ ψ : Polynomial R
    ⊢ Eq f g → Eq s t → Eq φ ψ → Eq (Polynomial.eval₂ f s φ) (Polynomial.eval₂ g t …
  -/
  rintro rfl rfl rfl; rfl
                      /-
                        🎉 no goals
                      -/


@[simp]
                                                    /-
                                                      R : Type u
                                                      S : Type v
                                                      inst✝¹ : Semiring R
                                                      inst✝ : Semiring S
                                                      f : RingHom R S
                                                      x : S
                                                      ⊢ Eq (Polynomial.eval₂ f x 0) 0
                                                    -/
theorem eval₂_zero : (0 : R[X]).eval₂ f x = 0 := by simp [eval₂_eq_sum]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
                                              /-
                                                R : Type u
                                                S : Type v
                                                a : R
                                                inst✝¹ : Semiring R
                                                inst✝ : Semiring S
                                                f : RingHom R S
                                                x : S
                                                ⊢ Eq (Polynomial.eval₂ f x (Polynomial.C a)) (f a)
                                              -/
theorem eval₂_C : (C a).eval₂ f x = f a := by simp [eval₂_eq_sum]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
                                        /-
                                          R : Type u
                                          S : Type v
                                          inst✝¹ : Semiring R
                                          inst✝ : Semiring S
                                          f : RingHom R S
                                          x : S
                                          ⊢ Eq (Polynomial.eval₂ f x Polynomial.X) x
                                        -/
theorem eval₂_X : X.eval₂ f x = x := by simp [eval₂_eq_sum]
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem eval₂_monomial {n : ℕ} {r : R} : (monomial n r).eval₂ f x = f r * x ^ n := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    n : Nat
    r : R
    ⊢ Eq (Polynomial.eval₂ f x ((Polynomial.monomial n) r)) (HMul.hMul (f r) (HPow …
  -/
  simp [eval₂_eq_sum]
  /-
    🎉 no goals
  -/


@[simp]
theorem eval₂_X_pow {n : ℕ} : (X ^ n).eval₂ f x = x ^ n := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    n : Nat
    ⊢ Eq (Polynomial.eval₂ f x (HPow.hPow Polynomial.X n)) (HPow.hPow x n)
  -/
  rw [X_pow_eq_monomial]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    n : Nat
    ⊢ Eq (Polynomial.eval₂ f x ((Polynomial.monomial n) 1)) (HPow.hPow x n)
  -/
  convert eval₂_monomial f x (n := n) (r := 1)
  /-
    case h.e'_3
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    n : Nat
    ⊢ Eq (HPow.hPow x n) (HMul.hMul (f 1) (HPow.hPow x n))
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem eval₂_add : (p + q).eval₂ f x = p.eval₂ f x + q.eval₂ f x := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p q : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    ⊢ Eq (Polynomial.eval₂ f x (HAdd.hAdd p q)) (HAdd.hAdd (Polynomial.eval₂ f x p …
  -/
  simp only [eval₂_eq_sum]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p q : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    ⊢ Eq ((HAdd.hAdd p q).sum fun e a => HMul.hMul (f a) (HPow.hPow x e)) (HAdd.hA …
  -/
                          /-
                            🎉 no goals
                          -/
  apply sum_add_index <;> simp [add_mul]
                          /-
                            🎉 no goals
                          -/


@[simp]
                                                   /-
                                                     R : Type u
                                                     S : Type v
                                                     inst✝¹ : Semiring R
                                                     inst✝ : Semiring S
                                                     f : RingHom R S
                                                     x : S
                                                     ⊢ Eq (Polynomial.eval₂ f x 1) 1
                                                   -/
theorem eval₂_one : (1 : R[X]).eval₂ f x = 1 := by rw [← C_1, eval₂_C, f.map_one]
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- `eval₂AddMonoidHom (f : R →+* S) (x : S)` is the `AddMonoidHom` from
`R[X]` to `S` obtained by evaluating the pushforward of `p` along `f` at `x`. -/
@[simps]
def eval₂AddMonoidHom : R[X] →+ S where
  toFun := eval₂ f x
  map_zero' := eval₂_zero _ _
  map_add' _ _ := eval₂_add _ _


@[simp]
theorem eval₂_natCast (n : ℕ) : (n : R[X]).eval₂ f x = n := by
  induction n with
  | zero => simp only [eval₂_zero, Nat.cast_zero]
  | succ n ih => rw [n.cast_succ, eval₂_add, ih, eval₂_one, n.cast_succ]


@[deprecated (since := "2024-04-17")]
alias eval₂_nat_cast := eval₂_natCast

-- See note [no_index around OfNat.ofNat]

@[simp]
lemma eval₂_ofNat {S : Type*} [Semiring S] (n : ℕ) [n.AtLeastTwo] (f : R →+* S) (a : S) :
    (no_index (OfNat.ofNat n : R[X])).eval₂ f a = OfNat.ofNat n := by
  /-
    R : Type u
    inst✝² : Semiring R
    S : Type u_1
    inst✝¹ : Semiring S
    n : Nat
    inst✝ : n.AtLeastTwo
    f : RingHom R S
    a : S
    ⊢ Eq (Polynomial.eval₂ f a (OfNat.ofNat n)) (OfNat.ofNat n)
  -/
  simp [OfNat.ofNat]
  /-
    🎉 no goals
  -/


theorem eval₂_sum (p : T[X]) (g : ℕ → T → R[X]) (x : S) :
    (p.sum g).eval₂ f x = p.sum fun n a => (g n a).eval₂ f x := by
  let T : R[X] →+ S :=
    { toFun := eval₂ f x
      map_zero' := eval₂_zero _ _
      map_add' := fun p q => eval₂_add _ _ }
  /-
    R : Type u
    S : Type v
    T✝ : Type w
    inst✝² : Semiring R
    inst✝¹ : Semiring S
    f : RingHom R S
    inst✝ : Semiring T✝
    p : Polynomial T✝
    g : Nat → T✝ → Polynomial R
    x : S
    T : AddMonoidHom (Polynomial R) S := { toFun := Polynomial.eval₂ f x, map_zero …
    ⊢ Eq (Polynomial.eval₂ f x (p.sum g)) (p.sum fun n a => Polynomial.eval₂ f x ( …
  -/
  have A : ∀ y, eval₂ f x y = T y := fun y => rfl
  /-
    R : Type u
    S : Type v
    T✝ : Type w
    inst✝² : Semiring R
    inst✝¹ : Semiring S
    f : RingHom R S
    inst✝ : Semiring T✝
    p : Polynomial T✝
    g : Nat → T✝ → Polynomial R
    x : S
    T : AddMonoidHom (Polynomial R) S := { toFun := Polynomial.eval₂ f x, map_zero …
    A : ∀ (y : Polynomial R), Eq (Polynomial.eval₂ f x y) (T y)
    ⊢ Eq (Polynomial.eval₂ f x (p.sum g)) (p.sum fun n a => Polynomial.eval₂ f x ( …
  -/
  simp only [A]
  /-
    R : Type u
    S : Type v
    T✝ : Type w
    inst✝² : Semiring R
    inst✝¹ : Semiring S
    f : RingHom R S
    inst✝ : Semiring T✝
    p : Polynomial T✝
    g : Nat → T✝ → Polynomial R
    x : S
    T : AddMonoidHom (Polynomial R) S := { toFun := Polynomial.eval₂ f x, map_zero …
    A : ∀ (y : Polynomial R), Eq (Polynomial.eval₂ f x y) (T y)
    ⊢ Eq (T (p.sum g)) (p.sum fun n a => T (g n a))
  -/
  rw [sum, map_sum, sum]
  /-
    🎉 no goals
  -/


theorem eval₂_list_sum (l : List R[X]) (x : S) : eval₂ f x l.sum = (l.map (eval₂ f x)).sum :=
  map_list_sum (eval₂AddMonoidHom f x) l


theorem eval₂_multiset_sum (s : Multiset R[X]) (x : S) :
    eval₂ f x s.sum = (s.map (eval₂ f x)).sum :=
  map_multiset_sum (eval₂AddMonoidHom f x) s


theorem eval₂_finset_sum (s : Finset ι) (g : ι → R[X]) (x : S) :
    (∑ i ∈ s, g i).eval₂ f x = ∑ i ∈ s, (g i).eval₂ f x :=
  map_sum (eval₂AddMonoidHom f x) _ _


theorem eval₂_ofFinsupp {f : R →+* S} {x : S} {p : R[ℕ]} :
    eval₂ f x (⟨p⟩ : R[X]) = liftNC (↑f) (powersHom S x) p := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    p : AddMonoidAlgebra R Nat
    ⊢ Eq (Polynomial.eval₂ f x { toFinsupp := p }) ((AddMonoidAlgebra.liftNC ↑f ⇑( …
  -/
  simp only [eval₂_eq_sum, sum, toFinsupp_sum, support, coeff]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    p : AddMonoidAlgebra R Nat
    ⊢ Eq (p.support.sum fun x_1 => HMul.hMul (f (p x_1)) (HPow.hPow x x_1)) ((AddM …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem eval₂_mul_noncomm (hf : ∀ k, Commute (f <| q.coeff k) x) :
    eval₂ f x (p * q) = eval₂ f x p * eval₂ f x q := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p q : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    hf : ∀ (k : Nat), Commute (f (q.coeff k)) x
    ⊢ Eq (Polynomial.eval₂ f x (HMul.hMul p q)) (HMul.hMul (Polynomial.eval₂ f x p …
  -/
  rcases p with ⟨p⟩; rcases q with ⟨q⟩
  /-
    case ofFinsupp.ofFinsupp
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    p q : AddMonoidAlgebra R Nat
    hf : ∀ (k : Nat), Commute (f ({ toFinsupp := q }.coeff k)) x
    ⊢ Eq (Polynomial.eval₂ f x (HMul.hMul { toFinsupp := p } { toFinsupp := q }))  …
  -/
  simp only [coeff] at hf
  /-
    case ofFinsupp.ofFinsupp
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    p q : AddMonoidAlgebra R Nat
    hf : ∀ (k : Nat), Commute (f (q k)) x
    ⊢ Eq (Polynomial.eval₂ f x (HMul.hMul { toFinsupp := p } { toFinsupp := q }))  …
  -/
  simp only [← ofFinsupp_mul, eval₂_ofFinsupp]
  /-
    case ofFinsupp.ofFinsupp
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    p q : AddMonoidAlgebra R Nat
    hf : ∀ (k : Nat), Commute (f (q k)) x
    ⊢ Eq ((AddMonoidAlgebra.liftNC ↑f ⇑((powersHom S) x)) (HMul.hMul p q)) (HMul.h …
  -/
  exact liftNC_mul _ _ p q fun {k n} _hn => (hf k).pow_right n
  /-
    🎉 no goals
  -/


@[simp]
theorem eval₂_mul_X : eval₂ f x (p * X) = eval₂ f x p * x := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    ⊢ Eq (Polynomial.eval₂ f x (HMul.hMul p Polynomial.X)) (HMul.hMul (Polynomial. …
  -/
  refine _root_.trans (eval₂_mul_noncomm _ _ fun k => ?_) (by rw [eval₂_X])
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    k : Nat
    ⊢ Commute (f (Polynomial.X.coeff k)) x
  -/
  rcases em (k = 1) with (rfl | hk)
    /-
      case inl
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      p : Polynomial R
      inst✝ : Semiring S
      f : RingHom R S
      x : S
      ⊢ Commute (f (Polynomial.X.coeff 1)) x
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      p : Polynomial R
      inst✝ : Semiring S
      f : RingHom R S
      x : S
      k : Nat
      hk : Not (Eq k 1)
      ⊢ Commute (f (Polynomial.X.coeff k)) x
    -/
  · simp [coeff_X_of_ne_one hk]
    /-
      🎉 no goals
    -/


@[simp]
                                                                /-
                                                                  R : Type u
                                                                  S : Type v
                                                                  inst✝¹ : Semiring R
                                                                  p : Polynomial R
                                                                  inst✝ : Semiring S
                                                                  f : RingHom R S
                                                                  x : S
                                                                  ⊢ Eq (Polynomial.eval₂ f x (HMul.hMul Polynomial.X p)) (HMul.hMul (Polynomial. …
                                                                -/
theorem eval₂_X_mul : eval₂ f x (X * p) = eval₂ f x p * x := by rw [X_mul, eval₂_mul_X]
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem eval₂_mul_C' (h : Commute (f a) x) : eval₂ f x (p * C a) = eval₂ f x p * f a := by
  /-
    R : Type u
    S : Type v
    a : R
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    h : Commute (f a) x
    ⊢ Eq (Polynomial.eval₂ f x (HMul.hMul p (Polynomial.C a))) (HMul.hMul (Polynom …
  -/
  rw [eval₂_mul_noncomm, eval₂_C]
  /-
    case hf
    R : Type u
    S : Type v
    a : R
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    h : Commute (f a) x
    ⊢ ∀ (k : Nat), Commute (f ((Polynomial.C a).coeff k)) x
  -/
  intro k
  /-
    case hf
    R : Type u
    S : Type v
    a : R
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    h : Commute (f a) x
    k : Nat
    ⊢ Commute (f ((Polynomial.C a).coeff k)) x
  -/
  by_cases hk : k = 0
    /-
      case pos
      R : Type u
      S : Type v
      a : R
      inst✝¹ : Semiring R
      p : Polynomial R
      inst✝ : Semiring S
      f : RingHom R S
      x : S
      h : Commute (f a) x
      k : Nat
      hk : Eq k 0
      ⊢ Commute (f ((Polynomial.C a).coeff k)) x
    -/
  · simp only [hk, h, coeff_C_zero, coeff_C_ne_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      S : Type v
      a : R
      inst✝¹ : Semiring R
      p : Polynomial R
      inst✝ : Semiring S
      f : RingHom R S
      x : S
      h : Commute (f a) x
      k : Nat
      hk : Not (Eq k 0)
      ⊢ Commute (f ((Polynomial.C a).coeff k)) x
    -/
  · simp only [coeff_C_ne_zero hk, RingHom.map_zero, Commute.zero_left]
    /-
      🎉 no goals
    -/


theorem eval₂_list_prod_noncomm (ps : List R[X])
    (hf : ∀ p ∈ ps, ∀ (k), Commute (f <| coeff p k) x) :
    eval₂ f x ps.prod = (ps.map (Polynomial.eval₂ f x)).prod := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    x : S
    ps : List (Polynomial R)
    hf : ∀ (p : Polynomial R), Membership.mem ps p → ∀ (k : Nat), Commute (f (p.co …
    ⊢ Eq (Polynomial.eval₂ f x ps.prod) (List.map (Polynomial.eval₂ f x) ps).prod
  -/
  induction' ps using List.reverseRecOn with ps p ihp
    /-
      case nil
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      x : S
      hf : ∀ (p : Polynomial R), Membership.mem List.nil p → ∀ (k : Nat), Commute (f …
      ⊢ Eq (Polynomial.eval₂ f x List.nil.prod) (List.map (Polynomial.eval₂ f x) Lis …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case append_singleton
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      x : S
      ps : List (Polynomial R)
      p : Polynomial R
      ihp : (∀ (p : Polynomial R), Membership.mem ps p → ∀ (k : Nat), Commute (f (p. …
      hf : ∀ (p_1 : Polynomial R), Membership.mem (HAppend.hAppend ps (List.cons p L …
      ⊢ Eq (Polynomial.eval₂ f x (HAppend.hAppend ps (List.cons p List.nil)).prod) ( …
    -/
  · simp only [List.forall_mem_append, List.forall_mem_singleton] at hf
    /-
      case append_singleton
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      x : S
      ps : List (Polynomial R)
      p : Polynomial R
      ihp : (∀ (p : Polynomial R), Membership.mem ps p → ∀ (k : Nat), Commute (f (p. …
      hf : And (∀ (x_1 : Polynomial R), Membership.mem ps x_1 → ∀ (k : Nat), Commute …
      ⊢ Eq (Polynomial.eval₂ f x (HAppend.hAppend ps (List.cons p List.nil)).prod) ( …
    -/
    simp [eval₂_mul_noncomm _ _ hf.2, ihp hf.1]
    /-
      🎉 no goals
    -/


/-- `eval₂` as a `RingHom` for noncommutative rings -/
@[simps]
def eval₂RingHom' (f : R →+* S) (x : S) (hf : ∀ a, Commute (f a) x) : R[X] →+* S where
  toFun := eval₂ f x
  map_add' _ _ := eval₂_add _ _
  map_zero' := eval₂_zero _ _
  map_mul' _p q := eval₂_mul_noncomm f x fun k => hf <| coeff q k
  map_one' := eval₂_one _ _


@[simp]
theorem eval₂_mul : (p * q).eval₂ f x = p.eval₂ f x * q.eval₂ f x :=
  eval₂_mul_noncomm _ _ fun _k => Commute.all _ _


theorem eval₂_mul_eq_zero_of_left (q : R[X]) (hp : p.eval₂ f x = 0) : (p * q).eval₂ f x = 0 := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : CommSemiring S
    f : RingHom R S
    x : S
    q : Polynomial R
    hp : Eq (Polynomial.eval₂ f x p) 0
    ⊢ Eq (Polynomial.eval₂ f x (HMul.hMul p q)) 0
  -/
  rw [eval₂_mul f x]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : CommSemiring S
    f : RingHom R S
    x : S
    q : Polynomial R
    hp : Eq (Polynomial.eval₂ f x p) 0
    ⊢ Eq (HMul.hMul (Polynomial.eval₂ f x p) (Polynomial.eval₂ f x q)) 0
  -/
  exact mul_eq_zero_of_left hp (q.eval₂ f x)
  /-
    🎉 no goals
  -/


theorem eval₂_mul_eq_zero_of_right (p : R[X]) (hq : q.eval₂ f x = 0) : (p * q).eval₂ f x = 0 := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    q : Polynomial R
    inst✝ : CommSemiring S
    f : RingHom R S
    x : S
    p : Polynomial R
    hq : Eq (Polynomial.eval₂ f x q) 0
    ⊢ Eq (Polynomial.eval₂ f x (HMul.hMul p q)) 0
  -/
  rw [eval₂_mul f x]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    q : Polynomial R
    inst✝ : CommSemiring S
    f : RingHom R S
    x : S
    p : Polynomial R
    hq : Eq (Polynomial.eval₂ f x q) 0
    ⊢ Eq (HMul.hMul (Polynomial.eval₂ f x p) (Polynomial.eval₂ f x q)) 0
  -/
  exact mul_eq_zero_of_right (p.eval₂ f x) hq
  /-
    🎉 no goals
  -/


/-- `eval₂` as a `RingHom` -/
def eval₂RingHom (f : R →+* S) (x : S) : R[X] →+* S :=
  { eval₂AddMonoidHom f x with
    map_one' := eval₂_one _ _
    map_mul' := fun _ _ => eval₂_mul _ _ }


@[simp]
theorem coe_eval₂RingHom (f : R →+* S) (x) : ⇑(eval₂RingHom f x) = eval₂ f x :=
  rfl


theorem eval₂_pow (n : ℕ) : (p ^ n).eval₂ f x = p.eval₂ f x ^ n :=
  (eval₂RingHom _ _).map_pow _ _


theorem eval₂_dvd : p ∣ q → eval₂ f x p ∣ eval₂ f x q :=
  (eval₂RingHom f x).map_dvd


theorem eval₂_eq_zero_of_dvd_of_eval₂_eq_zero (h : p ∣ q) (h0 : eval₂ f x p = 0) :
    eval₂ f x q = 0 :=
  zero_dvd_iff.mp (h0 ▸ eval₂_dvd f x h)


theorem eval₂_list_prod (l : List R[X]) (x : S) : eval₂ f x l.prod = (l.map (eval₂ f x)).prod :=
  map_list_prod (eval₂RingHom f x) l


/-- `eval x p` is the evaluation of the polynomial `p` at `x` -/
def eval : R → R[X] → R :=
  eval₂ (RingHom.id _)


theorem eval_eq_sum : p.eval x = p.sum fun e a => a * x ^ e := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    x : R
    ⊢ Eq (Polynomial.eval x p) (p.sum fun e a => HMul.hMul a (HPow.hPow x e))
  -/
  rw [eval, eval₂_eq_sum]
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    x : R
    ⊢ Eq (p.sum fun e a => HMul.hMul ((RingHom.id R) a) (HPow.hPow x e)) (p.sum fu …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem eval₂_at_apply {S : Type*} [Semiring S] (f : R →+* S) (r : R) :
    p.eval₂ f (f r) = f (p.eval r) := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    p : Polynomial R
    S : Type u_1
    inst✝ : Semiring S
    f : RingHom R S
    r : R
    ⊢ Eq (Polynomial.eval₂ f (f r) p) (f (Polynomial.eval r p))
  -/
  rw [eval₂_eq_sum, eval_eq_sum, sum, sum, map_sum f]
  /-
    R : Type u
    inst✝¹ : Semiring R
    p : Polynomial R
    S : Type u_1
    inst✝ : Semiring S
    f : RingHom R S
    r : R
    ⊢ Eq (p.support.sum fun n => HMul.hMul (f (p.coeff n)) (HPow.hPow (f r) n)) (p …
  -/
  simp only [f.map_mul, f.map_pow]
  /-
    🎉 no goals
  -/


@[simp]
theorem eval₂_at_one {S : Type*} [Semiring S] (f : R →+* S) : p.eval₂ f 1 = f (p.eval 1) := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    p : Polynomial R
    S : Type u_1
    inst✝ : Semiring S
    f : RingHom R S
    ⊢ Eq (Polynomial.eval₂ f 1 p) (f (Polynomial.eval 1 p))
  -/
  convert eval₂_at_apply (p := p) f 1
  /-
    case h.e'_2.h.e'_6
    R : Type u
    inst✝¹ : Semiring R
    p : Polynomial R
    S : Type u_1
    inst✝ : Semiring S
    f : RingHom R S
    ⊢ Eq 1 (f 1)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem eval₂_at_natCast {S : Type*} [Semiring S] (f : R →+* S) (n : ℕ) :
    p.eval₂ f n = f (p.eval n) := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    p : Polynomial R
    S : Type u_1
    inst✝ : Semiring S
    f : RingHom R S
    n : Nat
    ⊢ Eq (Polynomial.eval₂ f (↑n) p) (f (Polynomial.eval (↑n) p))
  -/
  convert eval₂_at_apply (p := p) f n
  /-
    case h.e'_2.h.e'_6
    R : Type u
    inst✝¹ : Semiring R
    p : Polynomial R
    S : Type u_1
    inst✝ : Semiring S
    f : RingHom R S
    n : Nat
    ⊢ Eq (↑n) (f ↑n)
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias eval₂_at_nat_cast := eval₂_at_natCast

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem eval₂_at_ofNat {S : Type*} [Semiring S] (f : R →+* S) (n : ℕ) [n.AtLeastTwo] :
    p.eval₂ f (no_index (OfNat.ofNat n)) = f (p.eval (OfNat.ofNat n)) := by
  /-
    R : Type u
    inst✝² : Semiring R
    p : Polynomial R
    S : Type u_1
    inst✝¹ : Semiring S
    f : RingHom R S
    n : Nat
    inst✝ : n.AtLeastTwo
    ⊢ Eq (Polynomial.eval₂ f (OfNat.ofNat n) p) (f (Polynomial.eval (OfNat.ofNat n …
  -/
  simp [OfNat.ofNat]
  /-
    🎉 no goals
  -/


@[simp]
theorem eval_C : (C a).eval x = a :=
  eval₂_C _ _


@[simp]
                                                           /-
                                                             R : Type u
                                                             inst✝ : Semiring R
                                                             x : R
                                                             n : Nat
                                                             ⊢ Eq (Polynomial.eval x ↑n) ↑n
                                                           -/
theorem eval_natCast {n : ℕ} : (n : R[X]).eval x = n := by simp only [← C_eq_natCast, eval_C]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[deprecated (since := "2024-04-17")]
alias eval_nat_cast := eval_natCast

-- See note [no_index around OfNat.ofNat]

@[simp]
lemma eval_ofNat (n : ℕ) [n.AtLeastTwo] (a : R) :
    (no_index (OfNat.ofNat n : R[X])).eval a = OfNat.ofNat n := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    n : Nat
    inst✝ : n.AtLeastTwo
    a : R
    ⊢ Eq (Polynomial.eval a (OfNat.ofNat n)) (OfNat.ofNat n)
  -/
  simp only [OfNat.ofNat, eval_natCast]
  /-
    🎉 no goals
  -/


@[simp]
theorem eval_X : X.eval x = x :=
  eval₂_X _ _


@[simp]
theorem eval_monomial {n a} : (monomial n a).eval x = a * x ^ n :=
  eval₂_monomial _ _


@[simp]
theorem eval_zero : (0 : R[X]).eval x = 0 :=
  eval₂_zero _ _


@[simp]
theorem eval_add : (p + q).eval x = p.eval x + q.eval x :=
  eval₂_add _ _


@[simp]
theorem eval_one : (1 : R[X]).eval x = 1 :=
  eval₂_one _ _


@[simp]
theorem eval_C_mul : (C a * p).eval x = a * p.eval x := by
  induction p using Polynomial.induction_on' with
  | h_add p q ph qh =>
    simp only [mul_add, eval_add, ph, qh]
  | h_monomial n b =>
    simp only [mul_assoc, C_mul_monomial, eval_monomial]


@[simp]
theorem eval_natCast_mul {n : ℕ} : ((n : R[X]) * p).eval x = n * p.eval x := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    x : R
    n : Nat
    ⊢ Eq (Polynomial.eval x (HMul.hMul (↑n) p)) (HMul.hMul (↑n) (Polynomial.eval x …
  -/
  rw [← C_eq_natCast, eval_C_mul]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias eval_nat_cast_mul := eval_natCast_mul


@[simp]
theorem eval_mul_X : (p * X).eval x = p.eval x * x := by
  induction p using Polynomial.induction_on' with
  | h_add p q ph qh =>
    simp only [add_mul, eval_add, ph, qh]
  | h_monomial n a =>
    simp only [← monomial_one_one_eq_X, monomial_mul_monomial, eval_monomial, mul_one, pow_succ,
      mul_assoc]


@[simp]
theorem eval_mul_X_pow {k : ℕ} : (p * X ^ k).eval x = p.eval x * x ^ k := by
  induction k with
  | zero => simp
  | succ k ih => simp [pow_succ, ← mul_assoc, ih]


theorem eval_sum (p : R[X]) (f : ℕ → R → R[X]) (x : R) :
    (p.sum f).eval x = p.sum fun n a => (f n a).eval x :=
  eval₂_sum _ _ _ _


theorem eval_finset_sum (s : Finset ι) (g : ι → R[X]) (x : R) :
    (∑ i ∈ s, g i).eval x = ∑ i ∈ s, (g i).eval x :=
  eval₂_finset_sum _ _ _ _


/-- `IsRoot p x` implies `x` is a root of `p`. The evaluation of `p` at `x` is zero -/
def IsRoot (p : R[X]) (a : R) : Prop :=
  p.eval a = 0


instance IsRoot.decidable [DecidableEq R] : Decidable (IsRoot p a) := by
  /-
    R : Type u
    S : Type v
    T : Type w
    ι : Type y
    a b : R
    m n : Nat
    inst✝¹ : Semiring R
    p q r : Polynomial R
    x : R
    inst✝ : DecidableEq R
    ⊢ Decidable (p.IsRoot a)
  -/
  unfold IsRoot; infer_instance
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem IsRoot.def : IsRoot p a ↔ p.eval a = 0 :=
  Iff.rfl


theorem IsRoot.eq_zero (h : IsRoot p x) : eval x p = 0 :=
  h


theorem IsRoot.dvd {R : Type*} [CommSemiring R] {p q : R[X]} {x : R} (h : p.IsRoot x)
    (hpq : p ∣ q) : q.IsRoot x := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    p q : Polynomial R
    x : R
    h : p.IsRoot x
    hpq : Dvd.dvd p q
    ⊢ q.IsRoot x
  -/
  rwa [IsRoot, eval, eval₂_eq_zero_of_dvd_of_eval₂_eq_zero _ _ hpq]
  /-
    🎉 no goals
  -/


                                                                    /-
                                                                      R : Type u
                                                                      inst✝ : Semiring R
                                                                      r a : R
                                                                      hr : Ne r 0
                                                                      ⊢ Not ((Polynomial.C r).IsRoot a)
                                                                    -/
theorem not_isRoot_C (r a : R) (hr : r ≠ 0) : ¬IsRoot (C r) a := by simpa using hr
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem eval_surjective (x : R) : Function.Surjective <| eval x := fun y => ⟨C y, eval_C⟩


/-- The composition of polynomials as a polynomial. -/
def comp (p q : R[X]) : R[X] :=
  p.eval₂ C q


                                                                         /-
                                                                           R : Type u
                                                                           inst✝ : Semiring R
                                                                           p q : Polynomial R
                                                                           ⊢ Eq (p.comp q) (p.sum fun e a => HMul.hMul (Polynomial.C a) (HPow.hPow q e))
                                                                         -/
theorem comp_eq_sum_left : p.comp q = p.sum fun e a => C a * q ^ e := by rw [comp, eval₂_eq_sum]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
theorem comp_X : p.comp X = p := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq (p.comp Polynomial.X) p
  -/
  simp only [comp, eval₂_def, C_mul_X_pow_eq_monomial]
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq (p.sum fun e a => (Polynomial.monomial e) a) p
  -/
  exact sum_monomial_eq _
  /-
    🎉 no goals
  -/


@[simp]
theorem X_comp : X.comp p = p :=
  eval₂_X _ _


@[simp]
                                                   /-
                                                     R : Type u
                                                     a : R
                                                     inst✝ : Semiring R
                                                     p : Polynomial R
                                                     ⊢ Eq (p.comp (Polynomial.C a)) (Polynomial.C (Polynomial.eval a p))
                                                   -/
theorem comp_C : p.comp (C a) = C (p.eval a) := by simp [comp, map_sum (C : R →+* _)]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem C_comp : (C a).comp p = C a :=
  eval₂_C _ _


@[simp]
                                                           /-
                                                             R : Type u
                                                             inst✝ : Semiring R
                                                             p : Polynomial R
                                                             n : Nat
                                                             ⊢ Eq ((↑n).comp p) ↑n
                                                           -/
theorem natCast_comp {n : ℕ} : (n : R[X]).comp p = n := by rw [← C_eq_natCast, C_comp]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[deprecated (since := "2024-04-17")]
alias nat_cast_comp := natCast_comp


@[simp]
theorem ofNat_comp (n : ℕ) [n.AtLeastTwo] : (no_index (OfNat.ofNat n) : R[X]).comp p = n :=
  natCast_comp


@[simp]
                                                           /-
                                                             R : Type u
                                                             inst✝ : Semiring R
                                                             p : Polynomial R
                                                             ⊢ Eq (p.comp 0) (Polynomial.C (Polynomial.eval 0 p))
                                                           -/
theorem comp_zero : p.comp (0 : R[X]) = C (p.eval 0) := by rw [← C_0, comp_C]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
                                                /-
                                                  R : Type u
                                                  inst✝ : Semiring R
                                                  p : Polynomial R
                                                  ⊢ Eq (Polynomial.comp 0 p) 0
                                                -/
theorem zero_comp : comp (0 : R[X]) p = 0 := by rw [← C_0, C_comp]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
                                                 /-
                                                   R : Type u
                                                   inst✝ : Semiring R
                                                   p : Polynomial R
                                                   ⊢ Eq (p.comp 1) (Polynomial.C (Polynomial.eval 1 p))
                                                 -/
theorem comp_one : p.comp 1 = C (p.eval 1) := by rw [← C_1, comp_C]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
                                               /-
                                                 R : Type u
                                                 inst✝ : Semiring R
                                                 p : Polynomial R
                                                 ⊢ Eq (Polynomial.comp 1 p) 1
                                               -/
theorem one_comp : comp (1 : R[X]) p = 1 := by rw [← C_1, C_comp]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem add_comp : (p + q).comp r = p.comp r + q.comp r :=
  eval₂_add _ _


@[simp]
theorem monomial_comp (n : ℕ) : (monomial n a).comp p = C a * p ^ n :=
  eval₂_monomial _ _


@[simp]
theorem mul_X_comp : (p * X).comp r = p.comp r * r := by
  induction p using Polynomial.induction_on' with
  | h_add p q hp hq =>
    simp only [hp, hq, add_mul, add_comp]
  | h_monomial n b =>
    simp only [pow_succ, mul_assoc, monomial_mul_X, monomial_comp]


@[simp]
theorem X_pow_comp {k : ℕ} : (X ^ k).comp p = p ^ k := by
  induction k with
  | zero => simp
  | succ k ih => simp [pow_succ, mul_X_comp, ih]


@[simp]
theorem mul_X_pow_comp {k : ℕ} : (p * X ^ k).comp r = p.comp r * r ^ k := by
  induction k with
  | zero => simp
  | succ k ih => simp [ih, pow_succ, ← mul_assoc, mul_X_comp]


@[simp]
theorem C_mul_comp : (C a * p).comp r = C a * p.comp r := by
  induction p using Polynomial.induction_on' with
  | h_add p q hp hq =>
    simp [hp, hq, mul_add]
  | h_monomial n b =>
    simp [mul_assoc]


@[simp]
theorem natCast_mul_comp {n : ℕ} : ((n : R[X]) * p).comp r = n * p.comp r := by
  /-
    R : Type u
    inst✝ : Semiring R
    p r : Polynomial R
    n : Nat
    ⊢ Eq ((HMul.hMul (↑n) p).comp r) (HMul.hMul (↑n) (p.comp r))
  -/
  rw [← C_eq_natCast, C_mul_comp]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias nat_cast_mul_comp := natCast_mul_comp


theorem mul_X_add_natCast_comp {n : ℕ} :
    (p * (X + (n : R[X]))).comp q = p.comp q * (q + n) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    n : Nat
    ⊢ Eq ((HMul.hMul p (HAdd.hAdd Polynomial.X ↑n)).comp q) (HMul.hMul (p.comp q)  …
  -/
  rw [mul_add, add_comp, mul_X_comp, ← Nat.cast_comm, natCast_mul_comp, Nat.cast_comm, mul_add]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias mul_X_add_nat_cast_comp := mul_X_add_natCast_comp


@[simp]
theorem mul_comp {R : Type*} [CommSemiring R] (p q r : R[X]) :
    (p * q).comp r = p.comp r * q.comp r :=
  eval₂_mul _ _


@[simp]
theorem pow_comp {R : Type*} [CommSemiring R] (p q : R[X]) (n : ℕ) :
    (p ^ n).comp q = p.comp q ^ n :=
  (MonoidHom.mk (OneHom.mk (fun r : R[X] => r.comp q) one_comp) fun r s => mul_comp r s q).map_pow
    p n


theorem comp_assoc {R : Type*} [CommSemiring R] (φ ψ χ : R[X]) :
    (φ.comp ψ).comp χ = φ.comp (ψ.comp χ) := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    φ ψ χ : Polynomial R
    ⊢ Eq ((φ.comp ψ).comp χ) (φ.comp (ψ.comp χ))
  -/
  refine Polynomial.induction_on φ ?_ ?_ ?_ <;>
      /-
        case refine_1
        R : Type u_1
        inst✝ : CommSemiring R
        φ ψ χ : Polynomial R
        ⊢ ∀ (a : R), Eq (((Polynomial.C a).comp ψ).comp χ) ((Polynomial.C a).comp (ψ.c …
      -/
      /-
        case refine_1
        R : Type u_1
        inst✝ : CommSemiring R
        φ ψ χ : Polynomial R
        a✝ : R
        ⊢ Eq (((Polynomial.C a✝).comp ψ).comp χ) ((Polynomial.C a✝).comp (ψ.comp χ))
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        R : Type u_1
        inst✝ : CommSemiring R
        φ ψ χ : Polynomial R
        n✝ : Nat
        a✝¹ : R
        a✝ : Eq (((HMul.hMul (Polynomial.C a✝¹) (HPow.hPow Polynomial.X n✝)).comp ψ).c …
        ⊢ Eq (((HMul.hMul (Polynomial.C a✝¹) (HPow.hPow Polynomial.X (HAdd.hAdd n✝ 1)) …
      -/
      simp_all only [add_comp, mul_comp, C_comp, X_comp, pow_succ, ← mul_assoc]
      /-
        🎉 no goals
      -/


@[simp] lemma sum_comp (s : Finset ι) (p : ι → R[X]) (q : R[X]) :
    (∑ i ∈ s, p i).comp q = ∑ i ∈ s, (p i).comp q := Polynomial.eval₂_finset_sum _ _ _ _


/-- `map f p` maps a polynomial `p` across a ring hom `f` -/
def map : R[X] → S[X] :=
  eval₂ (C.comp f) X


@[simp]
theorem map_C : (C a).map f = C (f a) :=
  eval₂_C _ _


@[simp]
theorem map_X : X.map f = X :=
  eval₂_X _ _


@[simp]
theorem map_monomial {n a} : (monomial n a).map f = monomial n (f a) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    n : Nat
    a : R
    ⊢ Eq (Polynomial.map f ((Polynomial.monomial n) a)) ((Polynomial.monomial n) ( …
  -/
  dsimp only [map]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    n : Nat
    a : R
    ⊢ Eq (Polynomial.eval₂ (Polynomial.C.comp f) Polynomial.X ((Polynomial.monomia …
  -/
  rw [eval₂_monomial, ← C_mul_X_pow_eq_monomial]; rfl
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
protected theorem map_zero : (0 : R[X]).map f = 0 :=
  eval₂_zero _ _


@[simp]
protected theorem map_add : (p + q).map f = p.map f + q.map f :=
  eval₂_add _ _


@[simp]
protected theorem map_one : (1 : R[X]).map f = 1 :=
  eval₂_one _ _


@[simp]
protected theorem map_mul : (p * q).map f = p.map f * q.map f := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p q : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    ⊢ Eq (Polynomial.map f (HMul.hMul p q)) (HMul.hMul (Polynomial.map f p) (Polyn …
  -/
  rw [map, eval₂_mul_noncomm]
  /-
    case hf
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p q : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    ⊢ ∀ (k : Nat), Commute ((Polynomial.C.comp f) (q.coeff k)) Polynomial.X
  -/
  exact fun k => (commute_X _).symm
  /-
    🎉 no goals
  -/

-- `map` is a ring-hom unconditionally, and theoretically the definition could be replaced,
-- but this turns out not to be easy because `p.map f` does not resolve to `Polynomial.map`
-- if `map` is a `RingHom` instead of a plain function; the elaborator does not try to coerce
-- to a function before trying field (dot) notation (this may be technically infeasible);
-- the relevant code is (both lines): https://github.com/leanprover-community/
-- lean/blob/487ac5d7e9b34800502e1ddf3c7c806c01cf9d51/src/frontends/lean/elaborator.cpp#L1876-L1913

/-- `Polynomial.map` as a `RingHom`. -/
def mapRingHom (f : R →+* S) : R[X] →+* S[X] where
  toFun := Polynomial.map f
  map_add' _ _ := Polynomial.map_add f
  map_zero' := Polynomial.map_zero f
  map_mul' _ _ := Polynomial.map_mul f
  map_one' := Polynomial.map_one f


@[simp]
theorem coe_mapRingHom (f : R →+* S) : ⇑(mapRingHom f) = map f :=
  rfl

-- This is protected to not clash with the global `map_natCast`.

@[simp]
protected theorem map_natCast (n : ℕ) : (n : R[X]).map f = n :=
  map_natCast (mapRingHom f) n


@[deprecated (since := "2024-04-17")]
alias map_nat_cast := map_natCast

-- See note [no_index around OfNat.ofNat]

@[simp]
protected theorem map_ofNat (n : ℕ) [n.AtLeastTwo] :
    (no_index (OfNat.ofNat n) : R[X]).map f = OfNat.ofNat n :=
                               /-
                                 R : Type u
                                 S : Type v
                                 inst✝² : Semiring R
                                 inst✝¹ : Semiring S
                                 f : RingHom R S
                                 n : Nat
                                 inst✝ : n.AtLeastTwo
                                 ⊢ Eq (Polynomial.map f ↑n) ↑n
                               -/
  show (n : R[X]).map f = n by rw [Polynomial.map_natCast]
                               /-
                                 🎉 no goals
                               -/

--TODO rename to `map_dvd_map`

theorem map_dvd (f : R →+* S) {x y : R[X]} : x ∣ y → x.map f ∣ y.map f :=
  (mapRingHom f).map_dvd


lemma mapRingHom_comp_C {R S} [CommRing R] [CommRing S] (f : R →+* S) :
                                           /-
                                             R : Type u_1
                                             S : Type u_2
                                             inst✝¹ : CommRing R
                                             inst✝ : CommRing S
                                             f : RingHom R S
                                             ⊢ Eq ((Polynomial.mapRingHom f).comp Polynomial.C) (Polynomial.C.comp f)
                                           -/
    (mapRingHom f).comp C = C.comp f := by ext; simp
                                                /-
                                                  🎉 no goals
                                                -/


theorem eval₂_eq_eval_map {x : S} : p.eval₂ f x = (p.map f).eval x := by
  induction p using Polynomial.induction_on' with
  | h_add p q hp hq =>
    simp [hp, hq]
  | h_monomial n r =>
    simp


protected theorem map_list_prod (L : List R[X]) : L.prod.map f = (L.map <| map f).prod :=
  Eq.symm <| List.prod_hom _ (mapRingHom f).toMonoidHom


@[simp]
protected theorem map_pow (n : ℕ) : (p ^ n).map f = p.map f ^ n :=
  (mapRingHom f).map_pow _ _


theorem eval_map (x : S) : (p.map f).eval x = p.eval₂ f x :=
  (eval₂_eq_eval_map f).symm


protected theorem map_sum {ι : Type*} (g : ι → R[X]) (s : Finset ι) :
    (∑ i ∈ s, g i).map f = ∑ i ∈ s, (g i).map f :=
  map_sum (mapRingHom f) _ _


theorem map_comp (p q : R[X]) : map f (p.comp q) = (map f p).comp (map f q) :=
                                /-
                                  R : Type u
                                  S : Type v
                                  inst✝¹ : Semiring R
                                  inst✝ : Semiring S
                                  f : RingHom R S
                                  p q : Polynomial R
                                  ⊢ ∀ (a : R), Eq (Polynomial.map f ((Polynomial.C a).comp q)) ((Polynomial.map  …
                                -/
  Polynomial.induction_on p (by simp only [map_C, forall_const, C_comp, eq_self_iff_true])
                                /-
                                  🎉 no goals
                                -/
    (by
      simp +contextual only [Polynomial.map_add, add_comp, forall_const,
        imp_true_iff, eq_self_iff_true])
    (by
      simp +contextual only [pow_succ, ← mul_assoc, comp, forall_const,
        eval₂_mul_X, imp_true_iff, eq_self_iff_true, map_X, Polynomial.map_mul])


@[simp]
theorem eval_mul : (p * q).eval x = p.eval x * q.eval x :=
  eval₂_mul _ _


/-- `eval r`, regarded as a ring homomorphism from `R[X]` to `R`. -/
def evalRingHom : R → R[X] →+* R :=
  eval₂RingHom (RingHom.id _)


@[simp]
theorem coe_evalRingHom (r : R) : (evalRingHom r : R[X] → R) = eval r :=
  rfl


@[simp]
theorem eval_pow (n : ℕ) : (p ^ n).eval x = p.eval x ^ n :=
  eval₂_pow _ _ _


@[simp]
theorem eval_comp : (p.comp q).eval x = p.eval (q.eval x) := by
  induction p using Polynomial.induction_on' with
  | h_add r s hr hs =>
    simp [add_comp, hr, hs]
  | h_monomial n a =>
    simp


lemma isRoot_comp {R} [CommSemiring R] {p q : R[X]} {r : R} :
                                                    /-
                                                      R : Type u_1
                                                      inst✝ : CommSemiring R
                                                      p q : Polynomial R
                                                      r : R
                                                      ⊢ Iff ((p.comp q).IsRoot r) (p.IsRoot (Polynomial.eval r q))
                                                    -/
    (p.comp q).IsRoot r ↔ p.IsRoot (q.eval r) := by simp_rw [IsRoot, eval_comp]
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- `comp p`, regarded as a ring homomorphism from `R[X]` to itself. -/
def compRingHom : R[X] → R[X] →+* R[X] :=
  eval₂RingHom C


@[simp]
theorem coe_compRingHom (q : R[X]) : (compRingHom q : R[X] → R[X]) = fun p => comp p q :=
  rfl


theorem coe_compRingHom_apply (p q : R[X]) : (compRingHom q : R[X] → R[X]) p = comp p q :=
  rfl


theorem root_mul_left_of_isRoot (p : R[X]) {q : R[X]} : IsRoot q a → IsRoot (p * q) a := fun H => by
  /-
    R : Type u
    a : R
    inst✝ : CommSemiring R
    p q : Polynomial R
    H : q.IsRoot a
    ⊢ (HMul.hMul p q).IsRoot a
  -/
  rw [IsRoot, eval_mul, IsRoot.def.1 H, mul_zero]
  /-
    🎉 no goals
  -/


theorem root_mul_right_of_isRoot {p : R[X]} (q : R[X]) : IsRoot p a → IsRoot (p * q) a := fun H =>
     /-
       R : Type u
       a : R
       inst✝ : CommSemiring R
       p q : Polynomial R
       H : p.IsRoot a
       ⊢ (HMul.hMul p q).IsRoot a
     -/
  by rw [IsRoot, eval_mul, IsRoot.def.1 H, zero_mul]
     /-
       🎉 no goals
     -/


theorem eval₂_multiset_prod (s : Multiset R[X]) (x : S) :
    eval₂ f x s.prod = (s.map (eval₂ f x)).prod :=
  map_multiset_prod (eval₂RingHom f x) s


theorem eval₂_finset_prod (s : Finset ι) (g : ι → R[X]) (x : S) :
    (∏ i ∈ s, g i).eval₂ f x = ∏ i ∈ s, (g i).eval₂ f x :=
  map_prod (eval₂RingHom f x) _ _


/-- Polynomial evaluation commutes with `List.prod`
-/
theorem eval_list_prod (l : List R[X]) (x : R) : eval x l.prod = (l.map (eval x)).prod :=
  map_list_prod (evalRingHom x) l


/-- Polynomial evaluation commutes with `Multiset.prod`
-/
theorem eval_multiset_prod (s : Multiset R[X]) (x : R) : eval x s.prod = (s.map (eval x)).prod :=
  (evalRingHom x).map_multiset_prod s


/-- Polynomial evaluation commutes with `Finset.prod`
-/
theorem eval_prod {ι : Type*} (s : Finset ι) (p : ι → R[X]) (x : R) :
    eval x (∏ j ∈ s, p j) = ∏ j ∈ s, eval x (p j) :=
  map_prod (evalRingHom x) _ _


theorem list_prod_comp (l : List R[X]) (q : R[X]) :
    l.prod.comp q = (l.map fun p : R[X] => p.comp q).prod :=
  map_list_prod (compRingHom q) _


theorem multiset_prod_comp (s : Multiset R[X]) (q : R[X]) :
    s.prod.comp q = (s.map fun p : R[X] => p.comp q).prod :=
  map_multiset_prod (compRingHom q) _


theorem prod_comp {ι : Type*} (s : Finset ι) (p : ι → R[X]) (q : R[X]) :
    (∏ j ∈ s, p j).comp q = ∏ j ∈ s, (p j).comp q :=
  map_prod (compRingHom q) _ _


theorem isRoot_prod {R} [CommRing R] [IsDomain R] {ι : Type*} (s : Finset ι) (p : ι → R[X])
    (x : R) : IsRoot (∏ j ∈ s, p j) x ↔ ∃ i ∈ s, IsRoot (p i) x := by
  /-
    R : Type u_2
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ι : Type u_1
    s : Finset ι
    p : ι → Polynomial R
    x : R
    ⊢ Iff ((s.prod fun j => p j).IsRoot x) (Exists fun i => And (Membership.mem s  …
  -/
  simp only [IsRoot, eval_prod, Finset.prod_eq_zero_iff]
  /-
    🎉 no goals
  -/


theorem eval_dvd : p ∣ q → eval x p ∣ eval x q :=
  eval₂_dvd _ _


theorem eval_eq_zero_of_dvd_of_eval_eq_zero : p ∣ q → eval x p = 0 → eval x q = 0 :=
  eval₂_eq_zero_of_dvd_of_eval₂_eq_zero _ _


@[simp]
theorem eval_geom_sum {R} [CommSemiring R] {n : ℕ} {x : R} :
                                                               /-
                                                                 R : Type u_1
                                                                 inst✝ : CommSemiring R
                                                                 n : Nat
                                                                 x : R
                                                                 ⊢ Eq (Polynomial.eval x ((Finset.range n).sum fun i => HPow.hPow Polynomial.X  …
                                                               -/
    eval x (∑ i ∈ range n, X ^ i) = ∑ i ∈ range n, x ^ i := by simp [eval_finset_sum]
                                                               /-
                                                                 🎉 no goals
                                                               -/


lemma root_mul : IsRoot (p * q) a ↔ IsRoot p a ∨ IsRoot q a := by
  /-
    R : Type u
    a : R
    inst✝¹ : CommSemiring R
    p q : Polynomial R
    inst✝ : NoZeroDivisors R
    ⊢ Iff ((HMul.hMul p q).IsRoot a) (Or (p.IsRoot a) (q.IsRoot a))
  -/
  simp_rw [IsRoot, eval_mul, mul_eq_zero]
  /-
    🎉 no goals
  -/


lemma root_or_root_of_root_mul (h : IsRoot (p * q) a) : IsRoot p a ∨ IsRoot q a :=
  root_mul.1 h


protected theorem map_multiset_prod (m : Multiset R[X]) : m.prod.map f = (m.map <| map f).prod :=
  Eq.symm <| Multiset.prod_hom _ (mapRingHom f).toMonoidHom


protected theorem map_prod {ι : Type*} (g : ι → R[X]) (s : Finset ι) :
    (∏ i ∈ s, g i).map f = ∏ i ∈ s, (g i).map f :=
  map_prod (mapRingHom f) _ _


@[simp]
protected theorem map_sub {S} [Ring S] (f : R →+* S) : (p - q).map f = p.map f - q.map f :=
  (mapRingHom f).map_sub p q


@[simp]
protected theorem map_neg {S} [Ring S] (f : R →+* S) : (-p).map f = -p.map f :=
  (mapRingHom f).map_neg p


@[simp] protected lemma map_intCast {S} [Ring S] (f : R →+* S) (n : ℤ) : map f ↑n = ↑n :=
  map_intCast (mapRingHom f) n


@[deprecated (since := "2024-04-17")]
alias map_int_cast := map_intCast


@[simp]
theorem eval_intCast {n : ℤ} {x : R} : (n : R[X]).eval x = n := by
  /-
    R : Type u
    inst✝ : Ring R
    n : Int
    x : R
    ⊢ Eq (Polynomial.eval x ↑n) ↑n
  -/
  simp only [← C_eq_intCast, eval_C]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias eval_int_cast := eval_intCast


@[simp]
theorem eval₂_neg {S} [Ring S] (f : R →+* S) {x : S} : (-p).eval₂ f x = -p.eval₂ f x := by
  /-
    R : Type u
    inst✝¹ : Ring R
    p : Polynomial R
    S : Type u_1
    inst✝ : Ring S
    f : RingHom R S
    x : S
    ⊢ Eq (Polynomial.eval₂ f x (Neg.neg p)) (Neg.neg (Polynomial.eval₂ f x p))
  -/
  rw [eq_neg_iff_add_eq_zero, ← eval₂_add, neg_add_cancel, eval₂_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem eval₂_sub {S} [Ring S] (f : R →+* S) {x : S} :
    (p - q).eval₂ f x = p.eval₂ f x - q.eval₂ f x := by
  /-
    R : Type u
    inst✝¹ : Ring R
    p q : Polynomial R
    S : Type u_1
    inst✝ : Ring S
    f : RingHom R S
    x : S
    ⊢ Eq (Polynomial.eval₂ f x (HSub.hSub p q)) (HSub.hSub (Polynomial.eval₂ f x p …
  -/
  rw [sub_eq_add_neg, eval₂_add, eval₂_neg, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem eval_neg (p : R[X]) (x : R) : (-p).eval x = -p.eval x :=
  eval₂_neg _


@[simp]
theorem eval_sub (p q : R[X]) (x : R) : (p - q).eval x = p.eval x - q.eval x :=
  eval₂_sub _


theorem root_X_sub_C : IsRoot (X - C a) b ↔ a = b := by
  /-
    R : Type u
    a b : R
    inst✝ : Ring R
    ⊢ Iff ((HSub.hSub Polynomial.X (Polynomial.C a)).IsRoot b) (Eq a b)
  -/
  rw [IsRoot.def, eval_sub, eval_X, eval_C, sub_eq_zero, eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem neg_comp : (-p).comp q = -p.comp q :=
  eval₂_neg _


@[simp]
theorem sub_comp : (p - q).comp r = p.comp r - q.comp r :=
  eval₂_sub _


@[simp]
                                                           /-
                                                             R : Type u
                                                             inst✝ : Ring R
                                                             p : Polynomial R
                                                             i : Int
                                                             ⊢ Eq ((↑i).comp p) ↑i
                                                           -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
theorem intCast_comp (i : ℤ) : comp (i : R[X]) p = i := by cases i <;> simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[deprecated (since := "2024-05-27")] alias cast_int_comp := intCast_comp


@[simp]
theorem eval₂_at_intCast {S : Type*} [Ring S] (f : R →+* S) (n : ℤ) :
    p.eval₂ f n = f (p.eval n) := by
  /-
    R : Type u
    inst✝¹ : Ring R
    p : Polynomial R
    S : Type u_1
    inst✝ : Ring S
    f : RingHom R S
    n : Int
    ⊢ Eq (Polynomial.eval₂ f (↑n) p) (f (Polynomial.eval (↑n) p))
  -/
  convert eval₂_at_apply (p := p) f n
  /-
    case h.e'_2.h.e'_6
    R : Type u
    inst✝¹ : Ring R
    p : Polynomial R
    S : Type u_1
    inst✝ : Ring S
    f : RingHom R S
    n : Int
    ⊢ Eq (↑n) (f ↑n)
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias eval₂_at_int_cast := eval₂_at_intCast


theorem mul_X_sub_intCast_comp {n : ℕ} :
    (p * (X - (n : R[X]))).comp q = p.comp q * (q - n) := by
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    n : Nat
    ⊢ Eq ((HMul.hMul p (HSub.hSub Polynomial.X ↑n)).comp q) (HMul.hMul (p.comp q)  …
  -/
  rw [mul_sub, sub_comp, mul_X_comp, ← Nat.cast_comm, natCast_mul_comp, Nat.cast_comm, mul_sub]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias mul_X_sub_int_cast_comp := mul_X_sub_intCast_comp


