/-- If `i ≤ N`, then `revAtFun N i` returns `N - i`, otherwise it returns `i`.
This is the map used by the embedding `revAt`.
-/
def revAtFun (N i : ℕ) : ℕ :=
  ite (i ≤ N) (N - i) i


theorem revAtFun_invol {N i : ℕ} : revAtFun N (revAtFun N i) = i := by
  /-
    N i : Nat
    ⊢ Eq (Polynomial.revAtFun N (Polynomial.revAtFun N i)) i
  -/
  unfold revAtFun
  /-
    N i : Nat
    ⊢ Eq (ite (LE.le (ite (LE.le i N) (HSub.hSub N i) i) N) (HSub.hSub N (ite (LE. …
  -/
  split_ifs with h j
    /-
      case pos
      N i : Nat
      h : LE.le i N
      j : LE.le (HSub.hSub N i) N
      ⊢ Eq (HSub.hSub N (HSub.hSub N i)) i
    -/
  · exact tsub_tsub_cancel_of_le h
    /-
      🎉 no goals
    -/
    /-
      case neg
      N i : Nat
      h : LE.le i N
      j : Not (LE.le (HSub.hSub N i) N)
      ⊢ Eq (HSub.hSub N i) i
    -/
  · exfalso
    /-
      case neg
      N i : Nat
      h : LE.le i N
      j : Not (LE.le (HSub.hSub N i) N)
      ⊢ False
    -/
    apply j
    /-
      case neg
      N i : Nat
      h : LE.le i N
      j : Not (LE.le (HSub.hSub N i) N)
      ⊢ LE.le (HSub.hSub N i) N
    -/
    exact Nat.sub_le N i
    /-
      🎉 no goals
    -/
    /-
      case neg
      N i : Nat
      h : Not (LE.le i N)
      ⊢ Eq i i
    -/
  · rfl
    /-
      🎉 no goals
    -/


theorem revAtFun_inj {N : ℕ} : Function.Injective (revAtFun N) := by
  /-
    N : Nat
    ⊢ Function.Injective (Polynomial.revAtFun N)
  -/
  intro a b hab
  /-
    N a b : Nat
    hab : Eq (Polynomial.revAtFun N a) (Polynomial.revAtFun N b)
    ⊢ Eq a b
  -/
  rw [← @revAtFun_invol N a, hab, revAtFun_invol]
  /-
    🎉 no goals
  -/


/-- If `i ≤ N`, then `revAt N i` returns `N - i`, otherwise it returns `i`.
Essentially, this embedding is only used for `i ≤ N`.
The advantage of `revAt N i` over `N - i` is that `revAt` is an involution.
-/
def revAt (N : ℕ) : Function.Embedding ℕ ℕ where
  toFun i := ite (i ≤ N) (N - i) i
  inj' := revAtFun_inj


/-- We prefer to use the bundled `revAt` over unbundled `revAtFun`. -/
@[simp]
theorem revAtFun_eq (N i : ℕ) : revAtFun N i = revAt N i :=
  rfl


@[simp]
theorem revAt_invol {N i : ℕ} : (revAt N) (revAt N i) = i :=
  revAtFun_invol


@[simp]
theorem revAt_le {N i : ℕ} (H : i ≤ N) : revAt N i = N - i :=
  if_pos H


                                                                      /-
                                                                        N i : Nat
                                                                        h : LT.lt N i
                                                                        ⊢ Eq ((Polynomial.revAt N) i) i
                                                                      -/
lemma revAt_eq_self_of_lt {N i : ℕ} (h : N < i) : revAt N i = i := by simp [revAt, Nat.not_le.mpr h]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem revAt_add {N O n o : ℕ} (hn : n ≤ N) (ho : o ≤ O) :
    revAt (N + O) (n + o) = revAt N n + revAt O o := by
  /-
    N O n o : Nat
    hn : LE.le n N
    ho : LE.le o O
    ⊢ Eq ((Polynomial.revAt (HAdd.hAdd N O)) (HAdd.hAdd n o)) (HAdd.hAdd ((Polynom …
  -/
  rcases Nat.le.dest hn with ⟨n', rfl⟩
  /-
    case intro
    O n o : Nat
    ho : LE.le o O
    n' : Nat
    hn : LE.le n (HAdd.hAdd n n')
    ⊢ Eq ((Polynomial.revAt (HAdd.hAdd (HAdd.hAdd n n') O)) (HAdd.hAdd n o)) (HAdd …
  -/
  rcases Nat.le.dest ho with ⟨o', rfl⟩
  /-
    case intro.intro
    n o n' : Nat
    hn : LE.le n (HAdd.hAdd n n')
    o' : Nat
    ho : LE.le o (HAdd.hAdd o o')
    ⊢ Eq ((Polynomial.revAt (HAdd.hAdd (HAdd.hAdd n n') (HAdd.hAdd o o'))) (HAdd.h …
  -/
  repeat' rw [revAt_le (le_add_right rfl.le)]
  /-
    case intro.intro
    n o n' : Nat
    hn : LE.le n (HAdd.hAdd n n')
    o' : Nat
    ho : LE.le o (HAdd.hAdd o o')
    ⊢ Eq ((Polynomial.revAt (HAdd.hAdd (HAdd.hAdd n n') (HAdd.hAdd o o'))) (HAdd.h …
  -/
  rw [add_assoc, add_left_comm n' o, ← add_assoc, revAt_le (le_add_right rfl.le)]
  /-
    case intro.intro
    n o n' : Nat
    hn : LE.le n (HAdd.hAdd n n')
    o' : Nat
    ho : LE.le o (HAdd.hAdd o o')
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd n o) (HAdd.hAdd n' o')) (HAdd.hAdd n o)) …
  -/
  repeat' rw [add_tsub_cancel_left]
  /-
    🎉 no goals
  -/


                                                 /-
                                                   N : Nat
                                                   ⊢ Eq ((Polynomial.revAt N) 0) N
                                                 -/
theorem revAt_zero (N : ℕ) : revAt N 0 = N := by simp
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- `reflect N f` is the polynomial such that `(reflect N f).coeff i = f.coeff (revAt N i)`.
In other words, the terms with exponent `[0, ..., N]` now have exponent `[N, ..., 0]`.

In practice, `reflect` is only used when `N` is at least as large as the degree of `f`.

Eventually, it will be used with `N` exactly equal to the degree of `f`. -/
noncomputable def reflect (N : ℕ) : R[X] → R[X]
  | ⟨f⟩ => ⟨Finsupp.embDomain (revAt N) f⟩


theorem reflect_support (N : ℕ) (f : R[X]) :
    (reflect N f).support = Finset.image (revAt N) f.support := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    N : Nat
    f : Polynomial R
    ⊢ Eq (Polynomial.reflect N f).support (Finset.image (⇑(Polynomial.revAt N)) f. …
  -/
  rcases f with ⟨⟩
  /-
    case ofFinsupp
    R : Type u_1
    inst✝ : Semiring R
    N : Nat
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq (Polynomial.reflect N { toFinsupp := toFinsupp✝ }).support (Finset.image  …
  -/
  ext1
  /-
    case ofFinsupp.h
    R : Type u_1
    inst✝ : Semiring R
    N : Nat
    toFinsupp✝ : AddMonoidAlgebra R Nat
    a✝ : Nat
    ⊢ Iff (Membership.mem (Polynomial.reflect N { toFinsupp := toFinsupp✝ }).suppo …
  -/
  simp only [reflect, support_ofFinsupp, support_embDomain, Finset.mem_map, Finset.mem_image]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_reflect (N : ℕ) (f : R[X]) (i : ℕ) : coeff (reflect N f) i = f.coeff (revAt N i) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    N : Nat
    f : Polynomial R
    i : Nat
    ⊢ Eq ((Polynomial.reflect N f).coeff i) (f.coeff ((Polynomial.revAt N) i))
  -/
  rcases f with ⟨f⟩
  /-
    case ofFinsupp
    R : Type u_1
    inst✝ : Semiring R
    N i : Nat
    f : AddMonoidAlgebra R Nat
    ⊢ Eq ((Polynomial.reflect N { toFinsupp := f }).coeff i) ({ toFinsupp := f }.c …
  -/
  simp only [reflect, coeff]
  calc
    Finsupp.embDomain (revAt N) f i = Finsupp.embDomain (revAt N) f (revAt N (revAt N i)) := by
      rw [revAt_invol]
    _ = f (revAt N i) := Finsupp.embDomain_apply _ _ _


@[simp]
theorem reflect_zero {N : ℕ} : reflect N (0 : R[X]) = 0 :=
  rfl


@[simp]
theorem reflect_eq_zero_iff {N : ℕ} {f : R[X]} : reflect N (f : R[X]) = 0 ↔ f = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    N : Nat
    f : Polynomial R
    ⊢ Iff (Eq (Polynomial.reflect N f) 0) (Eq f 0)
  -/
  rw [ofFinsupp_eq_zero, reflect, embDomain_eq_zero, ofFinsupp_eq_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem reflect_add (f g : R[X]) (N : ℕ) : reflect N (f + g) = reflect N f + reflect N g := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f g : Polynomial R
    N : Nat
    ⊢ Eq (Polynomial.reflect N (HAdd.hAdd f g)) (HAdd.hAdd (Polynomial.reflect N f …
  -/
  ext
  /-
    case a
    R : Type u_1
    inst✝ : Semiring R
    f g : Polynomial R
    N n✝ : Nat
    ⊢ Eq ((Polynomial.reflect N (HAdd.hAdd f g)).coeff n✝) ((HAdd.hAdd (Polynomial …
  -/
  simp only [coeff_add, coeff_reflect]
  /-
    🎉 no goals
  -/


@[simp]
theorem reflect_C_mul (f : R[X]) (r : R) (N : ℕ) : reflect N (C r * f) = C r * reflect N f := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    r : R
    N : Nat
    ⊢ Eq (Polynomial.reflect N (HMul.hMul (Polynomial.C r) f)) (HMul.hMul (Polynom …
  -/
  ext
  /-
    case a
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    r : R
    N n✝ : Nat
    ⊢ Eq ((Polynomial.reflect N (HMul.hMul (Polynomial.C r) f)).coeff n✝) ((HMul.h …
  -/
  simp only [coeff_reflect, coeff_C_mul]
  /-
    🎉 no goals
  -/


theorem reflect_C_mul_X_pow (N n : ℕ) {c : R} : reflect N (C c * X ^ n) = C c * X ^ revAt N n := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    N n : Nat
    c : R
    ⊢ Eq (Polynomial.reflect N (HMul.hMul (Polynomial.C c) (HPow.hPow Polynomial.X …
  -/
  ext
  /-
    case a
    R : Type u_1
    inst✝ : Semiring R
    N n : Nat
    c : R
    n✝ : Nat
    ⊢ Eq ((Polynomial.reflect N (HMul.hMul (Polynomial.C c) (HPow.hPow Polynomial. …
  -/
  rw [reflect_C_mul, coeff_C_mul, coeff_C_mul, coeff_X_pow, coeff_reflect]
  /-
    case a
    R : Type u_1
    inst✝ : Semiring R
    N n : Nat
    c : R
    n✝ : Nat
    ⊢ Eq (HMul.hMul c ((HPow.hPow Polynomial.X n).coeff ((Polynomial.revAt N) n✝)) …
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      N n : Nat
      c : R
      n✝ : Nat
      h : Eq n✝ ((Polynomial.revAt N) n)
      ⊢ Eq (HMul.hMul c ((HPow.hPow Polynomial.X n).coeff ((Polynomial.revAt N) n✝)) …
    -/
  · rw [h, revAt_invol, coeff_X_pow_self]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      N n : Nat
      c : R
      n✝ : Nat
      h : Not (Eq n✝ ((Polynomial.revAt N) n))
      ⊢ Eq (HMul.hMul c ((HPow.hPow Polynomial.X n).coeff ((Polynomial.revAt N) n✝)) …
    -/
  · rw [not_mem_support_iff.mp]
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      N n : Nat
      c : R
      n✝ : Nat
      h : Not (Eq n✝ ((Polynomial.revAt N) n))
      ⊢ Not (Membership.mem (HPow.hPow Polynomial.X n).support ((Polynomial.revAt N) …
    -/
    intro a
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      N n : Nat
      c : R
      n✝ : Nat
      h : Not (Eq n✝ ((Polynomial.revAt N) n))
      a : Membership.mem (HPow.hPow Polynomial.X n).support ((Polynomial.revAt N) n✝)
      ⊢ False
    -/
    rw [← one_mul (X ^ n), ← C_1] at a
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      N n : Nat
      c : R
      n✝ : Nat
      h : Not (Eq n✝ ((Polynomial.revAt N) n))
      a : Membership.mem (HMul.hMul (Polynomial.C 1) (HPow.hPow Polynomial.X n)).sup …
      ⊢ False
    -/
    apply h
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      N n : Nat
      c : R
      n✝ : Nat
      h : Not (Eq n✝ ((Polynomial.revAt N) n))
      a : Membership.mem (HMul.hMul (Polynomial.C 1) (HPow.hPow Polynomial.X n)).sup …
      ⊢ Eq n✝ ((Polynomial.revAt N) n)
    -/
    rw [← mem_support_C_mul_X_pow a, revAt_invol]
    /-
      🎉 no goals
    -/


@[simp]
theorem reflect_C (r : R) (N : ℕ) : reflect N (C r) = C r * X ^ N := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    r : R
    N : Nat
    ⊢ Eq (Polynomial.reflect N (Polynomial.C r)) (HMul.hMul (Polynomial.C r) (HPow …
  -/
  conv_lhs => rw [← mul_one (C r), ← pow_zero X, reflect_C_mul_X_pow, revAt_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem reflect_monomial (N n : ℕ) : reflect N ((X : R[X]) ^ n) = X ^ revAt N n := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    N n : Nat
    ⊢ Eq (Polynomial.reflect N (HPow.hPow Polynomial.X n)) (HPow.hPow Polynomial.X …
  -/
  rw [← one_mul (X ^ n), ← one_mul (X ^ revAt N n), ← C_1, reflect_C_mul_X_pow]
  /-
    🎉 no goals
  -/


@[simp] lemma reflect_one_X : reflect 1 (X : R[X]) = 1 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    ⊢ Eq (Polynomial.reflect 1 Polynomial.X) 1
  -/
  simpa using reflect_monomial 1 1 (R := R)
  /-
    🎉 no goals
  -/


lemma reflect_map {S : Type*} [Semiring S] (f : R →+* S) (p : R[X]) (n : ℕ) :
    (p.map f).reflect n = (p.reflect n).map f := by
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    S : Type u_2
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    n : Nat
    ⊢ Eq (Polynomial.reflect n (Polynomial.map f p)) (Polynomial.map f (Polynomial …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp]
lemma reflect_one (n : ℕ) : (1 : R[X]).reflect n = Polynomial.X ^ n := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq (Polynomial.reflect n 1) (HPow.hPow Polynomial.X n)
  -/
  rw [← C.map_one, reflect_C, map_one, one_mul]
  /-
    🎉 no goals
  -/


theorem reflect_mul_induction (cf cg : ℕ) :
    ∀ N O : ℕ,
      ∀ f g : R[X],
        #f.support ≤ cf.succ →
          #g.support ≤ cg.succ →
            f.natDegree ≤ N →
              g.natDegree ≤ O → reflect (N + O) (f * g) = reflect N f * reflect O g := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    cf cg : Nat
    ⊢ ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card cf.succ → LE.le g.s …
  -/
  induction' cf with cf hcf
  --first induction (left): base case
    /-
      case zero
      R : Type u_1
      inst✝ : Semiring R
      cg : Nat
      ⊢ ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card (Nat.succ 0) → LE.l …
    -/
  · induction' cg with cg hcg
    -- second induction (right): base case
      /-
        case zero.zero
        R : Type u_1
        inst✝ : Semiring R
        ⊢ ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card (Nat.succ 0) → LE.l …
      -/
    · intro N O f g Cf Cg Nf Og
      /-
        case zero.zero
        R : Type u_1
        inst✝ : Semiring R
        N O : Nat
        f g : Polynomial R
        Cf : LE.le f.support.card (Nat.succ 0)
        Cg : LE.le g.support.card (Nat.succ 0)
        Nf : LE.le f.natDegree N
        Og : LE.le g.natDegree O
        ⊢ Eq (Polynomial.reflect (HAdd.hAdd N O) (HMul.hMul f g)) (HMul.hMul (Polynomi …
      -/
      rw [← C_mul_X_pow_eq_self Cf, ← C_mul_X_pow_eq_self Cg]
      simp_rw [mul_assoc, X_pow_mul, mul_assoc, ← pow_add (X : R[X]), reflect_C_mul,
        reflect_monomial, add_comm, revAt_add Nf Og, mul_assoc, X_pow_mul, mul_assoc, ←
        pow_add (X : R[X]), add_comm]
    -- second induction (right): induction step
      /-
        case zero.succ
        R : Type u_1
        inst✝ : Semiring R
        cg : Nat
        hcg : ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card (Nat.succ 0) →  …
        ⊢ ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card (Nat.succ 0) → LE.l …
      -/
    · intro N O f g Cf Cg Nf Og
      /-
        case zero.succ
        R : Type u_1
        inst✝ : Semiring R
        cg : Nat
        hcg : ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card (Nat.succ 0) →  …
        N O : Nat
        f g : Polynomial R
        Cf : LE.le f.support.card (Nat.succ 0)
        Cg : LE.le g.support.card (HAdd.hAdd cg 1).succ
        Nf : LE.le f.natDegree N
        Og : LE.le g.natDegree O
        ⊢ Eq (Polynomial.reflect (HAdd.hAdd N O) (HMul.hMul f g)) (HMul.hMul (Polynomi …
      -/
      by_cases g0 : g = 0
        /-
          case pos
          R : Type u_1
          inst✝ : Semiring R
          cg : Nat
          hcg : ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card (Nat.succ 0) →  …
          N O : Nat
          f g : Polynomial R
          Cf : LE.le f.support.card (Nat.succ 0)
          Cg : LE.le g.support.card (HAdd.hAdd cg 1).succ
          Nf : LE.le f.natDegree N
          Og : LE.le g.natDegree O
          g0 : Eq g 0
          ⊢ Eq (Polynomial.reflect (HAdd.hAdd N O) (HMul.hMul f g)) (HMul.hMul (Polynomi …
        -/
      · rw [g0, reflect_zero, mul_zero, mul_zero, reflect_zero]
        /-
          🎉 no goals
        -/
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        cg : Nat
        hcg : ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card (Nat.succ 0) →  …
        N O : Nat
        f g : Polynomial R
        Cf : LE.le f.support.card (Nat.succ 0)
        Cg : LE.le g.support.card (HAdd.hAdd cg 1).succ
        Nf : LE.le f.natDegree N
        Og : LE.le g.natDegree O
        g0 : Not (Eq g 0)
        ⊢ Eq (Polynomial.reflect (HAdd.hAdd N O) (HMul.hMul f g)) (HMul.hMul (Polynomi …
      -/
      rw [← eraseLead_add_C_mul_X_pow g, mul_add, reflect_add, reflect_add, mul_add, hcg, hcg] <;>
        /-
          case neg.a
          R : Type u_1
          inst✝ : Semiring R
          cg : Nat
          hcg : ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card (Nat.succ 0) →  …
          N O : Nat
          f g : Polynomial R
          Cf : LE.le f.support.card (Nat.succ 0)
          Cg : LE.le g.support.card (HAdd.hAdd cg 1).succ
          Nf : LE.le f.natDegree N
          Og : LE.le g.natDegree O
          g0 : Not (Eq g 0)
          ⊢ LE.le f.support.card (Nat.succ 0)
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
        try assumption
        /-
          case neg.a
          R : Type u_1
          inst✝ : Semiring R
          cg : Nat
          hcg : ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card (Nat.succ 0) →  …
          N O : Nat
          f g : Polynomial R
          Cf : LE.le f.support.card (Nat.succ 0)
          Cg : LE.le g.support.card (HAdd.hAdd cg 1).succ
          Nf : LE.le f.natDegree N
          Og : LE.le g.natDegree O
          g0 : Not (Eq g 0)
          ⊢ LE.le (HMul.hMul (Polynomial.C g.leadingCoeff) (HPow.hPow Polynomial.X g.nat …
        -/
      · exact le_add_left card_support_C_mul_X_pow_le_one
        /-
          🎉 no goals
        -/
        /-
          case neg.a
          R : Type u_1
          inst✝ : Semiring R
          cg : Nat
          hcg : ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card (Nat.succ 0) →  …
          N O : Nat
          f g : Polynomial R
          Cf : LE.le f.support.card (Nat.succ 0)
          Cg : LE.le g.support.card (HAdd.hAdd cg 1).succ
          Nf : LE.le f.natDegree N
          Og : LE.le g.natDegree O
          g0 : Not (Eq g 0)
          ⊢ LE.le (HMul.hMul (Polynomial.C g.leadingCoeff) (HPow.hPow Polynomial.X g.nat …
        -/
      · exact le_trans (natDegree_C_mul_X_pow_le g.leadingCoeff g.natDegree) Og
        /-
          🎉 no goals
        -/
        /-
          case neg.a
          R : Type u_1
          inst✝ : Semiring R
          cg : Nat
          hcg : ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card (Nat.succ 0) →  …
          N O : Nat
          f g : Polynomial R
          Cf : LE.le f.support.card (Nat.succ 0)
          Cg : LE.le g.support.card (HAdd.hAdd cg 1).succ
          Nf : LE.le f.natDegree N
          Og : LE.le g.natDegree O
          g0 : Not (Eq g 0)
          ⊢ LE.le g.eraseLead.support.card cg.succ
        -/
      · exact Nat.lt_succ_iff.mp (gt_of_ge_of_gt Cg (eraseLead_support_card_lt g0))
        /-
          🎉 no goals
        -/
        /-
          case neg.a
          R : Type u_1
          inst✝ : Semiring R
          cg : Nat
          hcg : ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card (Nat.succ 0) →  …
          N O : Nat
          f g : Polynomial R
          Cf : LE.le f.support.card (Nat.succ 0)
          Cg : LE.le g.support.card (HAdd.hAdd cg 1).succ
          Nf : LE.le f.natDegree N
          Og : LE.le g.natDegree O
          g0 : Not (Eq g 0)
          ⊢ LE.le g.eraseLead.natDegree O
        -/
      · exact le_trans eraseLead_natDegree_le_aux Og
        /-
          🎉 no goals
        -/
  --first induction (left): induction step
    /-
      case succ
      R : Type u_1
      inst✝ : Semiring R
      cg cf : Nat
      hcf : ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card cf.succ → LE.le …
      ⊢ ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card (HAdd.hAdd cf 1).su …
    -/
  · intro N O f g Cf Cg Nf Og
    /-
      case succ
      R : Type u_1
      inst✝ : Semiring R
      cg cf : Nat
      hcf : ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card cf.succ → LE.le …
      N O : Nat
      f g : Polynomial R
      Cf : LE.le f.support.card (HAdd.hAdd cf 1).succ
      Cg : LE.le g.support.card cg.succ
      Nf : LE.le f.natDegree N
      Og : LE.le g.natDegree O
      ⊢ Eq (Polynomial.reflect (HAdd.hAdd N O) (HMul.hMul f g)) (HMul.hMul (Polynomi …
    -/
    by_cases f0 : f = 0
      /-
        case pos
        R : Type u_1
        inst✝ : Semiring R
        cg cf : Nat
        hcf : ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card cf.succ → LE.le …
        N O : Nat
        f g : Polynomial R
        Cf : LE.le f.support.card (HAdd.hAdd cf 1).succ
        Cg : LE.le g.support.card cg.succ
        Nf : LE.le f.natDegree N
        Og : LE.le g.natDegree O
        f0 : Eq f 0
        ⊢ Eq (Polynomial.reflect (HAdd.hAdd N O) (HMul.hMul f g)) (HMul.hMul (Polynomi …
      -/
    · rw [f0, reflect_zero, zero_mul, zero_mul, reflect_zero]
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      cg cf : Nat
      hcf : ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card cf.succ → LE.le …
      N O : Nat
      f g : Polynomial R
      Cf : LE.le f.support.card (HAdd.hAdd cf 1).succ
      Cg : LE.le g.support.card cg.succ
      Nf : LE.le f.natDegree N
      Og : LE.le g.natDegree O
      f0 : Not (Eq f 0)
      ⊢ Eq (Polynomial.reflect (HAdd.hAdd N O) (HMul.hMul f g)) (HMul.hMul (Polynomi …
    -/
    rw [← eraseLead_add_C_mul_X_pow f, add_mul, reflect_add, reflect_add, add_mul, hcf, hcf] <;>
      /-
        case neg.a
        R : Type u_1
        inst✝ : Semiring R
        cg cf : Nat
        hcf : ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card cf.succ → LE.le …
        N O : Nat
        f g : Polynomial R
        Cf : LE.le f.support.card (HAdd.hAdd cf 1).succ
        Cg : LE.le g.support.card cg.succ
        Nf : LE.le f.natDegree N
        Og : LE.le g.natDegree O
        f0 : Not (Eq f 0)
        ⊢ LE.le (HMul.hMul (Polynomial.C f.leadingCoeff) (HPow.hPow Polynomial.X f.nat …
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      try assumption
      /-
        🎉 no goals
      -/
      /-
        case neg.a
        R : Type u_1
        inst✝ : Semiring R
        cg cf : Nat
        hcf : ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card cf.succ → LE.le …
        N O : Nat
        f g : Polynomial R
        Cf : LE.le f.support.card (HAdd.hAdd cf 1).succ
        Cg : LE.le g.support.card cg.succ
        Nf : LE.le f.natDegree N
        Og : LE.le g.natDegree O
        f0 : Not (Eq f 0)
        ⊢ LE.le (HMul.hMul (Polynomial.C f.leadingCoeff) (HPow.hPow Polynomial.X f.nat …
      -/
    · exact le_add_left card_support_C_mul_X_pow_le_one
      /-
        🎉 no goals
      -/
      /-
        case neg.a
        R : Type u_1
        inst✝ : Semiring R
        cg cf : Nat
        hcf : ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card cf.succ → LE.le …
        N O : Nat
        f g : Polynomial R
        Cf : LE.le f.support.card (HAdd.hAdd cf 1).succ
        Cg : LE.le g.support.card cg.succ
        Nf : LE.le f.natDegree N
        Og : LE.le g.natDegree O
        f0 : Not (Eq f 0)
        ⊢ LE.le (HMul.hMul (Polynomial.C f.leadingCoeff) (HPow.hPow Polynomial.X f.nat …
      -/
    · exact le_trans (natDegree_C_mul_X_pow_le f.leadingCoeff f.natDegree) Nf
      /-
        🎉 no goals
      -/
      /-
        case neg.a
        R : Type u_1
        inst✝ : Semiring R
        cg cf : Nat
        hcf : ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card cf.succ → LE.le …
        N O : Nat
        f g : Polynomial R
        Cf : LE.le f.support.card (HAdd.hAdd cf 1).succ
        Cg : LE.le g.support.card cg.succ
        Nf : LE.le f.natDegree N
        Og : LE.le g.natDegree O
        f0 : Not (Eq f 0)
        ⊢ LE.le f.eraseLead.support.card cf.succ
      -/
    · exact Nat.lt_succ_iff.mp (gt_of_ge_of_gt Cf (eraseLead_support_card_lt f0))
      /-
        🎉 no goals
      -/
      /-
        case neg.a
        R : Type u_1
        inst✝ : Semiring R
        cg cf : Nat
        hcf : ∀ (N O : Nat) (f g : Polynomial R), LE.le f.support.card cf.succ → LE.le …
        N O : Nat
        f g : Polynomial R
        Cf : LE.le f.support.card (HAdd.hAdd cf 1).succ
        Cg : LE.le g.support.card cg.succ
        Nf : LE.le f.natDegree N
        Og : LE.le g.natDegree O
        f0 : Not (Eq f 0)
        ⊢ LE.le f.eraseLead.natDegree N
      -/
    · exact le_trans eraseLead_natDegree_le_aux Nf
      /-
        🎉 no goals
      -/


@[simp]
theorem reflect_mul (f g : R[X]) {F G : ℕ} (Ff : f.natDegree ≤ F) (Gg : g.natDegree ≤ G) :
    reflect (F + G) (f * g) = reflect F f * reflect G g :=
  reflect_mul_induction _ _ F G f g f.support.card.le_succ g.support.card.le_succ Ff Gg


theorem eval₂_reflect_mul_pow (i : R →+* S) (x : S) [Invertible x] (N : ℕ) (f : R[X])
    (hf : f.natDegree ≤ N) : eval₂ i (⅟ x) (reflect N f) * x ^ N = eval₂ i x f := by
  refine
    induction_with_natDegree_le (fun f => eval₂ i (⅟ x) (reflect N f) * x ^ N = eval₂ i x f) _ ?_ ?_
      ?_ f hf
    /-
      case refine_1
      R : Type u_1
      inst✝² : Semiring R
      S : Type u_2
      inst✝¹ : CommSemiring S
      i : RingHom R S
      x : S
      inst✝ : Invertible x
      N : Nat
      f : Polynomial R
      hf : LE.le f.natDegree N
      ⊢ (fun f => Eq (HMul.hMul (Polynomial.eval₂ i (Invertible.invOf x) (Polynomial …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝² : Semiring R
      S : Type u_2
      inst✝¹ : CommSemiring S
      i : RingHom R S
      x : S
      inst✝ : Invertible x
      N : Nat
      f : Polynomial R
      hf : LE.le f.natDegree N
      ⊢ ∀ (n : Nat) (r : R), Ne r 0 → LE.le n N → (fun f => Eq (HMul.hMul (Polynomia …
    -/
  · intro n r _ hnN
    /-
      case refine_2
      R : Type u_1
      inst✝² : Semiring R
      S : Type u_2
      inst✝¹ : CommSemiring S
      i : RingHom R S
      x : S
      inst✝ : Invertible x
      N : Nat
      f : Polynomial R
      hf : LE.le f.natDegree N
      n : Nat
      r : R
      a✝ : Ne r 0
      hnN : LE.le n N
      ⊢ Eq (HMul.hMul (Polynomial.eval₂ i (Invertible.invOf x) (Polynomial.reflect N …
    -/
    simp only [revAt_le hnN, reflect_C_mul_X_pow, eval₂_X_pow, eval₂_C, eval₂_mul]
    /-
      case refine_2
      R : Type u_1
      inst✝² : Semiring R
      S : Type u_2
      inst✝¹ : CommSemiring S
      i : RingHom R S
      x : S
      inst✝ : Invertible x
      N : Nat
      f : Polynomial R
      hf : LE.le f.natDegree N
      n : Nat
      r : R
      a✝ : Ne r 0
      hnN : LE.le n N
      ⊢ Eq (HMul.hMul (HMul.hMul (i r) (HPow.hPow (Invertible.invOf x) (HSub.hSub N  …
    -/
    conv in x ^ N => rw [← Nat.sub_add_cancel hnN]
    /-
      case refine_2
      R : Type u_1
      inst✝² : Semiring R
      S : Type u_2
      inst✝¹ : CommSemiring S
      i : RingHom R S
      x : S
      inst✝ : Invertible x
      N : Nat
      f : Polynomial R
      hf : LE.le f.natDegree N
      n : Nat
      r : R
      a✝ : Ne r 0
      hnN : LE.le n N
      ⊢ Eq (HMul.hMul (HMul.hMul (i r) (HPow.hPow (Invertible.invOf x) (HSub.hSub N  …
    -/
    rw [pow_add, ← mul_assoc, mul_assoc (i r), ← mul_pow, invOf_mul_self, one_pow, mul_one]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      inst✝² : Semiring R
      S : Type u_2
      inst✝¹ : CommSemiring S
      i : RingHom R S
      x : S
      inst✝ : Invertible x
      N : Nat
      f : Polynomial R
      hf : LE.le f.natDegree N
      ⊢ ∀ (f g : Polynomial R), LT.lt f.natDegree g.natDegree → LE.le g.natDegree N  …
    -/
  · intros
    /-
      case refine_3
      R : Type u_1
      inst✝² : Semiring R
      S : Type u_2
      inst✝¹ : CommSemiring S
      i : RingHom R S
      x : S
      inst✝ : Invertible x
      N : Nat
      f : Polynomial R
      hf : LE.le f.natDegree N
      f✝ g✝ : Polynomial R
      a✝³ : LT.lt f✝.natDegree g✝.natDegree
      a✝² : LE.le g✝.natDegree N
      a✝¹ : Eq (HMul.hMul (Polynomial.eval₂ i (Invertible.invOf x) (Polynomial.refle …
      a✝ : Eq (HMul.hMul (Polynomial.eval₂ i (Invertible.invOf x) (Polynomial.reflec …
      ⊢ Eq (HMul.hMul (Polynomial.eval₂ i (Invertible.invOf x) (Polynomial.reflect N …
    -/
    simp [*, add_mul]
    /-
      🎉 no goals
    -/


theorem eval₂_reflect_eq_zero_iff (i : R →+* S) (x : S) [Invertible x] (N : ℕ) (f : R[X])
    (hf : f.natDegree ≤ N) : eval₂ i (⅟ x) (reflect N f) = 0 ↔ eval₂ i x f = 0 := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    S : Type u_2
    inst✝¹ : CommSemiring S
    i : RingHom R S
    x : S
    inst✝ : Invertible x
    N : Nat
    f : Polynomial R
    hf : LE.le f.natDegree N
    ⊢ Iff (Eq (Polynomial.eval₂ i (Invertible.invOf x) (Polynomial.reflect N f)) 0 …
  -/
  conv_rhs => rw [← eval₂_reflect_mul_pow i x N f hf]
  /-
    R : Type u_1
    inst✝² : Semiring R
    S : Type u_2
    inst✝¹ : CommSemiring S
    i : RingHom R S
    x : S
    inst✝ : Invertible x
    N : Nat
    f : Polynomial R
    hf : LE.le f.natDegree N
    ⊢ Iff (Eq (Polynomial.eval₂ i (Invertible.invOf x) (Polynomial.reflect N f)) 0 …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝² : Semiring R
      S : Type u_2
      inst✝¹ : CommSemiring S
      i : RingHom R S
      x : S
      inst✝ : Invertible x
      N : Nat
      f : Polynomial R
      hf : LE.le f.natDegree N
      ⊢ Eq (Polynomial.eval₂ i (Invertible.invOf x) (Polynomial.reflect N f)) 0 → Eq …
    -/
  · intro h
    /-
      case mp
      R : Type u_1
      inst✝² : Semiring R
      S : Type u_2
      inst✝¹ : CommSemiring S
      i : RingHom R S
      x : S
      inst✝ : Invertible x
      N : Nat
      f : Polynomial R
      hf : LE.le f.natDegree N
      h : Eq (Polynomial.eval₂ i (Invertible.invOf x) (Polynomial.reflect N f)) 0
      ⊢ Eq (HMul.hMul (Polynomial.eval₂ i (Invertible.invOf x) (Polynomial.reflect N …
    -/
    rw [h, zero_mul]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝² : Semiring R
      S : Type u_2
      inst✝¹ : CommSemiring S
      i : RingHom R S
      x : S
      inst✝ : Invertible x
      N : Nat
      f : Polynomial R
      hf : LE.le f.natDegree N
      ⊢ Eq (HMul.hMul (Polynomial.eval₂ i (Invertible.invOf x) (Polynomial.reflect N …
    -/
  · intro h
    rw [← mul_one (eval₂ i (⅟ x) _), ← one_pow N, ← mul_invOf_self x, mul_pow, ← mul_assoc, h,
      zero_mul]


/-- The reverse of a polynomial f is the polynomial obtained by "reading f backwards".
Even though this is not the actual definition, `reverse f = f (1/X) * X ^ f.natDegree`. -/
noncomputable def reverse (f : R[X]) : R[X] :=
  reflect f.natDegree f


theorem coeff_reverse (f : R[X]) (n : ℕ) : f.reverse.coeff n = f.coeff (revAt f.natDegree n) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    n : Nat
    ⊢ Eq (f.reverse.coeff n) (f.coeff ((Polynomial.revAt f.natDegree) n))
  -/
  rw [reverse, coeff_reflect]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_zero_reverse (f : R[X]) : coeff (reverse f) 0 = leadingCoeff f := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ Eq (f.reverse.coeff 0) f.leadingCoeff
  -/
  rw [coeff_reverse, revAt_le (zero_le f.natDegree), tsub_zero, leadingCoeff]
  /-
    🎉 no goals
  -/


@[simp]
theorem reverse_zero : reverse (0 : R[X]) = 0 :=
  rfl


@[simp]
                                                      /-
                                                        R : Type u_1
                                                        inst✝ : Semiring R
                                                        f : Polynomial R
                                                        ⊢ Iff (Eq f.reverse 0) (Eq f 0)
                                                      -/
theorem reverse_eq_zero : f.reverse = 0 ↔ f = 0 := by simp [reverse]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem reverse_natDegree_le (f : R[X]) : f.reverse.natDegree ≤ f.natDegree := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ LE.le f.reverse.natDegree f.natDegree
  -/
  rw [natDegree_le_iff_degree_le, degree_le_iff_coeff_zero]
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ ∀ (m : Nat), LT.lt ↑f.natDegree ↑m → Eq (f.reverse.coeff m) 0
  -/
  intro n hn
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    n : Nat
    hn : LT.lt ↑f.natDegree ↑n
    ⊢ Eq (f.reverse.coeff n) 0
  -/
  rw [Nat.cast_lt] at hn
  rw [coeff_reverse, revAt, Function.Embedding.coeFn_mk, if_neg (not_le_of_gt hn),
    coeff_eq_zero_of_natDegree_lt hn]


theorem natDegree_eq_reverse_natDegree_add_natTrailingDegree (f : R[X]) :
    f.natDegree = f.reverse.natDegree + f.natTrailingDegree := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ Eq f.natDegree (HAdd.hAdd f.reverse.natDegree f.natTrailingDegree)
  -/
  by_cases hf : f = 0
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      hf : Eq f 0
      ⊢ Eq f.natDegree (HAdd.hAdd f.reverse.natDegree f.natTrailingDegree)
    -/
  · rw [hf, reverse_zero, natDegree_zero, natTrailingDegree_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    hf : Not (Eq f 0)
    ⊢ Eq f.natDegree (HAdd.hAdd f.reverse.natDegree f.natTrailingDegree)
  -/
  apply le_antisymm
    /-
      case neg.a
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      hf : Not (Eq f 0)
      ⊢ LE.le f.natDegree (HAdd.hAdd f.reverse.natDegree f.natTrailingDegree)
    -/
  · refine tsub_le_iff_right.mp ?_
    /-
      case neg.a
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      hf : Not (Eq f 0)
      ⊢ LE.le (HSub.hSub f.natDegree f.natTrailingDegree) f.reverse.natDegree
    -/
    apply le_natDegree_of_ne_zero
    /-
      case neg.a.h
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      hf : Not (Eq f 0)
      ⊢ Ne (f.reverse.coeff (HSub.hSub f.natDegree f.natTrailingDegree)) 0
    -/
    rw [reverse, coeff_reflect, ← revAt_le f.natTrailingDegree_le_natDegree, revAt_invol]
    /-
      case neg.a.h
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      hf : Not (Eq f 0)
      ⊢ Ne (f.coeff f.natTrailingDegree) 0
    -/
    exact trailingCoeff_nonzero_iff_nonzero.mpr hf
    /-
      🎉 no goals
    -/
    /-
      case neg.a
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      hf : Not (Eq f 0)
      ⊢ LE.le (HAdd.hAdd f.reverse.natDegree f.natTrailingDegree) f.natDegree
    -/
  · rw [← le_tsub_iff_left f.reverse_natDegree_le]
    /-
      case neg.a
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      hf : Not (Eq f 0)
      ⊢ LE.le f.natTrailingDegree (HSub.hSub f.natDegree f.reverse.natDegree)
    -/
    apply natTrailingDegree_le_of_ne_zero
    /-
      case neg.a.h
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      hf : Not (Eq f 0)
      ⊢ Ne (f.coeff (HSub.hSub f.natDegree f.reverse.natDegree)) 0
    -/
    have key := mt leadingCoeff_eq_zero.mp (mt reverse_eq_zero.mp hf)
    /-
      case neg.a.h
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      hf : Not (Eq f 0)
      key : Not (Eq f.reverse.leadingCoeff 0)
      ⊢ Ne (f.coeff (HSub.hSub f.natDegree f.reverse.natDegree)) 0
    -/
    rwa [leadingCoeff, coeff_reverse, revAt_le f.reverse_natDegree_le] at key
    /-
      🎉 no goals
    -/


theorem reverse_natDegree (f : R[X]) : f.reverse.natDegree = f.natDegree - f.natTrailingDegree := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ Eq f.reverse.natDegree (HSub.hSub f.natDegree f.natTrailingDegree)
  -/
  rw [f.natDegree_eq_reverse_natDegree_add_natTrailingDegree, add_tsub_cancel_right]
  /-
    🎉 no goals
  -/


theorem reverse_leadingCoeff (f : R[X]) : f.reverse.leadingCoeff = f.trailingCoeff := by
  rw [leadingCoeff, reverse_natDegree, ← revAt_le f.natTrailingDegree_le_natDegree,
    coeff_reverse, revAt_invol, trailingCoeff]


theorem natTrailingDegree_reverse (f : R[X]) : f.reverse.natTrailingDegree = 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ Eq f.reverse.natTrailingDegree 0
  -/
  rw [natTrailingDegree_eq_zero, reverse_eq_zero, coeff_zero_reverse, leadingCoeff_ne_zero]
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ Or (Eq f 0) (Ne f 0)
  -/
  exact eq_or_ne _ _
  /-
    🎉 no goals
  -/


theorem reverse_trailingCoeff (f : R[X]) : f.reverse.trailingCoeff = f.leadingCoeff := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ Eq f.reverse.trailingCoeff f.leadingCoeff
  -/
  rw [trailingCoeff, natTrailingDegree_reverse, coeff_zero_reverse]
  /-
    🎉 no goals
  -/


theorem reverse_mul {f g : R[X]} (fg : f.leadingCoeff * g.leadingCoeff ≠ 0) :
    reverse (f * g) = reverse f * reverse g := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f g : Polynomial R
    fg : Ne (HMul.hMul f.leadingCoeff g.leadingCoeff) 0
    ⊢ Eq (HMul.hMul f g).reverse (HMul.hMul f.reverse g.reverse)
  -/
  unfold reverse
  /-
    R : Type u_1
    inst✝ : Semiring R
    f g : Polynomial R
    fg : Ne (HMul.hMul f.leadingCoeff g.leadingCoeff) 0
    ⊢ Eq (Polynomial.reflect (HMul.hMul f g).natDegree (HMul.hMul f g)) (HMul.hMul …
  -/
  rw [natDegree_mul' fg, reflect_mul f g rfl.le rfl.le]
  /-
    🎉 no goals
  -/


@[simp]
theorem reverse_mul_of_domain {R : Type*} [Ring R] [NoZeroDivisors R] (f g : R[X]) :
    reverse (f * g) = reverse f * reverse g := by
  /-
    R : Type u_2
    inst✝¹ : Ring R
    inst✝ : NoZeroDivisors R
    f g : Polynomial R
    ⊢ Eq (HMul.hMul f g).reverse (HMul.hMul f.reverse g.reverse)
  -/
  by_cases f0 : f = 0
    /-
      case pos
      R : Type u_2
      inst✝¹ : Ring R
      inst✝ : NoZeroDivisors R
      f g : Polynomial R
      f0 : Eq f 0
      ⊢ Eq (HMul.hMul f g).reverse (HMul.hMul f.reverse g.reverse)
    -/
  · simp only [f0, zero_mul, reverse_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_2
    inst✝¹ : Ring R
    inst✝ : NoZeroDivisors R
    f g : Polynomial R
    f0 : Not (Eq f 0)
    ⊢ Eq (HMul.hMul f g).reverse (HMul.hMul f.reverse g.reverse)
  -/
  by_cases g0 : g = 0
    /-
      case pos
      R : Type u_2
      inst✝¹ : Ring R
      inst✝ : NoZeroDivisors R
      f g : Polynomial R
      f0 : Not (Eq f 0)
      g0 : Eq g 0
      ⊢ Eq (HMul.hMul f g).reverse (HMul.hMul f.reverse g.reverse)
    -/
  · rw [g0, mul_zero, reverse_zero, mul_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_2
    inst✝¹ : Ring R
    inst✝ : NoZeroDivisors R
    f g : Polynomial R
    f0 : Not (Eq f 0)
    g0 : Not (Eq g 0)
    ⊢ Eq (HMul.hMul f g).reverse (HMul.hMul f.reverse g.reverse)
  -/
  simp [reverse_mul, *]
  /-
    🎉 no goals
  -/


theorem trailingCoeff_mul {R : Type*} [Ring R] [NoZeroDivisors R] (p q : R[X]) :
    (p * q).trailingCoeff = p.trailingCoeff * q.trailingCoeff := by
  rw [← reverse_leadingCoeff, reverse_mul_of_domain, leadingCoeff_mul, reverse_leadingCoeff,
    reverse_leadingCoeff]


@[simp]
theorem coeff_one_reverse (f : R[X]) : coeff (reverse f) 1 = nextCoeff f := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ Eq (f.reverse.coeff 1) f.nextCoeff
  -/
  rw [coeff_reverse, nextCoeff]
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ Eq (f.coeff ((Polynomial.revAt f.natDegree) 1)) (ite (Eq f.natDegree 0) 0 (f …
  -/
  split_ifs with hf
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      hf : Eq f.natDegree 0
      ⊢ Eq (f.coeff ((Polynomial.revAt f.natDegree) 1)) 0
    -/
  · have : coeff f 1 = 0 := coeff_eq_zero_of_natDegree_lt (by simp only [hf, zero_lt_one])
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      hf : Eq f.natDegree 0
      this : Eq (f.coeff 1) 0
      ⊢ Eq (f.coeff ((Polynomial.revAt f.natDegree) 1)) 0
    -/
    simp [*, revAt]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      hf : Not (Eq f.natDegree 0)
      ⊢ Eq (f.coeff ((Polynomial.revAt f.natDegree) 1)) (f.coeff (HSub.hSub f.natDeg …
    -/
  · rw [revAt_le]
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      f : Polynomial R
      hf : Not (Eq f.natDegree 0)
      ⊢ LE.le 1 f.natDegree
    -/
    exact Nat.succ_le_iff.2 (pos_iff_ne_zero.2 hf)
    /-
      🎉 no goals
    -/


@[simp] lemma reverse_C (t : R) :
    reverse (C t) = C t := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    t : R
    ⊢ Eq (Polynomial.C t).reverse (Polynomial.C t)
  -/
  simp [reverse]
  /-
    🎉 no goals
  -/


@[simp] lemma reverse_mul_X (p : R[X]) : reverse (p * X) = reverse p := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq (HMul.hMul p Polynomial.X).reverse p.reverse
  -/
  nontriviality R
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    a✝ : Nontrivial R
    ⊢ Eq (HMul.hMul p Polynomial.X).reverse p.reverse
  -/
  rcases eq_or_ne p 0 with rfl | hp
    /-
      case inl
      R : Type u_1
      inst✝ : Semiring R
      a✝ : Nontrivial R
      ⊢ Eq (HMul.hMul 0 Polynomial.X).reverse (Polynomial.reverse 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      a✝ : Nontrivial R
      hp : Ne p 0
      ⊢ Eq (HMul.hMul p Polynomial.X).reverse p.reverse
    -/
  · simp [reverse, hp]
    /-
      🎉 no goals
    -/


@[simp] lemma reverse_X_mul (p : R[X]) : reverse (X * p) = reverse p := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq (HMul.hMul Polynomial.X p).reverse p.reverse
  -/
  rw [commute_X p, reverse_mul_X]
  /-
    🎉 no goals
  -/


@[simp] lemma reverse_mul_X_pow (p : R[X]) (n : ℕ) : reverse (p * X ^ n) = reverse p := by
  induction n with
  | zero => simp
  | succ n ih => rw [pow_succ, ← mul_assoc, reverse_mul_X, ih]


@[simp] lemma reverse_X_pow_mul (p : R[X]) (n : ℕ) : reverse (X ^ n * p) = reverse p := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow Polynomial.X n) p).reverse p.reverse
  -/
  rw [commute_X_pow p, reverse_mul_X_pow]
  /-
    🎉 no goals
  -/


@[simp] lemma reverse_add_C (p : R[X]) (t : R) :
    reverse (p + C t) = reverse p + C t * X ^ p.natDegree := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    t : R
    ⊢ Eq (HAdd.hAdd p (Polynomial.C t)).reverse (HAdd.hAdd p.reverse (HMul.hMul (P …
  -/
  simp [reverse]
  /-
    🎉 no goals
  -/


@[simp] lemma reverse_C_add (p : R[X]) (t : R) :
    reverse (C t + p) = C t * X ^ p.natDegree + reverse p := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    p : Polynomial R
    t : R
    ⊢ Eq (HAdd.hAdd (Polynomial.C t) p).reverse (HAdd.hAdd (HMul.hMul (Polynomial. …
  -/
  rw [add_comm, reverse_add_C, add_comm]
  /-
    🎉 no goals
  -/


theorem eval₂_reverse_mul_pow (i : R →+* S) (x : S) [Invertible x] (f : R[X]) :
    eval₂ i (⅟ x) (reverse f) * x ^ f.natDegree = eval₂ i x f :=
  eval₂_reflect_mul_pow i _ _ f le_rfl


@[simp]
theorem eval₂_reverse_eq_zero_iff (i : R →+* S) (x : S) [Invertible x] (f : R[X]) :
    eval₂ i (⅟ x) (reverse f) = 0 ↔ eval₂ i x f = 0 :=
  eval₂_reflect_eq_zero_iff i x _ _ le_rfl


@[simp]
theorem reflect_neg (f : R[X]) (N : ℕ) : reflect N (-f) = -reflect N f := by
  /-
    R : Type u_1
    inst✝ : Ring R
    f : Polynomial R
    N : Nat
    ⊢ Eq (Polynomial.reflect N (Neg.neg f)) (Neg.neg (Polynomial.reflect N f))
  -/
  rw [neg_eq_neg_one_mul, ← C_1, ← C_neg, reflect_C_mul, C_neg, C_1, ← neg_eq_neg_one_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem reflect_sub (f g : R[X]) (N : ℕ) : reflect N (f - g) = reflect N f - reflect N g := by
  /-
    R : Type u_1
    inst✝ : Ring R
    f g : Polynomial R
    N : Nat
    ⊢ Eq (Polynomial.reflect N (HSub.hSub f g)) (HSub.hSub (Polynomial.reflect N f …
  -/
  rw [sub_eq_add_neg, sub_eq_add_neg, reflect_add, reflect_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem reverse_neg (f : R[X]) : reverse (-f) = -reverse f := by
  /-
    R : Type u_1
    inst✝ : Ring R
    f : Polynomial R
    ⊢ Eq (Neg.neg f).reverse (Neg.neg f.reverse)
  -/
  rw [reverse, reverse, reflect_neg, natDegree_neg]
  /-
    🎉 no goals
  -/


