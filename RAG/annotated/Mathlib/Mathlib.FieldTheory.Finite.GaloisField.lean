instance FiniteField.isSplittingField_sub (K F : Type*) [Field K] [Fintype K]
    [Field F] [Algebra F K] : IsSplittingField F K (X ^ Fintype.card K - X) where
  splits' := by
    have h : (X ^ Fintype.card K - X : K[X]).natDegree = Fintype.card K :=
      FiniteField.X_pow_card_sub_X_natDegree_eq K Fintype.one_lt_card
    rw [← splits_id_iff_splits, splits_iff_card_roots, Polynomial.map_sub, Polynomial.map_pow,
      map_X, h, FiniteField.roots_X_pow_card_sub_X K, ← Finset.card_def, Finset.card_univ]
  adjoin_rootSet' := by
    classical
    trans Algebra.adjoin F ((roots (X ^ Fintype.card K - X : K[X])).toFinset : Set K)
    · simp only [rootSet, aroots, Polynomial.map_pow, map_X, Polynomial.map_sub]
    · rw [FiniteField.roots_X_pow_card_sub_X, val_toFinset, coe_univ, Algebra.adjoin_univ]


theorem galois_poly_separable {K : Type*} [Field K] (p q : ℕ) [CharP K p] (h : p ∣ q) :
    Separable (X ^ q - X : K[X]) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    p q : Nat
    inst✝ : CharP K p
    h : Dvd.dvd p q
    ⊢ (HSub.hSub (HPow.hPow Polynomial.X q) Polynomial.X).Separable
  -/
  use 1, X ^ q - X - 1
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    p q : Nat
    inst✝ : CharP K p
    h : Dvd.dvd p q
    ⊢ Eq (HAdd.hAdd (HMul.hMul 1 (HSub.hSub (HPow.hPow Polynomial.X q) Polynomial. …
  -/
  rw [← CharP.cast_eq_zero_iff K[X] p] at h
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    p q : Nat
    inst✝ : CharP K p
    h : Eq (↑q) 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul 1 (HSub.hSub (HPow.hPow Polynomial.X q) Polynomial. …
  -/
  rw [derivative_sub, derivative_X_pow, derivative_X, C_eq_natCast, h]
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    p q : Nat
    inst✝ : CharP K p
    h : Eq (↑q) 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul 1 (HSub.hSub (HPow.hPow Polynomial.X q) Polynomial. …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- A finite field with `p ^ n` elements.
Every field with the same cardinality is (non-canonically)
isomorphic to this field. -/
def GaloisField := SplittingField (X ^ p ^ n - X : (ZMod p)[X])
-- deriving Field -- Porting note: see https://github.com/leanprover-community/mathlib4/issues/5020


instance : Field (GaloisField p n) :=
  inferInstanceAs (Field (SplittingField _))


instance : Inhabited (@GaloisField 2 (Fact.mk Nat.prime_two) 1) := ⟨37⟩


instance : Algebra (ZMod p) (GaloisField p n) := SplittingField.algebra _


instance : IsSplittingField (ZMod p) (GaloisField p n) (X ^ p ^ n - X) :=
  Polynomial.IsSplittingField.splittingField _


instance : CharP (GaloisField p n) p :=
                                                          /-
                                                            p✝ : Nat
                                                            inst✝ : Fact (Nat.Prime p✝)
                                                            n✝ p : Nat
                                                            h_prime : Fact (Nat.Prime p)
                                                            n : Nat
                                                            ⊢ CharP (ZMod p) p
                                                          -/
  (Algebra.charP_iff (ZMod p) (GaloisField p n) p).mp (by infer_instance)
                                                          /-
                                                            🎉 no goals
                                                          -/


instance : FiniteDimensional (ZMod p) (GaloisField p n) := by
  /-
    p✝ : Nat
    inst✝ : Fact (Nat.Prime p✝)
    n✝ p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    ⊢ FiniteDimensional (ZMod p) (GaloisField p n)
  -/
  dsimp only [GaloisField]; infer_instance
                            /-
                              🎉 no goals
                            -/


instance : Finite (GaloisField p n) :=
  Module.finite_of_finite (ZMod p)


theorem finrank {n} (h : n ≠ 0) : Module.finrank (ZMod p) (GaloisField p n) = n := by
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    h : Ne n 0
    ⊢ Eq (Module.finrank (ZMod p) (GaloisField p n)) n
  -/
  haveI : Fintype (GaloisField p n) := Fintype.ofFinite (GaloisField p n)
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    h : Ne n 0
    this : Fintype (GaloisField p n)
    ⊢ Eq (Module.finrank (ZMod p) (GaloisField p n)) n
  -/
  set g_poly := (X ^ p ^ n - X : (ZMod p)[X])
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    h : Ne n 0
    this : Fintype (GaloisField p n)
    g_poly : Polynomial (ZMod p) := HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow p …
    ⊢ Eq (Module.finrank (ZMod p) (GaloisField p n)) n
  -/
  have hp : 1 < p := h_prime.out.one_lt
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    h : Ne n 0
    this : Fintype (GaloisField p n)
    g_poly : Polynomial (ZMod p) := HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow p …
    hp : LT.lt 1 p
    ⊢ Eq (Module.finrank (ZMod p) (GaloisField p n)) n
  -/
  have aux : g_poly ≠ 0 := FiniteField.X_pow_card_pow_sub_X_ne_zero _ h hp
  -- Porting note: in the statement of `key`, replaced `g_poly` by its value otherwise the
  -- proof fails
  have key : Fintype.card (g_poly.rootSet (GaloisField p n)) = g_poly.natDegree :=
    card_rootSet_eq_natDegree (galois_poly_separable p _ (dvd_pow (dvd_refl p) h))
      (SplittingField.splits (X ^ p ^ n - X : (ZMod p)[X]))
  have nat_degree_eq : g_poly.natDegree = p ^ n :=
    FiniteField.X_pow_card_pow_sub_X_natDegree_eq _ h hp
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    h : Ne n 0
    this : Fintype (GaloisField p n)
    g_poly : Polynomial (ZMod p) := HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow p …
    hp : LT.lt 1 p
    aux : Ne g_poly 0
    key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField p n))) g_poly.natDegree
    nat_degree_eq : Eq g_poly.natDegree (HPow.hPow p n)
    ⊢ Eq (Module.finrank (ZMod p) (GaloisField p n)) n
  -/
  rw [nat_degree_eq] at key
  suffices g_poly.rootSet (GaloisField p n) = Set.univ by
    simp_rw [this, ← Fintype.ofEquiv_card (Equiv.Set.univ _)] at key
    -- Porting note: prevents `card_eq_pow_finrank` from using a wrong instance for `Fintype`
    rw [@card_eq_pow_finrank (ZMod p) _ _ _ _ _ (_), ZMod.card] at key
    exact Nat.pow_right_injective (Nat.Prime.one_lt' p).out key
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    h : Ne n 0
    this : Fintype (GaloisField p n)
    g_poly : Polynomial (ZMod p) := HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow p …
    hp : LT.lt 1 p
    aux : Ne g_poly 0
    key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField p n))) (HPow.hPow p n)
    nat_degree_eq : Eq g_poly.natDegree (HPow.hPow p n)
    ⊢ Eq (g_poly.rootSet (GaloisField p n)) Set.univ
  -/
  rw [Set.eq_univ_iff_forall]
  suffices ∀ (x) (hx : x ∈ (⊤ : Subalgebra (ZMod p) (GaloisField p n))),
      x ∈ (X ^ p ^ n - X : (ZMod p)[X]).rootSet (GaloisField p n)
    by simpa
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    h : Ne n 0
    this : Fintype (GaloisField p n)
    g_poly : Polynomial (ZMod p) := HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow p …
    hp : LT.lt 1 p
    aux : Ne g_poly 0
    key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField p n))) (HPow.hPow p n)
    nat_degree_eq : Eq g_poly.natDegree (HPow.hPow p n)
    ⊢ ∀ (x : GaloisField p n), Membership.mem Top.top x → Membership.mem ((HSub.hS …
  -/
  rw [← SplittingField.adjoin_rootSet]
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    h : Ne n 0
    this : Fintype (GaloisField p n)
    g_poly : Polynomial (ZMod p) := HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow p …
    hp : LT.lt 1 p
    aux : Ne g_poly 0
    key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField p n))) (HPow.hPow p n)
    nat_degree_eq : Eq g_poly.natDegree (HPow.hPow p n)
    ⊢ ∀ (x : GaloisField p n), Membership.mem (Algebra.adjoin (ZMod p) ((HSub.hSub …
  -/
  simp_rw [Algebra.mem_adjoin_iff]
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    h : Ne n 0
    this : Fintype (GaloisField p n)
    g_poly : Polynomial (ZMod p) := HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow p …
    hp : LT.lt 1 p
    aux : Ne g_poly 0
    key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField p n))) (HPow.hPow p n)
    nat_degree_eq : Eq g_poly.natDegree (HPow.hPow p n)
    ⊢ ∀ (x : GaloisField p n), Membership.mem (Subring.closure (Union.union (Set.r …
  -/
  intro x hx
  -- We discharge the `p = 0` separately, to avoid typeclass issues on `ZMod p`.
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    h : Ne n 0
    this : Fintype (GaloisField p n)
    g_poly : Polynomial (ZMod p) := HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow p …
    hp : LT.lt 1 p
    aux : Ne g_poly 0
    key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField p n))) (HPow.hPow p n)
    nat_degree_eq : Eq g_poly.natDegree (HPow.hPow p n)
    x : GaloisField p n
    hx : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZMo …
    ⊢ Membership.mem ((HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow p n)) Polynomi …
  -/
  cases p; cases hp
  /-
    case succ
    n : Nat
    h : Ne n 0
    n✝ : Nat
    h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
    this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
    g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
    hp : LT.lt 1 (HAdd.hAdd n✝ 1)
    aux : Ne g_poly 0
    key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField (HAdd.hAdd n✝ 1) n))) (HP …
    nat_degree_eq : Eq g_poly.natDegree (HPow.hPow (HAdd.hAdd n✝ 1) n)
    x : GaloisField (HAdd.hAdd n✝ 1) n
    hx : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZMo …
    ⊢ Membership.mem ((HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow (HAdd.hAdd n✝  …
  -/
  refine Subring.closure_induction ?_ ?_ ?_ ?_ ?_ ?_ hx <;> simp_rw [mem_rootSet_of_ne aux]
    /-
      case succ.refine_1
      n : Nat
      h : Ne n 0
      n✝ : Nat
      h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
      this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
      g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
      hp : LT.lt 1 (HAdd.hAdd n✝ 1)
      aux : Ne g_poly 0
      key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField (HAdd.hAdd n✝ 1) n))) (HP …
      nat_degree_eq : Eq g_poly.natDegree (HPow.hPow (HAdd.hAdd n✝ 1) n)
      x : GaloisField (HAdd.hAdd n✝ 1) n
      hx : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZMo …
      ⊢ ∀ (x : GaloisField (HAdd.hAdd n✝ 1) n), Membership.mem (Union.union (Set.ran …
    -/
  · rintro x (⟨r, rfl⟩ | hx)
      /-
        case succ.refine_1.inl.intro
        n : Nat
        h : Ne n 0
        n✝ : Nat
        h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
        this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
        g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
        hp : LT.lt 1 (HAdd.hAdd n✝ 1)
        aux : Ne g_poly 0
        key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField (HAdd.hAdd n✝ 1) n))) (HP …
        nat_degree_eq : Eq g_poly.natDegree (HPow.hPow (HAdd.hAdd n✝ 1) n)
        x : GaloisField (HAdd.hAdd n✝ 1) n
        hx : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZMo …
        r : ZMod (HAdd.hAdd n✝ 1)
        ⊢ Eq ((Polynomial.aeval ((algebraMap (ZMod (HAdd.hAdd n✝ 1)) (GaloisField (HAd …
      -/
    · simp only [g_poly, map_sub, map_pow, aeval_X]
      /-
        case succ.refine_1.inl.intro
        n : Nat
        h : Ne n 0
        n✝ : Nat
        h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
        this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
        g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
        hp : LT.lt 1 (HAdd.hAdd n✝ 1)
        aux : Ne g_poly 0
        key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField (HAdd.hAdd n✝ 1) n))) (HP …
        nat_degree_eq : Eq g_poly.natDegree (HPow.hPow (HAdd.hAdd n✝ 1) n)
        x : GaloisField (HAdd.hAdd n✝ 1) n
        hx : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZMo …
        r : ZMod (HAdd.hAdd n✝ 1)
        ⊢ Eq (HSub.hSub (HPow.hPow ((algebraMap (ZMod (HAdd.hAdd n✝ 1)) (GaloisField ( …
      -/
      rw [← map_pow, ZMod.pow_card_pow, sub_self]
      /-
        🎉 no goals
      -/
      /-
        case succ.refine_1.inr
        n : Nat
        h : Ne n 0
        n✝ : Nat
        h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
        this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
        g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
        hp : LT.lt 1 (HAdd.hAdd n✝ 1)
        aux : Ne g_poly 0
        key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField (HAdd.hAdd n✝ 1) n))) (HP …
        nat_degree_eq : Eq g_poly.natDegree (HPow.hPow (HAdd.hAdd n✝ 1) n)
        x✝ : GaloisField (HAdd.hAdd n✝ 1) n
        hx✝ : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZM …
        x : GaloisField (HAdd.hAdd n✝ 1) n
        hx : Membership.mem ((HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow (HAdd.hAdd  …
        ⊢ Eq ((Polynomial.aeval x) g_poly) 0
      -/
    · dsimp only [GaloisField] at hx
      /-
        case succ.refine_1.inr
        n : Nat
        h : Ne n 0
        n✝ : Nat
        h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
        this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
        g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
        hp : LT.lt 1 (HAdd.hAdd n✝ 1)
        aux : Ne g_poly 0
        key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField (HAdd.hAdd n✝ 1) n))) (HP …
        nat_degree_eq : Eq g_poly.natDegree (HPow.hPow (HAdd.hAdd n✝ 1) n)
        x✝ : GaloisField (HAdd.hAdd n✝ 1) n
        hx✝ : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZM …
        x : GaloisField (HAdd.hAdd n✝ 1) n
        hx : Membership.mem ((HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow (HAdd.hAdd  …
        ⊢ Eq ((Polynomial.aeval x) g_poly) 0
      -/
      rwa [mem_rootSet_of_ne aux] at hx
      /-
        🎉 no goals
      -/
    /-
      case succ.refine_2
      n : Nat
      h : Ne n 0
      n✝ : Nat
      h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
      this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
      g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
      hp : LT.lt 1 (HAdd.hAdd n✝ 1)
      aux : Ne g_poly 0
      key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField (HAdd.hAdd n✝ 1) n))) (HP …
      nat_degree_eq : Eq g_poly.natDegree (HPow.hPow (HAdd.hAdd n✝ 1) n)
      x : GaloisField (HAdd.hAdd n✝ 1) n
      hx : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZMo …
      ⊢ Eq ((Polynomial.aeval 0) g_poly) 0
    -/
  · rw [← coeff_zero_eq_aeval_zero']
    simp only [g_poly, coeff_X_pow, coeff_X_zero, sub_zero, _root_.map_eq_zero, ite_eq_right_iff,
      one_ne_zero, coeff_sub]
    /-
      case succ.refine_2
      n : Nat
      h : Ne n 0
      n✝ : Nat
      h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
      this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
      g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
      hp : LT.lt 1 (HAdd.hAdd n✝ 1)
      aux : Ne g_poly 0
      key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField (HAdd.hAdd n✝ 1) n))) (HP …
      nat_degree_eq : Eq g_poly.natDegree (HPow.hPow (HAdd.hAdd n✝ 1) n)
      x : GaloisField (HAdd.hAdd n✝ 1) n
      hx : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZMo …
      ⊢ Eq 0 (HPow.hPow (HAdd.hAdd n✝ 1) n) → False
    -/
    intro hn
    /-
      case succ.refine_2
      n : Nat
      h : Ne n 0
      n✝ : Nat
      h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
      this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
      g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
      hp : LT.lt 1 (HAdd.hAdd n✝ 1)
      aux : Ne g_poly 0
      key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField (HAdd.hAdd n✝ 1) n))) (HP …
      nat_degree_eq : Eq g_poly.natDegree (HPow.hPow (HAdd.hAdd n✝ 1) n)
      x : GaloisField (HAdd.hAdd n✝ 1) n
      hx : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZMo …
      hn : Eq 0 (HPow.hPow (HAdd.hAdd n✝ 1) n)
      ⊢ False
    -/
    exact Nat.not_lt_zero 1 (pow_eq_zero hn.symm ▸ hp)
    /-
      🎉 no goals
    -/
    /-
      case succ.refine_3
      n : Nat
      h : Ne n 0
      n✝ : Nat
      h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
      this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
      g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
      hp : LT.lt 1 (HAdd.hAdd n✝ 1)
      aux : Ne g_poly 0
      key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField (HAdd.hAdd n✝ 1) n))) (HP …
      nat_degree_eq : Eq g_poly.natDegree (HPow.hPow (HAdd.hAdd n✝ 1) n)
      x : GaloisField (HAdd.hAdd n✝ 1) n
      hx : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZMo …
      ⊢ Eq ((Polynomial.aeval 1) g_poly) 0
    -/
  · simp [g_poly]
    /-
      🎉 no goals
    -/
    /-
      case succ.refine_4
      n : Nat
      h : Ne n 0
      n✝ : Nat
      h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
      this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
      g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
      hp : LT.lt 1 (HAdd.hAdd n✝ 1)
      aux : Ne g_poly 0
      key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField (HAdd.hAdd n✝ 1) n))) (HP …
      nat_degree_eq : Eq g_poly.natDegree (HPow.hPow (HAdd.hAdd n✝ 1) n)
      x : GaloisField (HAdd.hAdd n✝ 1) n
      hx : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZMo …
      ⊢ ∀ (x y : GaloisField (HAdd.hAdd n✝ 1) n), Membership.mem (Subring.closure (U …
    -/
  · simp only [g_poly, aeval_X_pow, aeval_X, map_sub, add_pow_char_pow, sub_eq_zero]
    /-
      case succ.refine_4
      n : Nat
      h : Ne n 0
      n✝ : Nat
      h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
      this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
      g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
      hp : LT.lt 1 (HAdd.hAdd n✝ 1)
      aux : Ne g_poly 0
      key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField (HAdd.hAdd n✝ 1) n))) (HP …
      nat_degree_eq : Eq g_poly.natDegree (HPow.hPow (HAdd.hAdd n✝ 1) n)
      x : GaloisField (HAdd.hAdd n✝ 1) n
      hx : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZMo …
      ⊢ ∀ (x y : GaloisField (HAdd.hAdd n✝ 1) n), Membership.mem (Subring.closure (U …
    -/
    intro x y _ _ hx hy
    /-
      case succ.refine_4
      n : Nat
      h : Ne n 0
      n✝ : Nat
      h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
      this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
      g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
      hp : LT.lt 1 (HAdd.hAdd n✝ 1)
      aux : Ne g_poly 0
      key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField (HAdd.hAdd n✝ 1) n))) (HP …
      nat_degree_eq : Eq g_poly.natDegree (HPow.hPow (HAdd.hAdd n✝ 1) n)
      x✝ : GaloisField (HAdd.hAdd n✝ 1) n
      hx✝¹ : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (Z …
      x y : GaloisField (HAdd.hAdd n✝ 1) n
      hx✝ : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZM …
      hy✝ : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZM …
      hx : Eq (HPow.hPow x (HPow.hPow (HAdd.hAdd n✝ 1) n)) x
      hy : Eq (HPow.hPow y (HPow.hPow (HAdd.hAdd n✝ 1) n)) y
      ⊢ Eq (HAdd.hAdd (HPow.hPow x (HPow.hPow (HAdd.hAdd n✝ 1) n)) (HPow.hPow y (HPo …
    -/
    rw [hx, hy]
    /-
      🎉 no goals
    -/
    /-
      case succ.refine_5
      n : Nat
      h : Ne n 0
      n✝ : Nat
      h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
      this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
      g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
      hp : LT.lt 1 (HAdd.hAdd n✝ 1)
      aux : Ne g_poly 0
      key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField (HAdd.hAdd n✝ 1) n))) (HP …
      nat_degree_eq : Eq g_poly.natDegree (HPow.hPow (HAdd.hAdd n✝ 1) n)
      x : GaloisField (HAdd.hAdd n✝ 1) n
      hx : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZMo …
      ⊢ ∀ (x : GaloisField (HAdd.hAdd n✝ 1) n), Membership.mem (Subring.closure (Uni …
    -/
  · intro x _ hx
    /-
      case succ.refine_5
      n : Nat
      h : Ne n 0
      n✝ : Nat
      h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
      this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
      g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
      hp : LT.lt 1 (HAdd.hAdd n✝ 1)
      aux : Ne g_poly 0
      key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField (HAdd.hAdd n✝ 1) n))) (HP …
      nat_degree_eq : Eq g_poly.natDegree (HPow.hPow (HAdd.hAdd n✝ 1) n)
      x✝ : GaloisField (HAdd.hAdd n✝ 1) n
      hx✝¹ : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (Z …
      x : GaloisField (HAdd.hAdd n✝ 1) n
      hx✝ : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZM …
      hx : Eq ((Polynomial.aeval x) g_poly) 0
      ⊢ Eq ((Polynomial.aeval (Neg.neg x)) g_poly) 0
    -/
    simp only [g_poly, sub_eq_zero, aeval_X_pow, aeval_X, map_sub, sub_neg_eq_add] at *
    /-
      case succ.refine_5
      n : Nat
      h : Ne n 0
      n✝ : Nat
      h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
      this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
      g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
      hp : LT.lt 1 (HAdd.hAdd n✝ 1)
      aux : Ne (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow (HAdd.hAdd n✝ 1) n)) Po …
      key : Eq (Fintype.card ↑((HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow (HAdd.h …
      nat_degree_eq : Eq (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow (HAdd.hAdd n✝ …
      x✝ : GaloisField (HAdd.hAdd n✝ 1) n
      hx✝¹ : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (Z …
      x : GaloisField (HAdd.hAdd n✝ 1) n
      hx✝ : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZM …
      hx : Eq (HPow.hPow x (HPow.hPow (HAdd.hAdd n✝ 1) n)) x
      ⊢ Eq (HAdd.hAdd (HPow.hPow (Neg.neg x) (HPow.hPow (HAdd.hAdd n✝ 1) n)) x) 0
    -/
    rw [neg_pow, hx, neg_one_pow_char_pow]
    /-
      case succ.refine_5
      n : Nat
      h : Ne n 0
      n✝ : Nat
      h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
      this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
      g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
      hp : LT.lt 1 (HAdd.hAdd n✝ 1)
      aux : Ne (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow (HAdd.hAdd n✝ 1) n)) Po …
      key : Eq (Fintype.card ↑((HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow (HAdd.h …
      nat_degree_eq : Eq (HSub.hSub (HPow.hPow Polynomial.X (HPow.hPow (HAdd.hAdd n✝ …
      x✝ : GaloisField (HAdd.hAdd n✝ 1) n
      hx✝¹ : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (Z …
      x : GaloisField (HAdd.hAdd n✝ 1) n
      hx✝ : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZM …
      hx : Eq (HPow.hPow x (HPow.hPow (HAdd.hAdd n✝ 1) n)) x
      ⊢ Eq (HAdd.hAdd (HMul.hMul (-1) x) x) 0
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case succ.refine_6
      n : Nat
      h : Ne n 0
      n✝ : Nat
      h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
      this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
      g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
      hp : LT.lt 1 (HAdd.hAdd n✝ 1)
      aux : Ne g_poly 0
      key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField (HAdd.hAdd n✝ 1) n))) (HP …
      nat_degree_eq : Eq g_poly.natDegree (HPow.hPow (HAdd.hAdd n✝ 1) n)
      x : GaloisField (HAdd.hAdd n✝ 1) n
      hx : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZMo …
      ⊢ ∀ (x y : GaloisField (HAdd.hAdd n✝ 1) n), Membership.mem (Subring.closure (U …
    -/
  · simp only [g_poly, aeval_X_pow, aeval_X, map_sub, mul_pow, sub_eq_zero]
    /-
      case succ.refine_6
      n : Nat
      h : Ne n 0
      n✝ : Nat
      h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
      this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
      g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
      hp : LT.lt 1 (HAdd.hAdd n✝ 1)
      aux : Ne g_poly 0
      key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField (HAdd.hAdd n✝ 1) n))) (HP …
      nat_degree_eq : Eq g_poly.natDegree (HPow.hPow (HAdd.hAdd n✝ 1) n)
      x : GaloisField (HAdd.hAdd n✝ 1) n
      hx : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZMo …
      ⊢ ∀ (x y : GaloisField (HAdd.hAdd n✝ 1) n), Membership.mem (Subring.closure (U …
    -/
    intro x y _ _ hx hy
    /-
      case succ.refine_6
      n : Nat
      h : Ne n 0
      n✝ : Nat
      h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
      this : Fintype (GaloisField (HAdd.hAdd n✝ 1) n)
      g_poly : Polynomial (ZMod (HAdd.hAdd n✝ 1)) := HSub.hSub (HPow.hPow Polynomial …
      hp : LT.lt 1 (HAdd.hAdd n✝ 1)
      aux : Ne g_poly 0
      key : Eq (Fintype.card ↑(g_poly.rootSet (GaloisField (HAdd.hAdd n✝ 1) n))) (HP …
      nat_degree_eq : Eq g_poly.natDegree (HPow.hPow (HAdd.hAdd n✝ 1) n)
      x✝ : GaloisField (HAdd.hAdd n✝ 1) n
      hx✝¹ : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (Z …
      x y : GaloisField (HAdd.hAdd n✝ 1) n
      hx✝ : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZM …
      hy✝ : Membership.mem (Subring.closure (Union.union (Set.range ⇑(algebraMap (ZM …
      hx : Eq (HPow.hPow x (HPow.hPow (HAdd.hAdd n✝ 1) n)) x
      hy : Eq (HPow.hPow y (HPow.hPow (HAdd.hAdd n✝ 1) n)) y
      ⊢ Eq (HMul.hMul (HPow.hPow x (HPow.hPow (HAdd.hAdd n✝ 1) n)) (HPow.hPow y (HPo …
    -/
    rw [hx, hy]
    /-
      🎉 no goals
    -/


theorem card (h : n ≠ 0) : Nat.card (GaloisField p n) = p ^ n := by
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    h : Ne n 0
    ⊢ Eq (Nat.card (GaloisField p n)) (HPow.hPow p n)
  -/
  let b := IsNoetherian.finsetBasis (ZMod p) (GaloisField p n)
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    h : Ne n 0
    b : Basis (Subtype fun x => Membership.mem (IsNoetherian.finsetBasisIndex (ZMo …
    ⊢ Eq (Nat.card (GaloisField p n)) (HPow.hPow p n)
  -/
  haveI : Fintype (GaloisField p n) := Fintype.ofFinite (GaloisField p n)
  rw [Nat.card_eq_fintype_card, Module.card_fintype b, ← Module.finrank_eq_card_basis b,
    ZMod.card, finrank p h]


theorem splits_zmod_X_pow_sub_X : Splits (RingHom.id (ZMod p)) (X ^ p - X) := by
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    ⊢ Polynomial.Splits (RingHom.id (ZMod p)) (HSub.hSub (HPow.hPow Polynomial.X p …
  -/
  have hp : 1 < p := h_prime.out.one_lt
  have h1 : roots (X ^ p - X : (ZMod p)[X]) = Finset.univ.val := by
    convert FiniteField.roots_X_pow_card_sub_X (ZMod p)
    exact (ZMod.card p).symm
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    hp : LT.lt 1 p
    h1 : Eq (HSub.hSub (HPow.hPow Polynomial.X p) Polynomial.X).roots Finset.univ. …
    ⊢ Polynomial.Splits (RingHom.id (ZMod p)) (HSub.hSub (HPow.hPow Polynomial.X p …
  -/
  have h2 := FiniteField.X_pow_card_sub_X_natDegree_eq (ZMod p) hp
  -- We discharge the `p = 0` separately, to avoid typeclass issues on `ZMod p`.
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    hp : LT.lt 1 p
    h1 : Eq (HSub.hSub (HPow.hPow Polynomial.X p) Polynomial.X).roots Finset.univ. …
    h2 : Eq (HSub.hSub (HPow.hPow Polynomial.X p) Polynomial.X).natDegree p
    ⊢ Polynomial.Splits (RingHom.id (ZMod p)) (HSub.hSub (HPow.hPow Polynomial.X p …
  -/
  cases p; cases hp
  /-
    case succ
    n✝ : Nat
    h_prime : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
    hp : LT.lt 1 (HAdd.hAdd n✝ 1)
    h1 : Eq (HSub.hSub (HPow.hPow Polynomial.X (HAdd.hAdd n✝ 1)) Polynomial.X).roo …
    h2 : Eq (HSub.hSub (HPow.hPow Polynomial.X (HAdd.hAdd n✝ 1)) Polynomial.X).nat …
    ⊢ Polynomial.Splits (RingHom.id (ZMod (HAdd.hAdd n✝ 1))) (HSub.hSub (HPow.hPow …
  -/
  rw [splits_iff_card_roots, h1, ← Finset.card_def, Finset.card_univ, h2, ZMod.card]
  /-
    🎉 no goals
  -/


/-- A Galois field with exponent 1 is equivalent to `ZMod` -/
def equivZmodP : GaloisField p 1 ≃ₐ[ZMod p] ZMod p :=
                                                                      /-
                                                                        p✝ : Nat
                                                                        inst✝ : Fact (Nat.Prime p✝)
                                                                        n✝ p : Nat
                                                                        h_prime : Fact (Nat.Prime p)
                                                                        n : Nat
                                                                        ⊢ Eq (HPow.hPow Polynomial.X (HPow.hPow p 1)) (HPow.hPow Polynomial.X (Fintype …
                                                                      -/
  let h : (X ^ p ^ 1 : (ZMod p)[X]) = X ^ Fintype.card (ZMod p) := by rw [pow_one, ZMod.card p]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
                                                                      /-
                                                                        p✝ : Nat
                                                                        inst✝ : Fact (Nat.Prime p✝)
                                                                        n✝ p : Nat
                                                                        h_prime : Fact (Nat.Prime p)
                                                                        n : Nat
                                                                        h : Eq (HPow.hPow Polynomial.X (HPow.hPow p 1)) (HPow.hPow Polynomial.X (Finty …
                                                                        ⊢ Polynomial.IsSplittingField (ZMod p) (ZMod p) (HSub.hSub (HPow.hPow Polynomi …
                                                                      -/
  let inst : IsSplittingField (ZMod p) (ZMod p) (X ^ p ^ 1 - X) := by rw [h]; infer_instance
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
  (@IsSplittingField.algEquiv _ (ZMod p) _ _ _ (X ^ p ^ 1 - X : (ZMod p)[X]) inst).symm


theorem _root_.FiniteField.splits_X_pow_card_sub_X :
    Splits (algebraMap (ZMod p) K) (X ^ Fintype.card K - X) :=
  (FiniteField.isSplittingField_sub K (ZMod p)).splits


@[deprecated (since := "2024-11-12")]
alias splits_X_pow_card_sub_X := FiniteField.splits_X_pow_card_sub_X


theorem _root_.FiniteField.isSplittingField_of_card_eq (h : Fintype.card K = p ^ n) :
    IsSplittingField (ZMod p) K (X ^ p ^ n - X) :=
  h ▸ FiniteField.isSplittingField_sub K (ZMod p)


@[deprecated (since := "2024-11-12")]
alias isSplittingField_of_card_eq := FiniteField.isSplittingField_of_card_eq


/-- Any finite field is (possibly non canonically) isomorphic to some Galois field. -/
def algEquivGaloisFieldOfFintype (h : Fintype.card K = p ^ n) : K ≃ₐ[ZMod p] GaloisField p n :=
  haveI := FiniteField.isSplittingField_of_card_eq _ _ h
  IsSplittingField.algEquiv _ _


theorem _root_.FiniteField.splits_X_pow_nat_card_sub_X [Finite K] :
    Splits (algebraMap (ZMod p) K) (X ^ Nat.card K - X) := by
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : Algebra (ZMod p) K
    inst✝ : Finite K
    ⊢ Polynomial.Splits (algebraMap (ZMod p) K) (HSub.hSub (HPow.hPow Polynomial.X …
  -/
  haveI : Fintype K := Fintype.ofFinite K
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : Algebra (ZMod p) K
    inst✝ : Finite K
    this : Fintype K
    ⊢ Polynomial.Splits (algebraMap (ZMod p) K) (HSub.hSub (HPow.hPow Polynomial.X …
  -/
  rw [Nat.card_eq_fintype_card]
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    K : Type u_1
    inst✝² : Field K
    inst✝¹ : Algebra (ZMod p) K
    inst✝ : Finite K
    this : Fintype K
    ⊢ Polynomial.Splits (algebraMap (ZMod p) K) (HSub.hSub (HPow.hPow Polynomial.X …
  -/
  exact (FiniteField.isSplittingField_sub K (ZMod p)).splits
  /-
    🎉 no goals
  -/


theorem _root_.FiniteField.isSplittingField_of_nat_card_eq (h : Nat.card K = p ^ n) :
    IsSplittingField (ZMod p) K (X ^ p ^ n - X) := by
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : Algebra (ZMod p) K
    h : Eq (Nat.card K) (HPow.hPow p n)
    ⊢ Polynomial.IsSplittingField (ZMod p) K (HSub.hSub (HPow.hPow Polynomial.X (H …
  -/
  haveI : Finite K := (Nat.card_pos_iff.mp (h ▸ pow_pos h_prime.1.pos n)).2
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : Algebra (ZMod p) K
    h : Eq (Nat.card K) (HPow.hPow p n)
    this : Finite K
    ⊢ Polynomial.IsSplittingField (ZMod p) K (HSub.hSub (HPow.hPow Polynomial.X (H …
  -/
  haveI : Fintype K := Fintype.ofFinite K
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : Algebra (ZMod p) K
    h : Eq (Nat.card K) (HPow.hPow p n)
    this✝ : Finite K
    this : Fintype K
    ⊢ Polynomial.IsSplittingField (ZMod p) K (HSub.hSub (HPow.hPow Polynomial.X (H …
  -/
  rw [← h, Nat.card_eq_fintype_card]
  /-
    p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : Algebra (ZMod p) K
    h : Eq (Nat.card K) (HPow.hPow p n)
    this✝ : Finite K
    this : Fintype K
    ⊢ Polynomial.IsSplittingField (ZMod p) K (HSub.hSub (HPow.hPow Polynomial.X (F …
  -/
  exact FiniteField.isSplittingField_sub K (ZMod p)
  /-
    🎉 no goals
  -/


instance (priority := 100) {K K' : Type*} [Field K] [Field K'] [Finite K'] [Algebra K K'] :
    IsGalois K K' := by
  /-
    p✝ : Nat
    inst✝⁶ : Fact (Nat.Prime p✝)
    n✝ p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    K✝ : Type u_1
    inst✝⁵ : Field K✝
    inst✝⁴ : Algebra (ZMod p) K✝
    K : Type u_2
    K' : Type u_3
    inst✝³ : Field K
    inst✝² : Field K'
    inst✝¹ : Finite K'
    inst✝ : Algebra K K'
    ⊢ IsGalois K K'
  -/
  cases nonempty_fintype K'
  /-
    case intro
    p✝ : Nat
    inst✝⁶ : Fact (Nat.Prime p✝)
    n✝ p : Nat
    h_prime : Fact (Nat.Prime p)
    n : Nat
    K✝ : Type u_1
    inst✝⁵ : Field K✝
    inst✝⁴ : Algebra (ZMod p) K✝
    K : Type u_2
    K' : Type u_3
    inst✝³ : Field K
    inst✝² : Field K'
    inst✝¹ : Finite K'
    inst✝ : Algebra K K'
    val✝ : Fintype K'
    ⊢ IsGalois K K'
  -/
  obtain ⟨p, hp⟩ := CharP.exists K
  /-
    case intro.intro
    p✝¹ : Nat
    inst✝⁶ : Fact (Nat.Prime p✝¹)
    n✝ p✝ : Nat
    h_prime : Fact (Nat.Prime p✝)
    n : Nat
    K✝ : Type u_1
    inst✝⁵ : Field K✝
    inst✝⁴ : Algebra (ZMod p✝) K✝
    K : Type u_2
    K' : Type u_3
    inst✝³ : Field K
    inst✝² : Field K'
    inst✝¹ : Finite K'
    inst✝ : Algebra K K'
    val✝ : Fintype K'
    p : Nat
    hp : CharP K p
    ⊢ IsGalois K K'
  -/
  haveI : CharP K p := hp
  /-
    case intro.intro
    p✝¹ : Nat
    inst✝⁶ : Fact (Nat.Prime p✝¹)
    n✝ p✝ : Nat
    h_prime : Fact (Nat.Prime p✝)
    n : Nat
    K✝ : Type u_1
    inst✝⁵ : Field K✝
    inst✝⁴ : Algebra (ZMod p✝) K✝
    K : Type u_2
    K' : Type u_3
    inst✝³ : Field K
    inst✝² : Field K'
    inst✝¹ : Finite K'
    inst✝ : Algebra K K'
    val✝ : Fintype K'
    p : Nat
    hp : CharP K p
    this : CharP K p
    ⊢ IsGalois K K'
  -/
  haveI : CharP K' p := charP_of_injective_algebraMap' K K' p
  exact IsGalois.of_separable_splitting_field
    (galois_poly_separable p (Fintype.card K')
      (let ⟨n, _, hn⟩ := FiniteField.card K' p
      hn.symm ▸ dvd_pow_self p n.ne_zero))


/-- Any finite field is (possibly non canonically) isomorphic to some Galois field. -/
def algEquivGaloisField (h : Nat.card K = p ^ n) : K ≃ₐ[ZMod p] GaloisField p n :=
  haveI := FiniteField.isSplittingField_of_nat_card_eq _ _ h
  IsSplittingField.algEquiv _ _


/-- Uniqueness of finite fields:
  Any two finite fields of the same cardinality are (possibly non canonically) isomorphic -/
def algEquivOfCardEq (p : ℕ) [h_prime : Fact p.Prime] [Algebra (ZMod p) K] [Algebra (ZMod p) K']
    (hKK' : Fintype.card K = Fintype.card K') : K ≃ₐ[ZMod p] K' := by
  /-
    p✝ : Nat
    inst✝⁶ : Fact (Nat.Prime p✝)
    n : Nat
    K : Type u_1
    inst✝⁵ : Field K
    inst✝⁴ : Fintype K
    K' : Type u_2
    inst✝³ : Field K'
    inst✝² : Fintype K'
    p : Nat
    h_prime : Fact (Nat.Prime p)
    inst✝¹ : Algebra (ZMod p) K
    inst✝ : Algebra (ZMod p) K'
    hKK' : Eq (Fintype.card K) (Fintype.card K')
    ⊢ AlgEquiv (ZMod p) K K'
  -/
  have : CharP K p := by rw [← Algebra.charP_iff (ZMod p) K p]; exact ZMod.charP p
  /-
    p✝ : Nat
    inst✝⁶ : Fact (Nat.Prime p✝)
    n : Nat
    K : Type u_1
    inst✝⁵ : Field K
    inst✝⁴ : Fintype K
    K' : Type u_2
    inst✝³ : Field K'
    inst✝² : Fintype K'
    p : Nat
    h_prime : Fact (Nat.Prime p)
    inst✝¹ : Algebra (ZMod p) K
    inst✝ : Algebra (ZMod p) K'
    hKK' : Eq (Fintype.card K) (Fintype.card K')
    this : CharP K p
    ⊢ AlgEquiv (ZMod p) K K'
  -/
  have : CharP K' p := by rw [← Algebra.charP_iff (ZMod p) K' p]; exact ZMod.charP p
  /-
    p✝ : Nat
    inst✝⁶ : Fact (Nat.Prime p✝)
    n : Nat
    K : Type u_1
    inst✝⁵ : Field K
    inst✝⁴ : Fintype K
    K' : Type u_2
    inst✝³ : Field K'
    inst✝² : Fintype K'
    p : Nat
    h_prime : Fact (Nat.Prime p)
    inst✝¹ : Algebra (ZMod p) K
    inst✝ : Algebra (ZMod p) K'
    hKK' : Eq (Fintype.card K) (Fintype.card K')
    this✝ : CharP K p
    this : CharP K' p
    ⊢ AlgEquiv (ZMod p) K K'
  -/
  choose n a hK using FiniteField.card K p
  /-
    p✝ : Nat
    inst✝⁶ : Fact (Nat.Prime p✝)
    n✝ : Nat
    K : Type u_1
    inst✝⁵ : Field K
    inst✝⁴ : Fintype K
    K' : Type u_2
    inst✝³ : Field K'
    inst✝² : Fintype K'
    p : Nat
    h_prime : Fact (Nat.Prime p)
    inst✝¹ : Algebra (ZMod p) K
    inst✝ : Algebra (ZMod p) K'
    hKK' : Eq (Fintype.card K) (Fintype.card K')
    this✝ : CharP K p
    this : CharP K' p
    n : PNat
    a : Nat.Prime p
    hK : Eq (Fintype.card K) (HPow.hPow p ↑n)
    ⊢ AlgEquiv (ZMod p) K K'
  -/
  choose n' a' hK' using FiniteField.card K' p
  /-
    p✝ : Nat
    inst✝⁶ : Fact (Nat.Prime p✝)
    n✝ : Nat
    K : Type u_1
    inst✝⁵ : Field K
    inst✝⁴ : Fintype K
    K' : Type u_2
    inst✝³ : Field K'
    inst✝² : Fintype K'
    p : Nat
    h_prime : Fact (Nat.Prime p)
    inst✝¹ : Algebra (ZMod p) K
    inst✝ : Algebra (ZMod p) K'
    hKK' : Eq (Fintype.card K) (Fintype.card K')
    this✝ : CharP K p
    this : CharP K' p
    n : PNat
    a : Nat.Prime p
    hK : Eq (Fintype.card K) (HPow.hPow p ↑n)
    n' : PNat
    a' : Nat.Prime p
    hK' : Eq (Fintype.card K') (HPow.hPow p ↑n')
    ⊢ AlgEquiv (ZMod p) K K'
  -/
  rw [hK, hK'] at hKK'
  /-
    p✝ : Nat
    inst✝⁶ : Fact (Nat.Prime p✝)
    n✝ : Nat
    K : Type u_1
    inst✝⁵ : Field K
    inst✝⁴ : Fintype K
    K' : Type u_2
    inst✝³ : Field K'
    inst✝² : Fintype K'
    p : Nat
    h_prime : Fact (Nat.Prime p)
    inst✝¹ : Algebra (ZMod p) K
    inst✝ : Algebra (ZMod p) K'
    this✝ : CharP K p
    this : CharP K' p
    n : PNat
    a : Nat.Prime p
    hK : Eq (Fintype.card K) (HPow.hPow p ↑n)
    n' : PNat
    hKK' : Eq (HPow.hPow p ↑n) (HPow.hPow p ↑n')
    a' : Nat.Prime p
    hK' : Eq (Fintype.card K') (HPow.hPow p ↑n')
    ⊢ AlgEquiv (ZMod p) K K'
  -/
  have hGalK := GaloisField.algEquivGaloisFieldOfFintype p n hK
  /-
    p✝ : Nat
    inst✝⁶ : Fact (Nat.Prime p✝)
    n✝ : Nat
    K : Type u_1
    inst✝⁵ : Field K
    inst✝⁴ : Fintype K
    K' : Type u_2
    inst✝³ : Field K'
    inst✝² : Fintype K'
    p : Nat
    h_prime : Fact (Nat.Prime p)
    inst✝¹ : Algebra (ZMod p) K
    inst✝ : Algebra (ZMod p) K'
    this✝ : CharP K p
    this : CharP K' p
    n : PNat
    a : Nat.Prime p
    hK : Eq (Fintype.card K) (HPow.hPow p ↑n)
    n' : PNat
    hKK' : Eq (HPow.hPow p ↑n) (HPow.hPow p ↑n')
    a' : Nat.Prime p
    hK' : Eq (Fintype.card K') (HPow.hPow p ↑n')
    hGalK : AlgEquiv (ZMod p) K (GaloisField p ↑n)
    ⊢ AlgEquiv (ZMod p) K K'
  -/
  have hK'Gal := (GaloisField.algEquivGaloisFieldOfFintype p n' hK').symm
  /-
    p✝ : Nat
    inst✝⁶ : Fact (Nat.Prime p✝)
    n✝ : Nat
    K : Type u_1
    inst✝⁵ : Field K
    inst✝⁴ : Fintype K
    K' : Type u_2
    inst✝³ : Field K'
    inst✝² : Fintype K'
    p : Nat
    h_prime : Fact (Nat.Prime p)
    inst✝¹ : Algebra (ZMod p) K
    inst✝ : Algebra (ZMod p) K'
    this✝ : CharP K p
    this : CharP K' p
    n : PNat
    a : Nat.Prime p
    hK : Eq (Fintype.card K) (HPow.hPow p ↑n)
    n' : PNat
    hKK' : Eq (HPow.hPow p ↑n) (HPow.hPow p ↑n')
    a' : Nat.Prime p
    hK' : Eq (Fintype.card K') (HPow.hPow p ↑n')
    hGalK : AlgEquiv (ZMod p) K (GaloisField p ↑n)
    hK'Gal : AlgEquiv (ZMod p) (GaloisField p ↑n') K'
    ⊢ AlgEquiv (ZMod p) K K'
  -/
  rw [Nat.pow_right_injective h_prime.out.one_lt hKK'] at *
  /-
    p✝ : Nat
    inst✝⁶ : Fact (Nat.Prime p✝)
    n✝ : Nat
    K : Type u_1
    inst✝⁵ : Field K
    inst✝⁴ : Fintype K
    K' : Type u_2
    inst✝³ : Field K'
    inst✝² : Fintype K'
    p : Nat
    h_prime : Fact (Nat.Prime p)
    inst✝¹ : Algebra (ZMod p) K
    inst✝ : Algebra (ZMod p) K'
    this✝ : CharP K p
    this : CharP K' p
    n : PNat
    a : Nat.Prime p
    hK : Eq (Fintype.card K) (HPow.hPow p ↑n)
    n' : PNat
    hKK' : Eq (HPow.hPow p ↑n') (HPow.hPow p ↑n')
    a' : Nat.Prime p
    hK' : Eq (Fintype.card K') (HPow.hPow p ↑n')
    hGalK : AlgEquiv (ZMod p) K (GaloisField p ↑n')
    hK'Gal : AlgEquiv (ZMod p) (GaloisField p ↑n') K'
    ⊢ AlgEquiv (ZMod p) K K'
  -/
  exact AlgEquiv.trans hGalK hK'Gal
  /-
    🎉 no goals
  -/


/-- Uniqueness of finite fields:
  Any two finite fields of the same cardinality are (possibly non canonically) isomorphic -/
def ringEquivOfCardEq (hKK' : Fintype.card K = Fintype.card K') : K ≃+* K' := by
  /-
    p : Nat
    inst✝⁴ : Fact (Nat.Prime p)
    n : Nat
    K : Type u_1
    inst✝³ : Field K
    inst✝² : Fintype K
    K' : Type u_2
    inst✝¹ : Field K'
    inst✝ : Fintype K'
    hKK' : Eq (Fintype.card K) (Fintype.card K')
    ⊢ RingEquiv K K'
  -/
  choose p _char_p_K using CharP.exists K
  /-
    p✝ : Nat
    inst✝⁴ : Fact (Nat.Prime p✝)
    n : Nat
    K : Type u_1
    inst✝³ : Field K
    inst✝² : Fintype K
    K' : Type u_2
    inst✝¹ : Field K'
    inst✝ : Fintype K'
    hKK' : Eq (Fintype.card K) (Fintype.card K')
    p : Nat
    _char_p_K : CharP K p
    ⊢ RingEquiv K K'
  -/
  choose p' _char_p'_K' using CharP.exists K'
  /-
    p✝ : Nat
    inst✝⁴ : Fact (Nat.Prime p✝)
    n : Nat
    K : Type u_1
    inst✝³ : Field K
    inst✝² : Fintype K
    K' : Type u_2
    inst✝¹ : Field K'
    inst✝ : Fintype K'
    hKK' : Eq (Fintype.card K) (Fintype.card K')
    p : Nat
    _char_p_K : CharP K p
    p' : Nat
    _char_p'_K' : CharP K' p'
    ⊢ RingEquiv K K'
  -/
  choose n hp hK using FiniteField.card K p
  /-
    p✝ : Nat
    inst✝⁴ : Fact (Nat.Prime p✝)
    n✝ : Nat
    K : Type u_1
    inst✝³ : Field K
    inst✝² : Fintype K
    K' : Type u_2
    inst✝¹ : Field K'
    inst✝ : Fintype K'
    hKK' : Eq (Fintype.card K) (Fintype.card K')
    p : Nat
    _char_p_K : CharP K p
    p' : Nat
    _char_p'_K' : CharP K' p'
    n : PNat
    hp : Nat.Prime p
    hK : Eq (Fintype.card K) (HPow.hPow p ↑n)
    ⊢ RingEquiv K K'
  -/
  choose n' hp' hK' using FiniteField.card K' p'
  have hpp' : p = p' := by
    by_contra hne
    have h2 := Nat.coprime_pow_primes n n' hp hp' hne
    rw [(Eq.congr hK hK').mp hKK', Nat.coprime_self, pow_eq_one_iff (PNat.ne_zero n')] at h2
    exact Nat.Prime.ne_one hp' h2
  /-
    p✝ : Nat
    inst✝⁴ : Fact (Nat.Prime p✝)
    n✝ : Nat
    K : Type u_1
    inst✝³ : Field K
    inst✝² : Fintype K
    K' : Type u_2
    inst✝¹ : Field K'
    inst✝ : Fintype K'
    hKK' : Eq (Fintype.card K) (Fintype.card K')
    p : Nat
    _char_p_K : CharP K p
    p' : Nat
    _char_p'_K' : CharP K' p'
    n : PNat
    hp : Nat.Prime p
    hK : Eq (Fintype.card K) (HPow.hPow p ↑n)
    n' : PNat
    hp' : Nat.Prime p'
    hK' : Eq (Fintype.card K') (HPow.hPow p' ↑n')
    hpp' : Eq p p'
    ⊢ RingEquiv K K'
  -/
  rw [← hpp'] at _char_p'_K'
  /-
    p✝ : Nat
    inst✝⁴ : Fact (Nat.Prime p✝)
    n✝ : Nat
    K : Type u_1
    inst✝³ : Field K
    inst✝² : Fintype K
    K' : Type u_2
    inst✝¹ : Field K'
    inst✝ : Fintype K'
    hKK' : Eq (Fintype.card K) (Fintype.card K')
    p : Nat
    _char_p_K : CharP K p
    p' : Nat
    _char_p'_K' : CharP K' p
    n : PNat
    hp : Nat.Prime p
    hK : Eq (Fintype.card K) (HPow.hPow p ↑n)
    n' : PNat
    hp' : Nat.Prime p'
    hK' : Eq (Fintype.card K') (HPow.hPow p' ↑n')
    hpp' : Eq p p'
    ⊢ RingEquiv K K'
  -/
  haveI := fact_iff.2 hp
  /-
    p✝ : Nat
    inst✝⁴ : Fact (Nat.Prime p✝)
    n✝ : Nat
    K : Type u_1
    inst✝³ : Field K
    inst✝² : Fintype K
    K' : Type u_2
    inst✝¹ : Field K'
    inst✝ : Fintype K'
    hKK' : Eq (Fintype.card K) (Fintype.card K')
    p : Nat
    _char_p_K : CharP K p
    p' : Nat
    _char_p'_K' : CharP K' p
    n : PNat
    hp : Nat.Prime p
    hK : Eq (Fintype.card K) (HPow.hPow p ↑n)
    n' : PNat
    hp' : Nat.Prime p'
    hK' : Eq (Fintype.card K') (HPow.hPow p' ↑n')
    hpp' : Eq p p'
    this : Fact (Nat.Prime p)
    ⊢ RingEquiv K K'
  -/
  letI : Algebra (ZMod p) K := ZMod.algebra _ _
  /-
    p✝ : Nat
    inst✝⁴ : Fact (Nat.Prime p✝)
    n✝ : Nat
    K : Type u_1
    inst✝³ : Field K
    inst✝² : Fintype K
    K' : Type u_2
    inst✝¹ : Field K'
    inst✝ : Fintype K'
    hKK' : Eq (Fintype.card K) (Fintype.card K')
    p : Nat
    _char_p_K : CharP K p
    p' : Nat
    _char_p'_K' : CharP K' p
    n : PNat
    hp : Nat.Prime p
    hK : Eq (Fintype.card K) (HPow.hPow p ↑n)
    n' : PNat
    hp' : Nat.Prime p'
    hK' : Eq (Fintype.card K') (HPow.hPow p' ↑n')
    hpp' : Eq p p'
    this✝ : Fact (Nat.Prime p)
    this : Algebra (ZMod p) K := ZMod.algebra K p
    ⊢ RingEquiv K K'
  -/
  letI : Algebra (ZMod p) K' := ZMod.algebra _ _
  /-
    p✝ : Nat
    inst✝⁴ : Fact (Nat.Prime p✝)
    n✝ : Nat
    K : Type u_1
    inst✝³ : Field K
    inst✝² : Fintype K
    K' : Type u_2
    inst✝¹ : Field K'
    inst✝ : Fintype K'
    hKK' : Eq (Fintype.card K) (Fintype.card K')
    p : Nat
    _char_p_K : CharP K p
    p' : Nat
    _char_p'_K' : CharP K' p
    n : PNat
    hp : Nat.Prime p
    hK : Eq (Fintype.card K) (HPow.hPow p ↑n)
    n' : PNat
    hp' : Nat.Prime p'
    hK' : Eq (Fintype.card K') (HPow.hPow p' ↑n')
    hpp' : Eq p p'
    this✝¹ : Fact (Nat.Prime p)
    this✝ : Algebra (ZMod p) K := ZMod.algebra K p
    this : Algebra (ZMod p) K' := ZMod.algebra K' p
    ⊢ RingEquiv K K'
  -/
  exact ↑(algEquivOfCardEq p hKK')
  /-
    🎉 no goals
  -/


