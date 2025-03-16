instance instCharP (p : ℕ) [h : CharP R p] : CharP R[X] p :=
  let ⟨h⟩ := h
               /-
                 R : Type u
                 S : Type u_1
                 inst✝ : Semiring R
                 p : Nat
                 h✝ : CharP R p
                 h : ∀ (x : Nat), Iff (Eq (↑x) 0) (Dvd.dvd p x)
                 n : Nat
                 ⊢ Iff (Eq (↑n) 0) (Dvd.dvd p n)
               -/
  ⟨fun n => by rw [← map_natCast C, ← C_0, C_inj, h]⟩
               /-
                 🎉 no goals
               -/


instance instExpChar (p : ℕ) [h : ExpChar R p] : ExpChar R[X] p := by
  /-
    R : Type u
    S : Type u_1
    inst✝ : Semiring R
    p : Nat
    h : ExpChar R p
    ⊢ ExpChar (Polynomial R) p
  -/
  cases h; exacts [ExpChar.zero, ExpChar.prime ‹_›]
           /-
             🎉 no goals
           -/


/-- The `R`-submodule of `R[X]` consisting of polynomials of degree ≤ `n`. -/
def degreeLE (n : WithBot ℕ) : Submodule R R[X] :=
  ⨅ k : ℕ, ⨅ _ : ↑k > n, LinearMap.ker (lcoeff R k)


/-- The `R`-submodule of `R[X]` consisting of polynomials of degree < `n`. -/
def degreeLT (n : ℕ) : Submodule R R[X] :=
  ⨅ k : ℕ, ⨅ (_ : k ≥ n), LinearMap.ker (lcoeff R k)


theorem mem_degreeLE {n : WithBot ℕ} {f : R[X]} : f ∈ degreeLE R n ↔ degree f ≤ n := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : WithBot Nat
    f : Polynomial R
    ⊢ Iff (Membership.mem (Polynomial.degreeLE R n) f) (LE.le f.degree n)
  -/
  simp only [degreeLE, Submodule.mem_iInf, degree_le_iff_coeff_zero, LinearMap.mem_ker]; rfl
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


@[mono]
theorem degreeLE_mono {m n : WithBot ℕ} (H : m ≤ n) : degreeLE R m ≤ degreeLE R n := fun _ hf =>
  mem_degreeLE.2 (le_trans (mem_degreeLE.1 hf) H)


theorem degreeLE_eq_span_X_pow [DecidableEq R] {n : ℕ} :
    degreeLE R n = Submodule.span R ↑((Finset.range (n + 1)).image fun n => (X : R[X]) ^ n) := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : DecidableEq R
    n : Nat
    ⊢ Eq (Polynomial.degreeLE R ↑n) (Submodule.span R ↑(Finset.image (fun n => HPo …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : DecidableEq R
      n : Nat
      ⊢ LE.le (Polynomial.degreeLE R ↑n) (Submodule.span R ↑(Finset.image (fun n =>  …
    -/
  · intro p hp
    /-
      case a
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : DecidableEq R
      n : Nat
      p : Polynomial R
      hp : Membership.mem (Polynomial.degreeLE R ↑n) p
      ⊢ Membership.mem (Submodule.span R ↑(Finset.image (fun n => HPow.hPow Polynomi …
    -/
    replace hp := mem_degreeLE.1 hp
    /-
      case a
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : DecidableEq R
      n : Nat
      p : Polynomial R
      hp : LE.le p.degree ↑n
      ⊢ Membership.mem (Submodule.span R ↑(Finset.image (fun n => HPow.hPow Polynomi …
    -/
    rw [← Polynomial.sum_monomial_eq p, Polynomial.sum]
    /-
      case a
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : DecidableEq R
      n : Nat
      p : Polynomial R
      hp : LE.le p.degree ↑n
      ⊢ Membership.mem (Submodule.span R ↑(Finset.image (fun n => HPow.hPow Polynomi …
    -/
    refine Submodule.sum_mem _ fun k hk => ?_
    /-
      case a
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : DecidableEq R
      n : Nat
      p : Polynomial R
      hp : LE.le p.degree ↑n
      k : Nat
      hk : Membership.mem p.support k
      ⊢ Membership.mem (Submodule.span R ↑(Finset.image (fun n => HPow.hPow Polynomi …
    -/
    have := WithBot.coe_le_coe.1 (Finset.sup_le_iff.1 hp k hk)
    /-
      case a
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : DecidableEq R
      n : Nat
      p : Polynomial R
      hp : LE.le p.degree ↑n
      k : Nat
      hk : Membership.mem p.support k
      this : LE.le k ↑n
      ⊢ Membership.mem (Submodule.span R ↑(Finset.image (fun n => HPow.hPow Polynomi …
    -/
    rw [← C_mul_X_pow_eq_monomial, C_mul']
    refine
      Submodule.smul_mem _ _
        (Submodule.subset_span <|
          Finset.mem_coe.2 <|
            Finset.mem_image.2 ⟨_, Finset.mem_range.2 (Nat.lt_succ_of_le this), rfl⟩)
  /-
    case a
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : DecidableEq R
    n : Nat
    ⊢ LE.le (Submodule.span R ↑(Finset.image (fun n => HPow.hPow Polynomial.X n) ( …
  -/
  rw [Submodule.span_le, Finset.coe_image, Set.image_subset_iff]
  /-
    case a
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : DecidableEq R
    n : Nat
    ⊢ HasSubset.Subset (↑(Finset.range (HAdd.hAdd n 1))) (Set.preimage (fun n => H …
  -/
  intro k hk
  /-
    case a
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : DecidableEq R
    n k : Nat
    hk : Membership.mem (↑(Finset.range (HAdd.hAdd n 1))) k
    ⊢ Membership.mem (Set.preimage (fun n => HPow.hPow Polynomial.X n) ↑(Polynomia …
  -/
  apply mem_degreeLE.2
  exact
    (degree_X_pow_le _).trans (WithBot.coe_le_coe.2 <| Nat.le_of_lt_succ <| Finset.mem_range.1 hk)


theorem mem_degreeLT {n : ℕ} {f : R[X]} : f ∈ degreeLT R n ↔ degree f < n := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    f : Polynomial R
    ⊢ Iff (Membership.mem (Polynomial.degreeLT R n) f) (LT.lt f.degree ↑n)
  -/
  rw [degreeLT, Submodule.mem_iInf]
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    f : Polynomial R
    ⊢ Iff (∀ (i : Nat), Membership.mem (iInf fun x => LinearMap.ker (Polynomial.lc …
  -/
  conv_lhs => intro i; rw [Submodule.mem_iInf]
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    f : Polynomial R
    ⊢ Iff (∀ (i : Nat), GE.ge i n → Membership.mem (LinearMap.ker (Polynomial.lcoe …
  -/
  rw [degree, Finset.max_eq_sup_coe]
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    f : Polynomial R
    ⊢ Iff (∀ (i : Nat), GE.ge i n → Membership.mem (LinearMap.ker (Polynomial.lcoe …
  -/
  rw [Finset.sup_lt_iff ?_]
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    f : Polynomial R
    ⊢ Iff (∀ (i : Nat), GE.ge i n → Membership.mem (LinearMap.ker (Polynomial.lcoe …
  -/
  rotate_left
    /-
      R : Type u
      inst✝ : Semiring R
      n : Nat
      f : Polynomial R
      ⊢ LT.lt Bot.bot ↑n
    -/
  · apply WithBot.bot_lt_coe
    /-
      🎉 no goals
    -/
  conv_rhs =>
    simp only [mem_support_iff]
    intro b
    rw [Nat.cast_withBot, WithBot.coe_lt_coe, lt_iff_not_le, Ne, not_imp_not]
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    f : Polynomial R
    ⊢ Iff (∀ (i : Nat), GE.ge i n → Membership.mem (LinearMap.ker (Polynomial.lcoe …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[mono]
theorem degreeLT_mono {m n : ℕ} (H : m ≤ n) : degreeLT R m ≤ degreeLT R n := fun _ hf =>
  mem_degreeLT.2 (lt_of_lt_of_le (mem_degreeLT.1 hf) <| WithBot.coe_le_coe.2 H)


theorem degreeLT_eq_span_X_pow [DecidableEq R] {n : ℕ} :
    degreeLT R n = Submodule.span R ↑((Finset.range n).image fun n => X ^ n : Finset R[X]) := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : DecidableEq R
    n : Nat
    ⊢ Eq (Polynomial.degreeLT R n) (Submodule.span R ↑(Finset.image (fun n => HPow …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : DecidableEq R
      n : Nat
      ⊢ LE.le (Polynomial.degreeLT R n) (Submodule.span R ↑(Finset.image (fun n => H …
    -/
  · intro p hp
    /-
      case a
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : DecidableEq R
      n : Nat
      p : Polynomial R
      hp : Membership.mem (Polynomial.degreeLT R n) p
      ⊢ Membership.mem (Submodule.span R ↑(Finset.image (fun n => HPow.hPow Polynomi …
    -/
    replace hp := mem_degreeLT.1 hp
    /-
      case a
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : DecidableEq R
      n : Nat
      p : Polynomial R
      hp : LT.lt p.degree ↑n
      ⊢ Membership.mem (Submodule.span R ↑(Finset.image (fun n => HPow.hPow Polynomi …
    -/
    rw [← Polynomial.sum_monomial_eq p, Polynomial.sum]
    /-
      case a
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : DecidableEq R
      n : Nat
      p : Polynomial R
      hp : LT.lt p.degree ↑n
      ⊢ Membership.mem (Submodule.span R ↑(Finset.image (fun n => HPow.hPow Polynomi …
    -/
    refine Submodule.sum_mem _ fun k hk => ?_
    /-
      case a
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : DecidableEq R
      n : Nat
      p : Polynomial R
      hp : LT.lt p.degree ↑n
      k : Nat
      hk : Membership.mem p.support k
      ⊢ Membership.mem (Submodule.span R ↑(Finset.image (fun n => HPow.hPow Polynomi …
    -/
    have := WithBot.coe_lt_coe.1 ((Finset.sup_lt_iff <| WithBot.bot_lt_coe n).1 hp k hk)
    /-
      case a
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : DecidableEq R
      n : Nat
      p : Polynomial R
      hp : LT.lt p.degree ↑n
      k : Nat
      hk : Membership.mem p.support k
      this : LT.lt k n
      ⊢ Membership.mem (Submodule.span R ↑(Finset.image (fun n => HPow.hPow Polynomi …
    -/
    rw [← C_mul_X_pow_eq_monomial, C_mul']
    refine
      Submodule.smul_mem _ _
        (Submodule.subset_span <|
          Finset.mem_coe.2 <| Finset.mem_image.2 ⟨_, Finset.mem_range.2 this, rfl⟩)
  /-
    case a
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : DecidableEq R
    n : Nat
    ⊢ LE.le (Submodule.span R ↑(Finset.image (fun n => HPow.hPow Polynomial.X n) ( …
  -/
  rw [Submodule.span_le, Finset.coe_image, Set.image_subset_iff]
  /-
    case a
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : DecidableEq R
    n : Nat
    ⊢ HasSubset.Subset (↑(Finset.range n)) (Set.preimage (fun n => HPow.hPow Polyn …
  -/
  intro k hk
  /-
    case a
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : DecidableEq R
    n k : Nat
    hk : Membership.mem (↑(Finset.range n)) k
    ⊢ Membership.mem (Set.preimage (fun n => HPow.hPow Polynomial.X n) ↑(Polynomia …
  -/
  apply mem_degreeLT.2
  /-
    case a
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : DecidableEq R
    n k : Nat
    hk : Membership.mem (↑(Finset.range n)) k
    ⊢ LT.lt ((fun n => HPow.hPow Polynomial.X n) k).degree ↑n
  -/
  exact lt_of_le_of_lt (degree_X_pow_le _) (WithBot.coe_lt_coe.2 <| Finset.mem_range.1 hk)
  /-
    🎉 no goals
  -/


/-- The first `n` coefficients on `degreeLT n` form a linear equivalence with `Fin n → R`. -/
def degreeLTEquiv (R) [Semiring R] (n : ℕ) : degreeLT R n ≃ₗ[R] Fin n → R where
  toFun p n := (↑p : R[X]).coeff n
  invFun f :=
    ⟨∑ i : Fin n, monomial i (f i),
      (degreeLT R n).sum_mem fun i _ =>
        mem_degreeLT.mpr
          (lt_of_le_of_lt (degree_monomial_le i (f i)) (WithBot.coe_lt_coe.mpr i.is_lt))⟩
  map_add' p q := by
    /-
      R✝ : Type u
      S : Type u_1
      inst✝¹ : Semiring R✝
      R : Type ?u.20452
      inst✝ : Semiring R
      n : Nat
      p q : Subtype fun x => Membership.mem (Polynomial.degreeLT R n) x
      ⊢ Eq ((fun p n_1 => (↑p).coeff ↑n_1) (HAdd.hAdd p q)) (HAdd.hAdd ((fun p n_1 = …
    -/
    ext
    /-
      case h
      R✝ : Type u
      S : Type u_1
      inst✝¹ : Semiring R✝
      R : Type ?u.20452
      inst✝ : Semiring R
      n : Nat
      p q : Subtype fun x => Membership.mem (Polynomial.degreeLT R n) x
      x✝ : Fin n
      ⊢ Eq ((fun p n_1 => (↑p).coeff ↑n_1) (HAdd.hAdd p q) x✝) (HAdd.hAdd ((fun p n_ …
    -/
    dsimp
    /-
      case h
      R✝ : Type u
      S : Type u_1
      inst✝¹ : Semiring R✝
      R : Type ?u.20452
      inst✝ : Semiring R
      n : Nat
      p q : Subtype fun x => Membership.mem (Polynomial.degreeLT R n) x
      x✝ : Fin n
      ⊢ Eq ((HAdd.hAdd ↑p ↑q).coeff ↑x✝) (HAdd.hAdd ((↑p).coeff ↑x✝) ((↑q).coeff ↑x✝))
    -/
    rw [coeff_add]
    /-
      🎉 no goals
    -/
  map_smul' x p := by
    /-
      R✝ : Type u
      S : Type u_1
      inst✝¹ : Semiring R✝
      R : Type ?u.20452
      inst✝ : Semiring R
      n : Nat
      x : R
      p : Subtype fun x => Membership.mem (Polynomial.degreeLT R n) x
      ⊢ Eq ({ toFun := fun p n_1 => (↑p).coeff ↑n_1, map_add' := ⋯ }.toFun (HSMul.hS …
    -/
    ext
    /-
      case h
      R✝ : Type u
      S : Type u_1
      inst✝¹ : Semiring R✝
      R : Type ?u.20452
      inst✝ : Semiring R
      n : Nat
      x : R
      p : Subtype fun x => Membership.mem (Polynomial.degreeLT R n) x
      x✝ : Fin n
      ⊢ Eq ({ toFun := fun p n_1 => (↑p).coeff ↑n_1, map_add' := ⋯ }.toFun (HSMul.hS …
    -/
    dsimp
    /-
      case h
      R✝ : Type u
      S : Type u_1
      inst✝¹ : Semiring R✝
      R : Type ?u.20452
      inst✝ : Semiring R
      n : Nat
      x : R
      p : Subtype fun x => Membership.mem (Polynomial.degreeLT R n) x
      x✝ : Fin n
      ⊢ Eq ((HSMul.hSMul x ↑p).coeff ↑x✝) (HMul.hMul x ((↑p).coeff ↑x✝))
    -/
    rw [coeff_smul]
    /-
      case h
      R✝ : Type u
      S : Type u_1
      inst✝¹ : Semiring R✝
      R : Type ?u.20452
      inst✝ : Semiring R
      n : Nat
      x : R
      p : Subtype fun x => Membership.mem (Polynomial.degreeLT R n) x
      x✝ : Fin n
      ⊢ Eq (HSMul.hSMul x ((↑p).coeff ↑x✝)) (HMul.hMul x ((↑p).coeff ↑x✝))
    -/
    rfl
    /-
      🎉 no goals
    -/
  left_inv := by
    /-
      R✝ : Type u
      S : Type u_1
      inst✝¹ : Semiring R✝
      R : Type ?u.20452
      inst✝ : Semiring R
      n : Nat
      ⊢ Function.LeftInverse (fun f => ⟨Finset.univ.sum fun i => (Polynomial.monomia …
    -/
    rintro ⟨p, hp⟩
    /-
      case mk
      R✝ : Type u
      S : Type u_1
      inst✝¹ : Semiring R✝
      R : Type ?u.20452
      inst✝ : Semiring R
      n : Nat
      p : Polynomial R
      hp : Membership.mem (Polynomial.degreeLT R n) p
      ⊢ Eq ((fun f => ⟨Finset.univ.sum fun i => (Polynomial.monomial ↑i) (f i), ⋯⟩)  …
    -/
    ext1
    /-
      case mk.a
      R✝ : Type u
      S : Type u_1
      inst✝¹ : Semiring R✝
      R : Type ?u.20452
      inst✝ : Semiring R
      n : Nat
      p : Polynomial R
      hp : Membership.mem (Polynomial.degreeLT R n) p
      ⊢ Eq ↑((fun f => ⟨Finset.univ.sum fun i => (Polynomial.monomial ↑i) (f i), ⋯⟩) …
    -/
    simp only [Submodule.coe_mk]
    /-
      case mk.a
      R✝ : Type u
      S : Type u_1
      inst✝¹ : Semiring R✝
      R : Type ?u.20452
      inst✝ : Semiring R
      n : Nat
      p : Polynomial R
      hp : Membership.mem (Polynomial.degreeLT R n) p
      ⊢ Eq (Finset.univ.sum fun x => (Polynomial.monomial ↑x) (p.coeff ↑x)) p
    -/
    by_cases hp0 : p = 0
      /-
        case pos
        R✝ : Type u
        S : Type u_1
        inst✝¹ : Semiring R✝
        R : Type ?u.20452
        inst✝ : Semiring R
        n : Nat
        p : Polynomial R
        hp : Membership.mem (Polynomial.degreeLT R n) p
        hp0 : Eq p 0
        ⊢ Eq (Finset.univ.sum fun x => (Polynomial.monomial ↑x) (p.coeff ↑x)) p
      -/
    · subst hp0
      /-
        case pos
        R✝ : Type u
        S : Type u_1
        inst✝¹ : Semiring R✝
        R : Type ?u.20452
        inst✝ : Semiring R
        n : Nat
        hp : Membership.mem (Polynomial.degreeLT R n) 0
        ⊢ Eq (Finset.univ.sum fun x => (Polynomial.monomial ↑x) (Polynomial.coeff 0 ↑x …
      -/
      simp only [coeff_zero, LinearMap.map_zero, Finset.sum_const_zero]
      /-
        🎉 no goals
      -/
    /-
      case neg
      R✝ : Type u
      S : Type u_1
      inst✝¹ : Semiring R✝
      R : Type ?u.20452
      inst✝ : Semiring R
      n : Nat
      p : Polynomial R
      hp : Membership.mem (Polynomial.degreeLT R n) p
      hp0 : Not (Eq p 0)
      ⊢ Eq (Finset.univ.sum fun x => (Polynomial.monomial ↑x) (p.coeff ↑x)) p
    -/
    rw [mem_degreeLT, degree_eq_natDegree hp0, Nat.cast_lt] at hp
    /-
      case neg
      R✝ : Type u
      S : Type u_1
      inst✝¹ : Semiring R✝
      R : Type ?u.20452
      inst✝ : Semiring R
      n : Nat
      p : Polynomial R
      hp : LT.lt p.natDegree n
      hp0 : Not (Eq p 0)
      ⊢ Eq (Finset.univ.sum fun x => (Polynomial.monomial ↑x) (p.coeff ↑x)) p
    -/
    conv_rhs => rw [p.as_sum_range' n hp, ← Fin.sum_univ_eq_sum_range]
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      R✝ : Type u
      S : Type u_1
      inst✝¹ : Semiring R✝
      R : Type ?u.20452
      inst✝ : Semiring R
      n : Nat
      f : Fin n → R
      ⊢ Eq ({ toFun := fun p n_1 => (↑p).coeff ↑n_1, map_add' := ⋯, map_smul' := ⋯ } …
    -/
    ext i
    /-
      case h
      R✝ : Type u
      S : Type u_1
      inst✝¹ : Semiring R✝
      R : Type ?u.20452
      inst✝ : Semiring R
      n : Nat
      f : Fin n → R
      i : Fin n
      ⊢ Eq ({ toFun := fun p n_1 => (↑p).coeff ↑n_1, map_add' := ⋯, map_smul' := ⋯ } …
    -/
    simp only [finset_sum_coeff, Submodule.coe_mk]
    /-
      case h
      R✝ : Type u
      S : Type u_1
      inst✝¹ : Semiring R✝
      R : Type ?u.20452
      inst✝ : Semiring R
      n : Nat
      f : Fin n → R
      i : Fin n
      ⊢ Eq (Finset.univ.sum fun b => ((Polynomial.monomial ↑b) (f b)).coeff ↑i) (f i)
    -/
    rw [Finset.sum_eq_single i, coeff_monomial, if_pos rfl]
      /-
        case h.h₀
        R✝ : Type u
        S : Type u_1
        inst✝¹ : Semiring R✝
        R : Type ?u.20452
        inst✝ : Semiring R
        n : Nat
        f : Fin n → R
        i : Fin n
        ⊢ ∀ (b : Fin n), Membership.mem Finset.univ b → Ne b i → Eq (((Polynomial.mono …
      -/
    · rintro j - hji
      /-
        case h.h₀
        R✝ : Type u
        S : Type u_1
        inst✝¹ : Semiring R✝
        R : Type ?u.20452
        inst✝ : Semiring R
        n : Nat
        f : Fin n → R
        i j : Fin n
        hji : Ne j i
        ⊢ Eq (((Polynomial.monomial ↑j) (f j)).coeff ↑i) 0
      -/
      rw [coeff_monomial, if_neg]
      /-
        case h.h₀.hnc
        R✝ : Type u
        S : Type u_1
        inst✝¹ : Semiring R✝
        R : Type ?u.20452
        inst✝ : Semiring R
        n : Nat
        f : Fin n → R
        i j : Fin n
        hji : Ne j i
        ⊢ Not (Eq ↑j ↑i)
      -/
      rwa [← Fin.ext_iff]
      /-
        🎉 no goals
      -/
      /-
        case h.h₁
        R✝ : Type u
        S : Type u_1
        inst✝¹ : Semiring R✝
        R : Type ?u.20452
        inst✝ : Semiring R
        n : Nat
        f : Fin n → R
        i : Fin n
        ⊢ Not (Membership.mem Finset.univ i) → Eq (((Polynomial.monomial ↑i) (f i)).co …
      -/
    · intro h
      /-
        case h.h₁
        R✝ : Type u
        S : Type u_1
        inst✝¹ : Semiring R✝
        R : Type ?u.20452
        inst✝ : Semiring R
        n : Nat
        f : Fin n → R
        i : Fin n
        h : Not (Membership.mem Finset.univ i)
        ⊢ Eq (((Polynomial.monomial ↑i) (f i)).coeff ↑i) 0
      -/
      exact (h (Finset.mem_univ _)).elim
      /-
        🎉 no goals
      -/

-- Porting note: removed @[simp] as simp can prove this

theorem degreeLTEquiv_eq_zero_iff_eq_zero {n : ℕ} {p : R[X]} (hp : p ∈ degreeLT R n) :
    degreeLTEquiv _ _ ⟨p, hp⟩ = 0 ↔ p = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    p : Polynomial R
    hp : Membership.mem (Polynomial.degreeLT R n) p
    ⊢ Iff (Eq ((Polynomial.degreeLTEquiv R n) ⟨p, hp⟩) 0) (Eq p 0)
  -/
  rw [LinearEquiv.map_eq_zero_iff, Submodule.mk_eq_zero]
  /-
    🎉 no goals
  -/


theorem eval_eq_sum_degreeLTEquiv {n : ℕ} {p : R[X]} (hp : p ∈ degreeLT R n) (x : R) :
    p.eval x = ∑ i, degreeLTEquiv _ _ ⟨p, hp⟩ i * x ^ (i : ℕ) := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    p : Polynomial R
    hp : Membership.mem (Polynomial.degreeLT R n) p
    x : R
    ⊢ Eq (Polynomial.eval x p) (Finset.univ.sum fun i => HMul.hMul ((Polynomial.de …
  -/
  simp_rw [eval_eq_sum]
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    p : Polynomial R
    hp : Membership.mem (Polynomial.degreeLT R n) p
    x : R
    ⊢ Eq (p.sum fun e a => HMul.hMul a (HPow.hPow x e)) (Finset.univ.sum fun i =>  …
  -/
  exact (sum_fin _ (by simp_rw [zero_mul, forall_const]) (mem_degreeLT.mp hp)).symm
  /-
    🎉 no goals
  -/


theorem degreeLT_succ_eq_degreeLE {n : ℕ} : degreeLT R (n + 1) = degreeLE R n := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq (Polynomial.degreeLT R (HAdd.hAdd n 1)) (Polynomial.degreeLE R ↑n)
  -/
  ext x
  /-
    case h
    R : Type u
    inst✝ : Semiring R
    n : Nat
    x : Polynomial R
    ⊢ Iff (Membership.mem (Polynomial.degreeLT R (HAdd.hAdd n 1)) x) (Membership.m …
  -/
  by_cases x_zero : x = 0
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      n : Nat
      x : Polynomial R
      x_zero : Eq x 0
      ⊢ Iff (Membership.mem (Polynomial.degreeLT R (HAdd.hAdd n 1)) x) (Membership.m …
    -/
  · simp_rw [x_zero, Submodule.zero_mem]
    /-
      🎉 no goals
    -/
  · rw [mem_degreeLT, mem_degreeLE, ← natDegree_lt_iff_degree_lt (by rwa [ne_eq]),
      ← natDegree_le_iff_degree_le, Nat.lt_succ]


/-- The equivalence between monic polynomials of degree `n` and polynomials of degree less than
`n`, formed by adding a term `X ^ n`. -/
def monicEquivDegreeLT [Nontrivial R] (n : ℕ) :
    { p : R[X] // p.Monic ∧ p.natDegree = n } ≃ degreeLT R n where
  toFun p := ⟨p.1.eraseLead, by
    /-
      R : Type u
      S : Type u_1
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      n : Nat
      p : Subtype fun p => And p.Monic (Eq p.natDegree n)
      ⊢ Membership.mem (Polynomial.degreeLT R n) (↑p).eraseLead
    -/
    rcases p with ⟨p, hp, rfl⟩
    /-
      case mk.intro
      R : Type u
      S : Type u_1
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      p : Polynomial R
      hp : p.Monic
      ⊢ Membership.mem (Polynomial.degreeLT R p.natDegree) (↑⟨p, ⋯⟩).eraseLead
    -/
    simp only [mem_degreeLT]
    /-
      case mk.intro
      R : Type u
      S : Type u_1
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      p : Polynomial R
      hp : p.Monic
      ⊢ LT.lt p.eraseLead.degree ↑p.natDegree
    -/
    refine lt_of_lt_of_le ?_ degree_le_natDegree
    /-
      case mk.intro
      R : Type u
      S : Type u_1
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      p : Polynomial R
      hp : p.Monic
      ⊢ LT.lt p.eraseLead.degree p.degree
    -/
    exact degree_eraseLead_lt (ne_zero_of_ne_zero_of_monic one_ne_zero hp)⟩
    /-
      🎉 no goals
    -/
  invFun := fun p =>
    ⟨X^n + p.1, monic_X_pow_add (mem_degreeLT.1 p.2), by
        /-
          R : Type u
          S : Type u_1
          inst✝¹ : Semiring R
          inst✝ : Nontrivial R
          n : Nat
          p : Subtype fun x => Membership.mem (Polynomial.degreeLT R n) x
          ⊢ Eq (HAdd.hAdd (HPow.hPow Polynomial.X n) ↑p).natDegree n
        -/
        rw [natDegree_add_eq_left_of_degree_lt]
          /-
            R : Type u
            S : Type u_1
            inst✝¹ : Semiring R
            inst✝ : Nontrivial R
            n : Nat
            p : Subtype fun x => Membership.mem (Polynomial.degreeLT R n) x
            ⊢ Eq (HPow.hPow Polynomial.X n).natDegree n
          -/
        · simp
          /-
            🎉 no goals
          -/
          /-
            R : Type u
            S : Type u_1
            inst✝¹ : Semiring R
            inst✝ : Nontrivial R
            n : Nat
            p : Subtype fun x => Membership.mem (Polynomial.degreeLT R n) x
            ⊢ LT.lt (↑p).degree (HPow.hPow Polynomial.X n).degree
          -/
        · simp [mem_degreeLT.1 p.2]⟩
          /-
            🎉 no goals
          -/
  left_inv := by
    /-
      R : Type u
      S : Type u_1
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      n : Nat
      ⊢ Function.LeftInverse (fun p => ⟨HAdd.hAdd (HPow.hPow Polynomial.X n) ↑p, ⋯⟩) …
    -/
    rintro ⟨p, hp, rfl⟩
    /-
      case mk.intro
      R : Type u
      S : Type u_1
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      p : Polynomial R
      hp : p.Monic
      ⊢ Eq ((fun p_1 => ⟨HAdd.hAdd (HPow.hPow Polynomial.X p.natDegree) ↑p_1, ⋯⟩) (( …
    -/
    ext1
    /-
      case mk.intro.a
      R : Type u
      S : Type u_1
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      p : Polynomial R
      hp : p.Monic
      ⊢ Eq ↑((fun p_1 => ⟨HAdd.hAdd (HPow.hPow Polynomial.X p.natDegree) ↑p_1, ⋯⟩) ( …
    -/
    simp only
    /-
      case mk.intro.a
      R : Type u
      S : Type u_1
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      p : Polynomial R
      hp : p.Monic
      ⊢ Eq (HAdd.hAdd (HPow.hPow Polynomial.X p.natDegree) p.eraseLead) p
    -/
    conv_rhs => rw [← eraseLead_add_C_mul_X_pow p]
    /-
      case mk.intro.a
      R : Type u
      S : Type u_1
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      p : Polynomial R
      hp : p.Monic
      ⊢ Eq (HAdd.hAdd (HPow.hPow Polynomial.X p.natDegree) p.eraseLead) (HAdd.hAdd p …
    -/
    simp [Monic.def.1 hp, add_comm]
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      R : Type u
      S : Type u_1
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      n : Nat
      ⊢ Function.RightInverse (fun p => ⟨HAdd.hAdd (HPow.hPow Polynomial.X n) ↑p, ⋯⟩ …
    -/
    rintro ⟨p, hp⟩
    /-
      case mk
      R : Type u
      S : Type u_1
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      n : Nat
      p : Polynomial R
      hp : Membership.mem (Polynomial.degreeLT R n) p
      ⊢ Eq ((fun p => ⟨(↑p).eraseLead, ⋯⟩) ((fun p => ⟨HAdd.hAdd (HPow.hPow Polynomi …
    -/
    ext1
    /-
      case mk.a
      R : Type u
      S : Type u_1
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      n : Nat
      p : Polynomial R
      hp : Membership.mem (Polynomial.degreeLT R n) p
      ⊢ Eq ↑((fun p => ⟨(↑p).eraseLead, ⋯⟩) ((fun p => ⟨HAdd.hAdd (HPow.hPow Polynom …
    -/
    simp only
    /-
      case mk.a
      R : Type u
      S : Type u_1
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      n : Nat
      p : Polynomial R
      hp : Membership.mem (Polynomial.degreeLT R n) p
      ⊢ Eq (HAdd.hAdd (HPow.hPow Polynomial.X n) p).eraseLead p
    -/
    rw [eraseLead_add_of_degree_lt_left]
      /-
        case mk.a
        R : Type u
        S : Type u_1
        inst✝¹ : Semiring R
        inst✝ : Nontrivial R
        n : Nat
        p : Polynomial R
        hp : Membership.mem (Polynomial.degreeLT R n) p
        ⊢ Eq (HAdd.hAdd (HPow.hPow Polynomial.X n).eraseLead p) p
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case mk.a
        R : Type u
        S : Type u_1
        inst✝¹ : Semiring R
        inst✝ : Nontrivial R
        n : Nat
        p : Polynomial R
        hp : Membership.mem (Polynomial.degreeLT R n) p
        ⊢ LT.lt p.degree (HPow.hPow Polynomial.X n).degree
      -/
    · simp [mem_degreeLT.1 hp]
      /-
        🎉 no goals
      -/


/-- For every polynomial `p` in the span of a set `s : Set R[X]`, there exists a polynomial of
  `p' ∈ s` with higher degree. See also `Polynomial.exists_degree_le_of_mem_span_of_finite`. -/
theorem exists_degree_le_of_mem_span {s : Set R[X]} {p : R[X]}
    (hs : s.Nonempty) (hp : p ∈ Submodule.span R s) :
    ∃ p' ∈ s, degree p ≤ degree p' := by
  /-
    R : Type u
    inst✝ : Semiring R
    s : Set (Polynomial R)
    p : Polynomial R
    hs : s.Nonempty
    hp : Membership.mem (Submodule.span R s) p
    ⊢ Exists fun p' => And (Membership.mem s p') (LE.le p.degree p'.degree)
  -/
  by_contra! h
  /-
    R : Type u
    inst✝ : Semiring R
    s : Set (Polynomial R)
    p : Polynomial R
    hs : s.Nonempty
    hp : Membership.mem (Submodule.span R s) p
    h : ∀ (p' : Polynomial R), Membership.mem s p' → LT.lt p'.degree p.degree
    ⊢ False
  -/
  by_cases hp_zero : p = 0
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      s : Set (Polynomial R)
      p : Polynomial R
      hs : s.Nonempty
      hp : Membership.mem (Submodule.span R s) p
      h : ∀ (p' : Polynomial R), Membership.mem s p' → LT.lt p'.degree p.degree
      hp_zero : Eq p 0
      ⊢ False
    -/
  · rw [hp_zero, degree_zero] at h
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      s : Set (Polynomial R)
      p : Polynomial R
      hs : s.Nonempty
      hp : Membership.mem (Submodule.span R s) p
      h : ∀ (p' : Polynomial R), Membership.mem s p' → LT.lt p'.degree Bot.bot
      hp_zero : Eq p 0
      ⊢ False
    -/
    rcases hs with ⟨x, hx⟩
    /-
      case pos.intro
      R : Type u
      inst✝ : Semiring R
      s : Set (Polynomial R)
      p : Polynomial R
      hp : Membership.mem (Submodule.span R s) p
      h : ∀ (p' : Polynomial R), Membership.mem s p' → LT.lt p'.degree Bot.bot
      hp_zero : Eq p 0
      x : Polynomial R
      hx : Membership.mem s x
      ⊢ False
    -/
    exact not_lt_bot (h x hx)
    /-
      🎉 no goals
    -/
  · have : p ∈ degreeLT R (natDegree p) := by
      refine (Submodule.span_le.mpr fun p' p'_mem => ?_) hp
      rw [SetLike.mem_coe, mem_degreeLT, Nat.cast_withBot]
      exact lt_of_lt_of_le (h p' p'_mem) degree_le_natDegree
    rwa [mem_degreeLT, Nat.cast_withBot, degree_eq_natDegree hp_zero,
      Nat.cast_withBot, lt_self_iff_false] at this


/-- A stronger version of `Polynomial.exists_degree_le_of_mem_span` under the assumption that the
  set `s : R[X]` is finite. There exists a polynomial `p' ∈ s` whose degree dominates the degree of
  every element of `p ∈ span R s`-/
theorem exists_degree_le_of_mem_span_of_finite {s : Set R[X]} (s_fin : s.Finite) (hs : s.Nonempty) :
    ∃ p' ∈ s, ∀ (p : R[X]), p ∈ Submodule.span R s → degree p ≤ degree p' := by
  /-
    R : Type u
    inst✝ : Semiring R
    s : Set (Polynomial R)
    s_fin : s.Finite
    hs : s.Nonempty
    ⊢ Exists fun p' => And (Membership.mem s p') (∀ (p : Polynomial R), Membership …
  -/
  rcases Set.Finite.exists_maximal_wrt degree s s_fin hs with ⟨a, has, hmax⟩
  /-
    case intro.intro
    R : Type u
    inst✝ : Semiring R
    s : Set (Polynomial R)
    s_fin : s.Finite
    hs : s.Nonempty
    a : Polynomial R
    has : Membership.mem s a
    hmax : ∀ (a' : Polynomial R), Membership.mem s a' → LE.le a.degree a'.degree → …
    ⊢ Exists fun p' => And (Membership.mem s p') (∀ (p : Polynomial R), Membership …
  -/
  refine ⟨a, has, fun p hp => ?_⟩
  /-
    case intro.intro
    R : Type u
    inst✝ : Semiring R
    s : Set (Polynomial R)
    s_fin : s.Finite
    hs : s.Nonempty
    a : Polynomial R
    has : Membership.mem s a
    hmax : ∀ (a' : Polynomial R), Membership.mem s a' → LE.le a.degree a'.degree → …
    p : Polynomial R
    hp : Membership.mem (Submodule.span R s) p
    ⊢ LE.le p.degree a.degree
  -/
  rcases exists_degree_le_of_mem_span hs hp with ⟨p', hp'⟩
  /-
    case intro.intro.intro
    R : Type u
    inst✝ : Semiring R
    s : Set (Polynomial R)
    s_fin : s.Finite
    hs : s.Nonempty
    a : Polynomial R
    has : Membership.mem s a
    hmax : ∀ (a' : Polynomial R), Membership.mem s a' → LE.le a.degree a'.degree → …
    p : Polynomial R
    hp : Membership.mem (Submodule.span R s) p
    p' : Polynomial R
    hp' : And (Membership.mem s p') (LE.le p.degree p'.degree)
    ⊢ LE.le p.degree a.degree
  -/
  by_cases h : degree a ≤ degree p'
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      s : Set (Polynomial R)
      s_fin : s.Finite
      hs : s.Nonempty
      a : Polynomial R
      has : Membership.mem s a
      hmax : ∀ (a' : Polynomial R), Membership.mem s a' → LE.le a.degree a'.degree → …
      p : Polynomial R
      hp : Membership.mem (Submodule.span R s) p
      p' : Polynomial R
      hp' : And (Membership.mem s p') (LE.le p.degree p'.degree)
      h : LE.le a.degree p'.degree
      ⊢ LE.le p.degree a.degree
    -/
  · rw [← hmax p' hp'.left h] at hp'; exact hp'.right
                                      /-
                                        🎉 no goals
                                      -/
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      s : Set (Polynomial R)
      s_fin : s.Finite
      hs : s.Nonempty
      a : Polynomial R
      has : Membership.mem s a
      hmax : ∀ (a' : Polynomial R), Membership.mem s a' → LE.le a.degree a'.degree → …
      p : Polynomial R
      hp : Membership.mem (Submodule.span R s) p
      p' : Polynomial R
      hp' : And (Membership.mem s p') (LE.le p.degree p'.degree)
      h : Not (LE.le a.degree p'.degree)
      ⊢ LE.le p.degree a.degree
    -/
  · exact le_trans hp'.right (not_le.mp h).le
    /-
      🎉 no goals
    -/


/-- The span of every finite set of polynomials is contained in a `degreeLE n` for some `n`. -/
theorem span_le_degreeLE_of_finite {s : Set R[X]} (s_fin : s.Finite) :
    ∃ n : ℕ, Submodule.span R s ≤ degreeLE R n := by
  /-
    R : Type u
    inst✝ : Semiring R
    s : Set (Polynomial R)
    s_fin : s.Finite
    ⊢ Exists fun n => LE.le (Submodule.span R s) (Polynomial.degreeLE R ↑n)
  -/
  by_cases s_emp : s.Nonempty
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      s : Set (Polynomial R)
      s_fin : s.Finite
      s_emp : s.Nonempty
      ⊢ Exists fun n => LE.le (Submodule.span R s) (Polynomial.degreeLE R ↑n)
    -/
  · rcases exists_degree_le_of_mem_span_of_finite s_fin s_emp with ⟨p', _, hp'max⟩
    /-
      case pos.intro.intro
      R : Type u
      inst✝ : Semiring R
      s : Set (Polynomial R)
      s_fin : s.Finite
      s_emp : s.Nonempty
      p' : Polynomial R
      left✝ : Membership.mem s p'
      hp'max : ∀ (p : Polynomial R), Membership.mem (Submodule.span R s) p → LE.le p …
      ⊢ Exists fun n => LE.le (Submodule.span R s) (Polynomial.degreeLE R ↑n)
    -/
    exact ⟨natDegree p', fun p hp => mem_degreeLE.mpr ((hp'max _ hp).trans degree_le_natDegree)⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      s : Set (Polynomial R)
      s_fin : s.Finite
      s_emp : Not s.Nonempty
      ⊢ Exists fun n => LE.le (Submodule.span R s) (Polynomial.degreeLE R ↑n)
    -/
  · rw [Set.not_nonempty_iff_eq_empty] at s_emp
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      s : Set (Polynomial R)
      s_fin : s.Finite
      s_emp : Eq s EmptyCollection.emptyCollection
      ⊢ Exists fun n => LE.le (Submodule.span R s) (Polynomial.degreeLE R ↑n)
    -/
    rw [s_emp, Submodule.span_empty]
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      s : Set (Polynomial R)
      s_fin : s.Finite
      s_emp : Eq s EmptyCollection.emptyCollection
      ⊢ Exists fun n => LE.le Bot.bot (Polynomial.degreeLE R ↑n)
    -/
    exact ⟨0, bot_le⟩
    /-
      🎉 no goals
    -/


/-- The span of every finite set of polynomials is contained in a `degreeLT n` for some `n`. -/
theorem span_of_finite_le_degreeLT {s : Set R[X]} (s_fin : s.Finite) :
    ∃ n : ℕ, Submodule.span R s ≤ degreeLT R n := by
  /-
    R : Type u
    inst✝ : Semiring R
    s : Set (Polynomial R)
    s_fin : s.Finite
    ⊢ Exists fun n => LE.le (Submodule.span R s) (Polynomial.degreeLT R n)
  -/
  rcases span_le_degreeLE_of_finite s_fin with ⟨n, _⟩
  /-
    case intro
    R : Type u
    inst✝ : Semiring R
    s : Set (Polynomial R)
    s_fin : s.Finite
    n : Nat
    h✝ : LE.le (Submodule.span R s) (Polynomial.degreeLE R ↑n)
    ⊢ Exists fun n => LE.le (Submodule.span R s) (Polynomial.degreeLT R n)
  -/
  exact ⟨n + 1, by rwa [degreeLT_succ_eq_degreeLE]⟩
  /-
    🎉 no goals
  -/


/-- If `R` is a nontrivial ring, the polynomials `R[X]` are not finite as an `R`-module. When `R` is
a field, this is equivalent to `R[X]` being an infinite-dimensional vector space over `R`. -/
theorem not_finite [Nontrivial R] : ¬ Module.Finite R R[X] := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    ⊢ Not (Module.Finite R (Polynomial R))
  -/
  rw [Module.finite_def, Submodule.fg_def]
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    ⊢ Not (Exists fun S => And S.Finite (Eq (Submodule.span R S) Top.top))
  -/
  push_neg
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    ⊢ ∀ (S : Set (Polynomial R)), S.Finite → Ne (Submodule.span R S) Top.top
  -/
  intro s hs contra
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    s : Set (Polynomial R)
    hs : s.Finite
    contra : Eq (Submodule.span R s) Top.top
    ⊢ False
  -/
  rcases span_le_degreeLE_of_finite hs with ⟨n,hn⟩
  have : ((X : R[X]) ^ (n + 1)) ∈ Polynomial.degreeLE R ↑n := by
    rw [contra] at hn
    exact hn Submodule.mem_top
  /-
    case intro
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    s : Set (Polynomial R)
    hs : s.Finite
    contra : Eq (Submodule.span R s) Top.top
    n : Nat
    hn : LE.le (Submodule.span R s) (Polynomial.degreeLE R ↑n)
    this : Membership.mem (Polynomial.degreeLE R ↑n) (HPow.hPow Polynomial.X (HAdd …
    ⊢ False
  -/
  rw [mem_degreeLE, degree_X_pow, Nat.cast_le, add_le_iff_nonpos_right, nonpos_iff_eq_zero] at this
  /-
    case intro
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    s : Set (Polynomial R)
    hs : s.Finite
    contra : Eq (Submodule.span R s) Top.top
    n : Nat
    hn : LE.le (Submodule.span R s) (Polynomial.degreeLE R ↑n)
    this : Eq 1 0
    ⊢ False
  -/
  exact one_ne_zero this
  /-
    🎉 no goals
  -/


/-- The finset of nonzero coefficients of a polynomial. -/
def coeffs (p : R[X]) : Finset R :=
  letI := Classical.decEq R
  Finset.image (fun n => p.coeff n) p.support


@[deprecated (since := "2024-05-17")] noncomputable alias frange := coeffs


@[simp]
theorem coeffs_zero : coeffs (0 : R[X]) = ∅ :=
  rfl


@[deprecated (since := "2024-05-17")] alias frange_zero := coeffs_zero


theorem mem_coeffs_iff {p : R[X]} {c : R} : c ∈ p.coeffs ↔ ∃ n ∈ p.support, c = p.coeff n := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    c : R
    ⊢ Iff (Membership.mem p.coeffs c) (Exists fun n => And (Membership.mem p.suppo …
  -/
  simp [coeffs, eq_comm, (Finset.mem_image)]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-05-17")] alias mem_frange_iff := mem_coeffs_iff


theorem coeffs_one : coeffs (1 : R[X]) ⊆ {1} := by
  classical
    simp_rw [coeffs, Finset.image_subset_iff]
    simp_all [coeff_one]


@[deprecated (since := "2024-05-17")] alias frange_one := coeffs_one


theorem coeff_mem_coeffs (p : R[X]) (n : ℕ) (h : p.coeff n ≠ 0) : p.coeff n ∈ p.coeffs := by
  classical
  simp only [coeffs, exists_prop, mem_support_iff, Finset.mem_image, Ne]
  exact ⟨n, h, rfl⟩


@[deprecated (since := "2024-05-17")] alias coeff_mem_frange := coeff_mem_coeffs


theorem coeffs_monomial (n : ℕ) {c : R} (hc : c ≠ 0) : (monomial n c).coeffs = {c} := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    c : R
    hc : Ne c 0
    ⊢ Eq ((Polynomial.monomial n) c).coeffs (Singleton.singleton c)
  -/
  rw [coeffs, support_monomial n hc]
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    c : R
    hc : Ne c 0
    ⊢ Eq (Finset.image (fun n_1 => ((Polynomial.monomial n) c).coeff n_1) (Singlet …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem geom_sum_X_comp_X_add_one_eq_sum (n : ℕ) :
    (∑ i ∈ range n, (X : R[X]) ^ i).comp (X + 1) =
      (Finset.range n).sum fun i : ℕ => (n.choose (i + 1) : R[X]) * X ^ i := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq (((Finset.range n).sum fun i => HPow.hPow Polynomial.X i).comp (HAdd.hAdd …
  -/
  ext i
  /-
    case a
    R : Type u
    inst✝ : Semiring R
    n i : Nat
    ⊢ Eq ((((Finset.range n).sum fun i => HPow.hPow Polynomial.X i).comp (HAdd.hAd …
  -/
  trans (n.choose (i + 1) : R); swap
    /-
      R : Type u
      inst✝ : Semiring R
      n i : Nat
      ⊢ Eq (↑(n.choose (HAdd.hAdd i 1))) (((Finset.range n).sum fun i => HMul.hMul ( …
    -/
  · simp only [finset_sum_coeff, ← C_eq_natCast, coeff_C_mul_X_pow]
    /-
      R : Type u
      inst✝ : Semiring R
      n i : Nat
      ⊢ Eq (↑(n.choose (HAdd.hAdd i 1))) ((Finset.range n).sum fun x => ite (Eq i x) …
    -/
    rw [Finset.sum_eq_single i, if_pos rfl]
    · simp +contextual only [@eq_comm _ i, if_false, eq_self_iff_true,
        imp_true_iff]
    · simp +contextual only [Nat.lt_add_one_iff, Nat.choose_eq_zero_of_lt,
        Nat.cast_zero, Finset.mem_range, not_lt, eq_self_iff_true, if_true, imp_true_iff]
  /-
    R : Type u
    inst✝ : Semiring R
    n i : Nat
    ⊢ Eq ((((Finset.range n).sum fun i => HPow.hPow Polynomial.X i).comp (HAdd.hAd …
  -/
  induction' n with n ih generalizing i
    /-
      case zero
      R : Type u
      inst✝ : Semiring R
      i : Nat
      ⊢ Eq ((((Finset.range 0).sum fun i => HPow.hPow Polynomial.X i).comp (HAdd.hAd …
    -/
  · dsimp; simp only [zero_comp, coeff_zero, Nat.cast_zero]
           /-
             🎉 no goals
           -/
  · simp only [geom_sum_succ', ih, add_comp, X_pow_comp, coeff_add, Nat.choose_succ_succ,
    Nat.cast_add, coeff_X_add_one_pow]


theorem Monic.geom_sum {P : R[X]} (hP : P.Monic) (hdeg : 0 < P.natDegree) {n : ℕ} (hn : n ≠ 0) :
    (∑ i ∈ range n, P ^ i).Monic := by
  /-
    R : Type u
    inst✝ : Semiring R
    P : Polynomial R
    hP : P.Monic
    hdeg : LT.lt 0 P.natDegree
    n : Nat
    hn : Ne n 0
    ⊢ ((Finset.range n).sum fun i => HPow.hPow P i).Monic
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : Semiring R
    P : Polynomial R
    hP : P.Monic
    hdeg : LT.lt 0 P.natDegree
    n : Nat
    hn : Ne n 0
    a✝ : Nontrivial R
    ⊢ ((Finset.range n).sum fun i => HPow.hPow P i).Monic
  -/
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hn
  /-
    case intro
    R : Type u
    inst✝ : Semiring R
    P : Polynomial R
    hP : P.Monic
    hdeg : LT.lt 0 P.natDegree
    a✝ : Nontrivial R
    n : Nat
    hn : Ne n.succ 0
    ⊢ ((Finset.range n.succ).sum fun i => HPow.hPow P i).Monic
  -/
  rw [geom_sum_succ']
  /-
    case intro
    R : Type u
    inst✝ : Semiring R
    P : Polynomial R
    hP : P.Monic
    hdeg : LT.lt 0 P.natDegree
    a✝ : Nontrivial R
    n : Nat
    hn : Ne n.succ 0
    ⊢ (HAdd.hAdd (HPow.hPow P n) ((Finset.range n).sum fun i => HPow.hPow P i)).Mo …
  -/
  refine (hP.pow _).add_of_left ?_
  /-
    case intro
    R : Type u
    inst✝ : Semiring R
    P : Polynomial R
    hP : P.Monic
    hdeg : LT.lt 0 P.natDegree
    a✝ : Nontrivial R
    n : Nat
    hn : Ne n.succ 0
    ⊢ LT.lt ((Finset.range n).sum fun i => HPow.hPow P i).degree (HPow.hPow P n).d …
  -/
  refine lt_of_le_of_lt (degree_sum_le _ _) ?_
  /-
    case intro
    R : Type u
    inst✝ : Semiring R
    P : Polynomial R
    hP : P.Monic
    hdeg : LT.lt 0 P.natDegree
    a✝ : Nontrivial R
    n : Nat
    hn : Ne n.succ 0
    ⊢ LT.lt ((Finset.range n).sup fun b => (HPow.hPow P b).degree) (HPow.hPow P n) …
  -/
  rw [Finset.sup_lt_iff]
    /-
      case intro
      R : Type u
      inst✝ : Semiring R
      P : Polynomial R
      hP : P.Monic
      hdeg : LT.lt 0 P.natDegree
      a✝ : Nontrivial R
      n : Nat
      hn : Ne n.succ 0
      ⊢ ∀ (b : Nat), Membership.mem (Finset.range n) b → LT.lt (HPow.hPow P b).degre …
    -/
  · simp only [Finset.mem_range, degree_eq_natDegree (hP.pow _).ne_zero]
    /-
      case intro
      R : Type u
      inst✝ : Semiring R
      P : Polynomial R
      hP : P.Monic
      hdeg : LT.lt 0 P.natDegree
      a✝ : Nontrivial R
      n : Nat
      hn : Ne n.succ 0
      ⊢ ∀ (b : Nat), LT.lt b n → LT.lt ↑(HPow.hPow P b).natDegree ↑(HPow.hPow P n).n …
    -/
    simp only [Nat.cast_lt, hP.natDegree_pow]
    /-
      case intro
      R : Type u
      inst✝ : Semiring R
      P : Polynomial R
      hP : P.Monic
      hdeg : LT.lt 0 P.natDegree
      a✝ : Nontrivial R
      n : Nat
      hn : Ne n.succ 0
      ⊢ ∀ (b : Nat), LT.lt b n → LT.lt (HMul.hMul b P.natDegree) (HMul.hMul n P.natD …
    -/
    intro k
    /-
      case intro
      R : Type u
      inst✝ : Semiring R
      P : Polynomial R
      hP : P.Monic
      hdeg : LT.lt 0 P.natDegree
      a✝ : Nontrivial R
      n : Nat
      hn : Ne n.succ 0
      k : Nat
      ⊢ LT.lt k n → LT.lt (HMul.hMul k P.natDegree) (HMul.hMul n P.natDegree)
    -/
    exact nsmul_lt_nsmul_left hdeg
    /-
      🎉 no goals
    -/
    /-
      case intro
      R : Type u
      inst✝ : Semiring R
      P : Polynomial R
      hP : P.Monic
      hdeg : LT.lt 0 P.natDegree
      a✝ : Nontrivial R
      n : Nat
      hn : Ne n.succ 0
      ⊢ LT.lt Bot.bot (HPow.hPow P n).degree
    -/
  · rw [bot_lt_iff_ne_bot, Ne, degree_eq_bot]
    /-
      case intro
      R : Type u
      inst✝ : Semiring R
      P : Polynomial R
      hP : P.Monic
      hdeg : LT.lt 0 P.natDegree
      a✝ : Nontrivial R
      n : Nat
      hn : Ne n.succ 0
      ⊢ Not (Eq (HPow.hPow P n) 0)
    -/
    exact (hP.pow _).ne_zero
    /-
      🎉 no goals
    -/


theorem Monic.geom_sum' {P : R[X]} (hP : P.Monic) (hdeg : 0 < P.degree) {n : ℕ} (hn : n ≠ 0) :
    (∑ i ∈ range n, P ^ i).Monic :=
  hP.geom_sum (natDegree_pos_iff_degree_pos.2 hdeg) hn


theorem monic_geom_sum_X {n : ℕ} (hn : n ≠ 0) : (∑ i ∈ range n, (X : R[X]) ^ i).Monic := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    hn : Ne n 0
    ⊢ ((Finset.range n).sum fun i => HPow.hPow Polynomial.X i).Monic
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    hn : Ne n 0
    a✝ : Nontrivial R
    ⊢ ((Finset.range n).sum fun i => HPow.hPow Polynomial.X i).Monic
  -/
  apply monic_X.geom_sum _ hn
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    hn : Ne n 0
    a✝ : Nontrivial R
    ⊢ LT.lt 0 Polynomial.X.natDegree
  -/
  simp only [natDegree_X, zero_lt_one]
  /-
    🎉 no goals
  -/


/-- Given a polynomial, return the polynomial whose coefficients are in
the ring closure of the original coefficients. -/
def restriction (p : R[X]) : Polynomial (Subring.closure (↑p.coeffs : Set R)) :=
  ∑ i ∈ p.support,
    monomial i
      (⟨p.coeff i,
          letI := Classical.decEq R
          if H : p.coeff i = 0 then H.symm ▸ (Subring.closure _).zero_mem
          else Subring.subset_closure (p.coeff_mem_coeffs _ H)⟩ :
        Subring.closure (↑p.coeffs : Set R))


@[simp]
theorem coeff_restriction {p : R[X]} {n : ℕ} : ↑(coeff (restriction p) n) = coeff p n := by
  classical
  simp only [restriction, coeff_monomial, finset_sum_coeff, mem_support_iff, Finset.sum_ite_eq',
    Ne, ite_not]
  split_ifs with h
  · rw [h]
    rfl
  · rfl

-- Porting note: removed @[simp] as simp can prove this

theorem coeff_restriction' {p : R[X]} {n : ℕ} : (coeff (restriction p) n).1 = coeff p n :=
  coeff_restriction


@[simp]
theorem support_restriction (p : R[X]) : support (restriction p) = support p := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    ⊢ Eq p.restriction.support p.support
  -/
  ext i
  /-
    case h
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    i : Nat
    ⊢ Iff (Membership.mem p.restriction.support i) (Membership.mem p.support i)
  -/
  simp only [mem_support_iff, not_iff_not, Ne]
  /-
    case h
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    i : Nat
    ⊢ Iff (Eq (p.restriction.coeff i) 0) (Eq (p.coeff i) 0)
  -/
  conv_rhs => rw [← coeff_restriction]
  /-
    case h
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    i : Nat
    ⊢ Iff (Eq (p.restriction.coeff i) 0) (Eq (↑(p.restriction.coeff i)) 0)
  -/
  exact ⟨fun H => by rw [H, ZeroMemClass.coe_zero], fun H => Subtype.coe_injective H⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem map_restriction {R : Type u} [CommRing R] (p : R[X]) :
    p.restriction.map (algebraMap _ _) = p :=
                  /-
                    R : Type u
                    inst✝ : CommRing R
                    p : Polynomial R
                    n : Nat
                    ⊢ Eq ((Polynomial.map (algebraMap (Subtype fun x => Membership.mem (Subring.cl …
                  -/
  ext fun n => by rw [coeff_map, Algebra.algebraMap_ofSubring_apply, coeff_restriction]
                  /-
                    🎉 no goals
                  -/


@[simp]
                                                                                /-
                                                                                  R : Type u
                                                                                  inst✝ : Ring R
                                                                                  p : Polynomial R
                                                                                  ⊢ Eq p.restriction.degree p.degree
                                                                                -/
theorem degree_restriction {p : R[X]} : (restriction p).degree = p.degree := by simp [degree]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[simp]
theorem natDegree_restriction {p : R[X]} : (restriction p).natDegree = p.natDegree := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    ⊢ Eq p.restriction.natDegree p.natDegree
  -/
  simp [natDegree]
  /-
    🎉 no goals
  -/


@[simp]
theorem monic_restriction {p : R[X]} : Monic (restriction p) ↔ Monic p := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    ⊢ Iff p.restriction.Monic p.Monic
  -/
  simp only [Monic, leadingCoeff, natDegree_restriction]
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    ⊢ Iff (Eq (p.restriction.coeff p.natDegree) 1) (Eq (p.coeff p.natDegree) 1)
  -/
  rw [← @coeff_restriction _ _ p]
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    ⊢ Iff (Eq (p.restriction.coeff p.natDegree) 1) (Eq (↑(p.restriction.coeff p.na …
  -/
  exact ⟨fun H => by rw [H, OneMemClass.coe_one], fun H => Subtype.coe_injective H⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem restriction_zero : restriction (0 : R[X]) = 0 := by
  /-
    R : Type u
    inst✝ : Ring R
    ⊢ Eq (Polynomial.restriction 0) 0
  -/
  simp only [restriction, Finset.sum_empty, support_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem restriction_one : restriction (1 : R[X]) = 1 :=
                                /-
                                  R : Type u
                                  inst✝ : Ring R
                                  i : Nat
                                  ⊢ Eq ↑((Polynomial.restriction 1).coeff i) ↑(Polynomial.coeff 1 i)
                                -/
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/
  ext fun i => Subtype.eq <| by rw [coeff_restriction', coeff_one, coeff_one]; split_ifs <;> rfl
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


theorem eval₂_restriction {p : R[X]} :
    eval₂ f x p =
      eval₂ (f.comp (Subring.subtype (Subring.closure (p.coeffs : Set R)))) x p.restriction := by
  simp only [eval₂_eq_sum, sum, support_restriction, ← @coeff_restriction _ _ p, RingHom.comp_apply,
    Subring.coeSubtype]


/-- Given a polynomial `p` and a subring `T` that contains the coefficients of `p`,
return the corresponding polynomial whose coefficients are in `T`. -/
def toSubring (hp : (↑p.coeffs : Set R) ⊆ T) : T[X] :=
  ∑ i ∈ p.support,
    monomial i
      (⟨p.coeff i,
        letI := Classical.decEq R
        if H : p.coeff i = 0 then H.symm ▸ T.zero_mem else hp (p.coeff_mem_coeffs _ H)⟩ : T)


@[simp]
theorem coeff_toSubring {n : ℕ} : ↑(coeff (toSubring p T hp) n) = coeff p n := by
  classical
  simp only [toSubring, coeff_monomial, finset_sum_coeff, mem_support_iff, Finset.sum_ite_eq',
    Ne, ite_not]
  split_ifs with h
  · rw [h]
    rfl
  · rfl

-- Porting note: removed @[simp] as simp can prove this

theorem coeff_toSubring' {n : ℕ} : (coeff (toSubring p T hp) n).1 = coeff p n :=
  coeff_toSubring _ _ hp


@[simp]
theorem support_toSubring : support (toSubring p T hp) = support p := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    T : Subring R
    hp : HasSubset.Subset ↑p.coeffs ↑T
    ⊢ Eq (p.toSubring T hp).support p.support
  -/
  ext i
  /-
    case h
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    T : Subring R
    hp : HasSubset.Subset ↑p.coeffs ↑T
    i : Nat
    ⊢ Iff (Membership.mem (p.toSubring T hp).support i) (Membership.mem p.support i)
  -/
  simp only [mem_support_iff, not_iff_not, Ne]
  /-
    case h
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    T : Subring R
    hp : HasSubset.Subset ↑p.coeffs ↑T
    i : Nat
    ⊢ Iff (Eq ((p.toSubring T hp).coeff i) 0) (Eq (p.coeff i) 0)
  -/
  conv_rhs => rw [← coeff_toSubring p T hp]
  /-
    case h
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    T : Subring R
    hp : HasSubset.Subset ↑p.coeffs ↑T
    i : Nat
    ⊢ Iff (Eq ((p.toSubring T hp).coeff i) 0) (Eq (↑((p.toSubring T hp).coeff i)) 0)
  -/
  exact ⟨fun H => by rw [H, ZeroMemClass.coe_zero], fun H => Subtype.coe_injective H⟩
  /-
    🎉 no goals
  -/


@[simp]
                                                                      /-
                                                                        R : Type u
                                                                        inst✝ : Ring R
                                                                        p : Polynomial R
                                                                        T : Subring R
                                                                        hp : HasSubset.Subset ↑p.coeffs ↑T
                                                                        ⊢ Eq (p.toSubring T hp).degree p.degree
                                                                      -/
theorem degree_toSubring : (toSubring p T hp).degree = p.degree := by simp [degree]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
                                                                               /-
                                                                                 R : Type u
                                                                                 inst✝ : Ring R
                                                                                 p : Polynomial R
                                                                                 T : Subring R
                                                                                 hp : HasSubset.Subset ↑p.coeffs ↑T
                                                                                 ⊢ Eq (p.toSubring T hp).natDegree p.natDegree
                                                                               -/
theorem natDegree_toSubring : (toSubring p T hp).natDegree = p.natDegree := by simp [natDegree]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[simp]
theorem monic_toSubring : Monic (toSubring p T hp) ↔ Monic p := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    T : Subring R
    hp : HasSubset.Subset ↑p.coeffs ↑T
    ⊢ Iff (p.toSubring T hp).Monic p.Monic
  -/
  simp_rw [Monic, leadingCoeff, natDegree_toSubring, ← coeff_toSubring p T hp]
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    T : Subring R
    hp : HasSubset.Subset ↑p.coeffs ↑T
    ⊢ Iff (Eq ((p.toSubring T hp).coeff p.natDegree) 1) (Eq (↑((p.toSubring T hp). …
  -/
  exact ⟨fun H => by rw [H, OneMemClass.coe_one], fun H => Subtype.coe_injective H⟩
  /-
    🎉 no goals
  -/


@[simp]
                                                    /-
                                                      R : Type u
                                                      S : Type u_1
                                                      inst✝¹ : Ring R
                                                      inst✝ : Semiring S
                                                      f : RingHom R S
                                                      x : S
                                                      p : Polynomial R
                                                      T : Subring R
                                                      hp : HasSubset.Subset ↑p.coeffs ↑T
                                                      ⊢ HasSubset.Subset ↑(Polynomial.coeffs 0) ↑T
                                                    -/
theorem toSubring_zero : toSubring (0 : R[X]) T (by simp [coeffs]) = 0 := by
                                                    /-
                                                      🎉 no goals
                                                    -/
  /-
    R : Type u
    inst✝ : Ring R
    T : Subring R
    ⊢ Eq (Polynomial.toSubring 0 T ⋯) 0
  -/
  ext i
  /-
    case a.a
    R : Type u
    inst✝ : Ring R
    T : Subring R
    i : Nat
    ⊢ Eq ↑((Polynomial.toSubring 0 T ⋯).coeff i) ↑(Polynomial.coeff 0 i)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem toSubring_one :
    toSubring (1 : R[X]) T
        (Set.Subset.trans coeffs_one <| Finset.singleton_subset_set_iff.2 T.one_mem) =
      1 :=
  ext fun i => Subtype.eq <| by
    rw [coeff_toSubring', coeff_one, coeff_one, apply_ite Subtype.val, ZeroMemClass.coe_zero,
      OneMemClass.coe_one]


@[simp]
theorem map_toSubring : (p.toSubring T hp).map (Subring.subtype T) = p := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    T : Subring R
    hp : HasSubset.Subset ↑p.coeffs ↑T
    ⊢ Eq (Polynomial.map T.subtype (p.toSubring T hp)) p
  -/
  ext n
  /-
    case a
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    T : Subring R
    hp : HasSubset.Subset ↑p.coeffs ↑T
    n : Nat
    ⊢ Eq ((Polynomial.map T.subtype (p.toSubring T hp)).coeff n) (p.coeff n)
  -/
  simp [coeff_map]
  /-
    🎉 no goals
  -/


/-- Given a polynomial whose coefficients are in some subring, return
the corresponding polynomial whose coefficients are in the ambient ring. -/
def ofSubring (p : T[X]) : R[X] :=
  ∑ i ∈ p.support, monomial i (p.coeff i : R)


theorem coeff_ofSubring (p : T[X]) (n : ℕ) : coeff (ofSubring T p) n = (coeff p n : T) := by
  simp only [ofSubring, coeff_monomial, finset_sum_coeff, mem_support_iff, Finset.sum_ite_eq',
    ite_eq_right_iff, Ne, ite_not, Classical.not_not, ite_eq_left_iff]
  /-
    R : Type u
    inst✝ : Ring R
    T : Subring R
    p : Polynomial (Subtype fun x => Membership.mem T x)
    n : Nat
    ⊢ Eq (p.coeff n) 0 → Eq 0 ↑(p.coeff n)
  -/
  intro h
  /-
    R : Type u
    inst✝ : Ring R
    T : Subring R
    p : Polynomial (Subtype fun x => Membership.mem T x)
    n : Nat
    h : Eq (p.coeff n) 0
    ⊢ Eq 0 ↑(p.coeff n)
  -/
  rw [h, ZeroMemClass.coe_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeffs_ofSubring {p : T[X]} : (↑(p.ofSubring T).coeffs : Set R) ⊆ T := by
  classical
  intro i hi
  simp only [coeffs, Set.mem_image, mem_support_iff, Ne, Finset.mem_coe,
    (Finset.coe_image)] at hi
  rcases hi with ⟨n, _, h'n⟩
  rw [← h'n, coeff_ofSubring]
  exact Subtype.mem (coeff p n : T)


@[deprecated (since := "2024-05-17")] alias frange_ofSubring := coeffs_ofSubring


/-- Transport an ideal of `R[X]` to an `R`-submodule of `R[X]`. -/
def ofPolynomial (I : Ideal R[X]) : Submodule R R[X] where
  carrier := I.carrier
  zero_mem' := I.zero_mem
  add_mem' := I.add_mem
  smul_mem' c x H := by
    /-
      R : Type u
      S : Type u_1
      inst✝ : Semiring R
      I : Ideal (Polynomial R)
      c : R
      x : Polynomial R
      H : Membership.mem { carrier := I.carrier, add_mem' := ⋯, zero_mem' := ⋯ }.car …
      ⊢ Membership.mem { carrier := I.carrier, add_mem' := ⋯, zero_mem' := ⋯ }.carri …
    -/
    rw [← C_mul']
    /-
      R : Type u
      S : Type u_1
      inst✝ : Semiring R
      I : Ideal (Polynomial R)
      c : R
      x : Polynomial R
      H : Membership.mem { carrier := I.carrier, add_mem' := ⋯, zero_mem' := ⋯ }.car …
      ⊢ Membership.mem { carrier := I.carrier, add_mem' := ⋯, zero_mem' := ⋯ }.carri …
    -/
    exact I.mul_mem_left _ H
    /-
      🎉 no goals
    -/


theorem mem_ofPolynomial (x) : x ∈ I.ofPolynomial ↔ x ∈ I :=
  Iff.rfl


/-- Given an ideal `I` of `R[X]`, make the `R`-submodule of `I`
consisting of polynomials of degree ≤ `n`. -/
def degreeLE (n : WithBot ℕ) : Submodule R R[X] :=
  Polynomial.degreeLE R n ⊓ I.ofPolynomial


/-- Given an ideal `I` of `R[X]`, make the ideal in `R` of
leading coefficients of polynomials in `I` with degree ≤ `n`. -/
def leadingCoeffNth (n : ℕ) : Ideal R :=
  (I.degreeLE n).map <| lcoeff R n


/-- Given an ideal `I` in `R[X]`, make the ideal in `R` of the
leading coefficients in `I`. -/
def leadingCoeff : Ideal R :=
  ⨆ n : ℕ, I.leadingCoeffNth n


/-- If every coefficient of a polynomial is in an ideal `I`, then so is the polynomial itself -/
theorem polynomial_mem_ideal_of_coeff_mem_ideal (I : Ideal R[X]) (p : R[X])
    (hp : ∀ n : ℕ, p.coeff n ∈ I.comap (C : R →+* R[X])) : p ∈ I :=
  sum_C_mul_X_pow_eq p ▸ Submodule.sum_mem I fun n _ => I.mul_mem_right _ (hp n)


/-- The push-forward of an ideal `I` of `R` to `R[X]` via inclusion
 is exactly the set of polynomials whose coefficients are in `I` -/
theorem mem_map_C_iff {I : Ideal R} {f : R[X]} :
    f ∈ (Ideal.map (C : R →+* R[X]) I : Ideal R[X]) ↔ ∀ n : ℕ, f.coeff n ∈ I := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal R
    f : Polynomial R
    ⊢ Iff (Membership.mem (Ideal.map Polynomial.C I) f) (∀ (n : Nat), Membership.m …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal R
      f : Polynomial R
      ⊢ Membership.mem (Ideal.map Polynomial.C I) f → ∀ (n : Nat), Membership.mem I  …
    -/
  · intro hf
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal R
      f : Polynomial R
      hf : Membership.mem (Ideal.map Polynomial.C I) f
      ⊢ ∀ (n : Nat), Membership.mem I (f.coeff n)
    -/
    refine Submodule.span_induction ?_ ?_ ?_ ?_ hf
      /-
        case mp.refine_1
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal R
        f : Polynomial R
        hf : Membership.mem (Ideal.map Polynomial.C I) f
        ⊢ ∀ (x : Polynomial R), Membership.mem (Set.image ⇑Polynomial.C ↑I) x → ∀ (n : …
      -/
    · intro f hf n
      /-
        case mp.refine_1
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal R
        f✝ : Polynomial R
        hf✝ : Membership.mem (Ideal.map Polynomial.C I) f✝
        f : Polynomial R
        hf : Membership.mem (Set.image ⇑Polynomial.C ↑I) f
        n : Nat
        ⊢ Membership.mem I (f.coeff n)
      -/
      cases' (Set.mem_image _ _ _).mp hf with x hx
      /-
        case mp.refine_1.intro
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal R
        f✝ : Polynomial R
        hf✝ : Membership.mem (Ideal.map Polynomial.C I) f✝
        f : Polynomial R
        hf : Membership.mem (Set.image ⇑Polynomial.C ↑I) f
        n : Nat
        x : R
        hx : And (Membership.mem (↑I) x) (Eq (Polynomial.C x) f)
        ⊢ Membership.mem I (f.coeff n)
      -/
      rw [← hx.right, coeff_C]
      /-
        case mp.refine_1.intro
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal R
        f✝ : Polynomial R
        hf✝ : Membership.mem (Ideal.map Polynomial.C I) f✝
        f : Polynomial R
        hf : Membership.mem (Set.image ⇑Polynomial.C ↑I) f
        n : Nat
        x : R
        hx : And (Membership.mem (↑I) x) (Eq (Polynomial.C x) f)
        ⊢ Membership.mem I (ite (Eq n 0) x 0)
      -/
      by_cases h : n = 0
        /-
          case pos
          R : Type u
          inst✝ : CommSemiring R
          I : Ideal R
          f✝ : Polynomial R
          hf✝ : Membership.mem (Ideal.map Polynomial.C I) f✝
          f : Polynomial R
          hf : Membership.mem (Set.image ⇑Polynomial.C ↑I) f
          n : Nat
          x : R
          hx : And (Membership.mem (↑I) x) (Eq (Polynomial.C x) f)
          h : Eq n 0
          ⊢ Membership.mem I (ite (Eq n 0) x 0)
        -/
      · simpa [h] using hx.left
        /-
          🎉 no goals
        -/
        /-
          case neg
          R : Type u
          inst✝ : CommSemiring R
          I : Ideal R
          f✝ : Polynomial R
          hf✝ : Membership.mem (Ideal.map Polynomial.C I) f✝
          f : Polynomial R
          hf : Membership.mem (Set.image ⇑Polynomial.C ↑I) f
          n : Nat
          x : R
          hx : And (Membership.mem (↑I) x) (Eq (Polynomial.C x) f)
          h : Not (Eq n 0)
          ⊢ Membership.mem I (ite (Eq n 0) x 0)
        -/
      · simp [h]
        /-
          🎉 no goals
        -/
      /-
        case mp.refine_2
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal R
        f : Polynomial R
        hf : Membership.mem (Ideal.map Polynomial.C I) f
        ⊢ ∀ (n : Nat), Membership.mem I (Polynomial.coeff 0 n)
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case mp.refine_3
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal R
        f : Polynomial R
        hf : Membership.mem (Ideal.map Polynomial.C I) f
        ⊢ ∀ (x y : Polynomial R), Membership.mem (Submodule.span (Polynomial R) (Set.i …
      -/
    · exact fun f g _ _ hf hg n => by simp [I.add_mem (hf n) (hg n)]
      /-
        🎉 no goals
      -/
      /-
        case mp.refine_4
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal R
        f : Polynomial R
        hf : Membership.mem (Ideal.map Polynomial.C I) f
        ⊢ ∀ (a x : Polynomial R), Membership.mem (Submodule.span (Polynomial R) (Set.i …
      -/
    · refine fun f g _ hg n => ?_
      /-
        case mp.refine_4
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal R
        f✝ : Polynomial R
        hf : Membership.mem (Ideal.map Polynomial.C I) f✝
        f g : Polynomial R
        x✝ : Membership.mem (Submodule.span (Polynomial R) (Set.image ⇑Polynomial.C ↑I …
        hg : ∀ (n : Nat), Membership.mem I (g.coeff n)
        n : Nat
        ⊢ Membership.mem I ((HSMul.hSMul f g).coeff n)
      -/
      rw [smul_eq_mul, coeff_mul]
      /-
        case mp.refine_4
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal R
        f✝ : Polynomial R
        hf : Membership.mem (Ideal.map Polynomial.C I) f✝
        f g : Polynomial R
        x✝ : Membership.mem (Submodule.span (Polynomial R) (Set.image ⇑Polynomial.C ↑I …
        hg : ∀ (n : Nat), Membership.mem I (g.coeff n)
        n : Nat
        ⊢ Membership.mem I ((Finset.HasAntidiagonal.antidiagonal n).sum fun x => HMul. …
      -/
      exact I.sum_mem fun c _ => I.mul_mem_left (f.coeff c.fst) (hg c.snd)
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal R
      f : Polynomial R
      ⊢ (∀ (n : Nat), Membership.mem I (f.coeff n)) → Membership.mem (Ideal.map Poly …
    -/
  · intro hf
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal R
      f : Polynomial R
      hf : ∀ (n : Nat), Membership.mem I (f.coeff n)
      ⊢ Membership.mem (Ideal.map Polynomial.C I) f
    -/
    rw [← sum_monomial_eq f]
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal R
      f : Polynomial R
      hf : ∀ (n : Nat), Membership.mem I (f.coeff n)
      ⊢ Membership.mem (Ideal.map Polynomial.C I) (f.sum fun n a => (Polynomial.mono …
    -/
    refine (I.map C : Ideal R[X]).sum_mem fun n _ => ?_
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal R
      f : Polynomial R
      hf : ∀ (n : Nat), Membership.mem I (f.coeff n)
      n : Nat
      x✝ : Membership.mem f.support n
      ⊢ Membership.mem (Ideal.map Polynomial.C I) ((fun n a => (Polynomial.monomial  …
    -/
    simp only [← C_mul_X_pow_eq_monomial, ne_eq]
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal R
      f : Polynomial R
      hf : ∀ (n : Nat), Membership.mem I (f.coeff n)
      n : Nat
      x✝ : Membership.mem f.support n
      ⊢ Membership.mem (Ideal.map Polynomial.C I) (HMul.hMul (Polynomial.C (f.coeff  …
    -/
    rw [mul_comm]
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal R
      f : Polynomial R
      hf : ∀ (n : Nat), Membership.mem I (f.coeff n)
      n : Nat
      x✝ : Membership.mem f.support n
      ⊢ Membership.mem (Ideal.map Polynomial.C I) (HMul.hMul (HPow.hPow Polynomial.X …
    -/
    exact (I.map C : Ideal R[X]).mul_mem_left _ (mem_map_of_mem _ (hf n))
    /-
      🎉 no goals
    -/


theorem _root_.Polynomial.ker_mapRingHom (f : R →+* S) :
    RingHom.ker (Polynomial.mapRingHom f) = f.ker.map (C : R →+* R[X]) := by
  /-
    R : Type u
    S : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Semiring S
    f : RingHom R S
    ⊢ Eq (RingHom.ker (Polynomial.mapRingHom f)) (Ideal.map Polynomial.C (RingHom. …
  -/
  ext
  /-
    case h
    R : Type u
    S : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Semiring S
    f : RingHom R S
    x✝ : Polynomial R
    ⊢ Iff (Membership.mem (RingHom.ker (Polynomial.mapRingHom f)) x✝) (Membership. …
  -/
  simp only [RingHom.mem_ker, coe_mapRingHom]
  /-
    case h
    R : Type u
    S : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Semiring S
    f : RingHom R S
    x✝ : Polynomial R
    ⊢ Iff (Eq (Polynomial.map f x✝) 0) (Membership.mem (Ideal.map Polynomial.C (Ri …
  -/
  rw [mem_map_C_iff, Polynomial.ext_iff]
  /-
    case h
    R : Type u
    S : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Semiring S
    f : RingHom R S
    x✝ : Polynomial R
    ⊢ Iff (∀ (n : Nat), Eq ((Polynomial.map f x✝).coeff n) (Polynomial.coeff 0 n)) …
  -/
  simp [RingHom.mem_ker]
  /-
    🎉 no goals
  -/


theorem mem_leadingCoeffNth (n : ℕ) (x) :
    x ∈ I.leadingCoeffNth n ↔ ∃ p ∈ I, degree p ≤ n ∧ p.leadingCoeff = x := by
  simp only [leadingCoeffNth, degreeLE, Submodule.mem_map, lcoeff_apply, Submodule.mem_inf,
    mem_degreeLE]
  /-
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal (Polynomial R)
    n : Nat
    x : R
    ⊢ Iff (Exists fun y => And (And (LE.le y.degree ↑n) (Membership.mem I.ofPolyno …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal (Polynomial R)
      n : Nat
      x : R
      ⊢ (Exists fun y => And (And (LE.le y.degree ↑n) (Membership.mem I.ofPolynomial …
    -/
  · rintro ⟨p, ⟨hpdeg, hpI⟩, rfl⟩
    /-
      case mp.intro.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal (Polynomial R)
      n : Nat
      p : Polynomial R
      hpdeg : LE.le p.degree ↑n
      hpI : Membership.mem I.ofPolynomial p
      ⊢ Exists fun p_1 => And (Membership.mem I p_1) (And (LE.le p_1.degree ↑n) (Eq  …
    -/
    rcases lt_or_eq_of_le hpdeg with hpdeg | hpdeg
      /-
        case mp.intro.intro.intro.inl
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal (Polynomial R)
        n : Nat
        p : Polynomial R
        hpdeg✝ : LE.le p.degree ↑n
        hpI : Membership.mem I.ofPolynomial p
        hpdeg : LT.lt p.degree ↑n
        ⊢ Exists fun p_1 => And (Membership.mem I p_1) (And (LE.le p_1.degree ↑n) (Eq  …
      -/
    · refine ⟨0, I.zero_mem, bot_le, ?_⟩
      /-
        case mp.intro.intro.intro.inl
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal (Polynomial R)
        n : Nat
        p : Polynomial R
        hpdeg✝ : LE.le p.degree ↑n
        hpI : Membership.mem I.ofPolynomial p
        hpdeg : LT.lt p.degree ↑n
        ⊢ Eq (Polynomial.leadingCoeff 0) (p.coeff n)
      -/
      rw [leadingCoeff_zero, eq_comm]
      /-
        case mp.intro.intro.intro.inl
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal (Polynomial R)
        n : Nat
        p : Polynomial R
        hpdeg✝ : LE.le p.degree ↑n
        hpI : Membership.mem I.ofPolynomial p
        hpdeg : LT.lt p.degree ↑n
        ⊢ Eq (p.coeff n) 0
      -/
      exact coeff_eq_zero_of_degree_lt hpdeg
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.inr
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal (Polynomial R)
        n : Nat
        p : Polynomial R
        hpdeg✝ : LE.le p.degree ↑n
        hpI : Membership.mem I.ofPolynomial p
        hpdeg : Eq p.degree ↑n
        ⊢ Exists fun p_1 => And (Membership.mem I p_1) (And (LE.le p_1.degree ↑n) (Eq  …
      -/
    · refine ⟨p, hpI, le_of_eq hpdeg, ?_⟩
      /-
        case mp.intro.intro.intro.inr
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal (Polynomial R)
        n : Nat
        p : Polynomial R
        hpdeg✝ : LE.le p.degree ↑n
        hpI : Membership.mem I.ofPolynomial p
        hpdeg : Eq p.degree ↑n
        ⊢ Eq p.leadingCoeff (p.coeff n)
      -/
      rw [Polynomial.leadingCoeff, natDegree, hpdeg, Nat.cast_withBot, WithBot.unbot'_coe]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal (Polynomial R)
      n : Nat
      x : R
      ⊢ (Exists fun p => And (Membership.mem I p) (And (LE.le p.degree ↑n) (Eq p.lea …
    -/
  · rintro ⟨p, hpI, hpdeg, rfl⟩
    have : natDegree p + (n - natDegree p) = n :=
      add_tsub_cancel_of_le (natDegree_le_of_degree_le hpdeg)
    /-
      case mpr.intro.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal (Polynomial R)
      n : Nat
      p : Polynomial R
      hpI : Membership.mem I p
      hpdeg : LE.le p.degree ↑n
      this : Eq (HAdd.hAdd p.natDegree (HSub.hSub n p.natDegree)) n
      ⊢ Exists fun y => And (And (LE.le y.degree ↑n) (Membership.mem I.ofPolynomial  …
    -/
    refine ⟨p * X ^ (n - natDegree p), ⟨?_, I.mul_mem_right _ hpI⟩, ?_⟩
      /-
        case mpr.intro.intro.intro.refine_1
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal (Polynomial R)
        n : Nat
        p : Polynomial R
        hpI : Membership.mem I p
        hpdeg : LE.le p.degree ↑n
        this : Eq (HAdd.hAdd p.natDegree (HSub.hSub n p.natDegree)) n
        ⊢ LE.le (HMul.hMul p (HPow.hPow Polynomial.X (HSub.hSub n p.natDegree))).degre …
      -/
    · apply le_trans (degree_mul_le _ _) _
      /-
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal (Polynomial R)
        n : Nat
        p : Polynomial R
        hpI : Membership.mem I p
        hpdeg : LE.le p.degree ↑n
        this : Eq (HAdd.hAdd p.natDegree (HSub.hSub n p.natDegree)) n
        ⊢ LE.le (HAdd.hAdd p.degree (HPow.hPow Polynomial.X (HSub.hSub n p.natDegree)) …
      -/
      apply le_trans (add_le_add degree_le_natDegree (degree_X_pow_le _)) _
      /-
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal (Polynomial R)
        n : Nat
        p : Polynomial R
        hpI : Membership.mem I p
        hpdeg : LE.le p.degree ↑n
        this : Eq (HAdd.hAdd p.natDegree (HSub.hSub n p.natDegree)) n
        ⊢ LE.le (HAdd.hAdd ↑p.natDegree ↑(HSub.hSub n p.natDegree)) ↑n
      -/
      rw [← Nat.cast_add, this]
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.intro.intro.refine_2
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal (Polynomial R)
        n : Nat
        p : Polynomial R
        hpI : Membership.mem I p
        hpdeg : LE.le p.degree ↑n
        this : Eq (HAdd.hAdd p.natDegree (HSub.hSub n p.natDegree)) n
        ⊢ Eq ((HMul.hMul p (HPow.hPow Polynomial.X (HSub.hSub n p.natDegree))).coeff n …
      -/
    · rw [Polynomial.leadingCoeff, ← coeff_mul_X_pow p (n - natDegree p), this]
      /-
        🎉 no goals
      -/


theorem mem_leadingCoeffNth_zero (x) : x ∈ I.leadingCoeffNth 0 ↔ C x ∈ I :=
  (mem_leadingCoeffNth _ _ _).trans
    ⟨fun ⟨p, hpI, hpdeg, hpx⟩ => by
      rwa [← hpx, Polynomial.leadingCoeff,
        Nat.eq_zero_of_le_zero (natDegree_le_of_degree_le hpdeg), ← eq_C_of_degree_le_zero hpdeg],
      fun hx => ⟨C x, hx, degree_C_le, leadingCoeff_C x⟩⟩


theorem leadingCoeffNth_mono {m n : ℕ} (H : m ≤ n) : I.leadingCoeffNth m ≤ I.leadingCoeffNth n := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal (Polynomial R)
    m n : Nat
    H : LE.le m n
    ⊢ LE.le (I.leadingCoeffNth m) (I.leadingCoeffNth n)
  -/
  intro r hr
  /-
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal (Polynomial R)
    m n : Nat
    H : LE.le m n
    r : R
    hr : Membership.mem (I.leadingCoeffNth m) r
    ⊢ Membership.mem (I.leadingCoeffNth n) r
  -/
  simp only [SetLike.mem_coe, mem_leadingCoeffNth] at hr ⊢
  /-
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal (Polynomial R)
    m n : Nat
    H : LE.le m n
    r : R
    hr : Exists fun p => And (Membership.mem I p) (And (LE.le p.degree ↑m) (Eq p.l …
    ⊢ Exists fun p => And (Membership.mem I p) (And (LE.le p.degree ↑n) (Eq p.lead …
  -/
  rcases hr with ⟨p, hpI, hpdeg, rfl⟩
  /-
    case intro.intro.intro
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal (Polynomial R)
    m n : Nat
    H : LE.le m n
    p : Polynomial R
    hpI : Membership.mem I p
    hpdeg : LE.le p.degree ↑m
    ⊢ Exists fun p_1 => And (Membership.mem I p_1) (And (LE.le p_1.degree ↑n) (Eq  …
  -/
  refine ⟨p * X ^ (n - m), I.mul_mem_right _ hpI, ?_, leadingCoeff_mul_X_pow⟩
  /-
    case intro.intro.intro
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal (Polynomial R)
    m n : Nat
    H : LE.le m n
    p : Polynomial R
    hpI : Membership.mem I p
    hpdeg : LE.le p.degree ↑m
    ⊢ LE.le (HMul.hMul p (HPow.hPow Polynomial.X (HSub.hSub n m))).degree ↑n
  -/
  refine le_trans (degree_mul_le _ _) ?_
  /-
    case intro.intro.intro
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal (Polynomial R)
    m n : Nat
    H : LE.le m n
    p : Polynomial R
    hpI : Membership.mem I p
    hpdeg : LE.le p.degree ↑m
    ⊢ LE.le (HAdd.hAdd p.degree (HPow.hPow Polynomial.X (HSub.hSub n m)).degree) ↑n
  -/
  refine le_trans (add_le_add hpdeg (degree_X_pow_le _)) ?_
  /-
    case intro.intro.intro
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal (Polynomial R)
    m n : Nat
    H : LE.le m n
    p : Polynomial R
    hpI : Membership.mem I p
    hpdeg : LE.le p.degree ↑m
    ⊢ LE.le (HAdd.hAdd ↑m ↑(HSub.hSub n m)) ↑n
  -/
  rw [← Nat.cast_add, add_tsub_cancel_of_le H]
  /-
    🎉 no goals
  -/


theorem mem_leadingCoeff (x) : x ∈ I.leadingCoeff ↔ ∃ p ∈ I, Polynomial.leadingCoeff p = x := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal (Polynomial R)
    x : R
    ⊢ Iff (Membership.mem I.leadingCoeff x) (Exists fun p => And (Membership.mem I …
  -/
  rw [leadingCoeff, Submodule.mem_iSup_of_directed]
    /-
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal (Polynomial R)
      x : R
      ⊢ Iff (Exists fun i => Membership.mem (I.leadingCoeffNth i) x) (Exists fun p = …
    -/
  · simp only [mem_leadingCoeffNth]
    /-
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal (Polynomial R)
      x : R
      ⊢ Iff (Exists fun i => Exists fun p => And (Membership.mem I p) (And (LE.le p. …
    -/
    constructor
      /-
        case mp
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal (Polynomial R)
        x : R
        ⊢ (Exists fun i => Exists fun p => And (Membership.mem I p) (And (LE.le p.degr …
      -/
    · rintro ⟨i, p, hpI, _, rfl⟩
      /-
        case mp.intro.intro.intro.intro
        R : Type u
        inst✝ : CommSemiring R
        I : Ideal (Polynomial R)
        i : Nat
        p : Polynomial R
        hpI : Membership.mem I p
        left✝ : LE.le p.degree ↑i
        ⊢ Exists fun p_1 => And (Membership.mem I p_1) (Eq p_1.leadingCoeff p.leadingC …
      -/
      exact ⟨p, hpI, rfl⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal (Polynomial R)
      x : R
      ⊢ (Exists fun p => And (Membership.mem I p) (Eq p.leadingCoeff x)) → Exists fu …
    -/
    rintro ⟨p, hpI, rfl⟩
    /-
      case mpr.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      I : Ideal (Polynomial R)
      p : Polynomial R
      hpI : Membership.mem I p
      ⊢ Exists fun i => Exists fun p_1 => And (Membership.mem I p_1) (And (LE.le p_1 …
    -/
    exact ⟨natDegree p, p, hpI, degree_le_natDegree, rfl⟩
    /-
      🎉 no goals
    -/
  /-
    case H
    R : Type u
    inst✝ : CommSemiring R
    I : Ideal (Polynomial R)
    x : R
    ⊢ Directed (fun x1 x2 => LE.le x1 x2) fun n => I.leadingCoeffNth n
  -/
  intro i j
  exact
    ⟨i + j, I.leadingCoeffNth_mono (Nat.le_add_right _ _),
      I.leadingCoeffNth_mono (Nat.le_add_left _ _)⟩


/-- If `I` is an ideal, and `pᵢ` is a finite family of polynomials each satisfying
`∀ k, (pᵢ)ₖ ∈ Iⁿⁱ⁻ᵏ` for some `nᵢ`, then `p = ∏ pᵢ` also satisfies `∀ k, pₖ ∈ Iⁿ⁻ᵏ` with `n = ∑ nᵢ`.
-/
theorem _root_.Polynomial.coeff_prod_mem_ideal_pow_tsub {ι : Type*} (s : Finset ι) (f : ι → R[X])
    (I : Ideal R) (n : ι → ℕ) (h : ∀ i ∈ s, ∀ (k), (f i).coeff k ∈ I ^ (n i - k)) (k : ℕ) :
    (s.prod f).coeff k ∈ I ^ (s.sum n - k) := by
  classical
    induction' s using Finset.induction with a s ha hs generalizing k
    · rw [sum_empty, prod_empty, coeff_one, zero_tsub, pow_zero, Ideal.one_eq_top]
      exact Submodule.mem_top
    · rw [sum_insert ha, prod_insert ha, coeff_mul]
      apply sum_mem
      rintro ⟨i, j⟩ e
      obtain rfl : i + j = k := mem_antidiagonal.mp e
      apply Ideal.pow_le_pow_right add_tsub_add_le_tsub_add_tsub
      rw [pow_add]
      exact
        Ideal.mul_mem_mul (h _ (Finset.mem_insert.mpr <| Or.inl rfl) _)
          (hs (fun i hi k => h _ (Finset.mem_insert.mpr <| Or.inr hi) _) j)


/-- `R[X]` is never a field for any ring `R`. -/
theorem polynomial_not_isField : ¬IsField R[X] := by
  /-
    R : Type u
    inst✝ : Ring R
    ⊢ Not (IsField (Polynomial R))
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : Ring R
    a✝ : Nontrivial R
    ⊢ Not (IsField (Polynomial R))
  -/
  intro hR
  /-
    R : Type u
    inst✝ : Ring R
    a✝ : Nontrivial R
    hR : IsField (Polynomial R)
    ⊢ False
  -/
  obtain ⟨p, hp⟩ := hR.mul_inv_cancel X_ne_zero
  /-
    case intro
    R : Type u
    inst✝ : Ring R
    a✝ : Nontrivial R
    hR : IsField (Polynomial R)
    p : Polynomial R
    hp : Eq (HMul.hMul Polynomial.X p) 1
    ⊢ False
  -/
  have hp0 : p ≠ 0 := right_ne_zero_of_mul_eq_one hp
  /-
    case intro
    R : Type u
    inst✝ : Ring R
    a✝ : Nontrivial R
    hR : IsField (Polynomial R)
    p : Polynomial R
    hp : Eq (HMul.hMul Polynomial.X p) 1
    hp0 : Ne p 0
    ⊢ False
  -/
  have := degree_lt_degree_mul_X hp0
  /-
    case intro
    R : Type u
    inst✝ : Ring R
    a✝ : Nontrivial R
    hR : IsField (Polynomial R)
    p : Polynomial R
    hp : Eq (HMul.hMul Polynomial.X p) 1
    hp0 : Ne p 0
    this : LT.lt p.degree (HMul.hMul p Polynomial.X).degree
    ⊢ False
  -/
  rw [← X_mul, congr_arg degree hp, degree_one, Nat.WithBot.lt_zero_iff, degree_eq_bot] at this
  /-
    case intro
    R : Type u
    inst✝ : Ring R
    a✝ : Nontrivial R
    hR : IsField (Polynomial R)
    p : Polynomial R
    hp : Eq (HMul.hMul Polynomial.X p) 1
    hp0 : Ne p 0
    this : Eq p 0
    ⊢ False
  -/
  exact hp0 this
  /-
    🎉 no goals
  -/


/-- The only constant in a maximal ideal over a field is `0`. -/
theorem eq_zero_of_constant_mem_of_maximal (hR : IsField R) (I : Ideal R[X]) [hI : I.IsMaximal]
    (x : R) (hx : C x ∈ I) : x = 0 := by
  /-
    R : Type u
    inst✝ : Ring R
    hR : IsField R
    I : Ideal (Polynomial R)
    hI : I.IsMaximal
    x : R
    hx : Membership.mem I (Polynomial.C x)
    ⊢ Eq x 0
  -/
  refine Classical.by_contradiction fun hx0 => hI.ne_top ((eq_top_iff_one I).2 ?_)
  /-
    R : Type u
    inst✝ : Ring R
    hR : IsField R
    I : Ideal (Polynomial R)
    hI : I.IsMaximal
    x : R
    hx : Membership.mem I (Polynomial.C x)
    hx0 : Not (Eq x 0)
    ⊢ Membership.mem I 1
  -/
  obtain ⟨y, hy⟩ := hR.mul_inv_cancel hx0
  /-
    case intro
    R : Type u
    inst✝ : Ring R
    hR : IsField R
    I : Ideal (Polynomial R)
    hI : I.IsMaximal
    x : R
    hx : Membership.mem I (Polynomial.C x)
    hx0 : Not (Eq x 0)
    y : R
    hy : Eq (HMul.hMul x y) 1
    ⊢ Membership.mem I 1
  -/
  convert I.mul_mem_left (C y) hx
  /-
    case h.e'_5
    R : Type u
    inst✝ : Ring R
    hR : IsField R
    I : Ideal (Polynomial R)
    hI : I.IsMaximal
    x : R
    hx : Membership.mem I (Polynomial.C x)
    hx0 : Not (Eq x 0)
    y : R
    hy : Eq (HMul.hMul x y) 1
    ⊢ Eq 1 (HMul.hMul (Polynomial.C y) (Polynomial.C x))
  -/
  rw [← C.map_mul, hR.mul_comm y x, hy, RingHom.map_one]
  /-
    🎉 no goals
  -/


/-- If `P` is a prime ideal of `R`, then `P.R[x]` is a prime ideal of `R[x]`. -/
theorem isPrime_map_C_iff_isPrime (P : Ideal R) :
    IsPrime (map (C : R →+* R[X]) P : Ideal R[X]) ↔ IsPrime P := by
  -- Note: the following proof avoids quotient rings
  -- It can be golfed substantially by using something like
  -- `(Quotient.isDomain_iff_prime (map C P : Ideal R[X]))`
  /-
    R : Type u
    inst✝ : CommRing R
    P : Ideal R
    ⊢ Iff (Ideal.map Polynomial.C P).IsPrime P.IsPrime
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : CommRing R
      P : Ideal R
      ⊢ (Ideal.map Polynomial.C P).IsPrime → P.IsPrime
    -/
  · intro H
    /-
      case mp
      R : Type u
      inst✝ : CommRing R
      P : Ideal R
      H : (Ideal.map Polynomial.C P).IsPrime
      ⊢ P.IsPrime
    -/
    have := comap_isPrime C (map C P)
    /-
      case mp
      R : Type u
      inst✝ : CommRing R
      P : Ideal R
      H : (Ideal.map Polynomial.C P).IsPrime
      this : (Ideal.comap Polynomial.C (Ideal.map Polynomial.C P)).IsPrime
      ⊢ P.IsPrime
    -/
    convert this using 1
    /-
      case h.e'_3
      R : Type u
      inst✝ : CommRing R
      P : Ideal R
      H : (Ideal.map Polynomial.C P).IsPrime
      this : (Ideal.comap Polynomial.C (Ideal.map Polynomial.C P)).IsPrime
      ⊢ Eq P (Ideal.comap Polynomial.C (Ideal.map Polynomial.C P))
    -/
    ext x
    /-
      case h.e'_3.h
      R : Type u
      inst✝ : CommRing R
      P : Ideal R
      H : (Ideal.map Polynomial.C P).IsPrime
      this : (Ideal.comap Polynomial.C (Ideal.map Polynomial.C P)).IsPrime
      x : R
      ⊢ Iff (Membership.mem P x) (Membership.mem (Ideal.comap Polynomial.C (Ideal.ma …
    -/
    simp only [mem_comap, mem_map_C_iff]
    /-
      case h.e'_3.h
      R : Type u
      inst✝ : CommRing R
      P : Ideal R
      H : (Ideal.map Polynomial.C P).IsPrime
      this : (Ideal.comap Polynomial.C (Ideal.map Polynomial.C P)).IsPrime
      x : R
      ⊢ Iff (Membership.mem P x) (∀ (n : Nat), Membership.mem P ((Polynomial.C x).co …
    -/
    constructor
      /-
        case h.e'_3.h.mp
        R : Type u
        inst✝ : CommRing R
        P : Ideal R
        H : (Ideal.map Polynomial.C P).IsPrime
        this : (Ideal.comap Polynomial.C (Ideal.map Polynomial.C P)).IsPrime
        x : R
        ⊢ Membership.mem P x → ∀ (n : Nat), Membership.mem P ((Polynomial.C x).coeff n)
      -/
    · rintro h (- | n)
        /-
          case h.e'_3.h.mp.zero
          R : Type u
          inst✝ : CommRing R
          P : Ideal R
          H : (Ideal.map Polynomial.C P).IsPrime
          this : (Ideal.comap Polynomial.C (Ideal.map Polynomial.C P)).IsPrime
          x : R
          h : Membership.mem P x
          ⊢ Membership.mem P ((Polynomial.C x).coeff 0)
        -/
      · rwa [coeff_C_zero]
        /-
          🎉 no goals
        -/
        /-
          case h.e'_3.h.mp.succ
          R : Type u
          inst✝ : CommRing R
          P : Ideal R
          H : (Ideal.map Polynomial.C P).IsPrime
          this : (Ideal.comap Polynomial.C (Ideal.map Polynomial.C P)).IsPrime
          x : R
          h : Membership.mem P x
          n : Nat
          ⊢ Membership.mem P ((Polynomial.C x).coeff (HAdd.hAdd n 1))
        -/
      · simp only [coeff_C_ne_zero (Nat.succ_ne_zero _), Submodule.zero_mem]
        /-
          🎉 no goals
        -/
      /-
        case h.e'_3.h.mpr
        R : Type u
        inst✝ : CommRing R
        P : Ideal R
        H : (Ideal.map Polynomial.C P).IsPrime
        this : (Ideal.comap Polynomial.C (Ideal.map Polynomial.C P)).IsPrime
        x : R
        ⊢ (∀ (n : Nat), Membership.mem P ((Polynomial.C x).coeff n)) → Membership.mem  …
      -/
    · intro h
      /-
        case h.e'_3.h.mpr
        R : Type u
        inst✝ : CommRing R
        P : Ideal R
        H : (Ideal.map Polynomial.C P).IsPrime
        this : (Ideal.comap Polynomial.C (Ideal.map Polynomial.C P)).IsPrime
        x : R
        h : ∀ (n : Nat), Membership.mem P ((Polynomial.C x).coeff n)
        ⊢ Membership.mem P x
      -/
      simpa only [coeff_C_zero] using h 0
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u
      inst✝ : CommRing R
      P : Ideal R
      ⊢ P.IsPrime → (Ideal.map Polynomial.C P).IsPrime
    -/
  · intro h
    /-
      case mpr
      R : Type u
      inst✝ : CommRing R
      P : Ideal R
      h : P.IsPrime
      ⊢ (Ideal.map Polynomial.C P).IsPrime
    -/
    constructor
      /-
        case mpr.ne_top'
        R : Type u
        inst✝ : CommRing R
        P : Ideal R
        h : P.IsPrime
        ⊢ Ne (Ideal.map Polynomial.C P) Top.top
      -/
    · rw [Ne, eq_top_iff_one, mem_map_C_iff, not_forall]
      /-
        case mpr.ne_top'
        R : Type u
        inst✝ : CommRing R
        P : Ideal R
        h : P.IsPrime
        ⊢ Exists fun x => Not (Membership.mem P (Polynomial.coeff 1 x))
      -/
      use 0
      /-
        case h
        R : Type u
        inst✝ : CommRing R
        P : Ideal R
        h : P.IsPrime
        ⊢ Not (Membership.mem P (Polynomial.coeff 1 0))
      -/
      rw [coeff_one_zero, ← eq_top_iff_one]
      /-
        case h
        R : Type u
        inst✝ : CommRing R
        P : Ideal R
        h : P.IsPrime
        ⊢ Not (Eq P Top.top)
      -/
      exact h.1
      /-
        🎉 no goals
      -/
      /-
        case mpr.mem_or_mem'
        R : Type u
        inst✝ : CommRing R
        P : Ideal R
        h : P.IsPrime
        ⊢ ∀ {x y : Polynomial R}, Membership.mem (Ideal.map Polynomial.C P) (HMul.hMul …
      -/
    · intro f g
      /-
        case mpr.mem_or_mem'
        R : Type u
        inst✝ : CommRing R
        P : Ideal R
        h : P.IsPrime
        f g : Polynomial R
        ⊢ Membership.mem (Ideal.map Polynomial.C P) (HMul.hMul f g) → Or (Membership.m …
      -/
      simp only [mem_map_C_iff]
      /-
        case mpr.mem_or_mem'
        R : Type u
        inst✝ : CommRing R
        P : Ideal R
        h : P.IsPrime
        f g : Polynomial R
        ⊢ (∀ (n : Nat), Membership.mem P ((HMul.hMul f g).coeff n)) → Or (∀ (n : Nat), …
      -/
      contrapose!
      /-
        case mpr.mem_or_mem'
        R : Type u
        inst✝ : CommRing R
        P : Ideal R
        h : P.IsPrime
        f g : Polynomial R
        ⊢ And (Exists fun n => Not (Membership.mem P (f.coeff n))) (Exists fun n => No …
      -/
      rintro ⟨hf, hg⟩
      classical
        let m := Nat.find hf
        let n := Nat.find hg
        refine ⟨m + n, ?_⟩
        rw [coeff_mul, ← Finset.insert_erase ((Finset.mem_antidiagonal (a := (m,n))).mpr rfl),
          Finset.sum_insert (Finset.not_mem_erase _ _), (P.add_mem_iff_left _).not]
        · apply mt h.2
          rw [not_or]
          exact ⟨Nat.find_spec hf, Nat.find_spec hg⟩
        apply P.sum_mem
        rintro ⟨i, j⟩ hij
        rw [Finset.mem_erase, Finset.mem_antidiagonal] at hij
        simp only [Ne, Prod.mk.inj_iff, not_and_or] at hij
        obtain hi | hj : i < m ∨ j < n := by
          rw [or_iff_not_imp_left, not_lt, le_iff_lt_or_eq]
          rintro (hmi | rfl)
          · rw [← not_le]
            intro hnj
            exact (add_lt_add_of_lt_of_le hmi hnj).ne hij.2.symm
          · simp only [eq_self_iff_true, not_true, false_or, add_right_inj, not_and_self_iff] at hij
        · rw [mul_comm]
          apply P.mul_mem_left
          exact Classical.not_not.1 (Nat.find_min hf hi)
        · apply P.mul_mem_left
          exact Classical.not_not.1 (Nat.find_min hg hj)


/-- If `P` is a prime ideal of `R`, then `P.R[x]` is a prime ideal of `R[x]`. -/
theorem isPrime_map_C_of_isPrime {P : Ideal R} (H : IsPrime P) :
    IsPrime (map (C : R →+* R[X]) P : Ideal R[X]) :=
  (isPrime_map_C_iff_isPrime P).mpr H


theorem is_fg_degreeLE [IsNoetherianRing R] (I : Ideal R[X]) (n : ℕ) :
    Submodule.FG (I.degreeLE n) :=
  letI := Classical.decEq R
  isNoetherian_submodule_left.1
    (isNoetherian_of_fg_of_noetherian _ ⟨_, degreeLE_eq_span_X_pow.symm⟩) _


/-- If the coefficients of a polynomial belong to an ideal, then that ideal contains
the ideal spanned by the coefficients of the polynomial. -/
theorem span_le_of_C_coeff_mem (cf : ∀ i : ℕ, C (f.coeff i) ∈ I) :
    Ideal.span { g | ∃ i, g = C (f.coeff i) } ≤ I := by
  /-
    R : Type u
    inst✝ : Semiring R
    f : Polynomial R
    I : Ideal (Polynomial R)
    cf : ∀ (i : Nat), Membership.mem I (Polynomial.C (f.coeff i))
    ⊢ LE.le (Ideal.span (setOf fun g => Exists fun i => Eq g (Polynomial.C (f.coef …
  -/
  simp only [@eq_comm _ _ (C _)]
  /-
    R : Type u
    inst✝ : Semiring R
    f : Polynomial R
    I : Ideal (Polynomial R)
    cf : ∀ (i : Nat), Membership.mem I (Polynomial.C (f.coeff i))
    ⊢ LE.le (Ideal.span (setOf fun g => Exists fun i => Eq (Polynomial.C (f.coeff  …
  -/
  exact (Ideal.span_le.trans range_subset_iff).mpr cf
  /-
    🎉 no goals
  -/


theorem mem_span_C_coeff : f ∈ Ideal.span { g : R[X] | ∃ i : ℕ, g = C (coeff f i) } := by
  /-
    R : Type u
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ Membership.mem (Ideal.span (setOf fun g => Exists fun i => Eq g (Polynomial. …
  -/
  let p := Ideal.span { g : R[X] | ∃ i : ℕ, g = C (coeff f i) }
  /-
    R : Type u
    inst✝ : Semiring R
    f : Polynomial R
    p : Ideal (Polynomial R) := Ideal.span (setOf fun g => Exists fun i => Eq g (P …
    ⊢ Membership.mem (Ideal.span (setOf fun g => Exists fun i => Eq g (Polynomial. …
  -/
  nth_rw 2 [(sum_C_mul_X_pow_eq f).symm]
  /-
    R : Type u
    inst✝ : Semiring R
    f : Polynomial R
    p : Ideal (Polynomial R) := Ideal.span (setOf fun g => Exists fun i => Eq g (P …
    ⊢ Membership.mem (Ideal.span (setOf fun g => Exists fun i => Eq g (Polynomial. …
  -/
  refine Submodule.sum_mem _ fun n _hn => ?_
  /-
    R : Type u
    inst✝ : Semiring R
    f : Polynomial R
    p : Ideal (Polynomial R) := Ideal.span (setOf fun g => Exists fun i => Eq g (P …
    n : Nat
    _hn : Membership.mem f.support n
    ⊢ Membership.mem (Ideal.span (setOf fun g => Exists fun i => Eq g (Polynomial. …
  -/
  dsimp
  have : C (coeff f n) ∈ p := by
    apply subset_span
    rw [mem_setOf_eq]
    use n
  /-
    R : Type u
    inst✝ : Semiring R
    f : Polynomial R
    p : Ideal (Polynomial R) := Ideal.span (setOf fun g => Exists fun i => Eq g (P …
    n : Nat
    _hn : Membership.mem f.support n
    this : Membership.mem p (Polynomial.C (f.coeff n))
    ⊢ Membership.mem (Ideal.span (setOf fun g => Exists fun i => Eq g (Polynomial. …
  -/
  have : monomial n (1 : R) • C (coeff f n) ∈ p := p.smul_mem _ this
  /-
    R : Type u
    inst✝ : Semiring R
    f : Polynomial R
    p : Ideal (Polynomial R) := Ideal.span (setOf fun g => Exists fun i => Eq g (P …
    n : Nat
    _hn : Membership.mem f.support n
    this✝ : Membership.mem p (Polynomial.C (f.coeff n))
    this : Membership.mem p (HSMul.hSMul ((Polynomial.monomial n) 1) (Polynomial.C …
    ⊢ Membership.mem (Ideal.span (setOf fun g => Exists fun i => Eq g (Polynomial. …
  -/
  convert this using 1
  /-
    case h.e'_5
    R : Type u
    inst✝ : Semiring R
    f : Polynomial R
    p : Ideal (Polynomial R) := Ideal.span (setOf fun g => Exists fun i => Eq g (P …
    n : Nat
    _hn : Membership.mem f.support n
    this✝ : Membership.mem p (Polynomial.C (f.coeff n))
    this : Membership.mem p (HSMul.hSMul ((Polynomial.monomial n) 1) (Polynomial.C …
    ⊢ Eq (HMul.hMul (Polynomial.C (f.coeff n)) (HPow.hPow Polynomial.X n)) (HSMul. …
  -/
  simp only [monomial_mul_C, one_mul, smul_eq_mul]
  /-
    case h.e'_5
    R : Type u
    inst✝ : Semiring R
    f : Polynomial R
    p : Ideal (Polynomial R) := Ideal.span (setOf fun g => Exists fun i => Eq g (P …
    n : Nat
    _hn : Membership.mem f.support n
    this✝ : Membership.mem p (Polynomial.C (f.coeff n))
    this : Membership.mem p (HSMul.hSMul ((Polynomial.monomial n) 1) (Polynomial.C …
    ⊢ Eq (HMul.hMul (Polynomial.C (f.coeff n)) (HPow.hPow Polynomial.X n)) ((Polyn …
  -/
  rw [← C_mul_X_pow_eq_monomial]
  /-
    🎉 no goals
  -/


theorem exists_C_coeff_not_mem : f ∉ I → ∃ i : ℕ, C (coeff f i) ∉ I :=
  Not.imp_symm fun cf => span_le_of_C_coeff_mem (not_exists_not.mp cf) mem_span_C_coeff


theorem prime_C_iff : Prime (C r) ↔ Prime r :=
  ⟨comap_prime C (evalRingHom (0 : R)) fun _ => eval_C, fun hr => by
    /-
      R : Type u
      inst✝ : CommRing R
      r : R
      hr : Prime r
      ⊢ Prime (Polynomial.C r)
    -/
    have := hr.1
    /-
      R : Type u
      inst✝ : CommRing R
      r : R
      hr : Prime r
      this : Ne r 0
      ⊢ Prime (Polynomial.C r)
    -/
    rw [← Ideal.span_singleton_prime] at hr ⊢
      /-
        R : Type u
        inst✝ : CommRing R
        r : R
        hr : (Ideal.span (Singleton.singleton r)).IsPrime
        this : Ne r 0
        ⊢ (Ideal.span (Singleton.singleton (Polynomial.C r))).IsPrime
      -/
    · rw [← Set.image_singleton, ← Ideal.map_span]
      /-
        R : Type u
        inst✝ : CommRing R
        r : R
        hr : (Ideal.span (Singleton.singleton r)).IsPrime
        this : Ne r 0
        ⊢ (Ideal.map Polynomial.C (Ideal.span (Singleton.singleton r))).IsPrime
      -/
      apply Ideal.isPrime_map_C_of_isPrime hr
      /-
        🎉 no goals
      -/
      /-
        R : Type u
        inst✝ : CommRing R
        r : R
        hr : (Ideal.span (Singleton.singleton r)).IsPrime
        this : Ne r 0
        ⊢ Ne (Polynomial.C r) 0
      -/
    · intro h; apply (this (C_eq_zero.mp h))
               /-
                 🎉 no goals
               -/
      /-
        R : Type u
        inst✝ : CommRing R
        r : R
        hr : Prime r
        this : Ne r 0
        ⊢ Ne r 0
      -/
    · assumption⟩
      /-
        🎉 no goals
      -/


private theorem prime_C_iff_of_fintype {R : Type u} (σ : Type v) {r : R} [CommRing R] [Fintype σ] :
    Prime (C r : MvPolynomial σ R) ↔ Prime r := by
  /-
    R : Type u
    σ : Type v
    r : R
    inst✝¹ : CommRing R
    inst✝ : Fintype σ
    ⊢ Iff (Prime (MvPolynomial.C r)) (Prime r)
  -/
  rw [← MulEquiv.prime_iff (renameEquiv R (Fintype.equivFin σ))]
  /-
    R : Type u
    σ : Type v
    r : R
    inst✝¹ : CommRing R
    inst✝ : Fintype σ
    ⊢ Iff (Prime ((MvPolynomial.renameEquiv R (Fintype.equivFin σ)) (MvPolynomial. …
  -/
  convert_to Prime (C r) ↔ _
    /-
      case h.e'_1.a
      R : Type u
      σ : Type v
      r : R
      inst✝¹ : CommRing R
      inst✝ : Fintype σ
      ⊢ Iff (Prime ((MvPolynomial.renameEquiv R (Fintype.equivFin σ)) (MvPolynomial. …
    -/
  · congr!
    /-
      case h.e'_1.a.a.h.e'_3
      R : Type u
      σ : Type v
      r : R
      inst✝¹ : CommRing R
      inst✝ : Fintype σ
      ⊢ Eq ((MvPolynomial.renameEquiv R (Fintype.equivFin σ)) (MvPolynomial.C r)) (M …
    -/
    simp only [renameEquiv_apply, algHom_C, algebraMap_eq]
    /-
      🎉 no goals
    -/
    /-
      case convert_3
      R : Type u
      σ : Type v
      r : R
      inst✝¹ : CommRing R
      inst✝ : Fintype σ
      ⊢ Iff (Prime (MvPolynomial.C r)) (Prime r)
    -/
  · induction' Fintype.card σ with d hd
      /-
        case convert_3.zero
        R : Type u
        σ : Type v
        r : R
        inst✝¹ : CommRing R
        inst✝ : Fintype σ
        ⊢ Iff (Prime (MvPolynomial.C r)) (Prime r)
      -/
    · exact MulEquiv.prime_iff (isEmptyAlgEquiv R (Fin 0)).symm (p := r)
      /-
        🎉 no goals
      -/
      /-
        case convert_3.succ
        R : Type u
        σ : Type v
        r : R
        inst✝¹ : CommRing R
        inst✝ : Fintype σ
        d : Nat
        hd : Iff (Prime (MvPolynomial.C r)) (Prime r)
        ⊢ Iff (Prime (MvPolynomial.C r)) (Prime r)
      -/
    · convert MulEquiv.prime_iff (finSuccEquiv R d).symm (p := Polynomial.C (C r))
        /-
          case h.e'_1.h.e'_3
          R : Type u
          σ : Type v
          r : R
          inst✝¹ : CommRing R
          inst✝ : Fintype σ
          d : Nat
          hd : Iff (Prime (MvPolynomial.C r)) (Prime r)
          ⊢ Eq (MvPolynomial.C r) ((MvPolynomial.finSuccEquiv R d).symm (Polynomial.C (M …
        -/
      · simp [← finSuccEquiv_comp_C_eq_C]
        /-
          🎉 no goals
        -/
        /-
          case h.e'_2.a
          R : Type u
          σ : Type v
          r : R
          inst✝¹ : CommRing R
          inst✝ : Fintype σ
          d : Nat
          hd : Iff (Prime (MvPolynomial.C r)) (Prime r)
          ⊢ Iff (Prime r) (Prime (Polynomial.C (MvPolynomial.C r)))
        -/
      · simp [← hd, Polynomial.prime_C_iff]
        /-
          🎉 no goals
        -/


theorem prime_C_iff : Prime (C r : MvPolynomial σ R) ↔ Prime r :=
  ⟨comap_prime C constantCoeff (constantCoeff_C _), fun hr =>
    ⟨fun h => hr.1 <| by
        /-
          R : Type u
          σ : Type v
          inst✝ : CommRing R
          r : R
          hr : Prime r
          h : Eq (MvPolynomial.C r) 0
          ⊢ Eq r 0
        -/
        rw [← C_inj, h]
        /-
          R : Type u
          σ : Type v
          inst✝ : CommRing R
          r : R
          hr : Prime r
          h : Eq (MvPolynomial.C r) 0
          ⊢ Eq 0 (MvPolynomial.C 0)
        -/
        simp,
        /-
          🎉 no goals
        -/
      fun h =>
      hr.2.1 <| by
        /-
          R : Type u
          σ : Type v
          inst✝ : CommRing R
          r : R
          hr : Prime r
          h : IsUnit (MvPolynomial.C r)
          ⊢ IsUnit r
        -/
        rw [← constantCoeff_C _ r]
        /-
          R : Type u
          σ : Type v
          inst✝ : CommRing R
          r : R
          hr : Prime r
          h : IsUnit (MvPolynomial.C r)
          ⊢ IsUnit (MvPolynomial.constantCoeff (MvPolynomial.C r))
        -/
        exact h.map _,
        /-
          🎉 no goals
        -/
      fun a b hd => by
      /-
        R : Type u
        σ : Type v
        inst✝ : CommRing R
        r : R
        hr : Prime r
        a b : MvPolynomial σ R
        hd : Dvd.dvd (MvPolynomial.C r) (HMul.hMul a b)
        ⊢ Or (Dvd.dvd (MvPolynomial.C r) a) (Dvd.dvd (MvPolynomial.C r) b)
      -/
      obtain ⟨s, a', b', rfl, rfl⟩ := exists_finset_rename₂ a b
      /-
        case intro.intro.intro.intro
        R : Type u
        σ : Type v
        inst✝ : CommRing R
        r : R
        hr : Prime r
        s : Finset σ
        a' b' : MvPolynomial (Subtype fun x => Membership.mem s x) R
        hd : Dvd.dvd (MvPolynomial.C r) (HMul.hMul ((MvPolynomial.rename Subtype.val)  …
        ⊢ Or (Dvd.dvd (MvPolynomial.C r) ((MvPolynomial.rename Subtype.val) a')) (Dvd. …
      -/
      rw [← algebraMap_eq] at hd
      have : algebraMap R _ r ∣ a' * b' := by
        convert killCompl Subtype.coe_injective |>.toRingHom.map_dvd hd <;> simp
      /-
        case intro.intro.intro.intro
        R : Type u
        σ : Type v
        inst✝ : CommRing R
        r : R
        hr : Prime r
        s : Finset σ
        a' b' : MvPolynomial (Subtype fun x => Membership.mem s x) R
        hd : Dvd.dvd ((algebraMap R (MvPolynomial σ R)) r) (HMul.hMul ((MvPolynomial.r …
        this : Dvd.dvd ((algebraMap R (MvPolynomial (Subtype fun x => Membership.mem s …
        ⊢ Or (Dvd.dvd (MvPolynomial.C r) ((MvPolynomial.rename Subtype.val) a')) (Dvd. …
      -/
      rw [← rename_C ((↑) : s → σ)]
      /-
        case intro.intro.intro.intro
        R : Type u
        σ : Type v
        inst✝ : CommRing R
        r : R
        hr : Prime r
        s : Finset σ
        a' b' : MvPolynomial (Subtype fun x => Membership.mem s x) R
        hd : Dvd.dvd ((algebraMap R (MvPolynomial σ R)) r) (HMul.hMul ((MvPolynomial.r …
        this : Dvd.dvd ((algebraMap R (MvPolynomial (Subtype fun x => Membership.mem s …
        ⊢ Or (Dvd.dvd ((MvPolynomial.rename Subtype.val) (MvPolynomial.C r)) ((MvPolyn …
      -/
      let f := (rename (R := R) ((↑) : s → σ)).toRingHom
      /-
        case intro.intro.intro.intro
        R : Type u
        σ : Type v
        inst✝ : CommRing R
        r : R
        hr : Prime r
        s : Finset σ
        a' b' : MvPolynomial (Subtype fun x => Membership.mem s x) R
        hd : Dvd.dvd ((algebraMap R (MvPolynomial σ R)) r) (HMul.hMul ((MvPolynomial.r …
        this : Dvd.dvd ((algebraMap R (MvPolynomial (Subtype fun x => Membership.mem s …
        f : RingHom (MvPolynomial (Subtype fun x => Membership.mem s x) R) (MvPolynomi …
        ⊢ Or (Dvd.dvd ((MvPolynomial.rename Subtype.val) (MvPolynomial.C r)) ((MvPolyn …
      -/
      exact (((prime_C_iff_of_fintype s).2 hr).2.2 a' b' this).imp f.map_dvd f.map_dvd⟩⟩
      /-
        🎉 no goals
      -/


theorem prime_rename_iff (s : Set σ) {p : MvPolynomial s R} :
    Prime (rename ((↑) : s → σ) p) ↔ Prime (p : MvPolynomial s R) := by
  classical
    symm
    let eqv :=
      (sumAlgEquiv R (↥sᶜ) s).symm.trans
        (renameEquiv R <| (Equiv.sumComm (↥sᶜ) s).trans <| Equiv.Set.sumCompl s)
    have : (rename (↑)).toRingHom = eqv.toAlgHom.toRingHom.comp C := by
      apply ringHom_ext
      · intro
        simp only [eqv, AlgHom.toRingHom_eq_coe, RingHom.coe_coe, rename_C,
          AlgEquiv.toAlgHom_eq_coe, AlgEquiv.toAlgHom_toRingHom, RingHom.coe_comp,
          AlgEquiv.coe_trans, Function.comp_apply, MvPolynomial.sumAlgEquiv_symm_apply,
          iterToSum_C_C, renameEquiv_apply, Equiv.coe_trans, Equiv.sumComm_apply]
      · intro
        simp only [eqv, AlgHom.toRingHom_eq_coe, RingHom.coe_coe, rename_X,
          AlgEquiv.toAlgHom_eq_coe, AlgEquiv.toAlgHom_toRingHom, RingHom.coe_comp,
          AlgEquiv.coe_trans, Function.comp_apply, MvPolynomial.sumAlgEquiv_symm_apply,
          iterToSum_C_X, renameEquiv_apply, Equiv.coe_trans, Equiv.sumComm_apply, Sum.swap_inr,
          Equiv.Set.sumCompl_apply_inl]
    apply_fun (· p) at this
    simp only [AlgHom.toRingHom_eq_coe, RingHom.coe_coe, AlgEquiv.toAlgHom_eq_coe,
      AlgEquiv.toAlgHom_toRingHom, RingHom.coe_comp, Function.comp_apply] at this
    rw [this, MulEquiv.prime_iff, prime_C_iff]


/-- Hilbert basis theorem: a polynomial ring over a noetherian ring is a noetherian ring. -/
protected theorem Polynomial.isNoetherianRing [inst : IsNoetherianRing R] : IsNoetherianRing R[X] :=
  isNoetherianRing_iff.2
    ⟨fun I : Ideal R[X] =>
      let M := inst.wf.min (Set.range I.leadingCoeffNth) ⟨_, ⟨0, rfl⟩⟩
      have hm : M ∈ Set.range I.leadingCoeffNth := WellFounded.min_mem _ _ _
      let ⟨N, HN⟩ := hm
      let ⟨s, hs⟩ := I.is_fg_degreeLE N
      have hm2 : ∀ k, I.leadingCoeffNth k ≤ M := fun k =>
        Or.casesOn (le_or_lt k N) (fun h => HN ▸ I.leadingCoeffNth_mono h) fun h _ hx =>
          Classical.by_contradiction fun hxm =>
            haveI : IsNoetherian R R := inst
            have : ¬M < I.leadingCoeffNth k := by
              /-
                R : Type u
                inst✝ : CommRing R
                inst : IsNoetherianRing R
                I : Ideal (Polynomial R)
                M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
                hm : Membership.mem (Set.range I.leadingCoeffNth) M
                N : Nat
                HN : Eq (I.leadingCoeffNth N) M
                s : Finset (Polynomial R)
                hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
                k : Nat
                h : LT.lt N k
                x✝ : R
                hx : Membership.mem (I.leadingCoeffNth k) x✝
                hxm : Not (Membership.mem M x✝)
                this : IsNoetherian R R
                ⊢ Not (LT.lt M (I.leadingCoeffNth k))
              -/
              refine WellFounded.not_lt_min inst.wf _ _ ?_; exact ⟨k, rfl⟩
                                                            /-
                                                              🎉 no goals
                                                            -/
            this ⟨HN ▸ I.leadingCoeffNth_mono (le_of_lt h), fun H => hxm (H hx)⟩
      have hs2 : ∀ {x}, x ∈ I.degreeLE N → x ∈ Ideal.span (↑s : Set R[X]) :=
        hs ▸ fun hx =>
          Submodule.span_induction (hx := hx) (fun _ hx => Ideal.subset_span hx) (Ideal.zero_mem _)
            (fun _ _ _ _ => Ideal.add_mem _) fun c f _ hf => f.C_mul' c ▸ Ideal.mul_mem_left _ _ hf
      ⟨s, le_antisymm (Ideal.span_le.2 fun x hx =>
          have : x ∈ I.degreeLE N := hs ▸ Submodule.subset_span hx
          this.2) <| by
        /-
          R : Type u
          inst✝ : CommRing R
          inst : IsNoetherianRing R
          I : Ideal (Polynomial R)
          M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
          hm : Membership.mem (Set.range I.leadingCoeffNth) M
          N : Nat
          HN : Eq (I.leadingCoeffNth N) M
          s : Finset (Polynomial R)
          hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
          hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
          hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
          ⊢ LE.le I (Submodule.span (Polynomial R) ↑s)
        -/
        have : Submodule.span R[X] ↑s = Ideal.span ↑s := rfl
        /-
          R : Type u
          inst✝ : CommRing R
          inst : IsNoetherianRing R
          I : Ideal (Polynomial R)
          M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
          hm : Membership.mem (Set.range I.leadingCoeffNth) M
          N : Nat
          HN : Eq (I.leadingCoeffNth N) M
          s : Finset (Polynomial R)
          hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
          hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
          hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
          this : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
          ⊢ LE.le I (Submodule.span (Polynomial R) ↑s)
        -/
        rw [this]
        /-
          R : Type u
          inst✝ : CommRing R
          inst : IsNoetherianRing R
          I : Ideal (Polynomial R)
          M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
          hm : Membership.mem (Set.range I.leadingCoeffNth) M
          N : Nat
          HN : Eq (I.leadingCoeffNth N) M
          s : Finset (Polynomial R)
          hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
          hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
          hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
          this : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
          ⊢ LE.le I (Ideal.span ↑s)
        -/
        intro p hp
        /-
          R : Type u
          inst✝ : CommRing R
          inst : IsNoetherianRing R
          I : Ideal (Polynomial R)
          M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
          hm : Membership.mem (Set.range I.leadingCoeffNth) M
          N : Nat
          HN : Eq (I.leadingCoeffNth N) M
          s : Finset (Polynomial R)
          hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
          hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
          hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
          this : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
          p : Polynomial R
          hp : Membership.mem I p
          ⊢ Membership.mem (Ideal.span ↑s) p
        -/
        generalize hn : p.natDegree = k
        /-
          R : Type u
          inst✝ : CommRing R
          inst : IsNoetherianRing R
          I : Ideal (Polynomial R)
          M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
          hm : Membership.mem (Set.range I.leadingCoeffNth) M
          N : Nat
          HN : Eq (I.leadingCoeffNth N) M
          s : Finset (Polynomial R)
          hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
          hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
          hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
          this : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
          p : Polynomial R
          hp : Membership.mem I p
          k : Nat
          hn : Eq p.natDegree k
          ⊢ Membership.mem (Ideal.span ↑s) p
        -/
        induction' k using Nat.strong_induction_on with k ih generalizing p
        /-
          case h
          R : Type u
          inst✝ : CommRing R
          inst : IsNoetherianRing R
          I : Ideal (Polynomial R)
          M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
          hm : Membership.mem (Set.range I.leadingCoeffNth) M
          N : Nat
          HN : Eq (I.leadingCoeffNth N) M
          s : Finset (Polynomial R)
          hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
          hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
          hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
          this : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
          k : Nat
          ih : ∀ (m : Nat), LT.lt m k → ∀ ⦃p : Polynomial R⦄, Membership.mem I p → Eq p. …
          p : Polynomial R
          hp : Membership.mem I p
          hn : Eq p.natDegree k
          ⊢ Membership.mem (Ideal.span ↑s) p
        -/
        rcases le_or_lt k N with h | h
          /-
            case h.inl
            R : Type u
            inst✝ : CommRing R
            inst : IsNoetherianRing R
            I : Ideal (Polynomial R)
            M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
            hm : Membership.mem (Set.range I.leadingCoeffNth) M
            N : Nat
            HN : Eq (I.leadingCoeffNth N) M
            s : Finset (Polynomial R)
            hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
            hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
            hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
            this : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
            k : Nat
            ih : ∀ (m : Nat), LT.lt m k → ∀ ⦃p : Polynomial R⦄, Membership.mem I p → Eq p. …
            p : Polynomial R
            hp : Membership.mem I p
            hn : Eq p.natDegree k
            h : LE.le k N
            ⊢ Membership.mem (Ideal.span ↑s) p
          -/
        · subst k
          refine hs2 ⟨Polynomial.mem_degreeLE.2
            (le_trans Polynomial.degree_le_natDegree <| WithBot.coe_le_coe.2 h), hp⟩
        · have hp0 : p ≠ 0 := by
            rintro rfl
            cases hn
            exact Nat.not_lt_zero _ h
          have : (0 : R) ≠ 1 := by
            intro h
            apply hp0
            ext i
            refine (mul_one _).symm.trans ?_
            rw [← h, mul_zero]
            rfl
          /-
            case h.inr
            R : Type u
            inst✝ : CommRing R
            inst : IsNoetherianRing R
            I : Ideal (Polynomial R)
            M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
            hm : Membership.mem (Set.range I.leadingCoeffNth) M
            N : Nat
            HN : Eq (I.leadingCoeffNth N) M
            s : Finset (Polynomial R)
            hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
            hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
            hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
            this✝ : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
            k : Nat
            ih : ∀ (m : Nat), LT.lt m k → ∀ ⦃p : Polynomial R⦄, Membership.mem I p → Eq p. …
            p : Polynomial R
            hp : Membership.mem I p
            hn : Eq p.natDegree k
            h : LT.lt N k
            hp0 : Ne p 0
            this : Ne 0 1
            ⊢ Membership.mem (Ideal.span ↑s) p
          -/
          haveI : Nontrivial R := ⟨⟨0, 1, this⟩⟩
          have : p.leadingCoeff ∈ I.leadingCoeffNth N := by
            rw [HN]
            exact hm2 k ((I.mem_leadingCoeffNth _ _).2
              ⟨_, hp, hn ▸ Polynomial.degree_le_natDegree, rfl⟩)
          /-
            case h.inr
            R : Type u
            inst✝ : CommRing R
            inst : IsNoetherianRing R
            I : Ideal (Polynomial R)
            M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
            hm : Membership.mem (Set.range I.leadingCoeffNth) M
            N : Nat
            HN : Eq (I.leadingCoeffNth N) M
            s : Finset (Polynomial R)
            hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
            hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
            hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
            this✝² : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
            k : Nat
            ih : ∀ (m : Nat), LT.lt m k → ∀ ⦃p : Polynomial R⦄, Membership.mem I p → Eq p. …
            p : Polynomial R
            hp : Membership.mem I p
            hn : Eq p.natDegree k
            h : LT.lt N k
            hp0 : Ne p 0
            this✝¹ : Ne 0 1
            this✝ : Nontrivial R
            this : Membership.mem (I.leadingCoeffNth N) p.leadingCoeff
            ⊢ Membership.mem (Ideal.span ↑s) p
          -/
          rw [I.mem_leadingCoeffNth] at this
          /-
            case h.inr
            R : Type u
            inst✝ : CommRing R
            inst : IsNoetherianRing R
            I : Ideal (Polynomial R)
            M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
            hm : Membership.mem (Set.range I.leadingCoeffNth) M
            N : Nat
            HN : Eq (I.leadingCoeffNth N) M
            s : Finset (Polynomial R)
            hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
            hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
            hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
            this✝² : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
            k : Nat
            ih : ∀ (m : Nat), LT.lt m k → ∀ ⦃p : Polynomial R⦄, Membership.mem I p → Eq p. …
            p : Polynomial R
            hp : Membership.mem I p
            hn : Eq p.natDegree k
            h : LT.lt N k
            hp0 : Ne p 0
            this✝¹ : Ne 0 1
            this✝ : Nontrivial R
            this : Exists fun p_1 => And (Membership.mem I p_1) (And (LE.le p_1.degree ↑N) …
            ⊢ Membership.mem (Ideal.span ↑s) p
          -/
          rcases this with ⟨q, hq, hdq, hlqp⟩
          have hq0 : q ≠ 0 := by
            intro H
            rw [← Polynomial.leadingCoeff_eq_zero] at H
            rw [hlqp, Polynomial.leadingCoeff_eq_zero] at H
            exact hp0 H
          have h1 : p.degree = (q * Polynomial.X ^ (k - q.natDegree)).degree := by
            rw [Polynomial.degree_mul', Polynomial.degree_X_pow]
            · rw [Polynomial.degree_eq_natDegree hp0, Polynomial.degree_eq_natDegree hq0]
              rw [← Nat.cast_add, add_tsub_cancel_of_le, hn]
              · refine le_trans (Polynomial.natDegree_le_of_degree_le hdq) (le_of_lt h)
            rw [Polynomial.leadingCoeff_X_pow, mul_one]
            exact mt Polynomial.leadingCoeff_eq_zero.1 hq0
          have h2 : p.leadingCoeff = (q * Polynomial.X ^ (k - q.natDegree)).leadingCoeff := by
            rw [← hlqp, Polynomial.leadingCoeff_mul_X_pow]
          /-
            case h.inr.intro.intro.intro
            R : Type u
            inst✝ : CommRing R
            inst : IsNoetherianRing R
            I : Ideal (Polynomial R)
            M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
            hm : Membership.mem (Set.range I.leadingCoeffNth) M
            N : Nat
            HN : Eq (I.leadingCoeffNth N) M
            s : Finset (Polynomial R)
            hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
            hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
            hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
            this✝¹ : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
            k : Nat
            ih : ∀ (m : Nat), LT.lt m k → ∀ ⦃p : Polynomial R⦄, Membership.mem I p → Eq p. …
            p : Polynomial R
            hp : Membership.mem I p
            hn : Eq p.natDegree k
            h : LT.lt N k
            hp0 : Ne p 0
            this✝ : Ne 0 1
            this : Nontrivial R
            q : Polynomial R
            hq : Membership.mem I q
            hdq : LE.le q.degree ↑N
            hlqp : Eq q.leadingCoeff p.leadingCoeff
            hq0 : Ne q 0
            h1 : Eq p.degree (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.natDegree …
            h2 : Eq p.leadingCoeff (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.nat …
            ⊢ Membership.mem (Ideal.span ↑s) p
          -/
          have := Polynomial.degree_sub_lt h1 hp0 h2
          /-
            case h.inr.intro.intro.intro
            R : Type u
            inst✝ : CommRing R
            inst : IsNoetherianRing R
            I : Ideal (Polynomial R)
            M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
            hm : Membership.mem (Set.range I.leadingCoeffNth) M
            N : Nat
            HN : Eq (I.leadingCoeffNth N) M
            s : Finset (Polynomial R)
            hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
            hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
            hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
            this✝² : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
            k : Nat
            ih : ∀ (m : Nat), LT.lt m k → ∀ ⦃p : Polynomial R⦄, Membership.mem I p → Eq p. …
            p : Polynomial R
            hp : Membership.mem I p
            hn : Eq p.natDegree k
            h : LT.lt N k
            hp0 : Ne p 0
            this✝¹ : Ne 0 1
            this✝ : Nontrivial R
            q : Polynomial R
            hq : Membership.mem I q
            hdq : LE.le q.degree ↑N
            hlqp : Eq q.leadingCoeff p.leadingCoeff
            hq0 : Ne q 0
            h1 : Eq p.degree (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.natDegree …
            h2 : Eq p.leadingCoeff (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.nat …
            this : LT.lt (HSub.hSub p (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q. …
            ⊢ Membership.mem (Ideal.span ↑s) p
          -/
          rw [Polynomial.degree_eq_natDegree hp0] at this
          /-
            case h.inr.intro.intro.intro
            R : Type u
            inst✝ : CommRing R
            inst : IsNoetherianRing R
            I : Ideal (Polynomial R)
            M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
            hm : Membership.mem (Set.range I.leadingCoeffNth) M
            N : Nat
            HN : Eq (I.leadingCoeffNth N) M
            s : Finset (Polynomial R)
            hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
            hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
            hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
            this✝² : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
            k : Nat
            ih : ∀ (m : Nat), LT.lt m k → ∀ ⦃p : Polynomial R⦄, Membership.mem I p → Eq p. …
            p : Polynomial R
            hp : Membership.mem I p
            hn : Eq p.natDegree k
            h : LT.lt N k
            hp0 : Ne p 0
            this✝¹ : Ne 0 1
            this✝ : Nontrivial R
            q : Polynomial R
            hq : Membership.mem I q
            hdq : LE.le q.degree ↑N
            hlqp : Eq q.leadingCoeff p.leadingCoeff
            hq0 : Ne q 0
            h1 : Eq p.degree (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.natDegree …
            h2 : Eq p.leadingCoeff (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.nat …
            this : LT.lt (HSub.hSub p (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q. …
            ⊢ Membership.mem (Ideal.span ↑s) p
          -/
          rw [← sub_add_cancel p (q * Polynomial.X ^ (k - q.natDegree))]
          /-
            case h.inr.intro.intro.intro
            R : Type u
            inst✝ : CommRing R
            inst : IsNoetherianRing R
            I : Ideal (Polynomial R)
            M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
            hm : Membership.mem (Set.range I.leadingCoeffNth) M
            N : Nat
            HN : Eq (I.leadingCoeffNth N) M
            s : Finset (Polynomial R)
            hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
            hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
            hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
            this✝² : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
            k : Nat
            ih : ∀ (m : Nat), LT.lt m k → ∀ ⦃p : Polynomial R⦄, Membership.mem I p → Eq p. …
            p : Polynomial R
            hp : Membership.mem I p
            hn : Eq p.natDegree k
            h : LT.lt N k
            hp0 : Ne p 0
            this✝¹ : Ne 0 1
            this✝ : Nontrivial R
            q : Polynomial R
            hq : Membership.mem I q
            hdq : LE.le q.degree ↑N
            hlqp : Eq q.leadingCoeff p.leadingCoeff
            hq0 : Ne q 0
            h1 : Eq p.degree (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.natDegree …
            h2 : Eq p.leadingCoeff (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.nat …
            this : LT.lt (HSub.hSub p (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q. …
            ⊢ Membership.mem (Ideal.span ↑s) (HAdd.hAdd (HSub.hSub p (HMul.hMul q (HPow.hP …
          -/
          convert (Ideal.span ↑s).add_mem _ ((Ideal.span (s : Set R[X])).mul_mem_right _ _)
            /-
              case h.inr.intro.intro.intro.convert_2
              R : Type u
              inst✝ : CommRing R
              inst : IsNoetherianRing R
              I : Ideal (Polynomial R)
              M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
              hm : Membership.mem (Set.range I.leadingCoeffNth) M
              N : Nat
              HN : Eq (I.leadingCoeffNth N) M
              s : Finset (Polynomial R)
              hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
              hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
              hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
              this✝² : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
              k : Nat
              ih : ∀ (m : Nat), LT.lt m k → ∀ ⦃p : Polynomial R⦄, Membership.mem I p → Eq p. …
              p : Polynomial R
              hp : Membership.mem I p
              hn : Eq p.natDegree k
              h : LT.lt N k
              hp0 : Ne p 0
              this✝¹ : Ne 0 1
              this✝ : Nontrivial R
              q : Polynomial R
              hq : Membership.mem I q
              hdq : LE.le q.degree ↑N
              hlqp : Eq q.leadingCoeff p.leadingCoeff
              hq0 : Ne q 0
              h1 : Eq p.degree (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.natDegree …
              h2 : Eq p.leadingCoeff (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.nat …
              this : LT.lt (HSub.hSub p (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q. …
              ⊢ Membership.mem (Ideal.span ↑s) (HSub.hSub p (HMul.hMul q (HPow.hPow Polynomi …
            -/
          · by_cases hpq : p - q * Polynomial.X ^ (k - q.natDegree) = 0
              /-
                case pos
                R : Type u
                inst✝ : CommRing R
                inst : IsNoetherianRing R
                I : Ideal (Polynomial R)
                M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
                hm : Membership.mem (Set.range I.leadingCoeffNth) M
                N : Nat
                HN : Eq (I.leadingCoeffNth N) M
                s : Finset (Polynomial R)
                hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
                hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
                hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
                this✝² : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
                k : Nat
                ih : ∀ (m : Nat), LT.lt m k → ∀ ⦃p : Polynomial R⦄, Membership.mem I p → Eq p. …
                p : Polynomial R
                hp : Membership.mem I p
                hn : Eq p.natDegree k
                h : LT.lt N k
                hp0 : Ne p 0
                this✝¹ : Ne 0 1
                this✝ : Nontrivial R
                q : Polynomial R
                hq : Membership.mem I q
                hdq : LE.le q.degree ↑N
                hlqp : Eq q.leadingCoeff p.leadingCoeff
                hq0 : Ne q 0
                h1 : Eq p.degree (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.natDegree …
                h2 : Eq p.leadingCoeff (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.nat …
                this : LT.lt (HSub.hSub p (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q. …
                hpq : Eq (HSub.hSub p (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.natD …
                ⊢ Membership.mem (Ideal.span ↑s) (HSub.hSub p (HMul.hMul q (HPow.hPow Polynomi …
              -/
            · rw [hpq]
              /-
                case pos
                R : Type u
                inst✝ : CommRing R
                inst : IsNoetherianRing R
                I : Ideal (Polynomial R)
                M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
                hm : Membership.mem (Set.range I.leadingCoeffNth) M
                N : Nat
                HN : Eq (I.leadingCoeffNth N) M
                s : Finset (Polynomial R)
                hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
                hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
                hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
                this✝² : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
                k : Nat
                ih : ∀ (m : Nat), LT.lt m k → ∀ ⦃p : Polynomial R⦄, Membership.mem I p → Eq p. …
                p : Polynomial R
                hp : Membership.mem I p
                hn : Eq p.natDegree k
                h : LT.lt N k
                hp0 : Ne p 0
                this✝¹ : Ne 0 1
                this✝ : Nontrivial R
                q : Polynomial R
                hq : Membership.mem I q
                hdq : LE.le q.degree ↑N
                hlqp : Eq q.leadingCoeff p.leadingCoeff
                hq0 : Ne q 0
                h1 : Eq p.degree (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.natDegree …
                h2 : Eq p.leadingCoeff (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.nat …
                this : LT.lt (HSub.hSub p (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q. …
                hpq : Eq (HSub.hSub p (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.natD …
                ⊢ Membership.mem (Ideal.span ↑s) 0
              -/
              exact Ideal.zero_mem _
              /-
                🎉 no goals
              -/
            /-
              case neg
              R : Type u
              inst✝ : CommRing R
              inst : IsNoetherianRing R
              I : Ideal (Polynomial R)
              M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
              hm : Membership.mem (Set.range I.leadingCoeffNth) M
              N : Nat
              HN : Eq (I.leadingCoeffNth N) M
              s : Finset (Polynomial R)
              hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
              hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
              hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
              this✝² : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
              k : Nat
              ih : ∀ (m : Nat), LT.lt m k → ∀ ⦃p : Polynomial R⦄, Membership.mem I p → Eq p. …
              p : Polynomial R
              hp : Membership.mem I p
              hn : Eq p.natDegree k
              h : LT.lt N k
              hp0 : Ne p 0
              this✝¹ : Ne 0 1
              this✝ : Nontrivial R
              q : Polynomial R
              hq : Membership.mem I q
              hdq : LE.le q.degree ↑N
              hlqp : Eq q.leadingCoeff p.leadingCoeff
              hq0 : Ne q 0
              h1 : Eq p.degree (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.natDegree …
              h2 : Eq p.leadingCoeff (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.nat …
              this : LT.lt (HSub.hSub p (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q. …
              hpq : Not (Eq (HSub.hSub p (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q …
              ⊢ Membership.mem (Ideal.span ↑s) (HSub.hSub p (HMul.hMul q (HPow.hPow Polynomi …
            -/
            refine ih _ ?_ (I.sub_mem hp (I.mul_mem_right _ hq)) rfl
            /-
              case neg
              R : Type u
              inst✝ : CommRing R
              inst : IsNoetherianRing R
              I : Ideal (Polynomial R)
              M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
              hm : Membership.mem (Set.range I.leadingCoeffNth) M
              N : Nat
              HN : Eq (I.leadingCoeffNth N) M
              s : Finset (Polynomial R)
              hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
              hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
              hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
              this✝² : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
              k : Nat
              ih : ∀ (m : Nat), LT.lt m k → ∀ ⦃p : Polynomial R⦄, Membership.mem I p → Eq p. …
              p : Polynomial R
              hp : Membership.mem I p
              hn : Eq p.natDegree k
              h : LT.lt N k
              hp0 : Ne p 0
              this✝¹ : Ne 0 1
              this✝ : Nontrivial R
              q : Polynomial R
              hq : Membership.mem I q
              hdq : LE.le q.degree ↑N
              hlqp : Eq q.leadingCoeff p.leadingCoeff
              hq0 : Ne q 0
              h1 : Eq p.degree (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.natDegree …
              h2 : Eq p.leadingCoeff (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.nat …
              this : LT.lt (HSub.hSub p (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q. …
              hpq : Not (Eq (HSub.hSub p (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q …
              ⊢ LT.lt (HSub.hSub p (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.natDe …
            -/
            rwa [Polynomial.degree_eq_natDegree hpq, Nat.cast_lt, hn] at this
            /-
              🎉 no goals
            -/
          /-
            case h.inr.intro.intro.intro.convert_5
            R : Type u
            inst✝ : CommRing R
            inst : IsNoetherianRing R
            I : Ideal (Polynomial R)
            M : Submodule R R := ⋯.min (Set.range I.leadingCoeffNth) ⋯
            hm : Membership.mem (Set.range I.leadingCoeffNth) M
            N : Nat
            HN : Eq (I.leadingCoeffNth N) M
            s : Finset (Polynomial R)
            hs : Eq (Submodule.span R ↑s) (I.degreeLE ↑N)
            hm2 : ∀ (k : Nat), LE.le (I.leadingCoeffNth k) M
            hs2 : ∀ {x : Polynomial R}, Membership.mem (I.degreeLE ↑N) x → Membership.mem  …
            this✝² : Eq (Submodule.span (Polynomial R) ↑s) (Ideal.span ↑s)
            k : Nat
            ih : ∀ (m : Nat), LT.lt m k → ∀ ⦃p : Polynomial R⦄, Membership.mem I p → Eq p. …
            p : Polynomial R
            hp : Membership.mem I p
            hn : Eq p.natDegree k
            h : LT.lt N k
            hp0 : Ne p 0
            this✝¹ : Ne 0 1
            this✝ : Nontrivial R
            q : Polynomial R
            hq : Membership.mem I q
            hdq : LE.le q.degree ↑N
            hlqp : Eq q.leadingCoeff p.leadingCoeff
            hq0 : Ne q 0
            h1 : Eq p.degree (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.natDegree …
            h2 : Eq p.leadingCoeff (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q.nat …
            this : LT.lt (HSub.hSub p (HMul.hMul q (HPow.hPow Polynomial.X (HSub.hSub k q. …
            ⊢ Membership.mem (Ideal.span ↑s) q
          -/
          exact hs2 ⟨Polynomial.mem_degreeLE.2 hdq, hq⟩⟩⟩
          /-
            🎉 no goals
          -/


theorem linearIndependent_powers_iff_aeval (f : M →ₗ[R] M) (v : M) :
    (LinearIndependent R fun n : ℕ => (f ^ n) v) ↔ ∀ p : R[X], aeval f p v = 0 → p = 0 := by
  /-
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    v : M
    ⊢ Iff (LinearIndependent R fun n => (HPow.hPow f n) v) (∀ (p : Polynomial R),  …
  -/
  rw [linearIndependent_iff]
  simp only [Finsupp.linearCombination_apply, aeval_endomorphism, forall_iff_forall_finsupp, Sum,
    support, coeff, ofFinsupp_eq_zero]
  /-
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    v : M
    ⊢ Iff (∀ (l : Finsupp Nat R), Eq (l.sum fun i a => HSMul.hSMul a ((HPow.hPow f …
  -/
  exact Iff.rfl
  /-
    🎉 no goals
  -/


theorem disjoint_ker_aeval_of_coprime (f : M →ₗ[R] M) {p q : R[X]} (hpq : IsCoprime p q) :
    Disjoint (LinearMap.ker (aeval f p)) (LinearMap.ker (aeval f q)) := by
  /-
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    hpq : IsCoprime p q
    ⊢ Disjoint (LinearMap.ker ((Polynomial.aeval f) p)) (LinearMap.ker ((Polynomia …
  -/
  rw [disjoint_iff_inf_le]
  /-
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    hpq : IsCoprime p q
    ⊢ LE.le (Min.min (LinearMap.ker ((Polynomial.aeval f) p)) (LinearMap.ker ((Pol …
  -/
  intro v hv
  /-
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    hpq : IsCoprime p q
    v : M
    hv : Membership.mem (Min.min (LinearMap.ker ((Polynomial.aeval f) p)) (LinearM …
    ⊢ Membership.mem Bot.bot v
  -/
  rcases hpq with ⟨p', q', hpq'⟩
  simpa [LinearMap.mem_ker.1 (Submodule.mem_inf.1 hv).1,
    LinearMap.mem_ker.1 (Submodule.mem_inf.1 hv).2] using
    congr_arg (fun p : R[X] => aeval f p v) hpq'.symm


theorem sup_aeval_range_eq_top_of_coprime (f : M →ₗ[R] M) {p q : R[X]} (hpq : IsCoprime p q) :
    LinearMap.range (aeval f p) ⊔ LinearMap.range (aeval f q) = ⊤ := by
  /-
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    hpq : IsCoprime p q
    ⊢ Eq (Max.max (LinearMap.range ((Polynomial.aeval f) p)) (LinearMap.range ((Po …
  -/
  rw [eq_top_iff]
  /-
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    hpq : IsCoprime p q
    ⊢ LE.le Top.top (Max.max (LinearMap.range ((Polynomial.aeval f) p)) (LinearMap …
  -/
  intro v _
  /-
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    hpq : IsCoprime p q
    v : M
    a✝ : Membership.mem Top.top v
    ⊢ Membership.mem (Max.max (LinearMap.range ((Polynomial.aeval f) p)) (LinearMa …
  -/
  rw [Submodule.mem_sup]
  /-
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    hpq : IsCoprime p q
    v : M
    a✝ : Membership.mem Top.top v
    ⊢ Exists fun y => And (Membership.mem (LinearMap.range ((Polynomial.aeval f) p …
  -/
  rcases hpq with ⟨p', q', hpq'⟩
  /-
    case intro.intro
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    v : M
    a✝ : Membership.mem Top.top v
    p' q' : Polynomial R
    hpq' : Eq (HAdd.hAdd (HMul.hMul p' p) (HMul.hMul q' q)) 1
    ⊢ Exists fun y => And (Membership.mem (LinearMap.range ((Polynomial.aeval f) p …
  -/
  use aeval f (p * p') v
  /-
    case h
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    v : M
    a✝ : Membership.mem Top.top v
    p' q' : Polynomial R
    hpq' : Eq (HAdd.hAdd (HMul.hMul p' p) (HMul.hMul q' q)) 1
    ⊢ And (Membership.mem (LinearMap.range ((Polynomial.aeval f) p)) (((Polynomial …
  -/
  use LinearMap.mem_range.2 ⟨aeval f p' v, by simp only [LinearMap.mul_apply, aeval_mul]⟩
  /-
    case right
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    v : M
    a✝ : Membership.mem Top.top v
    p' q' : Polynomial R
    hpq' : Eq (HAdd.hAdd (HMul.hMul p' p) (HMul.hMul q' q)) 1
    ⊢ Exists fun z => And (Membership.mem (LinearMap.range ((Polynomial.aeval f) q …
  -/
  use aeval f (q * q') v
  /-
    case h
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    v : M
    a✝ : Membership.mem Top.top v
    p' q' : Polynomial R
    hpq' : Eq (HAdd.hAdd (HMul.hMul p' p) (HMul.hMul q' q)) 1
    ⊢ And (Membership.mem (LinearMap.range ((Polynomial.aeval f) q)) (((Polynomial …
  -/
  use LinearMap.mem_range.2 ⟨aeval f q' v, by simp only [LinearMap.mul_apply, aeval_mul]⟩
  simpa only [mul_comm p p', mul_comm q q', aeval_one, aeval_add] using
    congr_arg (fun p : R[X] => aeval f p v) hpq'


theorem sup_ker_aeval_le_ker_aeval_mul {f : M →ₗ[R] M} {p q : R[X]} :
    LinearMap.ker (aeval f p) ⊔ LinearMap.ker (aeval f q) ≤ LinearMap.ker (aeval f (p * q)) := by
  /-
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    ⊢ LE.le (Max.max (LinearMap.ker ((Polynomial.aeval f) p)) (LinearMap.ker ((Pol …
  -/
  intro v hv
  /-
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    v : M
    hv : Membership.mem (Max.max (LinearMap.ker ((Polynomial.aeval f) p)) (LinearM …
    ⊢ Membership.mem (LinearMap.ker ((Polynomial.aeval f) (HMul.hMul p q))) v
  -/
  rcases Submodule.mem_sup.1 hv with ⟨x, hx, y, hy, hxy⟩
  have h_eval_x : aeval f (p * q) x = 0 := by
    rw [mul_comm, aeval_mul, LinearMap.mul_apply, LinearMap.mem_ker.1 hx, LinearMap.map_zero]
  have h_eval_y : aeval f (p * q) y = 0 := by
    rw [aeval_mul, LinearMap.mul_apply, LinearMap.mem_ker.1 hy, LinearMap.map_zero]
  /-
    case intro.intro.intro.intro
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    v : M
    hv : Membership.mem (Max.max (LinearMap.ker ((Polynomial.aeval f) p)) (LinearM …
    x : M
    hx : Membership.mem (LinearMap.ker ((Polynomial.aeval f) p)) x
    y : M
    hy : Membership.mem (LinearMap.ker ((Polynomial.aeval f) q)) y
    hxy : Eq (HAdd.hAdd x y) v
    h_eval_x : Eq (((Polynomial.aeval f) (HMul.hMul p q)) x) 0
    h_eval_y : Eq (((Polynomial.aeval f) (HMul.hMul p q)) y) 0
    ⊢ Membership.mem (LinearMap.ker ((Polynomial.aeval f) (HMul.hMul p q))) v
  -/
  rw [LinearMap.mem_ker, ← hxy, LinearMap.map_add, h_eval_x, h_eval_y, add_zero]
  /-
    🎉 no goals
  -/


theorem sup_ker_aeval_eq_ker_aeval_mul_of_coprime (f : M →ₗ[R] M) {p q : R[X]}
    (hpq : IsCoprime p q) :
    LinearMap.ker (aeval f p) ⊔ LinearMap.ker (aeval f q) = LinearMap.ker (aeval f (p * q)) := by
  /-
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    hpq : IsCoprime p q
    ⊢ Eq (Max.max (LinearMap.ker ((Polynomial.aeval f) p)) (LinearMap.ker ((Polyno …
  -/
  apply le_antisymm sup_ker_aeval_le_ker_aeval_mul
  /-
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    hpq : IsCoprime p q
    ⊢ LE.le (LinearMap.ker ((Polynomial.aeval f) (HMul.hMul p q))) (Max.max (Linea …
  -/
  intro v hv
  /-
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    hpq : IsCoprime p q
    v : M
    hv : Membership.mem (LinearMap.ker ((Polynomial.aeval f) (HMul.hMul p q))) v
    ⊢ Membership.mem (Max.max (LinearMap.ker ((Polynomial.aeval f) p)) (LinearMap. …
  -/
  rw [Submodule.mem_sup]
  /-
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    hpq : IsCoprime p q
    v : M
    hv : Membership.mem (LinearMap.ker ((Polynomial.aeval f) (HMul.hMul p q))) v
    ⊢ Exists fun y => And (Membership.mem (LinearMap.ker ((Polynomial.aeval f) p)) …
  -/
  rcases hpq with ⟨p', q', hpq'⟩
  have h_eval₂_qpp' :=
    calc
      aeval f (q * (p * p')) v = aeval f (p' * (p * q)) v := by
        rw [mul_comm, mul_assoc, mul_comm, mul_assoc, mul_comm q p]
      _ = 0 := by rw [aeval_mul, LinearMap.mul_apply, LinearMap.mem_ker.1 hv, LinearMap.map_zero]

  have h_eval₂_pqq' :=
    calc
      aeval f (p * (q * q')) v = aeval f (q' * (p * q)) v := by rw [← mul_assoc, mul_comm]
      _ = 0 := by rw [aeval_mul, LinearMap.mul_apply, LinearMap.mem_ker.1 hv, LinearMap.map_zero]

  /-
    case intro.intro
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    v : M
    hv : Membership.mem (LinearMap.ker ((Polynomial.aeval f) (HMul.hMul p q))) v
    p' q' : Polynomial R
    hpq' : Eq (HAdd.hAdd (HMul.hMul p' p) (HMul.hMul q' q)) 1
    h_eval₂_qpp' : Eq (((Polynomial.aeval f) (HMul.hMul q (HMul.hMul p p'))) v) 0
    h_eval₂_pqq' : Eq (((Polynomial.aeval f) (HMul.hMul p (HMul.hMul q q'))) v) 0
    ⊢ Exists fun y => And (Membership.mem (LinearMap.ker ((Polynomial.aeval f) p)) …
  -/
  rw [aeval_mul] at h_eval₂_qpp' h_eval₂_pqq'
  refine
    ⟨aeval f (q * q') v, LinearMap.mem_ker.1 h_eval₂_pqq', aeval f (p * p') v,
      LinearMap.mem_ker.1 h_eval₂_qpp', ?_⟩
  /-
    case intro.intro
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    v : M
    hv : Membership.mem (LinearMap.ker ((Polynomial.aeval f) (HMul.hMul p q))) v
    p' q' : Polynomial R
    hpq' : Eq (HAdd.hAdd (HMul.hMul p' p) (HMul.hMul q' q)) 1
    h_eval₂_qpp' : Eq ((HMul.hMul ((Polynomial.aeval f) q) ((Polynomial.aeval f) ( …
    h_eval₂_pqq' : Eq ((HMul.hMul ((Polynomial.aeval f) p) ((Polynomial.aeval f) ( …
    ⊢ Eq (HAdd.hAdd (((Polynomial.aeval f) (HMul.hMul q q')) v) (((Polynomial.aeva …
  -/
  rw [add_comm, mul_comm p p', mul_comm q q']
  /-
    case intro.intro
    R : Type u
    M : Type w
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    p q : Polynomial R
    v : M
    hv : Membership.mem (LinearMap.ker ((Polynomial.aeval f) (HMul.hMul p q))) v
    p' q' : Polynomial R
    hpq' : Eq (HAdd.hAdd (HMul.hMul p' p) (HMul.hMul q' q)) 1
    h_eval₂_qpp' : Eq ((HMul.hMul ((Polynomial.aeval f) q) ((Polynomial.aeval f) ( …
    h_eval₂_pqq' : Eq ((HMul.hMul ((Polynomial.aeval f) p) ((Polynomial.aeval f) ( …
    ⊢ Eq (HAdd.hAdd (((Polynomial.aeval f) (HMul.hMul p' p)) v) (((Polynomial.aeva …
  -/
  simpa only [map_add, map_mul, aeval_one] using congr_arg (fun p : R[X] => aeval f p v) hpq'
  /-
    🎉 no goals
  -/


lemma aeval_natDegree_le {R : Type*} [CommSemiring R] {m n : ℕ}
    (F : MvPolynomial σ R) (hF : F.totalDegree ≤ m)
    (f : σ → Polynomial R) (hf : ∀ i, (f i).natDegree ≤ n) :
    (MvPolynomial.aeval f F).natDegree ≤ m * n := by
  /-
    σ : Type v
    R : Type u_2
    inst✝ : CommSemiring R
    m n : Nat
    F : MvPolynomial σ R
    hF : LE.le F.totalDegree m
    f : σ → Polynomial R
    hf : ∀ (i : σ), LE.le (f i).natDegree n
    ⊢ LE.le ((MvPolynomial.aeval f) F).natDegree (HMul.hMul m n)
  -/
  rw [MvPolynomial.aeval_def, MvPolynomial.eval₂]
  /-
    σ : Type v
    R : Type u_2
    inst✝ : CommSemiring R
    m n : Nat
    F : MvPolynomial σ R
    hF : LE.le F.totalDegree m
    f : σ → Polynomial R
    hf : ∀ (i : σ), LE.le (f i).natDegree n
    ⊢ LE.le (Finsupp.sum F fun s a => HMul.hMul ((algebraMap R (Polynomial R)) a)  …
  -/
  apply (Polynomial.natDegree_sum_le _ _).trans
  /-
    σ : Type v
    R : Type u_2
    inst✝ : CommSemiring R
    m n : Nat
    F : MvPolynomial σ R
    hF : LE.le F.totalDegree m
    f : σ → Polynomial R
    hf : ∀ (i : σ), LE.le (f i).natDegree n
    ⊢ LE.le (Finset.fold Max.max 0 (Function.comp Polynomial.natDegree fun i => (f …
  -/
  apply Finset.sup_le
  /-
    case a
    σ : Type v
    R : Type u_2
    inst✝ : CommSemiring R
    m n : Nat
    F : MvPolynomial σ R
    hF : LE.le F.totalDegree m
    f : σ → Polynomial R
    hf : ∀ (i : σ), LE.le (f i).natDegree n
    ⊢ ∀ (b : Finsupp σ Nat), Membership.mem F.support b → LE.le (Function.comp Pol …
  -/
  intro d hd
  /-
    case a
    σ : Type v
    R : Type u_2
    inst✝ : CommSemiring R
    m n : Nat
    F : MvPolynomial σ R
    hF : LE.le F.totalDegree m
    f : σ → Polynomial R
    hf : ∀ (i : σ), LE.le (f i).natDegree n
    d : Finsupp σ Nat
    hd : Membership.mem F.support d
    ⊢ LE.le (Function.comp Polynomial.natDegree (fun i => (fun s a => HMul.hMul (( …
  -/
  simp_rw [Function.comp_apply, ← C_eq_algebraMap]
  /-
    case a
    σ : Type v
    R : Type u_2
    inst✝ : CommSemiring R
    m n : Nat
    F : MvPolynomial σ R
    hF : LE.le F.totalDegree m
    f : σ → Polynomial R
    hf : ∀ (i : σ), LE.le (f i).natDegree n
    d : Finsupp σ Nat
    hd : Membership.mem F.support d
    ⊢ LE.le (HMul.hMul (Polynomial.C (F d)) (d.prod fun n e => HPow.hPow (f n) e)) …
  -/
  apply (Polynomial.natDegree_C_mul_le _ _).trans
  /-
    case a
    σ : Type v
    R : Type u_2
    inst✝ : CommSemiring R
    m n : Nat
    F : MvPolynomial σ R
    hF : LE.le F.totalDegree m
    f : σ → Polynomial R
    hf : ∀ (i : σ), LE.le (f i).natDegree n
    d : Finsupp σ Nat
    hd : Membership.mem F.support d
    ⊢ LE.le (d.prod fun n e => HPow.hPow (f n) e).natDegree (HMul.hMul m n)
  -/
  apply (Polynomial.natDegree_prod_le _ _).trans
  have : ∑ i ∈ d.support, (d i) * n ≤ m * n := by
    rw [← Finset.sum_mul]
    apply mul_le_mul' (.trans _ hF) le_rfl
    rw [MvPolynomial.totalDegree]
    exact Finset.le_sup_of_le hd le_rfl
  /-
    case a
    σ : Type v
    R : Type u_2
    inst✝ : CommSemiring R
    m n : Nat
    F : MvPolynomial σ R
    hF : LE.le F.totalDegree m
    f : σ → Polynomial R
    hf : ∀ (i : σ), LE.le (f i).natDegree n
    d : Finsupp σ Nat
    hd : Membership.mem F.support d
    this : LE.le (d.support.sum fun i => HMul.hMul (d i) n) (HMul.hMul m n)
    ⊢ LE.le (d.support.sum fun i => ((fun n e => HPow.hPow (f n) e) i (d i)).natDe …
  -/
  apply (Finset.sum_le_sum _).trans this
  /-
    σ : Type v
    R : Type u_2
    inst✝ : CommSemiring R
    m n : Nat
    F : MvPolynomial σ R
    hF : LE.le F.totalDegree m
    f : σ → Polynomial R
    hf : ∀ (i : σ), LE.le (f i).natDegree n
    d : Finsupp σ Nat
    hd : Membership.mem F.support d
    this : LE.le (d.support.sum fun i => HMul.hMul (d i) n) (HMul.hMul m n)
    ⊢ ∀ (i : σ), Membership.mem d.support i → LE.le ((fun n e => HPow.hPow (f n) e …
  -/
  rintro i -
  /-
    σ : Type v
    R : Type u_2
    inst✝ : CommSemiring R
    m n : Nat
    F : MvPolynomial σ R
    hF : LE.le F.totalDegree m
    f : σ → Polynomial R
    hf : ∀ (i : σ), LE.le (f i).natDegree n
    d : Finsupp σ Nat
    hd : Membership.mem F.support d
    this : LE.le (d.support.sum fun i => HMul.hMul (d i) n) (HMul.hMul m n)
    i : σ
    ⊢ LE.le ((fun n e => HPow.hPow (f n) e) i (d i)).natDegree (HMul.hMul (d i) n)
  -/
  apply Polynomial.natDegree_pow_le.trans
  /-
    σ : Type v
    R : Type u_2
    inst✝ : CommSemiring R
    m n : Nat
    F : MvPolynomial σ R
    hF : LE.le F.totalDegree m
    f : σ → Polynomial R
    hf : ∀ (i : σ), LE.le (f i).natDegree n
    d : Finsupp σ Nat
    hd : Membership.mem F.support d
    this : LE.le (d.support.sum fun i => HMul.hMul (d i) n) (HMul.hMul m n)
    i : σ
    ⊢ LE.le (HMul.hMul (d i) (f i).natDegree) (HMul.hMul (d i) n)
  -/
  exact mul_le_mul' le_rfl (hf i)
  /-
    🎉 no goals
  -/


theorem isNoetherianRing_fin_0 [IsNoetherianRing R] :
    IsNoetherianRing (MvPolynomial (Fin 0) R) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    ⊢ IsNoetherianRing (MvPolynomial (Fin 0) R)
  -/
  apply isNoetherianRing_of_ringEquiv R
  /-
    case f
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    ⊢ RingEquiv R (MvPolynomial (Fin 0) R)
  -/
  symm; apply MvPolynomial.isEmptyRingEquiv R (Fin 0)
        /-
          🎉 no goals
        -/


theorem isNoetherianRing_fin [IsNoetherianRing R] :
    ∀ {n : ℕ}, IsNoetherianRing (MvPolynomial (Fin n) R)
  | 0 => isNoetherianRing_fin_0
  | n + 1 =>
    @isNoetherianRing_of_ringEquiv (Polynomial (MvPolynomial (Fin n) R)) _ _ _
      (MvPolynomial.finSuccEquiv _ n).toRingEquiv.symm
      (@Polynomial.isNoetherianRing (MvPolynomial (Fin n) R) _ isNoetherianRing_fin)


/-- The multivariate polynomial ring in finitely many variables over a noetherian ring
is itself a noetherian ring. -/
instance isNoetherianRing [Finite σ] [IsNoetherianRing R] :
    IsNoetherianRing (MvPolynomial σ R) := by
  /-
    R : Type u
    S : Type u_1
    σ : Type v
    M : Type w
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Finite σ
    inst✝ : IsNoetherianRing R
    ⊢ IsNoetherianRing (MvPolynomial σ R)
  -/
  cases nonempty_fintype σ
  exact
    @isNoetherianRing_of_ringEquiv (MvPolynomial (Fin (Fintype.card σ)) R) _ _ _
      (renameEquiv R (Fintype.equivFin σ).symm).toRingEquiv isNoetherianRing_fin


/-- Auxiliary lemma:
Multivariate polynomials over an integral domain
with variables indexed by `Fin n` form an integral domain.
This fact is proven inductively,
and then used to prove the general case without any finiteness hypotheses.
See `MvPolynomial.noZeroDivisors` for the general case. -/
theorem noZeroDivisors_fin (R : Type u) [CommSemiring R] [NoZeroDivisors R] :
    ∀ n : ℕ, NoZeroDivisors (MvPolynomial (Fin n) R)
  | 0 => (MvPolynomial.isEmptyAlgEquiv R _).injective.noZeroDivisors _ (map_zero _) (map_mul _)
  | n + 1 =>
    haveI := noZeroDivisors_fin R n
    (MvPolynomial.finSuccEquiv R n).injective.noZeroDivisors _ (map_zero _) (map_mul _)


/-- Auxiliary definition:
Multivariate polynomials in finitely many variables over an integral domain form an integral domain.
This fact is proven by transport of structure from the `MvPolynomial.noZeroDivisors_fin`,
and then used to prove the general case without finiteness hypotheses.
See `MvPolynomial.noZeroDivisors` for the general case. -/
theorem noZeroDivisors_of_finite (R : Type u) (σ : Type v) [CommSemiring R] [Finite σ]
    [NoZeroDivisors R] : NoZeroDivisors (MvPolynomial σ R) := by
  /-
    R : Type u
    σ : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Finite σ
    inst✝ : NoZeroDivisors R
    ⊢ NoZeroDivisors (MvPolynomial σ R)
  -/
  cases nonempty_fintype σ
  /-
    case intro
    R : Type u
    σ : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Finite σ
    inst✝ : NoZeroDivisors R
    val✝ : Fintype σ
    ⊢ NoZeroDivisors (MvPolynomial σ R)
  -/
  haveI := noZeroDivisors_fin R (Fintype.card σ)
  /-
    case intro
    R : Type u
    σ : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Finite σ
    inst✝ : NoZeroDivisors R
    val✝ : Fintype σ
    this : NoZeroDivisors (MvPolynomial (Fin (Fintype.card σ)) R)
    ⊢ NoZeroDivisors (MvPolynomial σ R)
  -/
  exact (renameEquiv R (Fintype.equivFin σ)).injective.noZeroDivisors _ (map_zero _) (map_mul _)
  /-
    🎉 no goals
  -/


instance {R : Type u} [CommSemiring R] [NoZeroDivisors R] {σ : Type v} :
    NoZeroDivisors (MvPolynomial σ R) where
  eq_zero_or_eq_zero_of_mul_eq_zero {p q} h := by
    /-
      R✝ : Type u
      S : Type u_1
      σ✝ : Type v
      M : Type w
      inst✝⁵ : CommRing R✝
      inst✝⁴ : CommRing S
      inst✝³ : AddCommGroup M
      inst✝² : Module R✝ M
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      σ : Type v
      p q : MvPolynomial σ R
      h : Eq (HMul.hMul p q) 0
      ⊢ Or (Eq p 0) (Eq q 0)
    -/
    obtain ⟨s, p, q, rfl, rfl⟩ := exists_finset_rename₂ p q
    /-
      case intro.intro.intro.intro
      R✝ : Type u
      S : Type u_1
      σ✝ : Type v
      M : Type w
      inst✝⁵ : CommRing R✝
      inst✝⁴ : CommRing S
      inst✝³ : AddCommGroup M
      inst✝² : Module R✝ M
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      σ : Type v
      s : Finset σ
      p q : MvPolynomial (Subtype fun x => Membership.mem s x) R
      h : Eq (HMul.hMul ((MvPolynomial.rename Subtype.val) p) ((MvPolynomial.rename  …
      ⊢ Or (Eq ((MvPolynomial.rename Subtype.val) p) 0) (Eq ((MvPolynomial.rename Su …
    -/
    let _nzd := MvPolynomial.noZeroDivisors_of_finite R s
    have : p * q = 0 := by
      apply rename_injective _ Subtype.val_injective
      simpa using h
    /-
      case intro.intro.intro.intro
      R✝ : Type u
      S : Type u_1
      σ✝ : Type v
      M : Type w
      inst✝⁵ : CommRing R✝
      inst✝⁴ : CommRing S
      inst✝³ : AddCommGroup M
      inst✝² : Module R✝ M
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      σ : Type v
      s : Finset σ
      p q : MvPolynomial (Subtype fun x => Membership.mem s x) R
      h : Eq (HMul.hMul ((MvPolynomial.rename Subtype.val) p) ((MvPolynomial.rename  …
      _nzd : NoZeroDivisors (MvPolynomial (Subtype fun x => Membership.mem s x) R) : …
      this : Eq (HMul.hMul p q) 0
      ⊢ Or (Eq ((MvPolynomial.rename Subtype.val) p) 0) (Eq ((MvPolynomial.rename Su …
    -/
    rw [mul_eq_zero] at this
    /-
      case intro.intro.intro.intro
      R✝ : Type u
      S : Type u_1
      σ✝ : Type v
      M : Type w
      inst✝⁵ : CommRing R✝
      inst✝⁴ : CommRing S
      inst✝³ : AddCommGroup M
      inst✝² : Module R✝ M
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      σ : Type v
      s : Finset σ
      p q : MvPolynomial (Subtype fun x => Membership.mem s x) R
      h : Eq (HMul.hMul ((MvPolynomial.rename Subtype.val) p) ((MvPolynomial.rename  …
      _nzd : NoZeroDivisors (MvPolynomial (Subtype fun x => Membership.mem s x) R) : …
      this : Or (Eq p 0) (Eq q 0)
      ⊢ Or (Eq ((MvPolynomial.rename Subtype.val) p) 0) (Eq ((MvPolynomial.rename Su …
    -/
                                      /-
                                        🎉 no goals
                                      -/
    apply this.imp <;> rintro rfl <;> simp
                                      /-
                                        🎉 no goals
                                      -/


/-- The multivariate polynomial ring over an integral domain is an integral domain. -/
instance isDomain {R : Type u} {σ : Type v} [CommRing R] [IsDomain R] :
    IsDomain (MvPolynomial σ R) := by
  /-
    R✝ : Type u
    S : Type u_1
    σ✝ : Type v
    M : Type w
    inst✝⁵ : CommRing R✝
    inst✝⁴ : CommRing S
    inst✝³ : AddCommGroup M
    inst✝² : Module R✝ M
    R : Type u
    σ : Type v
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ⊢ IsDomain (MvPolynomial σ R)
  -/
  apply @NoZeroDivisors.to_isDomain (MvPolynomial σ R) _ ?_ _
  /-
    R✝ : Type u
    S : Type u_1
    σ✝ : Type v
    M : Type w
    inst✝⁵ : CommRing R✝
    inst✝⁴ : CommRing S
    inst✝³ : AddCommGroup M
    inst✝² : Module R✝ M
    R : Type u
    σ : Type v
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ⊢ Nontrivial (MvPolynomial σ R)
  -/
  apply AddMonoidAlgebra.nontrivial
  /-
    🎉 no goals
  -/

-- instance {R : Type u} {σ : Type v} [CommRing R] [IsDomain R] :
--     IsDomain (MvPolynomial σ R)[X] := inferInstance


theorem map_mvPolynomial_eq_eval₂ {S : Type*} [CommRing S] [Finite σ] (ϕ : MvPolynomial σ R →+* S)
    (p : MvPolynomial σ R) :
    ϕ p = MvPolynomial.eval₂ (ϕ.comp MvPolynomial.C) (fun s => ϕ (MvPolynomial.X s)) p := by
  /-
    R : Type u
    σ : Type v
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : Finite σ
    ϕ : RingHom (MvPolynomial σ R) S
    p : MvPolynomial σ R
    ⊢ Eq (ϕ p) (MvPolynomial.eval₂ (ϕ.comp MvPolynomial.C) (fun s => ϕ (MvPolynomi …
  -/
  cases nonempty_fintype σ
  /-
    case intro
    R : Type u
    σ : Type v
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : Finite σ
    ϕ : RingHom (MvPolynomial σ R) S
    p : MvPolynomial σ R
    val✝ : Fintype σ
    ⊢ Eq (ϕ p) (MvPolynomial.eval₂ (ϕ.comp MvPolynomial.C) (fun s => ϕ (MvPolynomi …
  -/
  refine Trans.trans (congr_arg ϕ (MvPolynomial.as_sum p)) ?_
  /-
    case intro
    R : Type u
    σ : Type v
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : Finite σ
    ϕ : RingHom (MvPolynomial σ R) S
    p : MvPolynomial σ R
    val✝ : Fintype σ
    ⊢ Eq (ϕ (p.support.sum fun v => (MvPolynomial.monomial v) (MvPolynomial.coeff  …
  -/
  rw [MvPolynomial.eval₂_eq', map_sum ϕ]
  /-
    case intro
    R : Type u
    σ : Type v
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : Finite σ
    ϕ : RingHom (MvPolynomial σ R) S
    p : MvPolynomial σ R
    val✝ : Fintype σ
    ⊢ Eq (p.support.sum fun x => ϕ ((MvPolynomial.monomial x) (MvPolynomial.coeff  …
  -/
  congr
  /-
    case intro.e_f
    R : Type u
    σ : Type v
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : Finite σ
    ϕ : RingHom (MvPolynomial σ R) S
    p : MvPolynomial σ R
    val✝ : Fintype σ
    ⊢ Eq (fun x => ϕ ((MvPolynomial.monomial x) (MvPolynomial.coeff x p))) fun d = …
  -/
  ext
  /-
    case intro.e_f.h
    R : Type u
    σ : Type v
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : Finite σ
    ϕ : RingHom (MvPolynomial σ R) S
    p : MvPolynomial σ R
    val✝ : Fintype σ
    x✝ : Finsupp σ Nat
    ⊢ Eq (ϕ ((MvPolynomial.monomial x✝) (MvPolynomial.coeff x✝ p))) (HMul.hMul ((ϕ …
  -/
  simp only [monomial_eq, ϕ.map_pow, map_prod ϕ, ϕ.comp_apply, ϕ.map_mul, Finsupp.prod_pow]
  /-
    🎉 no goals
  -/


/-- If every coefficient of a polynomial is in an ideal `I`, then so is the polynomial itself,
multivariate version. -/
theorem mem_ideal_of_coeff_mem_ideal (I : Ideal (MvPolynomial σ R)) (p : MvPolynomial σ R)
    (hcoe : ∀ m : σ →₀ ℕ, p.coeff m ∈ I.comap (C : R →+* MvPolynomial σ R)) : p ∈ I := by
  /-
    R : Type u
    σ : Type v
    inst✝ : CommRing R
    I : Ideal (MvPolynomial σ R)
    p : MvPolynomial σ R
    hcoe : ∀ (m : Finsupp σ Nat), Membership.mem (Ideal.comap MvPolynomial.C I) (M …
    ⊢ Membership.mem I p
  -/
  rw [as_sum p]
  suffices ∀ m ∈ p.support, monomial m (MvPolynomial.coeff m p) ∈ I by
    exact Submodule.sum_mem I this
  /-
    R : Type u
    σ : Type v
    inst✝ : CommRing R
    I : Ideal (MvPolynomial σ R)
    p : MvPolynomial σ R
    hcoe : ∀ (m : Finsupp σ Nat), Membership.mem (Ideal.comap MvPolynomial.C I) (M …
    ⊢ ∀ (m : Finsupp σ Nat), Membership.mem p.support m → Membership.mem I ((MvPol …
  -/
  intro m _
  /-
    R : Type u
    σ : Type v
    inst✝ : CommRing R
    I : Ideal (MvPolynomial σ R)
    p : MvPolynomial σ R
    hcoe : ∀ (m : Finsupp σ Nat), Membership.mem (Ideal.comap MvPolynomial.C I) (M …
    m : Finsupp σ Nat
    a✝ : Membership.mem p.support m
    ⊢ Membership.mem I ((MvPolynomial.monomial m) (MvPolynomial.coeff m p))
  -/
  rw [← mul_one (coeff m p), ← C_mul_monomial]
  /-
    R : Type u
    σ : Type v
    inst✝ : CommRing R
    I : Ideal (MvPolynomial σ R)
    p : MvPolynomial σ R
    hcoe : ∀ (m : Finsupp σ Nat), Membership.mem (Ideal.comap MvPolynomial.C I) (M …
    m : Finsupp σ Nat
    a✝ : Membership.mem p.support m
    ⊢ Membership.mem I (HMul.hMul (MvPolynomial.C (MvPolynomial.coeff m p)) ((MvPo …
  -/
  suffices C (coeff m p) ∈ I by exact I.mul_mem_right (monomial m 1) this
  /-
    R : Type u
    σ : Type v
    inst✝ : CommRing R
    I : Ideal (MvPolynomial σ R)
    p : MvPolynomial σ R
    hcoe : ∀ (m : Finsupp σ Nat), Membership.mem (Ideal.comap MvPolynomial.C I) (M …
    m : Finsupp σ Nat
    a✝ : Membership.mem p.support m
    ⊢ Membership.mem I (MvPolynomial.C (MvPolynomial.coeff m p))
  -/
  simpa [Ideal.mem_comap] using hcoe m
  /-
    🎉 no goals
  -/


/-- The push-forward of an ideal `I` of `R` to `MvPolynomial σ R` via inclusion
 is exactly the set of polynomials whose coefficients are in `I` -/
theorem mem_map_C_iff {I : Ideal R} {f : MvPolynomial σ R} :
    f ∈ (Ideal.map (C : R →+* MvPolynomial σ R) I : Ideal (MvPolynomial σ R)) ↔
      ∀ m : σ →₀ ℕ, f.coeff m ∈ I := by
  classical
  constructor
  · intro hf
    refine Submodule.span_induction ?_ ?_ ?_ ?_ hf
    · intro f hf n
      cases' (Set.mem_image _ _ _).mp hf with x hx
      rw [← hx.right, coeff_C]
      by_cases h : n = 0
      · simpa [h] using hx.left
      · simp [Ne.symm h]
    · simp
    · exact fun f g _ _ hf hg n => by simp [I.add_mem (hf n) (hg n)]
    · refine fun f g _ hg n => ?_
      rw [smul_eq_mul, coeff_mul]
      exact I.sum_mem fun c _ => I.mul_mem_left (f.coeff c.fst) (hg c.snd)
  · intro hf
    rw [as_sum f]
    suffices ∀ m ∈ f.support, monomial m (coeff m f) ∈ (Ideal.map C I : Ideal (MvPolynomial σ R)) by
      exact Submodule.sum_mem _ this
    intro m _
    rw [← mul_one (coeff m f), ← C_mul_monomial]
    suffices C (coeff m f) ∈ (Ideal.map C I : Ideal (MvPolynomial σ R)) by
      exact Ideal.mul_mem_right _ _ this
    apply Ideal.mem_map_of_mem _
    exact hf m


theorem ker_map (f : R →+* S) :
    RingHom.ker (map f : MvPolynomial σ R →+* MvPolynomial σ S) =
    Ideal.map (C : R →+* MvPolynomial σ R) (RingHom.ker f) := by
  /-
    R : Type u
    S : Type u_1
    σ : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Eq (RingHom.ker (MvPolynomial.map f)) (Ideal.map MvPolynomial.C (RingHom.ker …
  -/
  ext
  /-
    case h
    R : Type u
    S : Type u_1
    σ : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    x✝ : MvPolynomial σ R
    ⊢ Iff (Membership.mem (RingHom.ker (MvPolynomial.map f)) x✝) (Membership.mem ( …
  -/
  rw [MvPolynomial.mem_map_C_iff, RingHom.mem_ker, MvPolynomial.ext_iff]
  /-
    case h
    R : Type u
    S : Type u_1
    σ : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    x✝ : MvPolynomial σ R
    ⊢ Iff (∀ (m : Finsupp σ Nat), Eq (MvPolynomial.coeff m ((MvPolynomial.map f) x …
  -/
  simp_rw [coeff_map, coeff_zero, RingHom.mem_ker]
  /-
    🎉 no goals
  -/


lemma ker_mapAlgHom {S₁ S₂ σ : Type*} [CommRing S₁] [CommRing S₂] [Algebra R S₁]
    [Algebra R S₂] (f : S₁ →ₐ[R] S₂) :
    RingHom.ker (MvPolynomial.mapAlgHom (σ := σ) f) = Ideal.map MvPolynomial.C (RingHom.ker f) :=
  MvPolynomial.ker_map (f.toRingHom : S₁ →+* S₂)


