private def directSumNeZeroMulHom {ι : Type} [DecidableEq ι] (p : ι → ℕ) (n : ι → ℕ) :
    (⨁ i : {i // n i ≠ 0}, ZMod (p i ^ n i)) →+ ⨁ i, ZMod (p i ^ n i) :=
  DirectSum.toAddMonoid fun i ↦ DirectSum.of (fun i ↦ ZMod (p i ^ n i)) i


private def directSumNeZeroMulEquiv (ι : Type) [DecidableEq ι] (p : ι → ℕ) (n : ι → ℕ) :
    (⨁ i : {i // n i ≠ 0}, ZMod (p i ^ n i)) ≃+ ⨁ i, ZMod (p i ^ n i) where
  toFun := directSumNeZeroMulHom p n
  invFun := DirectSum.toAddMonoid fun i ↦
    if h : n i = 0 then 0 else DirectSum.of (fun j : {i // n i ≠ 0} ↦ ZMod (p j ^ n j)) ⟨i, h⟩
  left_inv x := by
    /-
      ι : Type
      inst✝ : DecidableEq ι
      p n : ι → Nat
      x : DirectSum (Subtype fun i => Ne (n i) 0) fun i => ZMod (HPow.hPow (p ↑i) (n …
      ⊢ Eq ((DirectSum.toAddMonoid fun i => dite (Eq (n i) 0) (fun h => 0) fun h =>  …
    -/
    induction' x using DirectSum.induction_on with i x x y hx hy
      /-
        case H_zero
        ι : Type
        inst✝ : DecidableEq ι
        p n : ι → Nat
        ⊢ Eq ((DirectSum.toAddMonoid fun i => dite (Eq (n i) 0) (fun h => 0) fun h =>  …
      -/
    · simp
      /-
        🎉 no goals
      -/
    · rw [directSumNeZeroMulHom, DirectSum.toAddMonoid_of, DirectSum.toAddMonoid_of,
        dif_neg i.prop]
      /-
        case H_plus
        ι : Type
        inst✝ : DecidableEq ι
        p n : ι → Nat
        x y : DirectSum (Subtype fun i => Ne (n i) 0) fun i => ZMod (HPow.hPow (p ↑i)  …
        hx : Eq ((DirectSum.toAddMonoid fun i => dite (Eq (n i) 0) (fun h => 0) fun h  …
        hy : Eq ((DirectSum.toAddMonoid fun i => dite (Eq (n i) 0) (fun h => 0) fun h  …
        ⊢ Eq ((DirectSum.toAddMonoid fun i => dite (Eq (n i) 0) (fun h => 0) fun h =>  …
      -/
    · rw [map_add, map_add, hx, hy]
      /-
        🎉 no goals
      -/
  right_inv x := by
    /-
      ι : Type
      inst✝ : DecidableEq ι
      p n : ι → Nat
      x : DirectSum ι fun i => ZMod (HPow.hPow (p i) (n i))
      ⊢ Eq ((directSumNeZeroMulHom p n) ((DirectSum.toAddMonoid fun i => dite (Eq (n …
    -/
    induction' x using DirectSum.induction_on with i x x y hx hy
      /-
        case H_zero
        ι : Type
        inst✝ : DecidableEq ι
        p n : ι → Nat
        ⊢ Eq ((directSumNeZeroMulHom p n) ((DirectSum.toAddMonoid fun i => dite (Eq (n …
      -/
    · rw [map_zero, map_zero]
      /-
        🎉 no goals
      -/
      /-
        case H_basic
        ι : Type
        inst✝ : DecidableEq ι
        p n : ι → Nat
        i : ι
        x : ZMod (HPow.hPow (p i) (n i))
        ⊢ Eq ((directSumNeZeroMulHom p n) ((DirectSum.toAddMonoid fun i => dite (Eq (n …
      -/
    · rw [DirectSum.toAddMonoid_of]
      /-
        case H_basic
        ι : Type
        inst✝ : DecidableEq ι
        p n : ι → Nat
        i : ι
        x : ZMod (HPow.hPow (p i) (n i))
        ⊢ Eq ((directSumNeZeroMulHom p n) ((dite (Eq (n i) 0) (fun h => 0) fun h => Di …
      -/
      split_ifs with h
        /-
          case pos
          ι : Type
          inst✝ : DecidableEq ι
          p n : ι → Nat
          i : ι
          x : ZMod (HPow.hPow (p i) (n i))
          h : Eq (n i) 0
          ⊢ Eq ((directSumNeZeroMulHom p n) (0 x)) ((DirectSum.of (fun i => ZMod (HPow.h …
        -/
      · simp [(ZMod.subsingleton_iff.2 <| by rw [h, pow_zero]).elim x 0]
        /-
          🎉 no goals
        -/
        /-
          case neg
          ι : Type
          inst✝ : DecidableEq ι
          p n : ι → Nat
          i : ι
          x : ZMod (HPow.hPow (p i) (n i))
          h : Not (Eq (n i) 0)
          ⊢ Eq ((directSumNeZeroMulHom p n) ((DirectSum.of (fun j => ZMod (HPow.hPow (p  …
        -/
      · simp_rw [directSumNeZeroMulHom, DirectSum.toAddMonoid_of]
        /-
          🎉 no goals
        -/
      /-
        case H_plus
        ι : Type
        inst✝ : DecidableEq ι
        p n : ι → Nat
        x y : DirectSum ι fun i => ZMod (HPow.hPow (p i) (n i))
        hx : Eq ((directSumNeZeroMulHom p n) ((DirectSum.toAddMonoid fun i => dite (Eq …
        hy : Eq ((directSumNeZeroMulHom p n) ((DirectSum.toAddMonoid fun i => dite (Eq …
        ⊢ Eq ((directSumNeZeroMulHom p n) ((DirectSum.toAddMonoid fun i => dite (Eq (n …
      -/
    · rw [map_add, map_add, hx, hy]
      /-
        🎉 no goals
      -/
  map_add' := map_add (directSumNeZeroMulHom p n)


theorem finite_of_fg_torsion [AddCommGroup M] [Module ℤ M] [Module.Finite ℤ M]
    (hM : Module.IsTorsion ℤ M) : _root_.Finite M := by
  /-
    M : Type u
    inst✝² : AddCommGroup M
    inst✝¹ : Module Int M
    inst✝ : Module.Finite Int M
    hM : Module.IsTorsion Int M
    ⊢ Finite M
  -/
  rcases Module.equiv_directSum_of_isTorsion hM with ⟨ι, _, p, h, e, ⟨l⟩⟩
  haveI : ∀ i : ι, NeZero (p i ^ e i).natAbs := fun i =>
    ⟨Int.natAbs_ne_zero.mpr <| pow_ne_zero (e i) (h i).ne_zero⟩
  haveI : ∀ i : ι, _root_.Finite <| ℤ ⧸ Submodule.span ℤ {p i ^ e i} := fun i =>
    Finite.of_equiv _ (p i ^ e i).quotientSpanEquivZMod.symm.toEquiv
  haveI : _root_.Finite (⨁ i, ℤ ⧸ (Submodule.span ℤ {p i ^ e i} : Submodule ℤ ℤ)) :=
    Finite.of_equiv _ DFinsupp.equivFunOnFintype.symm
  /-
    case intro.intro.intro.intro.intro.intro
    M : Type u
    inst✝² : AddCommGroup M
    inst✝¹ : Module Int M
    inst✝ : Module.Finite Int M
    hM : Module.IsTorsion Int M
    ι : Type
    w✝ : Fintype ι
    p : ι → Int
    h : ∀ (i : ι), Irreducible (p i)
    e : ι → Nat
    l : LinearEquiv (RingHom.id Int) M (DirectSum ι fun i => HasQuotient.Quotient  …
    this✝¹ : ∀ (i : ι), NeZero (HPow.hPow (p i) (e i)).natAbs
    this✝ : ∀ (i : ι), Finite (HasQuotient.Quotient Int (Submodule.span Int (Singl …
    this : Finite (DirectSum ι fun i => HasQuotient.Quotient Int (Submodule.span I …
    ⊢ Finite M
  -/
  exact Finite.of_equiv _ l.symm.toEquiv
  /-
    🎉 no goals
  -/


/-- **Structure theorem of finitely generated abelian groups** : Any finitely generated abelian
group is the product of a power of `ℤ` and a direct sum of some `ZMod (p i ^ e i)` for some
prime powers `p i ^ e i`. -/
theorem equiv_free_prod_directSum_zmod [hG : AddGroup.FG G] :
    ∃ (n : ℕ) (ι : Type) (_ : Fintype ι) (p : ι → ℕ) (_ : ∀ i, Nat.Prime <| p i) (e : ι → ℕ),
      Nonempty <| G ≃+ (Fin n →₀ ℤ) × ⨁ i : ι, ZMod (p i ^ e i) := by
  obtain ⟨n, ι, fι, p, hp, e, ⟨f⟩⟩ :=
    @Module.equiv_free_prod_directSum _ _ _ _ _ _ _ (Module.Finite.iff_addGroup_fg.mpr hG)
  /-
    case intro.intro.intro.intro.intro.intro.intro
    G : Type u
    inst✝ : AddCommGroup G
    hG : AddGroup.FG G
    n : Nat
    ι : Type
    fι : Fintype ι
    p : ι → Int
    hp : ∀ (i : ι), Irreducible (p i)
    e : ι → Nat
    f : LinearEquiv (RingHom.id Int) G (Prod (Finsupp (Fin n) Int) (DirectSum ι fu …
    ⊢ Exists fun n => Exists fun ι => Exists fun x => Exists fun p => Exists fun x …
  -/
  refine ⟨n, ι, fι, fun i => (p i).natAbs, fun i => ?_, e, ⟨?_⟩⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_1
      G : Type u
      inst✝ : AddCommGroup G
      hG : AddGroup.FG G
      n : Nat
      ι : Type
      fι : Fintype ι
      p : ι → Int
      hp : ∀ (i : ι), Irreducible (p i)
      e : ι → Nat
      f : LinearEquiv (RingHom.id Int) G (Prod (Finsupp (Fin n) Int) (DirectSum ι fu …
      i : ι
      ⊢ Nat.Prime ((fun i => (p i).natAbs) i)
    -/
  · rw [← Int.prime_iff_natAbs_prime, ← irreducible_iff_prime]; exact hp i
                                                                /-
                                                                  🎉 no goals
                                                                -/
  exact
    f.toAddEquiv.trans
      ((AddEquiv.refl _).prodCongr <|
        DFinsupp.mapRange.addEquiv fun i =>
          ((Int.quotientSpanEquivZMod _).trans <|
              ZMod.ringEquivCongr <| (p i).natAbs_pow _).toAddEquiv)


/-- **Structure theorem of finite abelian groups** : Any finite abelian group is a direct sum of
some `ZMod (p i ^ e i)` for some prime powers `p i ^ e i`. -/
theorem equiv_directSum_zmod_of_finite [Finite G] :
    ∃ (ι : Type) (_ : Fintype ι) (p : ι → ℕ) (_ : ∀ i, Nat.Prime <| p i) (e : ι → ℕ),
      Nonempty <| G ≃+ ⨁ i : ι, ZMod (p i ^ e i) := by
  /-
    G : Type u
    inst✝¹ : AddCommGroup G
    inst✝ : Finite G
    ⊢ Exists fun ι => Exists fun x => Exists fun p => Exists fun x => Exists fun e …
  -/
  cases nonempty_fintype G
  /-
    case intro
    G : Type u
    inst✝¹ : AddCommGroup G
    inst✝ : Finite G
    val✝ : Fintype G
    ⊢ Exists fun ι => Exists fun x => Exists fun p => Exists fun x => Exists fun e …
  -/
  obtain ⟨n, ι, fι, p, hp, e, ⟨f⟩⟩ := equiv_free_prod_directSum_zmod G
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    G : Type u
    inst✝¹ : AddCommGroup G
    inst✝ : Finite G
    val✝ : Fintype G
    n : Nat
    ι : Type
    fι : Fintype ι
    p : ι → Nat
    hp : ∀ (i : ι), Nat.Prime (p i)
    e : ι → Nat
    f : AddEquiv G (Prod (Finsupp (Fin n) Int) (DirectSum ι fun i => ZMod (HPow.hP …
    ⊢ Exists fun ι => Exists fun x => Exists fun p => Exists fun x => Exists fun e …
  -/
  cases' n with n
  · have : Unique (Fin Nat.zero →₀ ℤ) :=
      { uniq := by simp only [eq_iff_true_of_subsingleton]; trivial }
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.zero
      G : Type u
      inst✝¹ : AddCommGroup G
      inst✝ : Finite G
      val✝ : Fintype G
      ι : Type
      fι : Fintype ι
      p : ι → Nat
      hp : ∀ (i : ι), Nat.Prime (p i)
      e : ι → Nat
      f : AddEquiv G (Prod (Finsupp (Fin 0) Int) (DirectSum ι fun i => ZMod (HPow.hP …
      this : Unique (Finsupp (Fin Nat.zero) Int)
      ⊢ Exists fun ι => Exists fun x => Exists fun p => Exists fun x => Exists fun e …
    -/
    exact ⟨ι, fι, p, hp, e, ⟨f.trans AddEquiv.uniqueProd⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.succ
      G : Type u
      inst✝¹ : AddCommGroup G
      inst✝ : Finite G
      val✝ : Fintype G
      ι : Type
      fι : Fintype ι
      p : ι → Nat
      hp : ∀ (i : ι), Nat.Prime (p i)
      e : ι → Nat
      n : Nat
      f : AddEquiv G (Prod (Finsupp (Fin (HAdd.hAdd n 1)) Int) (DirectSum ι fun i => …
      ⊢ Exists fun ι => Exists fun x => Exists fun p => Exists fun x => Exists fun e …
    -/
  · haveI := @Fintype.prodLeft _ _ _ (Fintype.ofEquiv G f.toEquiv) _
    exact
      (Fintype.ofSurjective (fun f : Fin n.succ →₀ ℤ => f 0) fun a =>
            ⟨Finsupp.single 0 a, Finsupp.single_eq_same⟩).false.elim


/-- **Structure theorem of finite abelian groups** : Any finite abelian group is a direct sum of
some `ZMod (n i)` for some natural numbers `n i > 1`. -/
lemma equiv_directSum_zmod_of_finite' (G : Type*) [AddCommGroup G] [Finite G] :
    ∃ (ι : Type) (_ : Fintype ι) (n : ι → ℕ),
      (∀ i, 1 < n i) ∧ Nonempty (G ≃+ ⨁ i, ZMod (n i)) := by
  classical
  obtain ⟨ι, hι, p, hp, n, ⟨e⟩⟩ := AddCommGroup.equiv_directSum_zmod_of_finite G
  refine ⟨{i : ι // n i ≠ 0}, inferInstance, fun i ↦ p i ^ n i, ?_,
    ⟨e.trans (directSumNeZeroMulEquiv ι _ _).symm⟩⟩
  rintro ⟨i, hi⟩
  exact one_lt_pow₀ (hp _).one_lt hi


theorem finite_of_fg_torsion [hG' : AddGroup.FG G] (hG : AddMonoid.IsTorsion G) : Finite G :=
  @Module.finite_of_fg_torsion _ _ _ (Module.Finite.iff_addGroup_fg.mpr hG') <|
    AddMonoid.isTorsion_iff_isTorsion_int.mp hG


theorem finite_of_fg_torsion [CommGroup G] [Group.FG G] (hG : Monoid.IsTorsion G) : Finite G :=
  @Finite.of_equiv _ _ (AddCommGroup.finite_of_fg_torsion (Additive G) hG) Multiplicative.ofAdd


/-- The **Structure Theorem For Finite Abelian Groups** in a multiplicative version:
A finite commutative group `G` is isomorphic to a finite product of finite cyclic groups. -/
theorem equiv_prod_multiplicative_zmod_of_finite (G : Type*) [CommGroup G] [Finite G] :
    ∃ (ι : Type) (_ : Fintype ι) (n : ι → ℕ),
       (∀ (i : ι), 1 < n i) ∧ Nonempty (G ≃* ((i : ι) → Multiplicative (ZMod (n i)))) := by
  /-
    G : Type u_1
    inst✝¹ : CommGroup G
    inst✝ : Finite G
    ⊢ Exists fun ι => Exists fun x => Exists fun n => And (∀ (i : ι), LT.lt 1 (n i …
  -/
  obtain ⟨ι, inst, n, h₁, h₂⟩ := AddCommGroup.equiv_directSum_zmod_of_finite' (Additive G)
  exact ⟨ι, inst, n, h₁, ⟨MulEquiv.toAdditive.symm <| h₂.some.trans <|
    (DirectSum.addEquivProd _).trans <| MulEquiv.toAdditive'' <| MulEquiv.piMultiplicative _⟩⟩


