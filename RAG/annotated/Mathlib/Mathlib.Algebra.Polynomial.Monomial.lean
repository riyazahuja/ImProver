theorem monomial_one_eq_iff [Nontrivial R] {i j : ℕ} :
    (monomial i 1 : R[X]) = monomial j 1 ↔ i = j := by
  -- Porting note: `ofFinsupp.injEq` is required.
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    i j : Nat
    ⊢ Iff (Eq ((Polynomial.monomial i) 1) ((Polynomial.monomial j) 1)) (Eq i j)
  -/
  simp_rw [← ofFinsupp_single, ofFinsupp.injEq]
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    i j : Nat
    ⊢ Iff (Eq (Finsupp.single i 1) (Finsupp.single j 1)) (Eq i j)
  -/
  exact AddMonoidAlgebra.of_injective.eq_iff
  /-
    🎉 no goals
  -/


instance infinite [Nontrivial R] : Infinite R[X] :=
                                                                /-
                                                                  R : Type u
                                                                  a b : R
                                                                  m✝ n✝ : Nat
                                                                  inst✝¹ : Semiring R
                                                                  p q r : Polynomial R
                                                                  inst✝ : Nontrivial R
                                                                  m n : Nat
                                                                  h : Eq ((fun i => (Polynomial.monomial i) 1) m) ((fun i => (Polynomial.monomia …
                                                                  ⊢ Eq m n
                                                                -/
  Infinite.of_injective (fun i => monomial i 1) fun m n h => by simpa [monomial_one_eq_iff] using h
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem card_support_le_one_iff_monomial {f : R[X]} :
    Finset.card f.support ≤ 1 ↔ ∃ n a, f = monomial n a := by
  /-
    R : Type u
    inst✝ : Semiring R
    f : Polynomial R
    ⊢ Iff (LE.le f.support.card 1) (Exists fun n => Exists fun a => Eq f ((Polynom …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : Semiring R
      f : Polynomial R
      ⊢ LE.le f.support.card 1 → Exists fun n => Exists fun a => Eq f ((Polynomial.m …
    -/
  · intro H
    /-
      case mp
      R : Type u
      inst✝ : Semiring R
      f : Polynomial R
      H : LE.le f.support.card 1
      ⊢ Exists fun n => Exists fun a => Eq f ((Polynomial.monomial n) a)
    -/
    rw [Finset.card_le_one_iff_subset_singleton] at H
    /-
      case mp
      R : Type u
      inst✝ : Semiring R
      f : Polynomial R
      H : Exists fun x => HasSubset.Subset f.support (Singleton.singleton x)
      ⊢ Exists fun n => Exists fun a => Eq f ((Polynomial.monomial n) a)
    -/
    rcases H with ⟨n, hn⟩
    /-
      case mp.intro
      R : Type u
      inst✝ : Semiring R
      f : Polynomial R
      n : Nat
      hn : HasSubset.Subset f.support (Singleton.singleton n)
      ⊢ Exists fun n => Exists fun a => Eq f ((Polynomial.monomial n) a)
    -/
    refine ⟨n, f.coeff n, ?_⟩
    /-
      case mp.intro
      R : Type u
      inst✝ : Semiring R
      f : Polynomial R
      n : Nat
      hn : HasSubset.Subset f.support (Singleton.singleton n)
      ⊢ Eq f ((Polynomial.monomial n) (f.coeff n))
    -/
    ext i
    /-
      case mp.intro.a
      R : Type u
      inst✝ : Semiring R
      f : Polynomial R
      n : Nat
      hn : HasSubset.Subset f.support (Singleton.singleton n)
      i : Nat
      ⊢ Eq (f.coeff i) (((Polynomial.monomial n) (f.coeff n)).coeff i)
    -/
    by_cases hi : i = n
      /-
        case pos
        R : Type u
        inst✝ : Semiring R
        f : Polynomial R
        n : Nat
        hn : HasSubset.Subset f.support (Singleton.singleton n)
        i : Nat
        hi : Eq i n
        ⊢ Eq (f.coeff i) (((Polynomial.monomial n) (f.coeff n)).coeff i)
      -/
    · simp [hi, coeff_monomial]
      /-
        🎉 no goals
      -/
    · have : f.coeff i = 0 := by
        rw [← not_mem_support_iff]
        exact fun hi' => hi (Finset.mem_singleton.1 (hn hi'))
      /-
        case neg
        R : Type u
        inst✝ : Semiring R
        f : Polynomial R
        n : Nat
        hn : HasSubset.Subset f.support (Singleton.singleton n)
        i : Nat
        hi : Not (Eq i n)
        this : Eq (f.coeff i) 0
        ⊢ Eq (f.coeff i) (((Polynomial.monomial n) (f.coeff n)).coeff i)
      -/
      simp [this, Ne.symm hi, coeff_monomial]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u
      inst✝ : Semiring R
      f : Polynomial R
      ⊢ (Exists fun n => Exists fun a => Eq f ((Polynomial.monomial n) a)) → LE.le f …
    -/
  · rintro ⟨n, a, rfl⟩
    /-
      case mpr.intro.intro
      R : Type u
      inst✝ : Semiring R
      n : Nat
      a : R
      ⊢ LE.le ((Polynomial.monomial n) a).support.card 1
    -/
    rw [← Finset.card_singleton n]
    /-
      case mpr.intro.intro
      R : Type u
      inst✝ : Semiring R
      n : Nat
      a : R
      ⊢ LE.le ((Polynomial.monomial n) a).support.card (Singleton.singleton n).card
    -/
    apply Finset.card_le_card
    /-
      case mpr.intro.intro.a
      R : Type u
      inst✝ : Semiring R
      n : Nat
      a : R
      ⊢ HasSubset.Subset ((Polynomial.monomial n) a).support (Singleton.singleton n)
    -/
    exact support_monomial' _ _
    /-
      🎉 no goals
    -/


theorem ringHom_ext {S} [Semiring S] {f g : R[X] →+* S} (h₁ : ∀ a, f (C a) = g (C a))
    (h₂ : f X = g X) : f = g := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type u_1
    inst✝ : Semiring S
    f g : RingHom (Polynomial R) S
    h₁ : ∀ (a : R), Eq (f (Polynomial.C a)) (g (Polynomial.C a))
    h₂ : Eq (f Polynomial.X) (g Polynomial.X)
    ⊢ Eq f g
  -/
  set f' := f.comp (toFinsuppIso R).symm.toRingHom with hf'
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type u_1
    inst✝ : Semiring S
    f g : RingHom (Polynomial R) S
    h₁ : ∀ (a : R), Eq (f (Polynomial.C a)) (g (Polynomial.C a))
    h₂ : Eq (f Polynomial.X) (g Polynomial.X)
    f' : RingHom (AddMonoidAlgebra R Nat) S := f.comp (Polynomial.toFinsuppIso R). …
    hf' : Eq f' (f.comp (Polynomial.toFinsuppIso R).symm.toRingHom)
    ⊢ Eq f g
  -/
  set g' := g.comp (toFinsuppIso R).symm.toRingHom with hg'
  have A : f' = g' := by
    ext
    simp [f', g', h₁, RingEquiv.toRingHom_eq_coe]
    simpa using h₂
  have B : f = f'.comp (toFinsuppIso R) := by
    rw [hf', RingHom.comp_assoc]
    ext x
    simp only [RingEquiv.toRingHom_eq_coe, RingEquiv.symm_apply_apply, Function.comp_apply,
      RingHom.coe_comp, RingEquiv.coe_toRingHom]
  have C' : g = g'.comp (toFinsuppIso R) := by
    rw [hg', RingHom.comp_assoc]
    ext x
    simp only [RingEquiv.toRingHom_eq_coe, RingEquiv.symm_apply_apply, Function.comp_apply,
      RingHom.coe_comp, RingEquiv.coe_toRingHom]
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type u_1
    inst✝ : Semiring S
    f g : RingHom (Polynomial R) S
    h₁ : ∀ (a : R), Eq (f (Polynomial.C a)) (g (Polynomial.C a))
    h₂ : Eq (f Polynomial.X) (g Polynomial.X)
    f' : RingHom (AddMonoidAlgebra R Nat) S := f.comp (Polynomial.toFinsuppIso R). …
    hf' : Eq f' (f.comp (Polynomial.toFinsuppIso R).symm.toRingHom)
    g' : RingHom (AddMonoidAlgebra R Nat) S := g.comp (Polynomial.toFinsuppIso R). …
    hg' : Eq g' (g.comp (Polynomial.toFinsuppIso R).symm.toRingHom)
    A : Eq f' g'
    B : Eq f (f'.comp ↑(Polynomial.toFinsuppIso R))
    C' : Eq g (g'.comp ↑(Polynomial.toFinsuppIso R))
    ⊢ Eq f g
  -/
  rw [B, C', A]
  /-
    🎉 no goals
  -/


@[ext high]
theorem ringHom_ext' {S} [Semiring S] {f g : R[X] →+* S} (h₁ : f.comp C = g.comp C)
    (h₂ : f X = g X) : f = g :=
  ringHom_ext (RingHom.congr_fun h₁) h₂


