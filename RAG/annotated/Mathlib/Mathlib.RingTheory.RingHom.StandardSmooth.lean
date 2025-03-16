/-- A ring homomorphism `R →+* S` is standard smooth if `S` is standard smooth as `R`-algebra. -/
@[algebraize RingHom.IsStandardSmooth.toAlgebra]
def IsStandardSmooth (f : R →+* S) : Prop :=
  @Algebra.IsStandardSmooth.{t, w} _ _ _ _ f.toAlgebra


/-- Helper lemma for the `algebraize` tactic.-/
lemma IsStandardSmooth.toAlgebra {f : R →+* S} (hf : IsStandardSmooth.{t, w} f) :
    @Algebra.IsStandardSmooth.{t, w} R S _ _ f.toAlgebra := hf


/-- A ring homomorphism `R →+* S` is standard smooth of relative dimension `n` if
`S` is standard smooth of relative dimension `n` as `R`-algebra. -/
@[algebraize RingHom.IsStandardSmoothOfRelativeDimension.toAlgebra]
def IsStandardSmoothOfRelativeDimension (f : R →+* S) : Prop :=
  @Algebra.IsStandardSmoothOfRelativeDimension.{t, w} n _ _ _ _ f.toAlgebra


/-- Helper lemma for the `algebraize` tactic.-/
lemma IsStandardSmoothOfRelativeDimension.toAlgebra {f : R →+* S}
    (hf : IsStandardSmoothOfRelativeDimension.{t, w} n f) :
    @Algebra.IsStandardSmoothOfRelativeDimension.{t, w} n R S _ _ f.toAlgebra := hf


lemma IsStandardSmoothOfRelativeDimension.isStandardSmooth (f : R →+* S)
    (hf : IsStandardSmoothOfRelativeDimension.{t, w} n f) :
    IsStandardSmooth.{t, w} f := by
  /-
    n : Nat
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : RingHom.IsStandardSmoothOfRelativeDimension n f
    ⊢ f.IsStandardSmooth
  -/
  algebraize [f]
  /-
    n : Nat
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : RingHom.IsStandardSmoothOfRelativeDimension n f
    algInst✝ : Algebra R S := f.toAlgebra
    algebraizeInst✝ : Algebra.IsStandardSmoothOfRelativeDimension n R S
    ⊢ f.IsStandardSmooth
  -/
  exact Algebra.IsStandardSmoothOfRelativeDimension.isStandardSmooth n
  /-
    🎉 no goals
  -/


variable (R) in
lemma IsStandardSmoothOfRelativeDimension.id :
    IsStandardSmoothOfRelativeDimension.{t, w} 0 (RingHom.id R) :=
  Algebra.IsStandardSmoothOfRelativeDimension.id R


lemma IsStandardSmoothOfRelativeDimension.equiv (e : R ≃+* S) :
    IsStandardSmoothOfRelativeDimension.{t, w} 0 (e : R →+* S) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    e : RingEquiv R S
    ⊢ RingHom.IsStandardSmoothOfRelativeDimension 0 ↑e
  -/
  algebraize [e.toRingHom]
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    e : RingEquiv R S
    algInst✝ : Algebra R S := e.toRingHom.toAlgebra
    ⊢ RingHom.IsStandardSmoothOfRelativeDimension 0 ↑e
  -/
  exact Algebra.IsStandardSmoothOfRelativeDimension.of_algebraMap_bijective e.bijective
  /-
    🎉 no goals
  -/


lemma IsStandardSmooth.comp {g : S →+* T} {f : R →+* S}
    (hg : IsStandardSmooth.{t', w'} g) (hf : IsStandardSmooth.{t, w} f) :
    IsStandardSmooth.{max t t', max w w'} (g.comp f) := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    T : Type u_1
    inst✝ : CommRing T
    g : RingHom S T
    f : RingHom R S
    hg : g.IsStandardSmooth
    hf : f.IsStandardSmooth
    ⊢ (g.comp f).IsStandardSmooth
  -/
  rw [IsStandardSmooth]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    T : Type u_1
    inst✝ : CommRing T
    g : RingHom S T
    f : RingHom R S
    hg : g.IsStandardSmooth
    hf : f.IsStandardSmooth
    ⊢ Algebra.IsStandardSmooth R T
  -/
  algebraize [f, g, (g.comp f)]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    T : Type u_1
    inst✝ : CommRing T
    g : RingHom S T
    f : RingHom R S
    hg : g.IsStandardSmooth
    hf : f.IsStandardSmooth
    algInst✝² : Algebra R S := f.toAlgebra
    algInst✝¹ : Algebra S T := g.toAlgebra
    algInst✝ : Algebra R T := (g.comp f).toAlgebra
    scalarTowerInst✝ : IsScalarTower R S T := IsScalarTower.of_algebraMap_eq' (Eq. …
    algebraizeInst✝¹ : Algebra.IsStandardSmooth S T
    algebraizeInst✝ : Algebra.IsStandardSmooth R S
    ⊢ Algebra.IsStandardSmooth R T
  -/
  exact Algebra.IsStandardSmooth.trans.{t, t', w, w'} R S T
  /-
    🎉 no goals
  -/


lemma IsStandardSmoothOfRelativeDimension.comp {g : S →+* T} {f : R →+* S}
    (hg : IsStandardSmoothOfRelativeDimension.{t', w'} n g)
    (hf : IsStandardSmoothOfRelativeDimension.{t, w} m f) :
    IsStandardSmoothOfRelativeDimension.{max t t', max w w'} (n + m) (g.comp f) := by
  /-
    n m : Nat
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    T : Type u_1
    inst✝ : CommRing T
    g : RingHom S T
    f : RingHom R S
    hg : RingHom.IsStandardSmoothOfRelativeDimension n g
    hf : RingHom.IsStandardSmoothOfRelativeDimension m f
    ⊢ RingHom.IsStandardSmoothOfRelativeDimension (HAdd.hAdd n m) (g.comp f)
  -/
  rw [IsStandardSmoothOfRelativeDimension]
  /-
    n m : Nat
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    T : Type u_1
    inst✝ : CommRing T
    g : RingHom S T
    f : RingHom R S
    hg : RingHom.IsStandardSmoothOfRelativeDimension n g
    hf : RingHom.IsStandardSmoothOfRelativeDimension m f
    ⊢ Algebra.IsStandardSmoothOfRelativeDimension (HAdd.hAdd n m) R T
  -/
  algebraize [f, g, (g.comp f)]
  /-
    n m : Nat
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    T : Type u_1
    inst✝ : CommRing T
    g : RingHom S T
    f : RingHom R S
    hg : RingHom.IsStandardSmoothOfRelativeDimension n g
    hf : RingHom.IsStandardSmoothOfRelativeDimension m f
    algInst✝² : Algebra R S := f.toAlgebra
    algInst✝¹ : Algebra S T := g.toAlgebra
    algInst✝ : Algebra R T := (g.comp f).toAlgebra
    scalarTowerInst✝ : IsScalarTower R S T := IsScalarTower.of_algebraMap_eq' (Eq. …
    algebraizeInst✝¹ : Algebra.IsStandardSmoothOfRelativeDimension n S T
    algebraizeInst✝ : Algebra.IsStandardSmoothOfRelativeDimension m R S
    ⊢ Algebra.IsStandardSmoothOfRelativeDimension (HAdd.hAdd n m) R T
  -/
  exact Algebra.IsStandardSmoothOfRelativeDimension.trans m n R S T
  /-
    🎉 no goals
  -/


lemma isStandardSmooth_stableUnderComposition :
    StableUnderComposition @IsStandardSmooth.{t, w} :=
  fun _ _ _ _ _ _ _ _ hf hg ↦ hg.comp hf


lemma isStandardSmooth_respectsIso : RespectsIso @IsStandardSmooth.{t, w} := by
  /-
    ⊢ RingHom.RespectsIso @RingHom.IsStandardSmooth
  -/
  apply isStandardSmooth_stableUnderComposition.respectsIso
  /-
    ⊢ ∀ {R S : Type u_2} [inst : CommRing R] [inst_1 : CommRing S] (e : RingEquiv  …
  -/
  introv
  /-
    R S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    e : RingEquiv R S
    ⊢ e.toRingHom.IsStandardSmooth
  -/
  exact (IsStandardSmoothOfRelativeDimension.equiv e).isStandardSmooth
  /-
    🎉 no goals
  -/


lemma isStandardSmoothOfRelativeDimension_respectsIso :
    RespectsIso (@IsStandardSmoothOfRelativeDimension.{t, w} n) where
  left {R S T _ _ _} f e hf := by
    /-
      n : Nat
      R S T : Type u_2
      x✝² : CommRing R
      x✝¹ : CommRing S
      x✝ : CommRing T
      f : RingHom R S
      e : RingEquiv S T
      hf : RingHom.IsStandardSmoothOfRelativeDimension n f
      ⊢ RingHom.IsStandardSmoothOfRelativeDimension n (e.toRingHom.comp f)
    -/
    rw [← zero_add n]
    /-
      n : Nat
      R S T : Type u_2
      x✝² : CommRing R
      x✝¹ : CommRing S
      x✝ : CommRing T
      f : RingHom R S
      e : RingEquiv S T
      hf : RingHom.IsStandardSmoothOfRelativeDimension n f
      ⊢ RingHom.IsStandardSmoothOfRelativeDimension (HAdd.hAdd 0 n) (e.toRingHom.com …
    -/
    exact (IsStandardSmoothOfRelativeDimension.equiv e).comp hf
    /-
      🎉 no goals
    -/
  right {R S T _ _ _} f e hf := by
    /-
      n : Nat
      R S T : Type u_2
      x✝² : CommRing R
      x✝¹ : CommRing S
      x✝ : CommRing T
      f : RingHom S T
      e : RingEquiv R S
      hf : RingHom.IsStandardSmoothOfRelativeDimension n f
      ⊢ RingHom.IsStandardSmoothOfRelativeDimension n (f.comp e.toRingHom)
    -/
    rw [← add_zero n]
    /-
      n : Nat
      R S T : Type u_2
      x✝² : CommRing R
      x✝¹ : CommRing S
      x✝ : CommRing T
      f : RingHom S T
      e : RingEquiv R S
      hf : RingHom.IsStandardSmoothOfRelativeDimension n f
      ⊢ RingHom.IsStandardSmoothOfRelativeDimension (HAdd.hAdd n 0) (f.comp e.toRing …
    -/
    exact hf.comp (IsStandardSmoothOfRelativeDimension.equiv e)
    /-
      🎉 no goals
    -/


lemma isStandardSmooth_isStableUnderBaseChange :
    IsStableUnderBaseChange @IsStandardSmooth.{t, w} := by
  /-
    ⊢ RingHom.IsStableUnderBaseChange @RingHom.IsStandardSmooth
  -/
  apply IsStableUnderBaseChange.mk
    /-
      case h₁
      ⊢ RingHom.RespectsIso @RingHom.IsStandardSmooth
    -/
  · exact isStandardSmooth_respectsIso
    /-
      🎉 no goals
    -/
    /-
      case h₂
      ⊢ ∀ ⦃R S T : Type u_2⦄ [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
    -/
  · introv h
    replace h : Algebra.IsStandardSmooth R T := by
      rw [RingHom.IsStandardSmooth] at h; convert h; ext; simp_rw [Algebra.smul_def]; rfl
    suffices Algebra.IsStandardSmooth S (S ⊗[R] T) by
      rw [RingHom.IsStandardSmooth]; convert this; ext; simp_rw [Algebra.smul_def]; rfl
    /-
      case h₂
      R S T : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra R S
      inst✝ : Algebra R T
      h : Algebra.IsStandardSmooth R T
      ⊢ Algebra.IsStandardSmooth S (TensorProduct R S T)
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma isStandardSmoothOfRelativeDimension_isStableUnderBaseChange :
    IsStableUnderBaseChange (@IsStandardSmoothOfRelativeDimension.{t, w} n) := by
  /-
    n : Nat
    ⊢ RingHom.IsStableUnderBaseChange (@RingHom.IsStandardSmoothOfRelativeDimensio …
  -/
  apply IsStableUnderBaseChange.mk
    /-
      case h₁
      n : Nat
      ⊢ RingHom.RespectsIso (@RingHom.IsStandardSmoothOfRelativeDimension n)
    -/
  · exact isStandardSmoothOfRelativeDimension_respectsIso
    /-
      🎉 no goals
    -/
    /-
      case h₂
      n : Nat
      ⊢ ∀ ⦃R S T : Type u_2⦄ [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
    -/
  · introv h
    replace h : Algebra.IsStandardSmoothOfRelativeDimension n R T := by
      rw [RingHom.IsStandardSmoothOfRelativeDimension] at h
      convert h; ext; simp_rw [Algebra.smul_def]; rfl
    suffices Algebra.IsStandardSmoothOfRelativeDimension n S (S ⊗[R] T) by
      rw [RingHom.IsStandardSmoothOfRelativeDimension]
      convert this; ext; simp_rw [Algebra.smul_def]; rfl
    /-
      case h₂
      n : Nat
      R S T : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra R S
      inst✝ : Algebra R T
      h : Algebra.IsStandardSmoothOfRelativeDimension n R T
      ⊢ Algebra.IsStandardSmoothOfRelativeDimension n S (TensorProduct R S T)
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma IsStandardSmoothOfRelativeDimension.algebraMap_isLocalizationAway {Rᵣ : Type*} [CommRing Rᵣ]
    [Algebra R Rᵣ] (r : R) [IsLocalization.Away r Rᵣ] :
    IsStandardSmoothOfRelativeDimension.{0, 0} 0 (algebraMap R Rᵣ) := by
  have : (algebraMap R Rᵣ).toAlgebra = ‹Algebra R Rᵣ› := by
    ext
    rw [Algebra.smul_def]
    rfl
  /-
    R : Type u
    inst✝³ : CommRing R
    Rᵣ : Type u_2
    inst✝² : CommRing Rᵣ
    inst✝¹ : Algebra R Rᵣ
    r : R
    inst✝ : IsLocalization.Away r Rᵣ
    this : Eq (algebraMap R Rᵣ).toAlgebra inst✝¹
    ⊢ RingHom.IsStandardSmoothOfRelativeDimension 0 (algebraMap R Rᵣ)
  -/
  rw [IsStandardSmoothOfRelativeDimension, this]
  /-
    R : Type u
    inst✝³ : CommRing R
    Rᵣ : Type u_2
    inst✝² : CommRing Rᵣ
    inst✝¹ : Algebra R Rᵣ
    r : R
    inst✝ : IsLocalization.Away r Rᵣ
    this : Eq (algebraMap R Rᵣ).toAlgebra inst✝¹
    ⊢ Algebra.IsStandardSmoothOfRelativeDimension 0 R Rᵣ
  -/
  exact Algebra.IsStandardSmoothOfRelativeDimension.localization_away r
  /-
    🎉 no goals
  -/


lemma isStandardSmooth_localizationPreserves : LocalizationPreserves IsStandardSmooth.{t, w} :=
  isStandardSmooth_isStableUnderBaseChange.localizationPreserves


lemma isStandardSmoothOfRelativeDimension_localizationPreserves :
    LocalizationPreserves (IsStandardSmoothOfRelativeDimension.{t, w} n) :=
  (isStandardSmoothOfRelativeDimension_isStableUnderBaseChange n).localizationPreserves


lemma isStandardSmooth_holdsForLocalizationAway :
    HoldsForLocalizationAway IsStandardSmooth.{0, 0} := by
  /-
    ⊢ RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => Ring …
  -/
  introv R h
  /-
    R S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    r : R
    h : IsLocalization.Away r S
    ⊢ (algebraMap R S).IsStandardSmooth
  -/
  exact (IsStandardSmoothOfRelativeDimension.algebraMap_isLocalizationAway r).isStandardSmooth
  /-
    🎉 no goals
  -/


lemma isStandardSmoothOfRelativeDimension_holdsForLocalizationAway :
    HoldsForLocalizationAway (IsStandardSmoothOfRelativeDimension.{0, 0} 0) := by
  /-
    ⊢ RingHom.HoldsForLocalizationAway fun {R S} [CommRing R] [CommRing S] => Ring …
  -/
  introv R h
  /-
    R S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    r : R
    h : IsLocalization.Away r S
    ⊢ RingHom.IsStandardSmoothOfRelativeDimension 0 (algebraMap R S)
  -/
  exact IsStandardSmoothOfRelativeDimension.algebraMap_isLocalizationAway r
  /-
    🎉 no goals
  -/


lemma isStandardSmooth_stableUnderCompositionWithLocalizationAway :
    StableUnderCompositionWithLocalizationAway IsStandardSmooth.{0, 0} :=
  isStandardSmooth_stableUnderComposition.stableUnderCompositionWithLocalizationAway
    isStandardSmooth_holdsForLocalizationAway


lemma isStandardSmoothOfRelativeDimension_stableUnderCompositionWithLocalizationAway :
    StableUnderCompositionWithLocalizationAway (IsStandardSmoothOfRelativeDimension.{0, 0} n) where
  left R S _ _ _ _ _ r _ _ hf :=
    have : (algebraMap R S).IsStandardSmoothOfRelativeDimension 0 :=
      IsStandardSmoothOfRelativeDimension.algebraMap_isLocalizationAway r
    add_zero n ▸ IsStandardSmoothOfRelativeDimension.comp hf this
  right _ S T _ _ _ _ s _ _ hf :=
    have : (algebraMap S T).IsStandardSmoothOfRelativeDimension 0 :=
      IsStandardSmoothOfRelativeDimension.algebraMap_isLocalizationAway s
    zero_add n ▸ IsStandardSmoothOfRelativeDimension.comp this hf


