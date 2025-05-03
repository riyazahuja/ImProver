/-- An `R` algebra `A` is formally étale if for every `R`-algebra, every square-zero ideal
`I : Ideal B` and `f : A →ₐ[R] B ⧸ I`, there exists exactly one lift `A →ₐ[R] B`.

See <https://stacks.math.columbia.edu/tag/00UQ> -/
@[mk_iff]
class FormallyEtale : Prop where
  comp_bijective :
    ∀ ⦃B : Type u⦄ [CommRing B],
      ∀ [Algebra R B] (I : Ideal B) (_ : I ^ 2 = ⊥),
        Function.Bijective ((Ideal.Quotient.mkₐ R I).comp : (A →ₐ[R] B) → A →ₐ[R] B ⧸ I)


theorem iff_unramified_and_smooth :
    FormallyEtale R A ↔ FormallyUnramified R A ∧ FormallySmooth R A := by
  /-
    R : Type u
    inst✝² : CommRing R
    A : Type u
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Iff (Algebra.FormallyEtale R A) (And (Algebra.FormallyUnramified R A) (Algeb …
  -/
  rw [FormallyUnramified.iff_comp_injective, formallySmooth_iff, formallyEtale_iff]
  /-
    R : Type u
    inst✝² : CommRing R
    A : Type u
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Iff (∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra R B] (I : Ideal B) …
  -/
  simp_rw [← forall_and, Function.Bijective]
  /-
    🎉 no goals
  -/


instance (priority := 100) to_unramified [h : FormallyEtale R A] :
    FormallyUnramified R A :=
  (FormallyEtale.iff_unramified_and_smooth.mp h).1


instance (priority := 100) to_smooth [h : FormallyEtale R A] : FormallySmooth R A :=
  (FormallyEtale.iff_unramified_and_smooth.mp h).2


theorem of_unramified_and_smooth [h₁ : FormallyUnramified R A]
    [h₂ : FormallySmooth R A] : FormallyEtale R A :=
  FormallyEtale.iff_unramified_and_smooth.mpr ⟨h₁, h₂⟩


theorem of_equiv [FormallyEtale R A] (e : A ≃ₐ[R] B) : FormallyEtale R B :=
  FormallyEtale.iff_unramified_and_smooth.mpr
    ⟨FormallyUnramified.of_equiv e, FormallySmooth.of_equiv e⟩


theorem iff_of_equiv (e : A ≃ₐ[R] B) : FormallyEtale R A ↔ FormallyEtale R B :=
  ⟨fun _ ↦ of_equiv e, fun _ ↦ of_equiv e.symm⟩


theorem comp [FormallyEtale R A] [FormallyEtale A B] : FormallyEtale R B :=
  FormallyEtale.iff_unramified_and_smooth.mpr
    ⟨FormallyUnramified.comp R A B, FormallySmooth.comp R A B⟩


instance base_change [FormallyEtale R A] : FormallyEtale B (B ⊗[R] A) :=
  FormallyEtale.iff_unramified_and_smooth.mpr ⟨inferInstance, inferInstance⟩


theorem of_isLocalization : FormallyEtale R Rₘ :=
  FormallyEtale.iff_unramified_and_smooth.mpr
    ⟨FormallyUnramified.of_isLocalization M, FormallySmooth.of_isLocalization M⟩


theorem localization_base [FormallyEtale R Sₘ] : FormallyEtale Rₘ Sₘ :=
  FormallyEtale.iff_unramified_and_smooth.mpr
    ⟨FormallyUnramified.localization_base M, FormallySmooth.localization_base M⟩


/-- The localization of a formally étale map is formally étale. -/
theorem localization_map [FormallyEtale R S] : FormallyEtale Rₘ Sₘ := by
  /-
    R S Rₘ Sₘ : Type u
    inst✝¹³ : CommRing R
    inst✝¹² : CommRing S
    inst✝¹¹ : CommRing Rₘ
    inst✝¹⁰ : CommRing Sₘ
    M : Submonoid R
    inst✝⁹ : Algebra R S
    inst✝⁸ : Algebra R Sₘ
    inst✝⁷ : Algebra S Sₘ
    inst✝⁶ : Algebra R Rₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : IsScalarTower R Rₘ Sₘ
    inst✝³ : IsScalarTower R S Sₘ
    inst✝² : IsLocalization M Rₘ
    inst✝¹ : IsLocalization (Submonoid.map (algebraMap R S) M) Sₘ
    inst✝ : Algebra.FormallyEtale R S
    ⊢ Algebra.FormallyEtale Rₘ Sₘ
  -/
  haveI : FormallyEtale S Sₘ := FormallyEtale.of_isLocalization (M.map (algebraMap R S))
  /-
    R S Rₘ Sₘ : Type u
    inst✝¹³ : CommRing R
    inst✝¹² : CommRing S
    inst✝¹¹ : CommRing Rₘ
    inst✝¹⁰ : CommRing Sₘ
    M : Submonoid R
    inst✝⁹ : Algebra R S
    inst✝⁸ : Algebra R Sₘ
    inst✝⁷ : Algebra S Sₘ
    inst✝⁶ : Algebra R Rₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : IsScalarTower R Rₘ Sₘ
    inst✝³ : IsScalarTower R S Sₘ
    inst✝² : IsLocalization M Rₘ
    inst✝¹ : IsLocalization (Submonoid.map (algebraMap R S) M) Sₘ
    inst✝ : Algebra.FormallyEtale R S
    this : Algebra.FormallyEtale S Sₘ
    ⊢ Algebra.FormallyEtale Rₘ Sₘ
  -/
  haveI : FormallyEtale R Sₘ := FormallyEtale.comp R S Sₘ
  /-
    R S Rₘ Sₘ : Type u
    inst✝¹³ : CommRing R
    inst✝¹² : CommRing S
    inst✝¹¹ : CommRing Rₘ
    inst✝¹⁰ : CommRing Sₘ
    M : Submonoid R
    inst✝⁹ : Algebra R S
    inst✝⁸ : Algebra R Sₘ
    inst✝⁷ : Algebra S Sₘ
    inst✝⁶ : Algebra R Rₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : IsScalarTower R Rₘ Sₘ
    inst✝³ : IsScalarTower R S Sₘ
    inst✝² : IsLocalization M Rₘ
    inst✝¹ : IsLocalization (Submonoid.map (algebraMap R S) M) Sₘ
    inst✝ : Algebra.FormallyEtale R S
    this✝ : Algebra.FormallyEtale S Sₘ
    this : Algebra.FormallyEtale R Sₘ
    ⊢ Algebra.FormallyEtale Rₘ Sₘ
  -/
  exact FormallyEtale.localization_base M
  /-
    🎉 no goals
  -/


/-- An `R`-algebra `A` is étale if it is formally étale and of finite presentation.

Note that the definition <https://stacks.math.columbia.edu/tag/00U1> in the stacks project is
different, but <https://stacks.math.columbia.edu/tag/00UR> shows that it is equivalent
to the definition here. -/
class Etale : Prop where
  formallyEtale : FormallyEtale R A := by infer_instance
  finitePresentation : FinitePresentation R A := by infer_instance


/-- Being étale is transported via algebra isomorphisms. -/
theorem of_equiv [Etale R A] (e : A ≃ₐ[R] B) : Etale R B where
  formallyEtale := FormallyEtale.of_equiv e
  finitePresentation := FinitePresentation.equiv e


/-- Etale is stable under composition. -/
theorem comp [Algebra A B] [IsScalarTower R A B] [Etale R A] [Etale A B] : Etale R B where
  formallyEtale := FormallyEtale.comp R A B
  finitePresentation := FinitePresentation.trans R A B


/-- Etale is stable under base change. -/
instance baseChange [Etale R A] : Etale B (B ⊗[R] A) where


/-- Localization at an element is étale. -/
theorem of_isLocalization_Away (r : R) [IsLocalization.Away r A] : Etale R A where
  formallyEtale := Algebra.FormallyEtale.of_isLocalization (Submonoid.powers r)
  finitePresentation := IsLocalization.Away.finitePresentation r


