lemma epi_iff_surjective_up_to_refinements (f : X ⟶ Y) :
    Epi f ↔ ∀ ⦃A : C⦄ (y : A ⟶ Y),
      ∃ (A' : C) (π : A' ⟶ A) (_ : Epi π) (x : A' ⟶ X), π ≫ y = x ≫ f := by
  /-
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_1, u_2} C
    inst✝ : CategoryTheory.Abelian C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Epi f) (∀ ⦃A : C⦄ (y : Quiver.Hom A Y), Exists fun A' => …
  -/
  constructor
    /-
      case mp
      C : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_1, u_2} C
      inst✝ : CategoryTheory.Abelian C
      X Y : C
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Epi f → ∀ ⦃A : C⦄ (y : Quiver.Hom A Y), Exists fun A' => Exis …
    -/
  · intro _ A a
    /-
      case mp
      C : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_1, u_2} C
      inst✝ : CategoryTheory.Abelian C
      X Y : C
      f : Quiver.Hom X Y
      a✝ : CategoryTheory.Epi f
      A : C
      a : Quiver.Hom A Y
      ⊢ Exists fun A' => Exists fun π => Exists fun x => Exists fun x => Eq (Categor …
    -/
    exact ⟨pullback a f, pullback.fst a f, inferInstance, pullback.snd a f, pullback.condition⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_1, u_2} C
      inst✝ : CategoryTheory.Abelian C
      X Y : C
      f : Quiver.Hom X Y
      ⊢ (∀ ⦃A : C⦄ (y : Quiver.Hom A Y), Exists fun A' => Exists fun π => Exists fun …
    -/
  · intro hf
    /-
      case mpr
      C : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_1, u_2} C
      inst✝ : CategoryTheory.Abelian C
      X Y : C
      f : Quiver.Hom X Y
      hf : ∀ ⦃A : C⦄ (y : Quiver.Hom A Y), Exists fun A' => Exists fun π => Exists f …
      ⊢ CategoryTheory.Epi f
    -/
    obtain ⟨A, π, hπ, a', fac⟩ := hf (𝟙 Y)
    /-
      case mpr.intro.intro.intro.intro
      C : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_1, u_2} C
      inst✝ : CategoryTheory.Abelian C
      X Y : C
      f : Quiver.Hom X Y
      hf : ∀ ⦃A : C⦄ (y : Quiver.Hom A Y), Exists fun A' => Exists fun π => Exists f …
      A : C
      π : Quiver.Hom A Y
      hπ : CategoryTheory.Epi π
      a' : Quiver.Hom A X
      fac : Eq (CategoryTheory.CategoryStruct.comp π (CategoryTheory.CategoryStruct. …
      ⊢ CategoryTheory.Epi f
    -/
    rw [comp_id] at fac
    /-
      case mpr.intro.intro.intro.intro
      C : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_1, u_2} C
      inst✝ : CategoryTheory.Abelian C
      X Y : C
      f : Quiver.Hom X Y
      hf : ∀ ⦃A : C⦄ (y : Quiver.Hom A Y), Exists fun A' => Exists fun π => Exists f …
      A : C
      π : Quiver.Hom A Y
      hπ : CategoryTheory.Epi π
      a' : Quiver.Hom A X
      fac : Eq π (CategoryTheory.CategoryStruct.comp a' f)
      ⊢ CategoryTheory.Epi f
    -/
    exact epi_of_epi_fac fac.symm
    /-
      🎉 no goals
    -/


lemma surjective_up_to_refinements_of_epi (f : X ⟶ Y) [Epi f] {A : C} (y : A ⟶ Y) :
    ∃ (A' : C) (π : A' ⟶ A) (_ : Epi π) (x : A' ⟶ X), π ≫ y = x ≫ f :=
  (epi_iff_surjective_up_to_refinements f).1 inferInstance y


lemma ShortComplex.exact_iff_exact_up_to_refinements :
    S.Exact ↔ ∀ ⦃A : C⦄ (x₂ : A ⟶ S.X₂) (_ : x₂ ≫ S.g = 0),
      ∃ (A' : C) (π : A' ⟶ A) (_ : Epi π) (x₁ : A' ⟶ S.X₁), π ≫ x₂ = x₁ ≫ S.f := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    ⊢ Iff S.Exact (∀ ⦃A : C⦄ (x₂ : Quiver.Hom A S.X₂), Eq (CategoryTheory.Category …
  -/
  rw [S.exact_iff_epi_toCycles, epi_iff_surjective_up_to_refinements]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    ⊢ Iff (∀ ⦃A : C⦄ (y : Quiver.Hom A S.cycles), Exists fun A' => Exists fun π => …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      ⊢ (∀ ⦃A : C⦄ (y : Quiver.Hom A S.cycles), Exists fun A' => Exists fun π => Exi …
    -/
  · intro hS A a ha
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      hS : ∀ ⦃A : C⦄ (y : Quiver.Hom A S.cycles), Exists fun A' => Exists fun π => E …
      A : C
      a : Quiver.Hom A S.X₂
      ha : Eq (CategoryTheory.CategoryStruct.comp a S.g) 0
      ⊢ Exists fun A' => Exists fun π => Exists fun x => Exists fun x₁ => Eq (Catego …
    -/
    obtain ⟨A', π, hπ, x₁, fac⟩ := hS (S.liftCycles a ha)
    /-
      case mp.intro.intro.intro.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      hS : ∀ ⦃A : C⦄ (y : Quiver.Hom A S.cycles), Exists fun A' => Exists fun π => E …
      A : C
      a : Quiver.Hom A S.X₂
      ha : Eq (CategoryTheory.CategoryStruct.comp a S.g) 0
      A' : C
      π : Quiver.Hom A' A
      hπ : CategoryTheory.Epi π
      x₁ : Quiver.Hom A' S.X₁
      fac : Eq (CategoryTheory.CategoryStruct.comp π (S.liftCycles a ha)) (CategoryT …
      ⊢ Exists fun A' => Exists fun π => Exists fun x => Exists fun x₁ => Eq (Catego …
    -/
    exact ⟨A', π, hπ, x₁, by simpa only [assoc, liftCycles_i, toCycles_i] using fac =≫ S.iCycles⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      ⊢ (∀ ⦃A : C⦄ (x₂ : Quiver.Hom A S.X₂), Eq (CategoryTheory.CategoryStruct.comp  …
    -/
  · intro hS A a
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      hS : ∀ ⦃A : C⦄ (x₂ : Quiver.Hom A S.X₂), Eq (CategoryTheory.CategoryStruct.com …
      A : C
      a : Quiver.Hom A S.cycles
      ⊢ Exists fun A' => Exists fun π => Exists fun x => Exists fun x => Eq (Categor …
    -/
    obtain ⟨A', π, hπ, x₁, fac⟩ := hS (a ≫ S.iCycles) (by simp)
    /-
      case mpr.intro.intro.intro.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      hS : ∀ ⦃A : C⦄ (x₂ : Quiver.Hom A S.X₂), Eq (CategoryTheory.CategoryStruct.com …
      A : C
      a : Quiver.Hom A S.cycles
      A' : C
      π : Quiver.Hom A' A
      hπ : CategoryTheory.Epi π
      x₁ : Quiver.Hom A' S.X₁
      fac : Eq (CategoryTheory.CategoryStruct.comp π (CategoryTheory.CategoryStruct. …
      ⊢ Exists fun A' => Exists fun π => Exists fun x => Exists fun x => Eq (Categor …
    -/
    exact ⟨A', π, hπ, x₁, by simp only [← cancel_mono S.iCycles, assoc, toCycles_i, fac]⟩
    /-
      🎉 no goals
    -/


lemma ShortComplex.Exact.exact_up_to_refinements
    (hS : S.Exact) {A : C} (x₂ : A ⟶ S.X₂) (hx₂ : x₂ ≫ S.g = 0) :
    ∃ (A' : C) (π : A' ⟶ A) (_ : Epi π) (x₁ : A' ⟶ S.X₁), π ≫ x₂ = x₁ ≫ S.f := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    A : C
    x₂ : Quiver.Hom A S.X₂
    hx₂ : Eq (CategoryTheory.CategoryStruct.comp x₂ S.g) 0
    ⊢ Exists fun A' => Exists fun π => Exists fun x => Exists fun x₁ => Eq (Catego …
  -/
  rw [ShortComplex.exact_iff_exact_up_to_refinements] at hS
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    hS : ∀ ⦃A : C⦄ (x₂ : Quiver.Hom A S.X₂), Eq (CategoryTheory.CategoryStruct.com …
    A : C
    x₂ : Quiver.Hom A S.X₂
    hx₂ : Eq (CategoryTheory.CategoryStruct.comp x₂ S.g) 0
    ⊢ Exists fun A' => Exists fun π => Exists fun x => Exists fun x₁ => Eq (Catego …
  -/
  exact hS x₂ hx₂
  /-
    🎉 no goals
  -/


lemma ShortComplex.eq_liftCycles_homologyπ_up_to_refinements {A : C} (γ : A ⟶ S.homology) :
    ∃ (A' : C) (π : A' ⟶ A) (_ : Epi π) (z : A' ⟶ S.X₂) (hz : z ≫ S.g = 0),
      π ≫ γ = S.liftCycles z hz ≫ S.homologyπ := by
  /-
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_1, u_2} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    A : C
    γ : Quiver.Hom A S.homology
    ⊢ Exists fun A' => Exists fun π => Exists fun x => Exists fun z => Exists fun  …
  -/
  obtain ⟨A', π, hπ, z, hz⟩ := surjective_up_to_refinements_of_epi S.homologyπ γ
  /-
    case intro.intro.intro.intro
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_1, u_2} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    A : C
    γ : Quiver.Hom A S.homology
    A' : C
    π : Quiver.Hom A' A
    hπ : CategoryTheory.Epi π
    z : Quiver.Hom A' S.cycles
    hz : Eq (CategoryTheory.CategoryStruct.comp π γ) (CategoryTheory.CategoryStruc …
    ⊢ Exists fun A' => Exists fun π => Exists fun x => Exists fun z => Exists fun  …
  -/
  refine ⟨A', π, hπ, z ≫ S.iCycles, by simp, ?_⟩
  /-
    case intro.intro.intro.intro
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_1, u_2} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    A : C
    γ : Quiver.Hom A S.homology
    A' : C
    π : Quiver.Hom A' A
    hπ : CategoryTheory.Epi π
    z : Quiver.Hom A' S.cycles
    hz : Eq (CategoryTheory.CategoryStruct.comp π γ) (CategoryTheory.CategoryStruc …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp π γ) (CategoryTheory.CategoryStruct.c …
  -/
  rw [hz]
  /-
    case intro.intro.intro.intro
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_1, u_2} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    A : C
    γ : Quiver.Hom A S.homology
    A' : C
    π : Quiver.Hom A' A
    hπ : CategoryTheory.Epi π
    z : Quiver.Hom A' S.cycles
    hz : Eq (CategoryTheory.CategoryStruct.comp π γ) (CategoryTheory.CategoryStruc …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp z S.homologyπ) (CategoryTheory.Catego …
  -/
  congr 1
  /-
    case intro.intro.intro.intro.e_a
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_1, u_2} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    A : C
    γ : Quiver.Hom A S.homology
    A' : C
    π : Quiver.Hom A' A
    hπ : CategoryTheory.Epi π
    z : Quiver.Hom A' S.cycles
    hz : Eq (CategoryTheory.CategoryStruct.comp π γ) (CategoryTheory.CategoryStruc …
    ⊢ Eq z (S.liftCycles (CategoryTheory.CategoryStruct.comp z S.iCycles) ⋯)
  -/
  rw [← cancel_mono S.iCycles, liftCycles_i]
  /-
    🎉 no goals
  -/


