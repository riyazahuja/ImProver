/--
For a preregular category, any sieve that contains an `EffectiveEpi` is a covering sieve of the
regular topology.
Note: This is one direction of `mem_sieves_iff_hasEffectiveEpi`, but is needed for the proof.
-/
theorem mem_sieves_of_hasEffectiveEpi (S : Sieve X) :
    (∃ (Y : C) (π : Y ⟶ X), EffectiveEpi π ∧ S.arrows π) → (S ∈ (regularTopology C) X) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preregular C
    X : C
    S : CategoryTheory.Sieve X
    ⊢ (Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpi π) (S.arro …
  -/
  rintro ⟨Y, π, h⟩
  have h_le : Sieve.generate (Presieve.ofArrows (fun () ↦ Y) (fun _ ↦ π)) ≤ S := by
    rw [Sieve.generate_le_iff (Presieve.ofArrows _ _) S]
    apply Presieve.le_of_factorsThru_sieve (Presieve.ofArrows _ _) S _
    intro W g f
    refine ⟨W, 𝟙 W, ?_⟩
    cases f
    exact ⟨π, ⟨h.2, Category.id_comp π⟩⟩
  /-
    case intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preregular C
    X : C
    S : CategoryTheory.Sieve X
    Y : C
    π : Quiver.Hom Y X
    h : And (CategoryTheory.EffectiveEpi π) (S.arrows π)
    h_le : LE.le (CategoryTheory.Sieve.generate (CategoryTheory.Presieve.ofArrows  …
    ⊢ Membership.mem ((CategoryTheory.regularTopology C) X) S
  -/
  apply Coverage.saturate_of_superset (regularCoverage C) h_le
  /-
    case intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preregular C
    X : C
    S : CategoryTheory.Sieve X
    Y : C
    π : Quiver.Hom Y X
    h : And (CategoryTheory.EffectiveEpi π) (S.arrows π)
    h_le : LE.le (CategoryTheory.Sieve.generate (CategoryTheory.Presieve.ofArrows  …
    ⊢ (CategoryTheory.regularCoverage C).Saturate X (CategoryTheory.Sieve.generate …
  -/
  exact Coverage.Saturate.of X _ ⟨Y, π, rfl, h.1⟩
  /-
    🎉 no goals
  -/


/-- Effective epis in a preregular category are stable under composition. -/
instance {Y Y' : C} (π : Y ⟶ X) [EffectiveEpi π]
    (π' : Y' ⟶ Y) [EffectiveEpi π'] : EffectiveEpi (π' ≫ π) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preregular C
    X Y Y' : C
    π : Quiver.Hom Y X
    inst✝¹ : CategoryTheory.EffectiveEpi π
    π' : Quiver.Hom Y' Y
    inst✝ : CategoryTheory.EffectiveEpi π'
    ⊢ CategoryTheory.EffectiveEpi (CategoryTheory.CategoryStruct.comp π' π)
  -/
  rw [effectiveEpi_iff_effectiveEpiFamily, ← Sieve.effectiveEpimorphic_family]
  suffices h₂ : (Sieve.generate (Presieve.ofArrows _ _)) ∈ (regularTopology C) X by
    change Nonempty _
    rw [← Sieve.forallYonedaIsSheaf_iff_colimit]
    exact fun W => regularTopology.isSheaf_yoneda_obj W _ h₂
  apply Coverage.Saturate.transitive X (Sieve.generate (Presieve.ofArrows (fun () ↦ Y)
      (fun () ↦ π)))
    /-
      case a
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preregular C
      X Y Y' : C
      π : Quiver.Hom Y X
      inst✝¹ : CategoryTheory.EffectiveEpi π
      π' : Quiver.Hom Y' Y
      inst✝ : CategoryTheory.EffectiveEpi π'
      ⊢ (CategoryTheory.regularCoverage C).Saturate X (CategoryTheory.Sieve.generate …
    -/
  · apply Coverage.Saturate.of
    /-
      case a.hS
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preregular C
      X Y Y' : C
      π : Quiver.Hom Y X
      inst✝¹ : CategoryTheory.EffectiveEpi π
      π' : Quiver.Hom Y' Y
      inst✝ : CategoryTheory.EffectiveEpi π'
      ⊢ Membership.mem ((CategoryTheory.regularCoverage C).covering X) (CategoryTheo …
    -/
    use Y, π
    /-
      🎉 no goals
    -/
    /-
      case a
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preregular C
      X Y Y' : C
      π : Quiver.Hom Y X
      inst✝¹ : CategoryTheory.EffectiveEpi π
      π' : Quiver.Hom Y' Y
      inst✝ : CategoryTheory.EffectiveEpi π'
      ⊢ ∀ ⦃Y_1 : C⦄ ⦃f : Quiver.Hom Y_1 X⦄, (CategoryTheory.Sieve.generate (Category …
    -/
  · intro V f ⟨Y₁, h, g, ⟨hY, hf⟩⟩
    /-
      case a
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preregular C
      X Y Y' : C
      π : Quiver.Hom Y X
      inst✝¹ : CategoryTheory.EffectiveEpi π
      π' : Quiver.Hom Y' Y
      inst✝ : CategoryTheory.EffectiveEpi π'
      V : C
      f : Quiver.Hom V X
      Y₁ : C
      h : Quiver.Hom V Y₁
      g : Quiver.Hom Y₁ X
      hY : CategoryTheory.Presieve.ofArrows (fun x => Y) (fun x => CategoryTheory.re …
      hf : Eq (CategoryTheory.CategoryStruct.comp h g) f
      ⊢ (CategoryTheory.regularCoverage C).Saturate V (CategoryTheory.Sieve.pullback …
    -/
    rw [← hf, Sieve.pullback_comp]
    /-
      case a
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preregular C
      X Y Y' : C
      π : Quiver.Hom Y X
      inst✝¹ : CategoryTheory.EffectiveEpi π
      π' : Quiver.Hom Y' Y
      inst✝ : CategoryTheory.EffectiveEpi π'
      V : C
      f : Quiver.Hom V X
      Y₁ : C
      h : Quiver.Hom V Y₁
      g : Quiver.Hom Y₁ X
      hY : CategoryTheory.Presieve.ofArrows (fun x => Y) (fun x => CategoryTheory.re …
      hf : Eq (CategoryTheory.CategoryStruct.comp h g) f
      ⊢ (CategoryTheory.regularCoverage C).Saturate V (CategoryTheory.Sieve.pullback …
    -/
    apply (regularTopology C).pullback_stable'
    /-
      case a.a
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preregular C
      X Y Y' : C
      π : Quiver.Hom Y X
      inst✝¹ : CategoryTheory.EffectiveEpi π
      π' : Quiver.Hom Y' Y
      inst✝ : CategoryTheory.EffectiveEpi π'
      V : C
      f : Quiver.Hom V X
      Y₁ : C
      h : Quiver.Hom V Y₁
      g : Quiver.Hom Y₁ X
      hY : CategoryTheory.Presieve.ofArrows (fun x => Y) (fun x => CategoryTheory.re …
      hf : Eq (CategoryTheory.CategoryStruct.comp h g) f
      ⊢ Membership.mem ((CategoryTheory.regularTopology C).sieves Y₁) (CategoryTheor …
    -/
    apply regularTopology.mem_sieves_of_hasEffectiveEpi
    /-
      case a.a.a
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preregular C
      X Y Y' : C
      π : Quiver.Hom Y X
      inst✝¹ : CategoryTheory.EffectiveEpi π
      π' : Quiver.Hom Y' Y
      inst✝ : CategoryTheory.EffectiveEpi π'
      V : C
      f : Quiver.Hom V X
      Y₁ : C
      h : Quiver.Hom V Y₁
      g : Quiver.Hom Y₁ X
      hY : CategoryTheory.Presieve.ofArrows (fun x => Y) (fun x => CategoryTheory.re …
      hf : Eq (CategoryTheory.CategoryStruct.comp h g) f
      ⊢ Exists fun Y_1 => Exists fun π_1 => And (CategoryTheory.EffectiveEpi π_1) (( …
    -/
    cases hY
    /-
      case a.a.a.mk
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preregular C
      X Y Y' : C
      π : Quiver.Hom Y X
      inst✝¹ : CategoryTheory.EffectiveEpi π
      π' : Quiver.Hom Y' Y
      inst✝ : CategoryTheory.EffectiveEpi π'
      V : C
      f : Quiver.Hom V X
      i✝ : Unit
      h : Quiver.Hom V Y
      hf : Eq (CategoryTheory.CategoryStruct.comp h (CategoryTheory.regularTopology. …
      ⊢ Exists fun Y_1 => Exists fun π_1 => And (CategoryTheory.EffectiveEpi π_1) (( …
    -/
    exact ⟨Y', π', inferInstance, Y', (𝟙 _), π' ≫ π, Presieve.ofArrows.mk (), (by simp)⟩
    /-
      🎉 no goals
    -/


/-- A sieve is a cover for the regular topology if and only if it contains an `EffectiveEpi`. -/
theorem mem_sieves_iff_hasEffectiveEpi (S : Sieve X) :
    (S ∈ (regularTopology C) X) ↔
    ∃ (Y : C) (π : Y ⟶ X), EffectiveEpi π ∧ (S.arrows π) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preregular C
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (Membership.mem ((CategoryTheory.regularTopology C) X) S) (Exists fun Y  …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preregular C
      X : C
      S : CategoryTheory.Sieve X
      ⊢ Membership.mem ((CategoryTheory.regularTopology C) X) S → Exists fun Y => Ex …
    -/
  · intro h
    induction h with
    | of Y T hS =>
      rcases hS with ⟨Y', π, h'⟩
      refine ⟨Y', π, h'.2, ?_⟩
      rcases h' with ⟨rfl, _⟩
      exact ⟨Y', 𝟙 Y', π, Presieve.ofArrows.mk (), (by simp)⟩
    | top Y => exact ⟨Y, (𝟙 Y), inferInstance, by simp only [Sieve.top_apply, forall_const]⟩
    | transitive Y R S _ _ a b =>
      rcases a with ⟨Y₁, π, ⟨h₁,h₂⟩⟩
      choose Y' π' _ H using b h₂
      exact ⟨Y', π' ≫ π, inferInstance, (by simpa using H)⟩
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preregular C
      X : C
      S : CategoryTheory.Sieve X
      ⊢ (Exists fun Y => Exists fun π => And (CategoryTheory.EffectiveEpi π) (S.arro …
    -/
  · exact regularTopology.mem_sieves_of_hasEffectiveEpi S
    /-
      🎉 no goals
    -/


