                                            /-
                                              C : Type u_1
                                              inst✝¹ : CategoryTheory.Category.{?u.162, u_1} C
                                              inst✝ : CategoryTheory.Abelian C
                                              R₁ R₂ : CategoryTheory.ComposableArrows C 3
                                              φ : Quiver.Hom R₁ R₂
                                              ⊢ LE.le 0 2
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
theorem mono_of_epi_of_mono_of_mono' (hR₁ : R₁.map' 0 2 = 0)
                                            /-
                                              🎉 no goals
                                            -/
                  /-
                    C : Type u_1
                    inst✝¹ : CategoryTheory.Category.{?u.162, u_1} C
                    inst✝ : CategoryTheory.Abelian C
                    R₁ R₂ : CategoryTheory.ComposableArrows C 3
                    φ : Quiver.Hom R₁ R₂
                    hR₁ : Eq (R₁.map' 0 2 ⋯ ⋯) 0
                    ⊢ LE.le 1 2
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
    (hR₁' : (mk₂ (R₁.map' 1 2) (R₁.map' 2 3)).Exact)
                                /-
                                  🎉 no goals
                                -/
                 /-
                   C : Type u_1
                   inst✝¹ : CategoryTheory.Category.{?u.162, u_1} C
                   inst✝ : CategoryTheory.Abelian C
                   R₁ R₂ : CategoryTheory.ComposableArrows C 3
                   φ : Quiver.Hom R₁ R₂
                   hR₁ : Eq (R₁.map' 0 2 ⋯ ⋯) 0
                   hR₁' : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯ …
                   ⊢ LE.le 0 1
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
    (hR₂ : (mk₂ (R₂.map' 0 1) (R₂.map' 1 2)).Exact)
                               /-
                                 🎉 no goals
                               -/
               /-
                 C : Type u_1
                 inst✝¹ : CategoryTheory.Category.{?u.162, u_1} C
                 inst✝ : CategoryTheory.Abelian C
                 R₁ R₂ : CategoryTheory.ComposableArrows C 3
                 φ : Quiver.Hom R₁ R₂
                 hR₁ : Eq (R₁.map' 0 2 ⋯ ⋯) 0
                 hR₁' : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯ …
                 hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
                 ⊢ LE.le 0 3
               -/
               /-
                 🎉 no goals
               -/
                                      /-
                                        🎉 no goals
                                      -/
    (h₀ : Epi (app' φ 0)) (h₁ : Mono (app' φ 1)) (h₃ : Mono (app' φ 3)) :
                                                             /-
                                                               🎉 no goals
                                                             -/
          /-
            C : Type u_1
            inst✝¹ : CategoryTheory.Category.{?u.162, u_1} C
            inst✝ : CategoryTheory.Abelian C
            R₁ R₂ : CategoryTheory.ComposableArrows C 3
            φ : Quiver.Hom R₁ R₂
            hR₁ : Eq (R₁.map' 0 2 ⋯ ⋯) 0
            hR₁' : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯ …
            hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
            h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
            h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
            h₃ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
            ⊢ LE.le 2 3
          -/
    Mono (app' φ 2) := by
          /-
            🎉 no goals
          -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 3
    φ : Quiver.Hom R₁ R₂
    hR₁ : Eq (R₁.map' 0 2 ⋯ ⋯) 0
    hR₁' : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯ …
    hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
    h₃ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
    ⊢ CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
  -/
  apply mono_of_cancel_zero
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 3
    φ : Quiver.Hom R₁ R₂
    hR₁ : Eq (R₁.map' 0 2 ⋯ ⋯) 0
    hR₁' : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯ …
    hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
    h₃ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
    ⊢ ∀ {P : C} (g : Quiver.Hom P (R₁.obj' 2 ⋯)), Eq (CategoryTheory.CategoryStruc …
  -/
  intro A f₂ h₁
  have h₂ : f₂ ≫ R₁.map' 2 3 = 0 := by
    rw [← cancel_mono (app' φ 3 _), assoc, NatTrans.naturality, reassoc_of% h₁,
      zero_comp, zero_comp]
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 3
    φ : Quiver.Hom R₁ R₂
    hR₁ : Eq (R₁.map' 0 2 ⋯ ⋯) 0
    hR₁' : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯ …
    hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₁✝ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
    h₃ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
    A : C
    f₂ : Quiver.Hom A (R₁.obj' 2 ⋯)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp f₂ (CategoryTheory.ComposableArrow …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ (R₁.map' 2 3 ⋯ ⋯)) 0
    ⊢ Eq f₂ 0
  -/
  obtain ⟨A₁, π₁, _, f₁, hf₁⟩ := (hR₁'.exact 0).exact_up_to_refinements f₂ h₂
  /-
    case h.intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 3
    φ : Quiver.Hom R₁ R₂
    hR₁ : Eq (R₁.map' 0 2 ⋯ ⋯) 0
    hR₁' : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯ …
    hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₁✝ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
    h₃ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
    A : C
    f₂ : Quiver.Hom A (R₁.obj' 2 ⋯)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp f₂ (CategoryTheory.ComposableArrow …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ (R₁.map' 2 3 ⋯ ⋯)) 0
    A₁ : C
    π₁ : Quiver.Hom A₁ A
    w✝ : CategoryTheory.Epi π₁
    f₁ : Quiver.Hom A₁ ((CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁ …
    hf₁ : Eq (CategoryTheory.CategoryStruct.comp π₁ f₂) (CategoryTheory.CategorySt …
    ⊢ Eq f₂ 0
  -/
  dsimp at hf₁
  have h₃ : (f₁ ≫ app' φ 1) ≫ R₂.map' 1 2 = 0 := by
    rw [assoc, ← NatTrans.naturality, ← reassoc_of% hf₁, h₁, comp_zero]
  /-
    case h.intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 3
    φ : Quiver.Hom R₁ R₂
    hR₁ : Eq (R₁.map' 0 2 ⋯ ⋯) 0
    hR₁' : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯ …
    hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₁✝ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
    h₃✝ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
    A : C
    f₂ : Quiver.Hom A (R₁.obj' 2 ⋯)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp f₂ (CategoryTheory.ComposableArrow …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ (R₁.map' 2 3 ⋯ ⋯)) 0
    A₁ : C
    π₁ : Quiver.Hom A₁ A
    w✝ : CategoryTheory.Epi π₁
    f₁ : Quiver.Hom A₁ ((CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁ …
    hf₁ : Eq (CategoryTheory.CategoryStruct.comp π₁ f₂) (CategoryTheory.CategorySt …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
    ⊢ Eq f₂ 0
  -/
  obtain ⟨A₂, π₂, _, g₀, hg₀⟩ := (hR₂.exact 0).exact_up_to_refinements _ h₃
  /-
    case h.intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 3
    φ : Quiver.Hom R₁ R₂
    hR₁ : Eq (R₁.map' 0 2 ⋯ ⋯) 0
    hR₁' : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯ …
    hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₁✝ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
    h₃✝ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
    A : C
    f₂ : Quiver.Hom A (R₁.obj' 2 ⋯)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp f₂ (CategoryTheory.ComposableArrow …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ (R₁.map' 2 3 ⋯ ⋯)) 0
    A₁ : C
    π₁ : Quiver.Hom A₁ A
    w✝¹ : CategoryTheory.Epi π₁
    f₁ : Quiver.Hom A₁ ((CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁ …
    hf₁ : Eq (CategoryTheory.CategoryStruct.comp π₁ f₂) (CategoryTheory.CategorySt …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
    A₂ : C
    π₂ : Quiver.Hom A₂ A₁
    w✝ : CategoryTheory.Epi π₂
    g₀ : Quiver.Hom A₂ ((CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂ …
    hg₀ : Eq (CategoryTheory.CategoryStruct.comp π₂ (CategoryTheory.CategoryStruct …
    ⊢ Eq f₂ 0
  -/
  obtain ⟨A₃, π₃, _, f₀, hf₀⟩ := surjective_up_to_refinements_of_epi (app' φ 0 _) g₀
  have h₄ : f₀ ≫ R₁.map' 0 1 = π₃ ≫ π₂ ≫ f₁ := by
    rw [← cancel_mono (app' φ 1 _), assoc, assoc, assoc, NatTrans.naturality,
      ← reassoc_of% hf₀, hg₀]
    rfl
  rw [← cancel_epi π₁, comp_zero, hf₁, ← cancel_epi π₂, ← cancel_epi π₃, comp_zero,
    comp_zero, ← reassoc_of% h₄, ← R₁.map'_comp 0 1 2, hR₁, comp_zero]


theorem mono_of_epi_of_mono_of_mono (hR₁ : R₁.Exact) (hR₂ : R₂.Exact)
               /-
                 C : Type u_1
                 inst✝¹ : CategoryTheory.Category.{?u.31397, u_1} C
                 inst✝ : CategoryTheory.Abelian C
                 R₁ R₂ : CategoryTheory.ComposableArrows C 3
                 φ : Quiver.Hom R₁ R₂
                 hR₁ : R₁.Exact
                 hR₂ : R₂.Exact
                 ⊢ LE.le 0 3
               -/
               /-
                 🎉 no goals
               -/
                                      /-
                                        🎉 no goals
                                      -/
    (h₀ : Epi (app' φ 0)) (h₁ : Mono (app' φ 1)) (h₃ : Mono (app' φ 3)) :
                                                             /-
                                                               🎉 no goals
                                                             -/
          /-
            C : Type u_1
            inst✝¹ : CategoryTheory.Category.{?u.31397, u_1} C
            inst✝ : CategoryTheory.Abelian C
            R₁ R₂ : CategoryTheory.ComposableArrows C 3
            φ : Quiver.Hom R₁ R₂
            hR₁ : R₁.Exact
            hR₂ : R₂.Exact
            h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
            h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
            h₃ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
            ⊢ LE.le 2 3
          -/
    Mono (app' φ 2) :=
          /-
            🎉 no goals
          -/
  mono_of_epi_of_mono_of_mono' φ
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Abelian C
          R₁ R₂ : CategoryTheory.ComposableArrows C 3
          φ : Quiver.Hom R₁ R₂
          hR₁ : R₁.Exact
          hR₂ : R₂.Exact
          h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
          h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
          h₃ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
          ⊢ Eq (R₁.map' 0 2 ⋯ ⋯) 0
        -/
    (by simpa only [R₁.map'_comp 0 1 2] using hR₁.toIsComplex.zero 0)
        /-
          🎉 no goals
        -/
     /-
       C : Type u_1
       inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
       inst✝ : CategoryTheory.Abelian C
       R₁ R₂ : CategoryTheory.ComposableArrows C 3
       φ : Quiver.Hom R₁ R₂
       hR₁ : R₁.Exact
       hR₂ : R₂.Exact
       h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
       h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
       h₃ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
       ⊢ LE.le (HAdd.hAdd 1 2) 3
     -/
     /-
       🎉 no goals
     -/
    (hR₁.exact 1).exact_toComposableArrows (hR₂.exact 0).exact_toComposableArrows h₀ h₁ h₃
                                            /-
                                              🎉 no goals
                                            -/


theorem epi_of_epi_of_epi_of_mono'
                 /-
                   C : Type u_1
                   inst✝¹ : CategoryTheory.Category.{?u.33826, u_1} C
                   inst✝ : CategoryTheory.Abelian C
                   R₁ R₂ : CategoryTheory.ComposableArrows C 3
                   φ : Quiver.Hom R₁ R₂
                   ⊢ LE.le 1 2
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
    (hR₁ : (mk₂ (R₁.map' 1 2) (R₁.map' 2 3)).Exact)
                               /-
                                 🎉 no goals
                               -/
                 /-
                   C : Type u_1
                   inst✝¹ : CategoryTheory.Category.{?u.33826, u_1} C
                   inst✝ : CategoryTheory.Abelian C
                   R₁ R₂ : CategoryTheory.ComposableArrows C 3
                   φ : Quiver.Hom R₁ R₂
                   hR₁ : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯) …
                   ⊢ LE.le 0 1
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
                                                            /-
                                                              🎉 no goals
                                                            -/
    (hR₂ : (mk₂ (R₂.map' 0 1) (R₂.map' 1 2)).Exact) (hR₂' : R₂.map' 1 3 = 0)
                                                            /-
                                                              🎉 no goals
                                                            -/
               /-
                 C : Type u_1
                 inst✝¹ : CategoryTheory.Category.{?u.33826, u_1} C
                 inst✝ : CategoryTheory.Abelian C
                 R₁ R₂ : CategoryTheory.ComposableArrows C 3
                 φ : Quiver.Hom R₁ R₂
                 hR₁ : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯) …
                 hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
                 hR₂' : Eq (R₂.map' 1 3 ⋯ ⋯) 0
                 ⊢ LE.le 0 3
               -/
               /-
                 🎉 no goals
               -/
                                     /-
                                       🎉 no goals
                                     -/
    (h₀ : Epi (app' φ 0)) (h₂ : Epi (app' φ 2)) (h₃ : Mono (app' φ 3)) :
                                                            /-
                                                              🎉 no goals
                                                            -/
         /-
           C : Type u_1
           inst✝¹ : CategoryTheory.Category.{?u.33826, u_1} C
           inst✝ : CategoryTheory.Abelian C
           R₁ R₂ : CategoryTheory.ComposableArrows C 3
           φ : Quiver.Hom R₁ R₂
           hR₁ : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯) …
           hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
           hR₂' : Eq (R₂.map' 1 3 ⋯ ⋯) 0
           h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
           h₂ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
           h₃ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
           ⊢ LE.le 1 3
         -/
    Epi (app' φ 1) := by
         /-
           🎉 no goals
         -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 3
    φ : Quiver.Hom R₁ R₂
    hR₁ : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯) …
    hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
    hR₂' : Eq (R₂.map' 1 3 ⋯ ⋯) 0
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₂ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
    h₃ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
    ⊢ CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
  -/
  rw [epi_iff_surjective_up_to_refinements]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 3
    φ : Quiver.Hom R₁ R₂
    hR₁ : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯) …
    hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
    hR₂' : Eq (R₂.map' 1 3 ⋯ ⋯) 0
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₂ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
    h₃ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
    ⊢ ∀ ⦃A : C⦄ (y : Quiver.Hom A (R₂.obj' 1 ⋯)), Exists fun A' => Exists fun π => …
  -/
  intro A g₁
  obtain ⟨A₁, π₁, _, f₂, h₁⟩ :=
    surjective_up_to_refinements_of_epi (app' φ 2 _) (g₁ ≫ R₂.map' 1 2)
  have h₂ : f₂ ≫ R₁.map' 2 3 = 0 := by
    rw [← cancel_mono (app' φ 3 _), assoc, zero_comp, NatTrans.naturality, ← reassoc_of% h₁,
      ← R₂.map'_comp 1 2 3, hR₂', comp_zero, comp_zero]
  /-
    case intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 3
    φ : Quiver.Hom R₁ R₂
    hR₁ : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯) …
    hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
    hR₂' : Eq (R₂.map' 1 3 ⋯ ⋯) 0
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₂✝ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
    h₃ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
    A : C
    g₁ : Quiver.Hom A (R₂.obj' 1 ⋯)
    A₁ : C
    π₁ : Quiver.Hom A₁ A
    w✝ : CategoryTheory.Epi π₁
    f₂ : Quiver.Hom A₁ (R₁.obj' 2 ⋯)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp π₁ (CategoryTheory.CategoryStruct. …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ (R₁.map' 2 3 ⋯ ⋯)) 0
    ⊢ Exists fun A' => Exists fun π => Exists fun x => Exists fun x => Eq (Categor …
  -/
  obtain ⟨A₂, π₂, _, f₁, h₃⟩ := (hR₁.exact 0).exact_up_to_refinements _ h₂
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 3
    φ : Quiver.Hom R₁ R₂
    hR₁ : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯) …
    hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
    hR₂' : Eq (R₂.map' 1 3 ⋯ ⋯) 0
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₂✝ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
    h₃✝ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
    A : C
    g₁ : Quiver.Hom A (R₂.obj' 1 ⋯)
    A₁ : C
    π₁ : Quiver.Hom A₁ A
    w✝¹ : CategoryTheory.Epi π₁
    f₂ : Quiver.Hom A₁ (R₁.obj' 2 ⋯)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp π₁ (CategoryTheory.CategoryStruct. …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ (R₁.map' 2 3 ⋯ ⋯)) 0
    A₂ : C
    π₂ : Quiver.Hom A₂ A₁
    w✝ : CategoryTheory.Epi π₂
    f₁ : Quiver.Hom A₂ ((CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁ …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp π₂ f₂) (CategoryTheory.CategoryStr …
    ⊢ Exists fun A' => Exists fun π => Exists fun x => Exists fun x => Eq (Categor …
  -/
  dsimp at f₁ h₃
  have h₄ : (π₂ ≫ π₁ ≫ g₁ - f₁ ≫ app' φ 1 _) ≫ R₂.map' 1 2 = 0 := by
    rw [sub_comp, assoc, assoc, assoc, ← NatTrans.naturality, ← reassoc_of% h₃, h₁, sub_self]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 3
    φ : Quiver.Hom R₁ R₂
    hR₁ : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯) …
    hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
    hR₂' : Eq (R₂.map' 1 3 ⋯ ⋯) 0
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₂✝ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
    h₃✝ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
    A : C
    g₁ : Quiver.Hom A (R₂.obj' 1 ⋯)
    A₁ : C
    π₁ : Quiver.Hom A₁ A
    w✝¹ : CategoryTheory.Epi π₁
    f₂ : Quiver.Hom A₁ (R₁.obj' 2 ⋯)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp π₁ (CategoryTheory.CategoryStruct. …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ (R₁.map' 2 3 ⋯ ⋯)) 0
    A₂ : C
    π₂ : Quiver.Hom A₂ A₁
    w✝ : CategoryTheory.Epi π₂
    f₁ : Quiver.Hom A₂ (R₁.obj 1)
    h₃ : Eq (CategoryTheory.CategoryStruct.comp π₂ f₂) (CategoryTheory.CategoryStr …
    h₄ : Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.Categor …
    ⊢ Exists fun A' => Exists fun π => Exists fun x => Exists fun x => Eq (Categor …
  -/
  obtain ⟨A₃, π₃, _, g₀, h₅⟩ := (hR₂.exact 0).exact_up_to_refinements _ h₄
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 3
    φ : Quiver.Hom R₁ R₂
    hR₁ : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯) …
    hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
    hR₂' : Eq (R₂.map' 1 3 ⋯ ⋯) 0
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₂✝ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
    h₃✝ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
    A : C
    g₁ : Quiver.Hom A (R₂.obj' 1 ⋯)
    A₁ : C
    π₁ : Quiver.Hom A₁ A
    w✝² : CategoryTheory.Epi π₁
    f₂ : Quiver.Hom A₁ (R₁.obj' 2 ⋯)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp π₁ (CategoryTheory.CategoryStruct. …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ (R₁.map' 2 3 ⋯ ⋯)) 0
    A₂ : C
    π₂ : Quiver.Hom A₂ A₁
    w✝¹ : CategoryTheory.Epi π₂
    f₁ : Quiver.Hom A₂ (R₁.obj 1)
    h₃ : Eq (CategoryTheory.CategoryStruct.comp π₂ f₂) (CategoryTheory.CategoryStr …
    h₄ : Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.Categor …
    A₃ : C
    π₃ : Quiver.Hom A₃ A₂
    w✝ : CategoryTheory.Epi π₃
    g₀ : Quiver.Hom A₃ ((CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂ …
    h₅ : Eq (CategoryTheory.CategoryStruct.comp π₃ (HSub.hSub (CategoryTheory.Cate …
    ⊢ Exists fun A' => Exists fun π => Exists fun x => Exists fun x => Eq (Categor …
  -/
  dsimp at g₀ h₅
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 3
    φ : Quiver.Hom R₁ R₂
    hR₁ : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯) …
    hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
    hR₂' : Eq (R₂.map' 1 3 ⋯ ⋯) 0
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₂✝ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
    h₃✝ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
    A : C
    g₁ : Quiver.Hom A (R₂.obj' 1 ⋯)
    A₁ : C
    π₁ : Quiver.Hom A₁ A
    w✝² : CategoryTheory.Epi π₁
    f₂ : Quiver.Hom A₁ (R₁.obj' 2 ⋯)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp π₁ (CategoryTheory.CategoryStruct. …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ (R₁.map' 2 3 ⋯ ⋯)) 0
    A₂ : C
    π₂ : Quiver.Hom A₂ A₁
    w✝¹ : CategoryTheory.Epi π₂
    f₁ : Quiver.Hom A₂ (R₁.obj 1)
    h₃ : Eq (CategoryTheory.CategoryStruct.comp π₂ f₂) (CategoryTheory.CategoryStr …
    h₄ : Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.Categor …
    A₃ : C
    π₃ : Quiver.Hom A₃ A₂
    w✝ : CategoryTheory.Epi π₃
    g₀ : Quiver.Hom A₃ (R₂.obj 0)
    h₅ : Eq (CategoryTheory.CategoryStruct.comp π₃ (HSub.hSub (CategoryTheory.Cate …
    ⊢ Exists fun A' => Exists fun π => Exists fun x => Exists fun x => Eq (Categor …
  -/
  rw [comp_sub] at h₅
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 3
    φ : Quiver.Hom R₁ R₂
    hR₁ : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯) …
    hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
    hR₂' : Eq (R₂.map' 1 3 ⋯ ⋯) 0
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₂✝ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
    h₃✝ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
    A : C
    g₁ : Quiver.Hom A (R₂.obj' 1 ⋯)
    A₁ : C
    π₁ : Quiver.Hom A₁ A
    w✝² : CategoryTheory.Epi π₁
    f₂ : Quiver.Hom A₁ (R₁.obj' 2 ⋯)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp π₁ (CategoryTheory.CategoryStruct. …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ (R₁.map' 2 3 ⋯ ⋯)) 0
    A₂ : C
    π₂ : Quiver.Hom A₂ A₁
    w✝¹ : CategoryTheory.Epi π₂
    f₁ : Quiver.Hom A₂ (R₁.obj 1)
    h₃ : Eq (CategoryTheory.CategoryStruct.comp π₂ f₂) (CategoryTheory.CategoryStr …
    h₄ : Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.Categor …
    A₃ : C
    π₃ : Quiver.Hom A₃ A₂
    w✝ : CategoryTheory.Epi π₃
    g₀ : Quiver.Hom A₃ (R₂.obj 0)
    h₅ : Eq (HSub.hSub (CategoryTheory.CategoryStruct.comp π₃ (CategoryTheory.Cate …
    ⊢ Exists fun A' => Exists fun π => Exists fun x => Exists fun x => Eq (Categor …
  -/
  obtain ⟨A₄, π₄, _, f₀, h₆⟩ := surjective_up_to_refinements_of_epi (app' φ 0 _) g₀
  refine ⟨A₄, π₄ ≫ π₃ ≫ π₂ ≫ π₁, inferInstance,
    π₄ ≫ π₃ ≫ f₁ + f₀ ≫ (by exact R₁.map' 0 1), ?_⟩
  rw [assoc, assoc, assoc, add_comp, assoc, assoc, assoc, NatTrans.naturality,
    ← reassoc_of% h₆, ← h₅, comp_sub]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 3
    φ : Quiver.Hom R₁ R₂
    hR₁ : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯) …
    hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
    hR₂' : Eq (R₂.map' 1 3 ⋯ ⋯) 0
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₂✝ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
    h₃✝ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
    A : C
    g₁ : Quiver.Hom A (R₂.obj' 1 ⋯)
    A₁ : C
    π₁ : Quiver.Hom A₁ A
    w✝³ : CategoryTheory.Epi π₁
    f₂ : Quiver.Hom A₁ (R₁.obj' 2 ⋯)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp π₁ (CategoryTheory.CategoryStruct. …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ (R₁.map' 2 3 ⋯ ⋯)) 0
    A₂ : C
    π₂ : Quiver.Hom A₂ A₁
    w✝² : CategoryTheory.Epi π₂
    f₁ : Quiver.Hom A₂ (R₁.obj 1)
    h₃ : Eq (CategoryTheory.CategoryStruct.comp π₂ f₂) (CategoryTheory.CategoryStr …
    h₄ : Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.Categor …
    A₃ : C
    π₃ : Quiver.Hom A₃ A₂
    w✝¹ : CategoryTheory.Epi π₃
    g₀ : Quiver.Hom A₃ (R₂.obj 0)
    h₅ : Eq (HSub.hSub (CategoryTheory.CategoryStruct.comp π₃ (CategoryTheory.Cate …
    A₄ : C
    π₄ : Quiver.Hom A₄ A₃
    w✝ : CategoryTheory.Epi π₄
    f₀ : Quiver.Hom A₄ (R₁.obj' 0 ⋯)
    h₆ : Eq (CategoryTheory.CategoryStruct.comp π₄ g₀) (CategoryTheory.CategoryStr …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp π₄ (CategoryTheory.CategoryStruct.com …
  -/
  dsimp
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 3
    φ : Quiver.Hom R₁ R₂
    hR₁ : (CategoryTheory.ComposableArrows.mk₂ (R₁.map' 1 2 ⋯ ⋯) (R₁.map' 2 3 ⋯ ⋯) …
    hR₂ : (CategoryTheory.ComposableArrows.mk₂ (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯ ⋯) …
    hR₂' : Eq (R₂.map' 1 3 ⋯ ⋯) 0
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₂✝ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
    h₃✝ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
    A : C
    g₁ : Quiver.Hom A (R₂.obj' 1 ⋯)
    A₁ : C
    π₁ : Quiver.Hom A₁ A
    w✝³ : CategoryTheory.Epi π₁
    f₂ : Quiver.Hom A₁ (R₁.obj' 2 ⋯)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp π₁ (CategoryTheory.CategoryStruct. …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp f₂ (R₁.map' 2 3 ⋯ ⋯)) 0
    A₂ : C
    π₂ : Quiver.Hom A₂ A₁
    w✝² : CategoryTheory.Epi π₂
    f₁ : Quiver.Hom A₂ (R₁.obj 1)
    h₃ : Eq (CategoryTheory.CategoryStruct.comp π₂ f₂) (CategoryTheory.CategoryStr …
    h₄ : Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.Categor …
    A₃ : C
    π₃ : Quiver.Hom A₃ A₂
    w✝¹ : CategoryTheory.Epi π₃
    g₀ : Quiver.Hom A₃ (R₂.obj 0)
    h₅ : Eq (HSub.hSub (CategoryTheory.CategoryStruct.comp π₃ (CategoryTheory.Cate …
    A₄ : C
    π₄ : Quiver.Hom A₄ A₃
    w✝ : CategoryTheory.Epi π₄
    f₀ : Quiver.Hom A₄ (R₁.obj' 0 ⋯)
    h₆ : Eq (CategoryTheory.CategoryStruct.comp π₄ g₀) (CategoryTheory.CategoryStr …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp π₄ (CategoryTheory.CategoryStruct.com …
  -/
  rw [add_sub_cancel]
  /-
    🎉 no goals
  -/


theorem epi_of_epi_of_epi_of_mono (hR₁ : R₁.Exact) (hR₂ : R₂.Exact)
               /-
                 C : Type u_1
                 inst✝¹ : CategoryTheory.Category.{?u.74504, u_1} C
                 inst✝ : CategoryTheory.Abelian C
                 R₁ R₂ : CategoryTheory.ComposableArrows C 3
                 φ : Quiver.Hom R₁ R₂
                 hR₁ : R₁.Exact
                 hR₂ : R₂.Exact
                 ⊢ LE.le 0 3
               -/
               /-
                 🎉 no goals
               -/
                                     /-
                                       🎉 no goals
                                     -/
    (h₀ : Epi (app' φ 0)) (h₂ : Epi (app' φ 2)) (h₃ : Mono (app' φ 3)) :
                                                            /-
                                                              🎉 no goals
                                                            -/
         /-
           C : Type u_1
           inst✝¹ : CategoryTheory.Category.{?u.74504, u_1} C
           inst✝ : CategoryTheory.Abelian C
           R₁ R₂ : CategoryTheory.ComposableArrows C 3
           φ : Quiver.Hom R₁ R₂
           hR₁ : R₁.Exact
           hR₂ : R₂.Exact
           h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
           h₂ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
           h₃ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
           ⊢ LE.le 1 3
         -/
    Epi (app' φ 1) :=
         /-
           🎉 no goals
         -/
                                /-
                                  C : Type u_1
                                  inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                                  inst✝ : CategoryTheory.Abelian C
                                  R₁ R₂ : CategoryTheory.ComposableArrows C 3
                                  φ : Quiver.Hom R₁ R₂
                                  hR₁ : R₁.Exact
                                  hR₂ : R₂.Exact
                                  h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
                                  h₂ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
                                  h₃ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
                                  ⊢ LE.le (HAdd.hAdd 1 2) 3
                                -/
  epi_of_epi_of_epi_of_mono' φ (hR₁.exact 1).exact_toComposableArrows
                                /-
                                  🎉 no goals
                                -/
     /-
       C : Type u_1
       inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
       inst✝ : CategoryTheory.Abelian C
       R₁ R₂ : CategoryTheory.ComposableArrows C 3
       φ : Quiver.Hom R₁ R₂
       hR₁ : R₁.Exact
       hR₂ : R₂.Exact
       h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
       h₂ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
       h₃ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
       ⊢ LE.le (HAdd.hAdd 0 2) 3
     -/
    (hR₂.exact 0).exact_toComposableArrows
     /-
       🎉 no goals
     -/
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Abelian C
          R₁ R₂ : CategoryTheory.ComposableArrows C 3
          φ : Quiver.Hom R₁ R₂
          hR₁ : R₁.Exact
          hR₂ : R₂.Exact
          h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
          h₂ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
          h₃ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
          ⊢ Eq (R₂.map' 1 3 ⋯ ⋯) 0
        -/
    (by simpa only [R₂.map'_comp 1 2 3] using hR₂.toIsComplex.zero 1) h₀ h₂ h₃
        /-
          🎉 no goals
        -/


set_option simprocs false in
/-- The five lemma. -/
                                                          /-
                                                            C : Type u_1
                                                            inst✝¹ : CategoryTheory.Category.{?u.77003, u_1} C
                                                            inst✝ : CategoryTheory.Abelian C
                                                            R₁ R₂ : CategoryTheory.ComposableArrows C 4
                                                            hR₁ : R₁.Exact
                                                            hR₂ : R₂.Exact
                                                            φ : Quiver.Hom R₁ R₂
                                                            ⊢ LE.le 0 4
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
theorem isIso_of_epi_of_isIso_of_isIso_of_mono (h₀ : Epi (app' φ 0)) (h₁ : IsIso (app' φ 1))
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
                 /-
                   C : Type u_1
                   inst✝¹ : CategoryTheory.Category.{?u.77003, u_1} C
                   inst✝ : CategoryTheory.Abelian C
                   R₁ R₂ : CategoryTheory.ComposableArrows C 4
                   hR₁ : R₁.Exact
                   hR₂ : R₂.Exact
                   φ : Quiver.Hom R₁ R₂
                   h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
                   h₁ : CategoryTheory.IsIso (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
                   ⊢ LE.le 3 4
                 -/
                 /-
                   🎉 no goals
                 -/
                                        /-
                                          🎉 no goals
                                        -/
    (h₂ : IsIso (app' φ 3)) (h₃ : Mono (app' φ 4)) : IsIso (app' φ 2) := by
                                                            /-
                                                              🎉 no goals
                                                            -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 4
    hR₁ : R₁.Exact
    hR₂ : R₂.Exact
    φ : Quiver.Hom R₁ R₂
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₁ : CategoryTheory.IsIso (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
    h₂ : CategoryTheory.IsIso (CategoryTheory.ComposableArrows.app' φ 3 ⋯)
    h₃ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 4 ⋯)
    ⊢ CategoryTheory.IsIso (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
  -/
  dsimp at h₀ h₁ h₂ h₃
  have : Mono (app' φ 2) := by
    apply mono_of_epi_of_mono_of_mono (δlastFunctor.map φ) (R₁.exact_iff_δlast.1 hR₁).1
      (R₂.exact_iff_δlast.1 hR₂).1 <;> dsimp <;> infer_instance
  have : Epi (app' φ 2) := by
    apply epi_of_epi_of_epi_of_mono (δ₀Functor.map φ) (R₁.exact_iff_δ₀.1 hR₁).2
      (R₂.exact_iff_δ₀.1 hR₂).2 <;> dsimp <;> infer_instance
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 4
    hR₁ : R₁.Exact
    hR₂ : R₂.Exact
    φ : Quiver.Hom R₁ R₂
    h₀ : CategoryTheory.Epi (φ.app 0)
    h₁ : CategoryTheory.IsIso (φ.app 1)
    h₂ : CategoryTheory.IsIso (φ.app ⟨3, ⋯⟩)
    h₃ : CategoryTheory.Mono (φ.app ⟨4, ⋯⟩)
    this✝ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
    this : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
    ⊢ CategoryTheory.IsIso (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
  -/
  apply isIso_of_mono_of_epi
  /-
    🎉 no goals
  -/


attribute [local simp] Precomp.map


                                        /-
                                          C : Type u_1
                                          inst✝¹ : CategoryTheory.Category.{?u.84318, u_1} C
                                          inst✝ : CategoryTheory.Abelian C
                                          R₁ R₂ : CategoryTheory.ComposableArrows C 2
                                          φ : Quiver.Hom R₁ R₂
                                          ⊢ LE.le 0 2
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
theorem mono_of_epi_of_epi_mono' (hR₁ : R₁.map' 0 2 = 0) (hR₁' : Epi (R₁.map' 1 2))
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
                                /-
                                  C : Type u_1
                                  inst✝¹ : CategoryTheory.Category.{?u.84318, u_1} C
                                  inst✝ : CategoryTheory.Abelian C
                                  R₁ R₂ : CategoryTheory.ComposableArrows C 2
                                  φ : Quiver.Hom R₁ R₂
                                  hR₁ : Eq (R₁.map' 0 2 ⋯ ⋯) 0
                                  hR₁' : CategoryTheory.Epi (R₁.map' 1 2 ⋯ ⋯)
                                  hR₂ : R₂.Exact
                                  ⊢ LE.le 0 2
                                -/
                                /-
                                  🎉 no goals
                                -/
    (hR₂ : R₂.Exact) (h₀ : Epi (app' φ 0)) (h₁ : Mono (app' φ 1)) :
                                                       /-
                                                         🎉 no goals
                                                       -/
          /-
            C : Type u_1
            inst✝¹ : CategoryTheory.Category.{?u.84318, u_1} C
            inst✝ : CategoryTheory.Abelian C
            R₁ R₂ : CategoryTheory.ComposableArrows C 2
            φ : Quiver.Hom R₁ R₂
            hR₁ : Eq (R₁.map' 0 2 ⋯ ⋯) 0
            hR₁' : CategoryTheory.Epi (R₁.map' 1 2 ⋯ ⋯)
            hR₂ : R₂.Exact
            h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
            h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
            ⊢ LE.le 2 2
          -/
    Mono (app' φ 2) := by
          /-
            🎉 no goals
          -/
  let ψ : mk₃ (R₁.map' 0 1) (R₁.map' 1 2) (0 : _ ⟶ R₁.obj' 0) ⟶
    mk₃ (R₂.map' 0 1) (R₂.map' 1 2) (0 : _ ⟶ R₁.obj' 0) := homMk₃ (app' φ 0) (app' φ 1)
      (app' φ 2) (𝟙 _) (naturality' φ 0 1) (naturality' φ 1 2) (by simp)
  refine mono_of_epi_of_mono_of_mono' ψ ?_ (exact₂_mk _ (by simp) ?_)
    (hR₂.exact 0).exact_toComposableArrows h₀ h₁ (by dsimp [ψ]; infer_instance)
    /-
      case refine_1
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      R₁ R₂ : CategoryTheory.ComposableArrows C 2
      φ : Quiver.Hom R₁ R₂
      hR₁ : Eq (R₁.map' 0 2 ⋯ ⋯) 0
      hR₁' : CategoryTheory.Epi (R₁.map' 1 2 ⋯ ⋯)
      hR₂ : R₂.Exact
      h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
      h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
      ψ : Quiver.Hom (CategoryTheory.ComposableArrows.mk₃ (R₁.map' 0 1 ⋯ ⋯) (R₁.map' …
      ⊢ Eq ((CategoryTheory.ComposableArrows.mk₃ (R₁.map' 0 1 ⋯ ⋯) (R₁.map' 1 2 ⋯ ⋯) …
    -/
  · dsimp
    /-
      case refine_1
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      R₁ R₂ : CategoryTheory.ComposableArrows C 2
      φ : Quiver.Hom R₁ R₂
      hR₁ : Eq (R₁.map' 0 2 ⋯ ⋯) 0
      hR₁' : CategoryTheory.Epi (R₁.map' 1 2 ⋯ ⋯)
      hR₂ : R₂.Exact
      h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
      h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
      ψ : Quiver.Hom (CategoryTheory.ComposableArrows.mk₃ (R₁.map' 0 1 ⋯ ⋯) (R₁.map' …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (R₁.map (CategoryTheory.homOfLE ⋯)) ( …
    -/
    rw [← Functor.map_comp]
    /-
      case refine_1
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      R₁ R₂ : CategoryTheory.ComposableArrows C 2
      φ : Quiver.Hom R₁ R₂
      hR₁ : Eq (R₁.map' 0 2 ⋯ ⋯) 0
      hR₁' : CategoryTheory.Epi (R₁.map' 1 2 ⋯ ⋯)
      hR₂ : R₂.Exact
      h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
      h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
      ψ : Quiver.Hom (CategoryTheory.ComposableArrows.mk₃ (R₁.map' 0 1 ⋯ ⋯) (R₁.map' …
      ⊢ Eq (R₁.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.homOfLE ⋯) (C …
    -/
    exact hR₁
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      R₁ R₂ : CategoryTheory.ComposableArrows C 2
      φ : Quiver.Hom R₁ R₂
      hR₁ : Eq (R₁.map' 0 2 ⋯ ⋯) 0
      hR₁' : CategoryTheory.Epi (R₁.map' 1 2 ⋯ ⋯)
      hR₂ : R₂.Exact
      h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
      h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
      ψ : Quiver.Hom (CategoryTheory.ComposableArrows.mk₃ (R₁.map' 0 1 ⋯ ⋯) (R₁.map' …
      ⊢ (CategoryTheory.ShortComplex.mk ((CategoryTheory.ComposableArrows.mk₂ ((Cate …
    -/
  · rw [ShortComplex.exact_iff_epi _ (by simp)]
    /-
      case refine_2
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      R₁ R₂ : CategoryTheory.ComposableArrows C 2
      φ : Quiver.Hom R₁ R₂
      hR₁ : Eq (R₁.map' 0 2 ⋯ ⋯) 0
      hR₁' : CategoryTheory.Epi (R₁.map' 1 2 ⋯ ⋯)
      hR₂ : R₂.Exact
      h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
      h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
      ψ : Quiver.Hom (CategoryTheory.ComposableArrows.mk₃ (R₁.map' 0 1 ⋯ ⋯) (R₁.map' …
      ⊢ CategoryTheory.Epi (CategoryTheory.ShortComplex.mk ((CategoryTheory.Composab …
    -/
    exact hR₁'
    /-
      🎉 no goals
    -/


theorem mono_of_epi_of_epi_of_mono (hR₁ : R₁.Exact) (hR₂ : R₂.Exact)
                 /-
                   C : Type u_1
                   inst✝¹ : CategoryTheory.Category.{?u.113861, u_1} C
                   inst✝ : CategoryTheory.Abelian C
                   R₁ R₂ : CategoryTheory.ComposableArrows C 2
                   φ : Quiver.Hom R₁ R₂
                   hR₁ : R₁.Exact
                   hR₂ : R₂.Exact
                   ⊢ LE.le 1 2
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
    (hR₁' : Epi (R₁.map' 1 2)) (h₀ : Epi (app' φ 0)) (h₁ : Mono (app' φ 1)) :
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
          /-
            C : Type u_1
            inst✝¹ : CategoryTheory.Category.{?u.113861, u_1} C
            inst✝ : CategoryTheory.Abelian C
            R₁ R₂ : CategoryTheory.ComposableArrows C 2
            φ : Quiver.Hom R₁ R₂
            hR₁ : R₁.Exact
            hR₂ : R₂.Exact
            hR₁' : CategoryTheory.Epi (R₁.map' 1 2 ⋯ ⋯)
            h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
            h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
            ⊢ LE.le 2 2
          -/
    Mono (app' φ 2) :=
          /-
            🎉 no goals
          -/
                                 /-
                                   C : Type u_1
                                   inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                                   inst✝ : CategoryTheory.Abelian C
                                   R₁ R₂ : CategoryTheory.ComposableArrows C 2
                                   φ : Quiver.Hom R₁ R₂
                                   hR₁ : R₁.Exact
                                   hR₂ : R₂.Exact
                                   hR₁' : CategoryTheory.Epi (R₁.map' 1 2 ⋯ ⋯)
                                   h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
                                   h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
                                   ⊢ Eq (R₁.map' 0 2 ⋯ ⋯) 0
                                 -/
  mono_of_epi_of_epi_mono' φ (by simpa only [map'_comp R₁ 0 1 2] using hR₁.toIsComplex.zero 0)
                                 /-
                                   🎉 no goals
                                 -/
    hR₁' hR₂ h₀ h₁


                                                            /-
                                                              C : Type u_1
                                                              inst✝¹ : CategoryTheory.Category.{?u.116040, u_1} C
                                                              inst✝ : CategoryTheory.Abelian C
                                                              R₁ R₂ : CategoryTheory.ComposableArrows C 2
                                                              φ : Quiver.Hom R₁ R₂
                                                              hR₁ : R₁.Exact
                                                              ⊢ LE.le 0 2
                                                            -/
                                                            /-
                                                              🎉 no goals
                                                            -/
theorem epi_of_mono_of_epi_of_mono' (hR₁ : R₁.Exact) (hR₂ : R₂.map' 0 2 = 0)
                                                            /-
                                                              🎉 no goals
                                                            -/
                  /-
                    C : Type u_1
                    inst✝¹ : CategoryTheory.Category.{?u.116040, u_1} C
                    inst✝ : CategoryTheory.Abelian C
                    R₁ R₂ : CategoryTheory.ComposableArrows C 2
                    φ : Quiver.Hom R₁ R₂
                    hR₁ : R₁.Exact
                    hR₂ : Eq (R₂.map' 0 2 ⋯ ⋯) 0
                    ⊢ LE.le 0 1
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
    (hR₂' : Mono (R₂.map' 0 1)) (h₀ : Epi (app' φ 1)) (h₁ : Mono (app' φ 2)) :
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
         /-
           C : Type u_1
           inst✝¹ : CategoryTheory.Category.{?u.116040, u_1} C
           inst✝ : CategoryTheory.Abelian C
           R₁ R₂ : CategoryTheory.ComposableArrows C 2
           φ : Quiver.Hom R₁ R₂
           hR₁ : R₁.Exact
           hR₂ : Eq (R₂.map' 0 2 ⋯ ⋯) 0
           hR₂' : CategoryTheory.Mono (R₂.map' 0 1 ⋯ ⋯)
           h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
           h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
           ⊢ LE.le 0 2
         -/
    Epi (app' φ 0) := by
         /-
           🎉 no goals
         -/
  let ψ : mk₃ (0 : R₁.obj' 0 ⟶ _) (R₁.map' 0 1) (R₁.map' 1 2) ⟶
    mk₃ (0 : R₁.obj' 0 ⟶ _) (R₂.map' 0 1) (R₂.map' 1 2) := homMk₃ (𝟙 _) (app' φ 0) (app' φ 1)
      (app' φ 2) (by simp) (naturality' φ 0 1) (naturality' φ 1 2)
  refine epi_of_epi_of_epi_of_mono' ψ (hR₁.exact 0).exact_toComposableArrows
    (exact₂_mk _ (by simp) ?_) ?_ (by dsimp [ψ]; infer_instance) h₀ h₁
    /-
      case refine_1
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      R₁ R₂ : CategoryTheory.ComposableArrows C 2
      φ : Quiver.Hom R₁ R₂
      hR₁ : R₁.Exact
      hR₂ : Eq (R₂.map' 0 2 ⋯ ⋯) 0
      hR₂' : CategoryTheory.Mono (R₂.map' 0 1 ⋯ ⋯)
      h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
      h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
      ψ : Quiver.Hom (CategoryTheory.ComposableArrows.mk₃ 0 (R₁.map' 0 1 ⋯ ⋯) (R₁.ma …
      ⊢ (CategoryTheory.ShortComplex.mk ((CategoryTheory.ComposableArrows.mk₂ ((Cate …
    -/
  · rw [ShortComplex.exact_iff_mono _ (by simp)]
    /-
      case refine_1
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      R₁ R₂ : CategoryTheory.ComposableArrows C 2
      φ : Quiver.Hom R₁ R₂
      hR₁ : R₁.Exact
      hR₂ : Eq (R₂.map' 0 2 ⋯ ⋯) 0
      hR₂' : CategoryTheory.Mono (R₂.map' 0 1 ⋯ ⋯)
      h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
      h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
      ψ : Quiver.Hom (CategoryTheory.ComposableArrows.mk₃ 0 (R₁.map' 0 1 ⋯ ⋯) (R₁.ma …
      ⊢ CategoryTheory.Mono (CategoryTheory.ShortComplex.mk ((CategoryTheory.Composa …
    -/
    exact hR₂'
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      R₁ R₂ : CategoryTheory.ComposableArrows C 2
      φ : Quiver.Hom R₁ R₂
      hR₁ : R₁.Exact
      hR₂ : Eq (R₂.map' 0 2 ⋯ ⋯) 0
      hR₂' : CategoryTheory.Mono (R₂.map' 0 1 ⋯ ⋯)
      h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
      h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
      ψ : Quiver.Hom (CategoryTheory.ComposableArrows.mk₃ 0 (R₁.map' 0 1 ⋯ ⋯) (R₁.ma …
      ⊢ Eq ((CategoryTheory.ComposableArrows.mk₃ 0 (R₂.map' 0 1 ⋯ ⋯) (R₂.map' 1 2 ⋯  …
    -/
  · dsimp
    /-
      case refine_2
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      R₁ R₂ : CategoryTheory.ComposableArrows C 2
      φ : Quiver.Hom R₁ R₂
      hR₁ : R₁.Exact
      hR₂ : Eq (R₂.map' 0 2 ⋯ ⋯) 0
      hR₂' : CategoryTheory.Mono (R₂.map' 0 1 ⋯ ⋯)
      h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
      h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
      ψ : Quiver.Hom (CategoryTheory.ComposableArrows.mk₃ 0 (R₁.map' 0 1 ⋯ ⋯) (R₁.ma …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (R₂.map (CategoryTheory.homOfLE ⋯)) ( …
    -/
    rw [← Functor.map_comp]
    /-
      case refine_2
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      R₁ R₂ : CategoryTheory.ComposableArrows C 2
      φ : Quiver.Hom R₁ R₂
      hR₁ : R₁.Exact
      hR₂ : Eq (R₂.map' 0 2 ⋯ ⋯) 0
      hR₂' : CategoryTheory.Mono (R₂.map' 0 1 ⋯ ⋯)
      h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
      h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
      ψ : Quiver.Hom (CategoryTheory.ComposableArrows.mk₃ 0 (R₁.map' 0 1 ⋯ ⋯) (R₁.ma …
      ⊢ Eq (R₂.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.homOfLE ⋯) (C …
    -/
    exact hR₂
    /-
      🎉 no goals
    -/


theorem epi_of_mono_of_epi_of_mono (hR₁ : R₁.Exact) (hR₂ : R₂.Exact)
                  /-
                    C : Type u_1
                    inst✝¹ : CategoryTheory.Category.{?u.143162, u_1} C
                    inst✝ : CategoryTheory.Abelian C
                    R₁ R₂ : CategoryTheory.ComposableArrows C 2
                    φ : Quiver.Hom R₁ R₂
                    hR₁ : R₁.Exact
                    hR₂ : R₂.Exact
                    ⊢ LE.le 0 1
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
    (hR₂' : Mono (R₂.map' 0 1)) (h₀ : Epi (app' φ 1)) (h₁ : Mono (app' φ 2)) :
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
         /-
           C : Type u_1
           inst✝¹ : CategoryTheory.Category.{?u.143162, u_1} C
           inst✝ : CategoryTheory.Abelian C
           R₁ R₂ : CategoryTheory.ComposableArrows C 2
           φ : Quiver.Hom R₁ R₂
           hR₁ : R₁.Exact
           hR₂ : R₂.Exact
           hR₂' : CategoryTheory.Mono (R₂.map' 0 1 ⋯ ⋯)
           h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
           h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
           ⊢ LE.le 0 2
         -/
    Epi (app' φ 0) :=
         /-
           🎉 no goals
         -/
  epi_of_mono_of_epi_of_mono' φ hR₁
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Abelian C
          R₁ R₂ : CategoryTheory.ComposableArrows C 2
          φ : Quiver.Hom R₁ R₂
          hR₁ : R₁.Exact
          hR₂ : R₂.Exact
          hR₂' : CategoryTheory.Mono (R₂.map' 0 1 ⋯ ⋯)
          h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 1 ⋯)
          h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
          ⊢ Eq (R₂.map' 0 2 ⋯ ⋯) 0
        -/
    (by simpa only [map'_comp R₂ 0 1 2] using hR₂.toIsComplex.zero 0) hR₂' h₀ h₁
        /-
          🎉 no goals
        -/


theorem mono_of_mono_of_mono_of_mono (hR₁ : R₁.Exact)
                  /-
                    C : Type u_1
                    inst✝¹ : CategoryTheory.Category.{?u.145470, u_1} C
                    inst✝ : CategoryTheory.Abelian C
                    R₁ R₂ : CategoryTheory.ComposableArrows C 2
                    φ : Quiver.Hom R₁ R₂
                    hR₁ : R₁.Exact
                    ⊢ LE.le 0 1
                  -/
                  /-
                    🎉 no goals
                  -/
    (hR₂' : Mono (R₂.map' 0 1))
                  /-
                    🎉 no goals
                  -/
                /-
                  C : Type u_1
                  inst✝¹ : CategoryTheory.Category.{?u.145470, u_1} C
                  inst✝ : CategoryTheory.Abelian C
                  R₁ R₂ : CategoryTheory.ComposableArrows C 2
                  φ : Quiver.Hom R₁ R₂
                  hR₁ : R₁.Exact
                  hR₂' : CategoryTheory.Mono (R₂.map' 0 1 ⋯ ⋯)
                  ⊢ LE.le 0 2
                -/
    (h₀ : Mono (app' φ 0))
                /-
                  🎉 no goals
                -/
                /-
                  C : Type u_1
                  inst✝¹ : CategoryTheory.Category.{?u.145470, u_1} C
                  inst✝ : CategoryTheory.Abelian C
                  R₁ R₂ : CategoryTheory.ComposableArrows C 2
                  φ : Quiver.Hom R₁ R₂
                  hR₁ : R₁.Exact
                  hR₂' : CategoryTheory.Mono (R₂.map' 0 1 ⋯ ⋯)
                  h₀ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
                  ⊢ LE.le 2 2
                -/
    (h₁ : Mono (app' φ 2)) :
                /-
                  🎉 no goals
                -/
          /-
            C : Type u_1
            inst✝¹ : CategoryTheory.Category.{?u.145470, u_1} C
            inst✝ : CategoryTheory.Abelian C
            R₁ R₂ : CategoryTheory.ComposableArrows C 2
            φ : Quiver.Hom R₁ R₂
            hR₁ : R₁.Exact
            hR₂' : CategoryTheory.Mono (R₂.map' 0 1 ⋯ ⋯)
            h₀ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
            h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
            ⊢ LE.le 1 2
          -/
    Mono (app' φ 1) := by
          /-
            🎉 no goals
          -/
  let ψ : mk₃ (0 : R₁.obj' 0 ⟶ _) (R₁.map' 0 1) (R₁.map' 1 2) ⟶
    mk₃ (0 : R₁.obj' 0 ⟶ _) (R₂.map' 0 1) (R₂.map' 1 2) := homMk₃ (𝟙 _) (app' φ 0) (app' φ 1)
      (app' φ 2) (by simp) (naturality' φ 0 1) (naturality' φ 1 2)
  refine mono_of_epi_of_mono_of_mono' ψ (by simp)
    (hR₁.exact 0).exact_toComposableArrows
    (exact₂_mk _ (by simp) ?_) (by dsimp [ψ]; infer_instance) h₀ h₁
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 2
    φ : Quiver.Hom R₁ R₂
    hR₁ : R₁.Exact
    hR₂' : CategoryTheory.Mono (R₂.map' 0 1 ⋯ ⋯)
    h₀ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
    ψ : Quiver.Hom (CategoryTheory.ComposableArrows.mk₃ 0 (R₁.map' 0 1 ⋯ ⋯) (R₁.ma …
    ⊢ (CategoryTheory.ShortComplex.mk ((CategoryTheory.ComposableArrows.mk₂ ((Cate …
  -/
  rw [ShortComplex.exact_iff_mono _ (by simp)]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 2
    φ : Quiver.Hom R₁ R₂
    hR₁ : R₁.Exact
    hR₂' : CategoryTheory.Mono (R₂.map' 0 1 ⋯ ⋯)
    h₀ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₁ : CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
    ψ : Quiver.Hom (CategoryTheory.ComposableArrows.mk₃ 0 (R₁.map' 0 1 ⋯ ⋯) (R₁.ma …
    ⊢ CategoryTheory.Mono (CategoryTheory.ShortComplex.mk ((CategoryTheory.Composa …
  -/
  exact hR₂'
  /-
    🎉 no goals
  -/


                                                               /-
                                                                 C : Type u_1
                                                                 inst✝¹ : CategoryTheory.Category.{?u.169639, u_1} C
                                                                 inst✝ : CategoryTheory.Abelian C
                                                                 R₁ R₂ : CategoryTheory.ComposableArrows C 2
                                                                 φ : Quiver.Hom R₁ R₂
                                                                 hR₂ : R₂.Exact
                                                                 ⊢ LE.le 1 2
                                                               -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
theorem epi_of_epi_of_epi_of_epi (hR₂ : R₂.Exact) (hR₁' : Epi (R₁.map' 1 2))
                                                               /-
                                                                 🎉 no goals
                                                               -/
               /-
                 C : Type u_1
                 inst✝¹ : CategoryTheory.Category.{?u.169639, u_1} C
                 inst✝ : CategoryTheory.Abelian C
                 R₁ R₂ : CategoryTheory.ComposableArrows C 2
                 φ : Quiver.Hom R₁ R₂
                 hR₂ : R₂.Exact
                 hR₁' : CategoryTheory.Epi (R₁.map' 1 2 ⋯ ⋯)
                 ⊢ LE.le 0 2
               -/
               /-
                 🎉 no goals
               -/
    (h₀ : Epi (app' φ 0)) (h₁ : Epi (app' φ 2)) :
                                     /-
                                       🎉 no goals
                                     -/
         /-
           C : Type u_1
           inst✝¹ : CategoryTheory.Category.{?u.169639, u_1} C
           inst✝ : CategoryTheory.Abelian C
           R₁ R₂ : CategoryTheory.ComposableArrows C 2
           φ : Quiver.Hom R₁ R₂
           hR₂ : R₂.Exact
           hR₁' : CategoryTheory.Epi (R₁.map' 1 2 ⋯ ⋯)
           h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
           h₁ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
           ⊢ LE.le 1 2
         -/
    Epi (app' φ 1) := by
         /-
           🎉 no goals
         -/
  let ψ : mk₃ (R₁.map' 0 1) (R₁.map' 1 2) (0 : _ ⟶ R₁.obj' 0) ⟶
    mk₃ (R₂.map' 0 1) (R₂.map' 1 2) (0 : _ ⟶ R₁.obj' 0) := homMk₃ (app' φ 0) (app' φ 1)
      (app' φ 2) (𝟙 _) (naturality' φ 0 1) (naturality' φ 1 2) (by simp)
  refine epi_of_epi_of_epi_of_mono' ψ (exact₂_mk _ (by simp) ?_)
    (hR₂.exact 0).exact_toComposableArrows (by simp)
    h₀ h₁ (by dsimp [ψ]; infer_instance)
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 2
    φ : Quiver.Hom R₁ R₂
    hR₂ : R₂.Exact
    hR₁' : CategoryTheory.Epi (R₁.map' 1 2 ⋯ ⋯)
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₁ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
    ψ : Quiver.Hom (CategoryTheory.ComposableArrows.mk₃ (R₁.map' 0 1 ⋯ ⋯) (R₁.map' …
    ⊢ (CategoryTheory.ShortComplex.mk ((CategoryTheory.ComposableArrows.mk₂ ((Cate …
  -/
  rw [ShortComplex.exact_iff_epi _ (by simp)]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    R₁ R₂ : CategoryTheory.ComposableArrows C 2
    φ : Quiver.Hom R₁ R₂
    hR₂ : R₂.Exact
    hR₁' : CategoryTheory.Epi (R₁.map' 1 2 ⋯ ⋯)
    h₀ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 0 ⋯)
    h₁ : CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' φ 2 ⋯)
    ψ : Quiver.Hom (CategoryTheory.ComposableArrows.mk₃ (R₁.map' 0 1 ⋯ ⋯) (R₁.map' …
    ⊢ CategoryTheory.Epi (CategoryTheory.ShortComplex.mk ((CategoryTheory.Composab …
  -/
  exact hR₁'
  /-
    🎉 no goals
  -/


