/-- To show that `ε_X` is a coequalizer for `(FUε_X, ε_FUX)`, it suffices to assume it's always a
coequalizer of something (i.e. a regular epi).
-/
def counitCoequalises [∀ X : B, RegularEpi (adj₁.counit.app X)] (X : B) :
    IsColimit (Cofork.ofπ (adj₁.counit.app X) (adj₁.counit_naturality _)) :=
  Cofork.IsColimit.mk' _ fun s => by
    /-
      A : Type u₁
      B : Type u₂
      C : Type u₃
      inst✝³ : CategoryTheory.Category.{v₁, u₁} A
      inst✝² : CategoryTheory.Category.{v₂, u₂} B
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      U : CategoryTheory.Functor B C
      F : CategoryTheory.Functor C B
      R : CategoryTheory.Functor A B
      F' : CategoryTheory.Functor C A
      adj₁ : CategoryTheory.Adjunction F U
      adj₂ : CategoryTheory.Adjunction F' (R.comp U)
      inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
      X : B
      s : CategoryTheory.Limits.Cofork (F.map (U.map (adj₁.counit.app X))) (adj₁.cou …
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory …
    -/
    refine ⟨(RegularEpi.desc' (adj₁.counit.app X) s.π ?_).1, ?_, ?_⟩
      /-
        case refine_1
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
        inst✝² : CategoryTheory.Category.{v₂, u₂} B
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor B C
        F : CategoryTheory.Functor C B
        R : CategoryTheory.Functor A B
        F' : CategoryTheory.Functor C A
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction F' (R.comp U)
        inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
        X : B
        s : CategoryTheory.Limits.Cofork (F.map (U.map (adj₁.counit.app X))) (adj₁.cou …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.RegularEpi.left s.π) ( …
      -/
    · rw [← cancel_epi (adj₁.counit.app (RegularEpi.W (adj₁.counit.app X)))]
      /-
        case refine_1
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
        inst✝² : CategoryTheory.Category.{v₂, u₂} B
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor B C
        F : CategoryTheory.Functor C B
        R : CategoryTheory.Functor A B
        F' : CategoryTheory.Functor C A
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction F' (R.comp U)
        inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
        X : B
        s : CategoryTheory.Limits.Cofork (F.map (U.map (adj₁.counit.app X))) (adj₁.cou …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₁.counit.app (CategoryTheory.Regu …
      -/
      rw [← adj₁.counit_naturality_assoc RegularEpi.left]
      /-
        case refine_1
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
        inst✝² : CategoryTheory.Category.{v₂, u₂} B
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor B C
        F : CategoryTheory.Functor C B
        R : CategoryTheory.Functor A B
        F' : CategoryTheory.Functor C A
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction F' (R.comp U)
        inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
        X : B
        s : CategoryTheory.Limits.Cofork (F.map (U.map (adj₁.counit.app X))) (adj₁.cou …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (U.map CategoryTheory.RegularE …
      -/
      dsimp only [Functor.comp_obj]
      rw [← s.condition, ← F.map_comp_assoc, ← U.map_comp, RegularEpi.w, U.map_comp,
        F.map_comp_assoc, s.condition, ← adj₁.counit_naturality_assoc RegularEpi.right]
      /-
        case refine_2
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
        inst✝² : CategoryTheory.Category.{v₂, u₂} B
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor B C
        F : CategoryTheory.Functor C B
        R : CategoryTheory.Functor A B
        F' : CategoryTheory.Functor C A
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction F' (R.comp U)
        inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
        X : B
        s : CategoryTheory.Limits.Cofork (F.map (U.map (adj₁.counit.app X))) (adj₁.cou …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ (ad …
      -/
    · apply (RegularEpi.desc' (adj₁.counit.app X) s.π _).2
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
        inst✝² : CategoryTheory.Category.{v₂, u₂} B
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor B C
        F : CategoryTheory.Functor C B
        R : CategoryTheory.Functor A B
        F' : CategoryTheory.Functor C A
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction F' (R.comp U)
        inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
        X : B
        s : CategoryTheory.Limits.Cofork (F.map (U.map (adj₁.counit.app X))) (adj₁.cou …
        ⊢ ∀ {m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.Walk …
      -/
    · intro m hm
      /-
        case refine_3
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
        inst✝² : CategoryTheory.Category.{v₂, u₂} B
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor B C
        F : CategoryTheory.Functor C B
        R : CategoryTheory.Functor A B
        F' : CategoryTheory.Functor C A
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction F' (R.comp U)
        inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
        X : B
        s : CategoryTheory.Limits.Cofork (F.map (U.map (adj₁.counit.app X))) (adj₁.cou …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ  …
        ⊢ Eq m ↑(CategoryTheory.RegularEpi.desc' (adj₁.counit.app X) s.π ⋯)
      -/
      rw [← cancel_epi (adj₁.counit.app X)]
      /-
        case refine_3
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
        inst✝² : CategoryTheory.Category.{v₂, u₂} B
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor B C
        F : CategoryTheory.Functor C B
        R : CategoryTheory.Functor A B
        F' : CategoryTheory.Functor C A
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction F' (R.comp U)
        inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
        X : B
        s : CategoryTheory.Limits.Cofork (F.map (U.map (adj₁.counit.app X))) (adj₁.cou …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ  …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj₁.counit.app X) m) (CategoryTheor …
      -/
      apply hm.trans (RegularEpi.desc' (adj₁.counit.app X) s.π _).2.symm
      /-
        🎉 no goals
      -/


/-- (Implementation)
To construct the left adjoint, we use the coequalizer of `F' U ε_Y` with the composite

`F' U F U X ⟶ F' U F U R F U' X ⟶ F' U R F' U X ⟶ F' U X`

where the first morphism is `F' U F ι_UX`, the second is `F' U ε_RF'UX`, and the third is `δ_F'UX`.
We will show that this coequalizer exists and that it forms the object map for a left adjoint to
`R`.
-/
def otherMap (X) : F'.obj (U.obj (F.obj (U.obj X))) ⟶ F'.obj (U.obj X) :=
  F'.map (U.map (F.map (adj₂.unit.app _) ≫ adj₁.counit.app _)) ≫ adj₂.counit.app _


/-- `(F'Uε_X, otherMap X)` is a reflexive pair: in particular if `A` has reflexive coequalizers then
this pair has a coequalizer.
-/
instance (X : B) :
    IsReflexivePair (F'.map (U.map (adj₁.counit.app X))) (otherMap _ _ adj₁ adj₂ X) :=
  IsReflexivePair.mk' (F'.map (adj₁.unit.app (U.obj X)))
    (by
      /-
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝² : CategoryTheory.Category.{v₁, u₁} A
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor B C
        F : CategoryTheory.Functor C B
        R : CategoryTheory.Functor A B
        F' : CategoryTheory.Functor C A
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction F' (R.comp U)
        X : B
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F'.map (adj₁.unit.app (U.obj X))) (F …
      -/
      rw [← F'.map_comp, adj₁.right_triangle_components]
      /-
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝² : CategoryTheory.Category.{v₁, u₁} A
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor B C
        F : CategoryTheory.Functor C B
        R : CategoryTheory.Functor A B
        F' : CategoryTheory.Functor C A
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction F' (R.comp U)
        X : B
        ⊢ Eq (F'.map (CategoryTheory.CategoryStruct.id (U.obj X))) (CategoryTheory.Cat …
      -/
      apply F'.map_id)
      /-
        🎉 no goals
      -/
    (by
      /-
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝² : CategoryTheory.Category.{v₁, u₁} A
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor B C
        F : CategoryTheory.Functor C B
        R : CategoryTheory.Functor A B
        F' : CategoryTheory.Functor C A
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction F' (R.comp U)
        X : B
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F'.map (adj₁.unit.app (U.obj X))) (C …
      -/
      dsimp [otherMap]
      rw [← F'.map_comp_assoc, U.map_comp, adj₁.unit_naturality_assoc,
        adj₁.right_triangle_components, comp_id, adj₂.left_triangle_components])


/-- Construct the object part of the desired left adjoint as the coequalizer of `F'Uε_Y` with
`otherMap`.
-/
noncomputable def constructLeftAdjointObj (Y : B) : A :=
  coequalizer (F'.map (U.map (adj₁.counit.app Y))) (otherMap _ _ adj₁ adj₂ Y)


set_option linter.unusedVariables false in
/-- The homset equivalence which helps show that `R` is a right adjoint. -/
@[simps!] -- Porting note: Originally `@[simps (config := { rhsMd := semireducible })]`
noncomputable def constructLeftAdjointEquiv [∀ X : B, RegularEpi (adj₁.counit.app X)] (Y : A)
    (X : B) : (constructLeftAdjointObj _ _ adj₁ adj₂ X ⟶ Y) ≃ (X ⟶ R.obj Y) :=
  calc
    (constructLeftAdjointObj _ _ adj₁ adj₂ X ⟶ Y) ≃
        { f : F'.obj (U.obj X) ⟶ Y //
          F'.map (U.map (adj₁.counit.app X)) ≫ f = otherMap _ _ adj₁ adj₂ _ ≫ f } :=
      Cofork.IsColimit.homIso (colimit.isColimit _) _
    _ ≃ { g : U.obj X ⟶ U.obj (R.obj Y) //
          U.map (F.map g ≫ adj₁.counit.app _) = U.map (adj₁.counit.app _) ≫ g } := by
      /-
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
        inst✝³ : CategoryTheory.Category.{v₂, u₂} B
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor B C
        F : CategoryTheory.Functor C B
        R : CategoryTheory.Functor A B
        F' : CategoryTheory.Functor C A
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction F' (R.comp U)
        inst✝¹ : CategoryTheory.Limits.HasReflexiveCoequalizers A
        inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
        Y : A
        X : B
        ⊢ Equiv (Subtype fun f => Eq (CategoryTheory.CategoryStruct.comp (F'.map (U.ma …
      -/
      apply (adj₂.homEquiv _ _).subtypeEquiv _
      /-
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
        inst✝³ : CategoryTheory.Category.{v₂, u₂} B
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor B C
        F : CategoryTheory.Functor C B
        R : CategoryTheory.Functor A B
        F' : CategoryTheory.Functor C A
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction F' (R.comp U)
        inst✝¹ : CategoryTheory.Limits.HasReflexiveCoequalizers A
        inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
        Y : A
        X : B
        ⊢ ∀ (a : Quiver.Hom (F'.obj (U.obj X)) Y), Iff (Eq (CategoryTheory.CategoryStr …
      -/
      intro f
      rw [← (adj₂.homEquiv _ _).injective.eq_iff, eq_comm, adj₂.homEquiv_naturality_left,
        otherMap, assoc, adj₂.homEquiv_naturality_left, ← adj₂.counit_naturality,
        adj₂.homEquiv_naturality_left, adj₂.homEquiv_unit, adj₂.right_triangle_components,
        comp_id, Functor.comp_map, ← U.map_comp, assoc, ← adj₁.counit_naturality,
        adj₂.homEquiv_unit, adj₂.homEquiv_unit, F.map_comp, assoc]
      /-
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
        inst✝³ : CategoryTheory.Category.{v₂, u₂} B
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor B C
        F : CategoryTheory.Functor C B
        R : CategoryTheory.Functor A B
        F' : CategoryTheory.Functor C A
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction F' (R.comp U)
        inst✝¹ : CategoryTheory.Limits.HasReflexiveCoequalizers A
        inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
        Y : A
        X : B
        f : Quiver.Hom (F'.obj (U.obj X)) Y
        ⊢ Iff (Eq (U.map (CategoryTheory.CategoryStruct.comp (F.map (adj₂.unit.app (U. …
      -/
      rfl
      /-
        🎉 no goals
      -/
    _ ≃ { z : F.obj (U.obj X) ⟶ R.obj Y // _ } := by
      /-
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
        inst✝³ : CategoryTheory.Category.{v₂, u₂} B
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor B C
        F : CategoryTheory.Functor C B
        R : CategoryTheory.Functor A B
        F' : CategoryTheory.Functor C A
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction F' (R.comp U)
        inst✝¹ : CategoryTheory.Limits.HasReflexiveCoequalizers A
        inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
        Y : A
        X : B
        ⊢ Equiv (Subtype fun g => Eq (U.map (CategoryTheory.CategoryStruct.comp (F.map …
      -/
      apply (adj₁.homEquiv _ _).symm.subtypeEquiv
      /-
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
        inst✝³ : CategoryTheory.Category.{v₂, u₂} B
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor B C
        F : CategoryTheory.Functor C B
        R : CategoryTheory.Functor A B
        F' : CategoryTheory.Functor C A
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction F' (R.comp U)
        inst✝¹ : CategoryTheory.Limits.HasReflexiveCoequalizers A
        inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
        Y : A
        X : B
        ⊢ ∀ (a : Quiver.Hom (U.obj X) (U.obj (R.obj Y))), Iff (Eq (U.map (CategoryTheo …
      -/
      intro g
      rw [← (adj₁.homEquiv _ _).symm.injective.eq_iff, adj₁.homEquiv_counit,
        adj₁.homEquiv_counit, adj₁.homEquiv_counit, F.map_comp, assoc, U.map_comp, F.map_comp,
        assoc, adj₁.counit_naturality, adj₁.counit_naturality_assoc]
      /-
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
        inst✝³ : CategoryTheory.Category.{v₂, u₂} B
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor B C
        F : CategoryTheory.Functor C B
        R : CategoryTheory.Functor A B
        F' : CategoryTheory.Functor C A
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction F' (R.comp U)
        inst✝¹ : CategoryTheory.Limits.HasReflexiveCoequalizers A
        inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
        Y : A
        X : B
        g : Quiver.Hom (U.obj X) (U.obj (R.obj Y))
        ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp (adj₁.counit.app (F.obj (U.obj X …
      -/
      apply eq_comm
      /-
        🎉 no goals
      -/
    _ ≃ (X ⟶ R.obj Y) := (Cofork.IsColimit.homIso (counitCoequalises adj₁ X) _).symm


/-- Construct the left adjoint to `R`, with object map `constructLeftAdjointObj`. -/
noncomputable def constructLeftAdjoint [∀ X : B, RegularEpi (adj₁.counit.app X)] : B ⥤ A := by
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
    inst✝³ : CategoryTheory.Category.{v₂, u₂} B
    inst✝² : CategoryTheory.Category.{v₃, u₃} C
    U : CategoryTheory.Functor B C
    F : CategoryTheory.Functor C B
    R : CategoryTheory.Functor A B
    F' : CategoryTheory.Functor C A
    adj₁ : CategoryTheory.Adjunction F U
    adj₂ : CategoryTheory.Adjunction F' (R.comp U)
    inst✝¹ : CategoryTheory.Limits.HasReflexiveCoequalizers A
    inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
    ⊢ CategoryTheory.Functor B A
  -/
  refine Adjunction.leftAdjointOfEquiv (fun X Y => constructLeftAdjointEquiv R _ adj₁ adj₂ Y X) ?_
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
    inst✝³ : CategoryTheory.Category.{v₂, u₂} B
    inst✝² : CategoryTheory.Category.{v₃, u₃} C
    U : CategoryTheory.Functor B C
    F : CategoryTheory.Functor C B
    R : CategoryTheory.Functor A B
    F' : CategoryTheory.Functor C A
    adj₁ : CategoryTheory.Adjunction F U
    adj₂ : CategoryTheory.Adjunction F' (R.comp U)
    inst✝¹ : CategoryTheory.Limits.HasReflexiveCoequalizers A
    inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
    ⊢ ∀ (X : B) (Y Y' : A) (g : Quiver.Hom Y Y') (h : Quiver.Hom (CategoryTheory.L …
  -/
  intro X Y Y' g h
  rw [constructLeftAdjointEquiv_apply, constructLeftAdjointEquiv_apply,
    Equiv.symm_apply_eq, Subtype.ext_iff]
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
    inst✝³ : CategoryTheory.Category.{v₂, u₂} B
    inst✝² : CategoryTheory.Category.{v₃, u₃} C
    U : CategoryTheory.Functor B C
    F : CategoryTheory.Functor C B
    R : CategoryTheory.Functor A B
    F' : CategoryTheory.Functor C A
    adj₁ : CategoryTheory.Adjunction F U
    adj₂ : CategoryTheory.Adjunction F' (R.comp U)
    inst✝¹ : CategoryTheory.Limits.HasReflexiveCoequalizers A
    inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
    X : B
    Y Y' : A
    g : Quiver.Hom Y Y'
    h : Quiver.Hom (CategoryTheory.LiftLeftAdjoint.constructLeftAdjointObj R F' ad …
    ⊢ Eq ↑⟨(adj₁.homEquiv (U.obj X) (R.obj Y')).symm ((adj₂.homEquiv (U.obj X) Y') …
  -/
  dsimp
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
    inst✝³ : CategoryTheory.Category.{v₂, u₂} B
    inst✝² : CategoryTheory.Category.{v₃, u₃} C
    U : CategoryTheory.Functor B C
    F : CategoryTheory.Functor C B
    R : CategoryTheory.Functor A B
    F' : CategoryTheory.Functor C A
    adj₁ : CategoryTheory.Adjunction F U
    adj₂ : CategoryTheory.Adjunction F' (R.comp U)
    inst✝¹ : CategoryTheory.Limits.HasReflexiveCoequalizers A
    inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
    X : B
    Y Y' : A
    g : Quiver.Hom Y Y'
    h : Quiver.Hom (CategoryTheory.LiftLeftAdjoint.constructLeftAdjointObj R F' ad …
    ⊢ Eq ((adj₁.homEquiv (U.obj X) (R.obj Y')).symm ((adj₂.homEquiv (U.obj X) Y')  …
  -/
  erw [Cofork.IsColimit.homIso_natural, Cofork.IsColimit.homIso_natural]
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
    inst✝³ : CategoryTheory.Category.{v₂, u₂} B
    inst✝² : CategoryTheory.Category.{v₃, u₃} C
    U : CategoryTheory.Functor B C
    F : CategoryTheory.Functor C B
    R : CategoryTheory.Functor A B
    F' : CategoryTheory.Functor C A
    adj₁ : CategoryTheory.Adjunction F U
    adj₂ : CategoryTheory.Adjunction F' (R.comp U)
    inst✝¹ : CategoryTheory.Limits.HasReflexiveCoequalizers A
    inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
    X : B
    Y Y' : A
    g : Quiver.Hom Y Y'
    h : Quiver.Hom (CategoryTheory.LiftLeftAdjoint.constructLeftAdjointObj R F' ad …
    ⊢ Eq ((adj₁.homEquiv (U.obj X) (R.obj Y')).symm ((adj₂.homEquiv (U.obj X) Y')  …
  -/
  erw [adj₂.homEquiv_naturality_right]
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
    inst✝³ : CategoryTheory.Category.{v₂, u₂} B
    inst✝² : CategoryTheory.Category.{v₃, u₃} C
    U : CategoryTheory.Functor B C
    F : CategoryTheory.Functor C B
    R : CategoryTheory.Functor A B
    F' : CategoryTheory.Functor C A
    adj₁ : CategoryTheory.Adjunction F U
    adj₂ : CategoryTheory.Adjunction F' (R.comp U)
    inst✝¹ : CategoryTheory.Limits.HasReflexiveCoequalizers A
    inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
    X : B
    Y Y' : A
    g : Quiver.Hom Y Y'
    h : Quiver.Hom (CategoryTheory.LiftLeftAdjoint.constructLeftAdjointObj R F' ad …
    ⊢ Eq ((adj₁.homEquiv (U.obj X) (R.obj Y')).symm (CategoryTheory.CategoryStruct …
  -/
  simp_rw [Functor.comp_map]
  -- This used to be `simp`, but we need `aesop_cat` after https://github.com/leanprover/lean4/pull/2644
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
    inst✝³ : CategoryTheory.Category.{v₂, u₂} B
    inst✝² : CategoryTheory.Category.{v₃, u₃} C
    U : CategoryTheory.Functor B C
    F : CategoryTheory.Functor C B
    R : CategoryTheory.Functor A B
    F' : CategoryTheory.Functor C A
    adj₁ : CategoryTheory.Adjunction F U
    adj₂ : CategoryTheory.Adjunction F' (R.comp U)
    inst✝¹ : CategoryTheory.Limits.HasReflexiveCoequalizers A
    inst✝ : (X : B) → CategoryTheory.RegularEpi (adj₁.counit.app X)
    X : B
    Y Y' : A
    g : Quiver.Hom Y Y'
    h : Quiver.Hom (CategoryTheory.LiftLeftAdjoint.constructLeftAdjointObj R F' ad …
    ⊢ Eq ((adj₁.homEquiv (U.obj X) (R.obj Y')).symm (CategoryTheory.CategoryStruct …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


/-- The adjoint triangle theorem: Suppose `U : B ⥤ C` has a left adjoint `F` such that each counit
`ε_X : FUX ⟶ X` is a regular epimorphism. Then if a category `A` has coequalizers of reflexive
pairs, then a functor `R : A ⥤ B` has a left adjoint if the composite `R ⋙ U` does.

Note the converse is true (with weaker assumptions), by `Adjunction.comp`.
See https://ncatlab.org/nlab/show/adjoint+triangle+theorem
-/
lemma isRightAdjoint_triangle_lift {U : B ⥤ C} {F : C ⥤ B} (R : A ⥤ B) (adj₁ : F ⊣ U)
    [∀ X : B, RegularEpi (adj₁.counit.app X)] [HasReflexiveCoequalizers A]
    [(R ⋙ U).IsRightAdjoint ] : R.IsRightAdjoint where
  exists_leftAdjoint :=
    ⟨LiftLeftAdjoint.constructLeftAdjoint R _ adj₁ (Adjunction.ofIsRightAdjoint _),
      ⟨Adjunction.adjunctionOfEquivLeft _ _⟩⟩


/-- If `R ⋙ U` has a left adjoint, the domain of `R` has reflexive coequalizers and `U` is a monadic
functor, then `R` has a left adjoint.
This is a special case of `isRightAdjoint_triangle_lift` which is often more useful in practice.
-/
lemma isRightAdjoint_triangle_lift_monadic (U : B ⥤ C) [MonadicRightAdjoint U] {R : A ⥤ B}
    [HasReflexiveCoequalizers A] [(R ⋙ U).IsRightAdjoint] : R.IsRightAdjoint := by
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    U : CategoryTheory.Functor B C
    inst✝² : CategoryTheory.MonadicRightAdjoint U
    R : CategoryTheory.Functor A B
    inst✝¹ : CategoryTheory.Limits.HasReflexiveCoequalizers A
    inst✝ : (R.comp U).IsRightAdjoint
    ⊢ R.IsRightAdjoint
  -/
  let R' : A ⥤ _ := R ⋙ Monad.comparison (monadicAdjunction U)
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    U : CategoryTheory.Functor B C
    inst✝² : CategoryTheory.MonadicRightAdjoint U
    R : CategoryTheory.Functor A B
    inst✝¹ : CategoryTheory.Limits.HasReflexiveCoequalizers A
    inst✝ : (R.comp U).IsRightAdjoint
    R' : CategoryTheory.Functor A (CategoryTheory.monadicAdjunction U).toMonad.Alg …
    ⊢ R.IsRightAdjoint
  -/
  rsuffices : R'.IsRightAdjoint
  · let this : (R' ⋙ (Monad.comparison (monadicAdjunction U)).inv).IsRightAdjoint := by
      infer_instance
    refine ((Adjunction.ofIsRightAdjoint
      (R' ⋙ (Monad.comparison (monadicAdjunction U)).inv)).ofNatIsoRight ?_).isRightAdjoint
    /-
      A : Type u₁
      B : Type u₂
      C : Type u₃
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
      inst✝³ : CategoryTheory.Category.{v₃, u₃} C
      U : CategoryTheory.Functor B C
      inst✝² : CategoryTheory.MonadicRightAdjoint U
      R : CategoryTheory.Functor A B
      inst✝¹ : CategoryTheory.Limits.HasReflexiveCoequalizers A
      inst✝ : (R.comp U).IsRightAdjoint
      R' : CategoryTheory.Functor A (CategoryTheory.monadicAdjunction U).toMonad.Alg …
      this✝ : R'.IsRightAdjoint
      this : (R'.comp (CategoryTheory.Monad.comparison (CategoryTheory.monadicAdjunc …
      ⊢ CategoryTheory.Iso (R'.comp (CategoryTheory.Monad.comparison (CategoryTheory …
    -/
    exact isoWhiskerLeft R (Monad.comparison _).asEquivalence.unitIso.symm ≪≫ R.rightUnitor
    /-
      🎉 no goals
    -/
  let this : (R' ⋙ Monad.forget (monadicAdjunction U).toMonad).IsRightAdjoint := by
    refine ((Adjunction.ofIsRightAdjoint (R ⋙ U)).ofNatIsoRight ?_).isRightAdjoint
    exact isoWhiskerLeft R (Monad.comparisonForget (monadicAdjunction U)).symm
  let this : ∀ X, RegularEpi ((Monad.adj (monadicAdjunction U).toMonad).counit.app X) := by
    intro X
    simp only [Monad.adj_counit]
    exact ⟨_, _, _, _, Monad.beckAlgebraCoequalizer X⟩
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    U : CategoryTheory.Functor B C
    inst✝² : CategoryTheory.MonadicRightAdjoint U
    R : CategoryTheory.Functor A B
    inst✝¹ : CategoryTheory.Limits.HasReflexiveCoequalizers A
    inst✝ : (R.comp U).IsRightAdjoint
    R' : CategoryTheory.Functor A (CategoryTheory.monadicAdjunction U).toMonad.Alg …
    this✝ : (R'.comp (CategoryTheory.monadicAdjunction U).toMonad.forget).IsRightA …
    this : (X : (CategoryTheory.monadicAdjunction U).toMonad.Algebra) → CategoryTh …
    ⊢ R'.IsRightAdjoint
  -/
  exact isRightAdjoint_triangle_lift R' (Monad.adj _)
  /-
    🎉 no goals
  -/


/-- Suppose we have a commutative square of functors

```
      Q
    A → B
  U ↓   ↓ V
    C → D
      R
```

where `U` has a left adjoint, `A` has reflexive coequalizers and `V` has a left adjoint such that
each component of the counit is a regular epi.
Then `Q` has a left adjoint if `R` has a left adjoint.

See https://ncatlab.org/nlab/show/adjoint+lifting+theorem
-/
lemma isRightAdjoint_square_lift (Q : A ⥤ B) (V : B ⥤ D) (U : A ⥤ C) (R : C ⥤ D)
    (comm : U ⋙ R ≅ Q ⋙ V) [U.IsRightAdjoint] [V.IsRightAdjoint] [R.IsRightAdjoint]
    [∀ X, RegularEpi ((Adjunction.ofIsRightAdjoint V).counit.app X)] [HasReflexiveCoequalizers A] :
    Q.IsRightAdjoint :=
  have := ((Adjunction.ofIsRightAdjoint (U ⋙ R)).ofNatIsoRight comm).isRightAdjoint
  isRightAdjoint_triangle_lift Q (Adjunction.ofIsRightAdjoint V)


/-- Suppose we have a commutative square of functors

```
      Q
    A → B
  U ↓   ↓ V
    C → D
      R
```

where `U` has a left adjoint, `A` has reflexive coequalizers and `V` is monadic.
Then `Q` has a left adjoint if `R` has a left adjoint.

See https://ncatlab.org/nlab/show/adjoint+lifting+theorem
-/
lemma isRightAdjoint_square_lift_monadic (Q : A ⥤ B) (V : B ⥤ D) (U : A ⥤ C) (R : C ⥤ D)
    (comm : U ⋙ R ≅ Q ⋙ V) [U.IsRightAdjoint] [MonadicRightAdjoint V] [R.IsRightAdjoint]
    [HasReflexiveCoequalizers A] : Q.IsRightAdjoint :=
  have := ((Adjunction.ofIsRightAdjoint (U ⋙ R)).ofNatIsoRight comm).isRightAdjoint
  isRightAdjoint_triangle_lift_monadic V


