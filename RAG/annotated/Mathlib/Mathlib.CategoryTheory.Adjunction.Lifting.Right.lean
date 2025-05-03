/-- To show that `η_X` is an equalizer for `(UFη_X, η_UFX)`, it suffices to assume it's always an
equalizer of something (i.e. a regular mono).
-/
def unitEqualises [∀ X : B, RegularMono (adj₁.unit.app X)] (X : B) :
    IsLimit (Fork.ofι (adj₁.unit.app X) (adj₁.unit_naturality _)) :=
  Fork.IsLimit.mk' _ fun s => by
    /-
      A : Type u₁
      B : Type u₂
      C : Type u₃
      inst✝³ : CategoryTheory.Category.{v₁, u₁} A
      inst✝² : CategoryTheory.Category.{v₂, u₂} B
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
      U : CategoryTheory.Functor A B
      F : CategoryTheory.Functor B A
      L : CategoryTheory.Functor C B
      U' : CategoryTheory.Functor A C
      adj₁ : CategoryTheory.Adjunction F U
      adj₂ : CategoryTheory.Adjunction (L.comp F) U'
      inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
      X : B
      s : CategoryTheory.Limits.Fork (U.map (F.map (adj₁.unit.app X))) (adj₁.unit.ap …
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheo …
    -/
    refine ⟨(RegularMono.lift' (adj₁.unit.app X) s.ι ?_).1, ?_, ?_⟩
      /-
        case refine_1
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
        inst✝² : CategoryTheory.Category.{v₂, u₂} B
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor A B
        F : CategoryTheory.Functor B A
        L : CategoryTheory.Functor C B
        U' : CategoryTheory.Functor A C
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction (L.comp F) U'
        inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
        X : B
        s : CategoryTheory.Limits.Fork (U.map (F.map (adj₁.unit.app X))) (adj₁.unit.ap …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp s.ι CategoryTheory.RegularMono.left)  …
      -/
    · rw [← cancel_mono (adj₁.unit.app (RegularMono.Z (adj₁.unit.app X)))]
      /-
        case refine_1
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
        inst✝² : CategoryTheory.Category.{v₂, u₂} B
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor A B
        F : CategoryTheory.Functor B A
        L : CategoryTheory.Functor C B
        U' : CategoryTheory.Functor A C
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction (L.comp F) U'
        inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
        X : B
        s : CategoryTheory.Limits.Fork (U.map (F.map (adj₁.unit.app X))) (adj₁.unit.ap …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp s …
      -/
      rw [assoc, ← adj₁.unit_naturality RegularMono.left]
      /-
        case refine_1
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
        inst✝² : CategoryTheory.Category.{v₂, u₂} B
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor A B
        F : CategoryTheory.Functor B A
        L : CategoryTheory.Functor C B
        U' : CategoryTheory.Functor A C
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction (L.comp F) U'
        inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
        X : B
        s : CategoryTheory.Limits.Fork (U.map (F.map (adj₁.unit.app X))) (adj₁.unit.ap …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp s.ι (CategoryTheory.CategoryStruct.co …
      -/
      dsimp only [Functor.comp_obj]
      erw [← assoc, ← s.condition, assoc, ← U.map_comp, ← F.map_comp, RegularMono.w, F.map_comp,
        U.map_comp, s.condition_assoc, assoc, ← adj₁.unit_naturality RegularMono.right]
      /-
        case refine_1
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
        inst✝² : CategoryTheory.Category.{v₂, u₂} B
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor A B
        F : CategoryTheory.Functor B A
        L : CategoryTheory.Functor C B
        U' : CategoryTheory.Functor A C
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction (L.comp F) U'
        inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
        X : B
        s : CategoryTheory.Limits.Fork (U.map (F.map (adj₁.unit.app X))) (adj₁.unit.ap …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp s.ι (CategoryTheory.CategoryStruct.co …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
        inst✝² : CategoryTheory.Category.{v₂, u₂} B
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor A B
        F : CategoryTheory.Functor B A
        L : CategoryTheory.Functor C B
        U' : CategoryTheory.Functor A C
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction (L.comp F) U'
        inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
        X : B
        s : CategoryTheory.Limits.Fork (U.map (F.map (adj₁.unit.app X))) (adj₁.unit.ap …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (↑(CategoryTheory.RegularMono.lift' ( …
      -/
    · apply (RegularMono.lift' (adj₁.unit.app X) s.ι _).2
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
        U : CategoryTheory.Functor A B
        F : CategoryTheory.Functor B A
        L : CategoryTheory.Functor C B
        U' : CategoryTheory.Functor A C
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction (L.comp F) U'
        inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
        X : B
        s : CategoryTheory.Limits.Fork (U.map (F.map (adj₁.unit.app X))) (adj₁.unit.ap …
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
        U : CategoryTheory.Functor A B
        F : CategoryTheory.Functor B A
        L : CategoryTheory.Functor C B
        U' : CategoryTheory.Functor A C
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction (L.comp F) U'
        inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
        X : B
        s : CategoryTheory.Limits.Fork (U.map (F.map (adj₁.unit.app X))) (adj₁.unit.ap …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        hm : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ofι  …
        ⊢ Eq m ↑(CategoryTheory.RegularMono.lift' (adj₁.unit.app X) s.ι ⋯)
      -/
      rw [← cancel_mono (adj₁.unit.app X)]
      /-
        case refine_3
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
        inst✝² : CategoryTheory.Category.{v₂, u₂} B
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor A B
        F : CategoryTheory.Functor B A
        L : CategoryTheory.Functor C B
        U' : CategoryTheory.Functor A C
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction (L.comp F) U'
        inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
        X : B
        s : CategoryTheory.Limits.Fork (U.map (F.map (adj₁.unit.app X))) (adj₁.unit.ap …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        hm : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ofι  …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (adj₁.unit.app X)) (CategoryTheory. …
      -/
      apply hm.trans (RegularMono.lift' (adj₁.unit.app X) s.ι _).2.symm
      /-
        🎉 no goals
      -/


/-- (Implementation)
To construct the right adjoint, we use the equalizer of `U' F η_X` with the composite

`U' F X ⟶ U' F L U' F X ⟶ U' F U F L U' F X ⟶ U' F U F X`

where the first morphism is `ι_U'FX`, the second is `U' F η_LU'FX` and the third is `U' F U δ_FX`.
We will show that this equalizer exists and that it forms the object map for a right adjoint to `L`.
-/
def otherMap (X : B) : U'.obj (F.obj X) ⟶  U'.obj (F.obj (U.obj (F.obj X))) :=
  adj₂.unit.app _ ≫ U'.map (F.map (adj₁.unit.app _ ≫ (U.map (adj₂.counit.app _))))


/-- `(U'Fη_X, otherMap X)` is a coreflexive pair: in particular if `C` has coreflexive equalizers
then this pair has an equalizer.
-/
instance (X : B) :
    IsCoreflexivePair (U'.map (F.map (adj₁.unit.app X))) (otherMap _ _ adj₁ adj₂ X) :=
  IsCoreflexivePair.mk' (U'.map (adj₁.counit.app (F.obj X)))
        /-
          A : Type u₁
          B : Type u₂
          C : Type u₃
          inst✝² : CategoryTheory.Category.{v₁, u₁} A
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
          inst✝ : CategoryTheory.Category.{v₃, u₃} C
          U : CategoryTheory.Functor A B
          F : CategoryTheory.Functor B A
          L : CategoryTheory.Functor C B
          U' : CategoryTheory.Functor A C
          adj₁ : CategoryTheory.Adjunction F U
          adj₂ : CategoryTheory.Adjunction (L.comp F) U'
          X : B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (U'.map (F.map (adj₁.unit.app X))) (U …
        -/
    (by simp [← Functor.map_comp])
        /-
          🎉 no goals
        -/
        /-
          A : Type u₁
          B : Type u₂
          C : Type u₃
          inst✝² : CategoryTheory.Category.{v₁, u₁} A
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} B
          inst✝ : CategoryTheory.Category.{v₃, u₃} C
          U : CategoryTheory.Functor A B
          F : CategoryTheory.Functor B A
          L : CategoryTheory.Functor C B
          U' : CategoryTheory.Functor A C
          adj₁ : CategoryTheory.Adjunction F U
          adj₂ : CategoryTheory.Adjunction (L.comp F) U'
          X : B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.LiftRightAdjoint.othe …
        -/
    (by simp only [otherMap, assoc, ← Functor.map_comp]; simp)
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- Construct the object part of the desired right adjoint as the equalizer of `U'Fη_Y` with
`otherMap`.
-/
noncomputable def constructRightAdjointObj (Y : B) : C :=
  equalizer (U'.map (F.map (adj₁.unit.app Y))) (otherMap _ _ adj₁ adj₂ Y)


/-- The homset equivalence which helps show that `L` is a left adjoint. -/
@[simps!]
noncomputable def constructRightAdjointEquiv [∀ X : B, RegularMono (adj₁.unit.app X)] (Y : C)
    (X : B) : (Y ⟶ constructRightAdjointObj _ _ adj₁ adj₂ X) ≃ (L.obj Y ⟶ X) :=
  calc
    (Y ⟶ constructRightAdjointObj _ _ adj₁ adj₂ X) ≃
        { f : Y ⟶ U'.obj (F.obj X) //
          f ≫ U'.map (F.map (adj₁.unit.app X)) = f ≫ (otherMap _ _ adj₁ adj₂ X) } :=
      Fork.IsLimit.homIso (limit.isLimit _) _
    _ ≃ { g : F.obj (L.obj Y) ⟶ F.obj X // F.map (adj₁.unit.app _≫ U.map g) =
        g ≫ F.map (adj₁.unit.app _) } := by
      /-
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
        inst✝³ : CategoryTheory.Category.{v₂, u₂} B
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor A B
        F : CategoryTheory.Functor B A
        L : CategoryTheory.Functor C B
        U' : CategoryTheory.Functor A C
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction (L.comp F) U'
        inst✝¹ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
        inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
        Y : C
        X : B
        ⊢ Equiv (Subtype fun f => Eq (CategoryTheory.CategoryStruct.comp f (U'.map (F. …
      -/
      apply (adj₂.homEquiv _ _).symm.subtypeEquiv _
      /-
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
        inst✝³ : CategoryTheory.Category.{v₂, u₂} B
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor A B
        F : CategoryTheory.Functor B A
        L : CategoryTheory.Functor C B
        U' : CategoryTheory.Functor A C
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction (L.comp F) U'
        inst✝¹ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
        inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
        Y : C
        X : B
        ⊢ ∀ (a : Quiver.Hom Y (U'.obj (F.obj X))), Iff (Eq (CategoryTheory.CategoryStr …
      -/
      intro f
      rw [← (adj₂.homEquiv _ _).injective.eq_iff, eq_comm, otherMap,
        ← adj₂.homEquiv_naturality_right_symm, adj₂.homEquiv_unit, ← adj₂.unit_naturality_assoc,
        adj₂.homEquiv_counit]
      /-
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
        inst✝³ : CategoryTheory.Category.{v₂, u₂} B
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor A B
        F : CategoryTheory.Functor B A
        L : CategoryTheory.Functor C B
        U' : CategoryTheory.Functor A C
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction (L.comp F) U'
        inst✝¹ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
        inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
        Y : C
        X : B
        f : Quiver.Hom Y (U'.obj (F.obj X))
        ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp (adj₂.unit.app Y) (CategoryTheor …
      -/
      simp
      /-
        🎉 no goals
      -/
    _ ≃ { z : L.obj Y ⟶ U.obj (F.obj X) //
        z ≫ U.map (F.map (adj₁.unit.app X)) = z ≫ adj₁.unit.app (U.obj (F.obj X)) } := by
      /-
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
        inst✝³ : CategoryTheory.Category.{v₂, u₂} B
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor A B
        F : CategoryTheory.Functor B A
        L : CategoryTheory.Functor C B
        U' : CategoryTheory.Functor A C
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction (L.comp F) U'
        inst✝¹ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
        inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
        Y : C
        X : B
        ⊢ Equiv (Subtype fun g => Eq (F.map (CategoryTheory.CategoryStruct.comp (adj₁. …
      -/
      apply (adj₁.homEquiv _ _).subtypeEquiv
      /-
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
        inst✝³ : CategoryTheory.Category.{v₂, u₂} B
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor A B
        F : CategoryTheory.Functor B A
        L : CategoryTheory.Functor C B
        U' : CategoryTheory.Functor A C
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction (L.comp F) U'
        inst✝¹ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
        inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
        Y : C
        X : B
        ⊢ ∀ (a : Quiver.Hom (F.obj (L.obj Y)) (F.obj X)), Iff (Eq (F.map (CategoryTheo …
      -/
      intro g
      rw [← (adj₁.homEquiv _ _).injective.eq_iff, adj₁.homEquiv_unit,
        adj₁.homEquiv_unit, adj₁.homEquiv_unit, eq_comm]
      /-
        A : Type u₁
        B : Type u₂
        C : Type u₃
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
        inst✝³ : CategoryTheory.Category.{v₂, u₂} B
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        U : CategoryTheory.Functor A B
        F : CategoryTheory.Functor B A
        L : CategoryTheory.Functor C B
        U' : CategoryTheory.Functor A C
        adj₁ : CategoryTheory.Adjunction F U
        adj₂ : CategoryTheory.Adjunction (L.comp F) U'
        inst✝¹ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
        inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
        Y : C
        X : B
        g : Quiver.Hom (F.obj (L.obj Y)) (F.obj X)
        ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp (adj₁.unit.app ((CategoryTheory. …
      -/
      simp
      /-
        🎉 no goals
      -/
    _ ≃ (L.obj Y ⟶ X) := (Fork.IsLimit.homIso (unitEqualises adj₁ X) _).symm


/-- Construct the right adjoint to `L`, with object map `constructRightAdjointObj`. -/
noncomputable def constructRightAdjoint [∀ X : B, RegularMono (adj₁.unit.app X)] : B ⥤ C := by
  refine Adjunction.rightAdjointOfEquiv
    (fun X Y => (constructRightAdjointEquiv L _ adj₁ adj₂ X Y).symm) ?_
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
    inst✝³ : CategoryTheory.Category.{v₂, u₂} B
    inst✝² : CategoryTheory.Category.{v₃, u₃} C
    U : CategoryTheory.Functor A B
    F : CategoryTheory.Functor B A
    L : CategoryTheory.Functor C B
    U' : CategoryTheory.Functor A C
    adj₁ : CategoryTheory.Adjunction F U
    adj₂ : CategoryTheory.Adjunction (L.comp F) U'
    inst✝¹ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
    inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
    ⊢ ∀ (X' X : C) (Y : B) (f : Quiver.Hom X' X) (g : Quiver.Hom (L.obj X) Y), Eq  …
  -/
  intro X Y Y' g h
  rw [constructRightAdjointEquiv_symm_apply, constructRightAdjointEquiv_symm_apply,
    Equiv.symm_apply_eq, Subtype.ext_iff]
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
    inst✝³ : CategoryTheory.Category.{v₂, u₂} B
    inst✝² : CategoryTheory.Category.{v₃, u₃} C
    U : CategoryTheory.Functor A B
    F : CategoryTheory.Functor B A
    L : CategoryTheory.Functor C B
    U' : CategoryTheory.Functor A C
    adj₁ : CategoryTheory.Adjunction F U
    adj₂ : CategoryTheory.Adjunction (L.comp F) U'
    inst✝¹ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
    inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
    X Y : C
    Y' : B
    g : Quiver.Hom X Y
    h : Quiver.Hom (L.obj Y) Y'
    ⊢ Eq ↑⟨(adj₂.homEquiv X (F.obj Y')) ((adj₁.homEquiv (L.obj X) (F.obj Y')).symm …
  -/
  dsimp
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
    inst✝³ : CategoryTheory.Category.{v₂, u₂} B
    inst✝² : CategoryTheory.Category.{v₃, u₃} C
    U : CategoryTheory.Functor A B
    F : CategoryTheory.Functor B A
    L : CategoryTheory.Functor C B
    U' : CategoryTheory.Functor A C
    adj₁ : CategoryTheory.Adjunction F U
    adj₂ : CategoryTheory.Adjunction (L.comp F) U'
    inst✝¹ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
    inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
    X Y : C
    Y' : B
    g : Quiver.Hom X Y
    h : Quiver.Hom (L.obj Y) Y'
    ⊢ Eq ((adj₂.homEquiv X (F.obj Y')) ((adj₁.homEquiv (L.obj X) (F.obj Y')).symm  …
  -/
  simp only [Adjunction.homEquiv_unit, Adjunction.homEquiv_counit]
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
    inst✝³ : CategoryTheory.Category.{v₂, u₂} B
    inst✝² : CategoryTheory.Category.{v₃, u₃} C
    U : CategoryTheory.Functor A B
    F : CategoryTheory.Functor B A
    L : CategoryTheory.Functor C B
    U' : CategoryTheory.Functor A C
    adj₁ : CategoryTheory.Adjunction F U
    adj₂ : CategoryTheory.Adjunction (L.comp F) U'
    inst✝¹ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
    inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
    X Y : C
    Y' : B
    g : Quiver.Hom X Y
    h : Quiver.Hom (L.obj Y) Y'
    ⊢ Eq ((adj₂.homEquiv X (F.obj Y')) (CategoryTheory.CategoryStruct.comp (F.map  …
  -/
  erw [Fork.IsLimit.homIso_natural, Fork.IsLimit.homIso_natural]
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
    inst✝³ : CategoryTheory.Category.{v₂, u₂} B
    inst✝² : CategoryTheory.Category.{v₃, u₃} C
    U : CategoryTheory.Functor A B
    F : CategoryTheory.Functor B A
    L : CategoryTheory.Functor C B
    U' : CategoryTheory.Functor A C
    adj₁ : CategoryTheory.Adjunction F U
    adj₂ : CategoryTheory.Adjunction (L.comp F) U'
    inst✝¹ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
    inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
    X Y : C
    Y' : B
    g : Quiver.Hom X Y
    h : Quiver.Hom (L.obj Y) Y'
    ⊢ Eq ((adj₂.homEquiv X (F.obj Y')) (CategoryTheory.CategoryStruct.comp (F.map  …
  -/
  simp only [Fork.ofι_pt, Functor.map_comp, assoc, limit.cone_x]
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
    inst✝³ : CategoryTheory.Category.{v₂, u₂} B
    inst✝² : CategoryTheory.Category.{v₃, u₃} C
    U : CategoryTheory.Functor A B
    F : CategoryTheory.Functor B A
    L : CategoryTheory.Functor C B
    U' : CategoryTheory.Functor A C
    adj₁ : CategoryTheory.Adjunction F U
    adj₂ : CategoryTheory.Adjunction (L.comp F) U'
    inst✝¹ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
    inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
    X Y : C
    Y' : B
    g : Quiver.Hom X Y
    h : Quiver.Hom (L.obj Y) Y'
    ⊢ Eq ((adj₂.homEquiv X (F.obj Y')) (CategoryTheory.CategoryStruct.comp (F.map  …
  -/
  erw [adj₂.homEquiv_naturality_left, Equiv.rightInverse_symm]
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
    inst✝³ : CategoryTheory.Category.{v₂, u₂} B
    inst✝² : CategoryTheory.Category.{v₃, u₃} C
    U : CategoryTheory.Functor A B
    F : CategoryTheory.Functor B A
    L : CategoryTheory.Functor C B
    U' : CategoryTheory.Functor A C
    adj₁ : CategoryTheory.Adjunction F U
    adj₂ : CategoryTheory.Adjunction (L.comp F) U'
    inst✝¹ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
    inst✝ : (X : B) → CategoryTheory.RegularMono (adj₁.unit.app X)
    X Y : C
    Y' : B
    g : Quiver.Hom X Y
    h : Quiver.Hom (L.obj Y) Y'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp g ((adj₂.homEquiv Y (F.obj Y')) (Cate …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The adjoint triangle theorem: Suppose `U : A ⥤ B` has a left adjoint `F` such that each unit
`η_X : X ⟶ UFX` is a regular monomorphism. Then if a category `C` has equalizers of coreflexive
pairs, then a functor `L : C ⥤ B` has a right adjoint if the composite `L ⋙ F` does.

Note the converse is true (with weaker assumptions), by `Adjunction.comp`.
See https://ncatlab.org/nlab/show/adjoint+triangle+theorem
-/
lemma isLeftAdjoint_triangle_lift {U : A ⥤ B} {F : B ⥤ A} (L : C ⥤ B) (adj₁ : F ⊣ U)
    [∀ X, RegularMono (adj₁.unit.app X)] [HasCoreflexiveEqualizers C]
    [(L ⋙ F).IsLeftAdjoint ] : L.IsLeftAdjoint where
  exists_rightAdjoint :=
    ⟨LiftRightAdjoint.constructRightAdjoint L _ adj₁ (Adjunction.ofIsLeftAdjoint _),
      ⟨Adjunction.adjunctionOfEquivRight _ _⟩⟩


/-- If `L ⋙ F` has a right adjoint, the domain of `L` has coreflexive equalizers and `F` is a
comonadic functor, then `L` has a right adjoint.
This is a special case of `isLeftAdjoint_triangle_lift` which is often more useful in practice.
-/
lemma isLeftAdjoint_triangle_lift_comonadic (F : B ⥤ A) [ComonadicLeftAdjoint F] {L : C ⥤ B}
    [HasCoreflexiveEqualizers C] [(L ⋙ F).IsLeftAdjoint] : L.IsLeftAdjoint := by
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor B A
    inst✝² : CategoryTheory.ComonadicLeftAdjoint F
    L : CategoryTheory.Functor C B
    inst✝¹ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
    inst✝ : (L.comp F).IsLeftAdjoint
    ⊢ L.IsLeftAdjoint
  -/
  let L' : _ ⥤ _ := L ⋙ Comonad.comparison (comonadicAdjunction F)
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor B A
    inst✝² : CategoryTheory.ComonadicLeftAdjoint F
    L : CategoryTheory.Functor C B
    inst✝¹ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
    inst✝ : (L.comp F).IsLeftAdjoint
    L' : CategoryTheory.Functor C (CategoryTheory.comonadicAdjunction F).toComonad …
    ⊢ L.IsLeftAdjoint
  -/
  rsuffices : L'.IsLeftAdjoint
  · let this : (L' ⋙ (Comonad.comparison (comonadicAdjunction F)).inv).IsLeftAdjoint := by
      infer_instance
    refine ((Adjunction.ofIsLeftAdjoint
      (L' ⋙ (Comonad.comparison (comonadicAdjunction F)).inv)).ofNatIsoLeft ?_).isLeftAdjoint
    /-
      A : Type u₁
      B : Type u₂
      C : Type u₃
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
      inst✝³ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor B A
      inst✝² : CategoryTheory.ComonadicLeftAdjoint F
      L : CategoryTheory.Functor C B
      inst✝¹ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
      inst✝ : (L.comp F).IsLeftAdjoint
      L' : CategoryTheory.Functor C (CategoryTheory.comonadicAdjunction F).toComonad …
      this✝ : L'.IsLeftAdjoint
      this : (L'.comp (CategoryTheory.Comonad.comparison (CategoryTheory.comonadicAd …
      ⊢ CategoryTheory.Iso (L'.comp (CategoryTheory.Comonad.comparison (CategoryTheo …
    -/
    exact isoWhiskerLeft L (Comonad.comparison _).asEquivalence.unitIso.symm ≪≫ L.leftUnitor
    /-
      🎉 no goals
    -/
  let this : (L' ⋙ Comonad.forget (comonadicAdjunction F).toComonad).IsLeftAdjoint := by
    refine ((Adjunction.ofIsLeftAdjoint (L ⋙ F)).ofNatIsoLeft ?_).isLeftAdjoint
    exact isoWhiskerLeft L (Comonad.comparisonForget (comonadicAdjunction F)).symm
  let this : ∀ X, RegularMono ((Comonad.adj (comonadicAdjunction F).toComonad).unit.app X) := by
    intro X
    simp only [Comonad.adj_unit]
    exact ⟨_, _, _, _, Comonad.beckCoalgebraEqualizer X⟩
  /-
    A : Type u₁
    B : Type u₂
    C : Type u₃
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} B
    inst✝³ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor B A
    inst✝² : CategoryTheory.ComonadicLeftAdjoint F
    L : CategoryTheory.Functor C B
    inst✝¹ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
    inst✝ : (L.comp F).IsLeftAdjoint
    L' : CategoryTheory.Functor C (CategoryTheory.comonadicAdjunction F).toComonad …
    this✝ : (L'.comp (CategoryTheory.comonadicAdjunction F).toComonad.forget).IsLe …
    this : (X : (CategoryTheory.comonadicAdjunction F).toComonad.Coalgebra) → Cate …
    ⊢ L'.IsLeftAdjoint
  -/
  exact isLeftAdjoint_triangle_lift L' (Comonad.adj _)
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

where `U` has a right adjoint, `A` has coreflexive equalizers and `V` has a right adjoint such that
each component of the counit is a regular mono.
Then `Q` has a right adjoint if `L` has a right adjoint.

See https://ncatlab.org/nlab/show/adjoint+lifting+theorem
-/
lemma isLeftAdjoint_square_lift (Q : A ⥤ B) (V : B ⥤ D) (U : A ⥤ C) (L : C ⥤ D)
    (comm : U ⋙ L ≅ Q ⋙ V) [U.IsLeftAdjoint] [V.IsLeftAdjoint] [L.IsLeftAdjoint]
    [∀ X, RegularMono ((Adjunction.ofIsLeftAdjoint V).unit.app X)] [HasCoreflexiveEqualizers A] :
    Q.IsLeftAdjoint :=
  have := ((Adjunction.ofIsLeftAdjoint (U ⋙ L)).ofNatIsoLeft comm).isLeftAdjoint
  isLeftAdjoint_triangle_lift Q (Adjunction.ofIsLeftAdjoint V)


/-- Suppose we have a commutative square of functors

```
      Q
    A → B
  U ↓   ↓ V
    C → D
      R
```

where `U` has a right adjoint, `A` has reflexive equalizers and `V` is comonadic.
Then `Q` has a right adjoint if `L` has a right adjoint.

See https://ncatlab.org/nlab/show/adjoint+lifting+theorem
-/
lemma isLeftAdjoint_square_lift_comonadic (Q : A ⥤ B) (V : B ⥤ D) (U : A ⥤ C) (L : C ⥤ D)
    (comm : U ⋙ L ≅ Q ⋙ V) [U.IsLeftAdjoint] [ComonadicLeftAdjoint V] [L.IsLeftAdjoint]
    [HasCoreflexiveEqualizers A] : Q.IsLeftAdjoint :=
  have := ((Adjunction.ofIsLeftAdjoint (U ⋙ L)).ofNatIsoLeft comm).isLeftAdjoint
  isLeftAdjoint_triangle_lift_comonadic V


