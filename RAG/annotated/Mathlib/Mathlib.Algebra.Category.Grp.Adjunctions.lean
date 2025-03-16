/-- The free functor `Type u ⥤ AddCommGroup` sending a type `X` to the
free abelian group with generators `x : X`.
-/
def free : Type u ⥤ AddCommGrp where
  obj α := of (FreeAbelianGroup α)
  map := FreeAbelianGroup.map
  map_id _ := AddMonoidHom.ext FreeAbelianGroup.map_id_apply
  map_comp _ _ := AddMonoidHom.ext FreeAbelianGroup.map_comp_apply


@[simp]
theorem free_obj_coe {α : Type u} : (free.obj α : Type u) = FreeAbelianGroup α :=
  rfl

-- This currently can't be a `simp` lemma,
-- because `free_obj_coe` will simplify implicit arguments in the LHS.
-- (The `simpNF` linter will, correctly, complain.)

theorem free_map_coe {α β : Type u} {f : α → β} (x : FreeAbelianGroup α) :
    (free.map f) x = f <$> x :=
  rfl


/-- The free-forgetful adjunction for abelian groups.
-/
def adj : free ⊣ forget AddCommGrp.{u} :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun _ _ => FreeAbelianGroup.lift.symm
      -- Porting note (https://github.com/leanprover-community/mathlib4/pull/11041): used to be just `by intros; ext; rfl`.
      homEquiv_naturality_left_symm := by
        /-
          ⊢ ∀ {X' X : Type u} {Y : AddCommGrp} (f : Quiver.Hom X' X) (g : Quiver.Hom X ( …
        -/
        intros
        /-
          X'✝ X✝ : Type u
          Y✝ : AddCommGrp
          f✝ : Quiver.Hom X'✝ X✝
          g✝ : Quiver.Hom X✝ ((CategoryTheory.forget AddCommGrp).obj Y✝)
          ⊢ Eq (((fun x x_1 => FreeAbelianGroup.lift.symm) X'✝ Y✝).symm (CategoryTheory. …
        -/
        ext
        /-
          case w
          X'✝ X✝ : Type u
          Y✝ : AddCommGrp
          f✝ : Quiver.Hom X'✝ X✝
          g✝ : Quiver.Hom X✝ ((CategoryTheory.forget AddCommGrp).obj Y✝)
          x✝ : ↑(AddCommGrp.free.obj X'✝)
          ⊢ Eq ((((fun x x_1 => FreeAbelianGroup.lift.symm) X'✝ Y✝).symm (CategoryTheory …
        -/
        simp only [Equiv.symm_symm]
        /-
          case w
          X'✝ X✝ : Type u
          Y✝ : AddCommGrp
          f✝ : Quiver.Hom X'✝ X✝
          g✝ : Quiver.Hom X✝ ((CategoryTheory.forget AddCommGrp).obj Y✝)
          x✝ : ↑(AddCommGrp.free.obj X'✝)
          ⊢ Eq ((FreeAbelianGroup.lift (CategoryTheory.CategoryStruct.comp f✝ g✝)) x✝) ( …
        -/
        apply FreeAbelianGroup.lift_comp }
        /-
          🎉 no goals
        -/


instance : free.{u}.IsLeftAdjoint :=
  ⟨_, ⟨adj⟩⟩


instance : (forget AddCommGrp.{u}).IsRightAdjoint :=
  ⟨_, ⟨adj⟩⟩


instance : AddCommGrp.free.{u}.IsLeftAdjoint :=
  ⟨_, ⟨adj⟩⟩


instance : (free.{u}).PreservesMonomorphisms where
  preserves {X Y} f _ := by
    /-
      X Y : Type u
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.Mono f
      ⊢ CategoryTheory.Mono (AddCommGrp.free.map f)
    -/
    by_cases hX : IsEmpty X
      /-
        case pos
        X Y : Type u
        f : Quiver.Hom X Y
        x✝ : CategoryTheory.Mono f
        hX : IsEmpty X
        ⊢ CategoryTheory.Mono (AddCommGrp.free.map f)
      -/
    · constructor
      /-
        case pos.right_cancellation
        X Y : Type u
        f : Quiver.Hom X Y
        x✝ : CategoryTheory.Mono f
        hX : IsEmpty X
        ⊢ ∀ {Z : AddCommGrp} (g h : Quiver.Hom Z (AddCommGrp.free.obj X)), Eq (Categor …
      -/
      intros
      apply (IsInitial.isInitialObj free _
        ((Types.initial_iff_empty X).2 hX).some).isZero.eq_of_tgt
      /-
        case neg
        X Y : Type u
        f : Quiver.Hom X Y
        x✝ : CategoryTheory.Mono f
        hX : Not (IsEmpty X)
        ⊢ CategoryTheory.Mono (AddCommGrp.free.map f)
      -/
    · simp only [not_isEmpty_iff] at hX
      /-
        case neg
        X Y : Type u
        f : Quiver.Hom X Y
        x✝ : CategoryTheory.Mono f
        hX : Nonempty X
        ⊢ CategoryTheory.Mono (AddCommGrp.free.map f)
      -/
      have hf : Function.Injective f := by rwa [← mono_iff_injective]
      /-
        case neg
        X Y : Type u
        f : Quiver.Hom X Y
        x✝ : CategoryTheory.Mono f
        hX : Nonempty X
        hf : Function.Injective f
        ⊢ CategoryTheory.Mono (AddCommGrp.free.map f)
      -/
      obtain ⟨g, hg⟩ := hf.hasLeftInverse
      /-
        case neg.intro
        X Y : Type u
        f : Quiver.Hom X Y
        x✝ : CategoryTheory.Mono f
        hX : Nonempty X
        hf : Function.Injective f
        g : Y → X
        hg : Function.LeftInverse g f
        ⊢ CategoryTheory.Mono (AddCommGrp.free.map f)
      -/
      have : IsSplitMono f := IsSplitMono.mk' { retraction := g }
      /-
        case neg.intro
        X Y : Type u
        f : Quiver.Hom X Y
        x✝ : CategoryTheory.Mono f
        hX : Nonempty X
        hf : Function.Injective f
        g : Y → X
        hg : Function.LeftInverse g f
        this : CategoryTheory.IsSplitMono f
        ⊢ CategoryTheory.Mono (AddCommGrp.free.map f)
      -/
      infer_instance
      /-
        🎉 no goals
      -/


/-- The free functor `Type u ⥤ Group` sending a type `X` to the free group with generators `x : X`.
-/
def free : Type u ⥤ Grp where
  obj α := of (FreeGroup α)
  map := FreeGroup.map
  map_id := by
    -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    /-
      ⊢ ∀ (X : Type u), Eq ({ obj := fun α => Grp.of (FreeGroup α), map := fun {X Y} …
    -/
                                                              /-
                                                                🎉 no goals
                                                              -/
    intros; ext1; erw [← FreeGroup.map.unique] <;> intros <;> rfl
                                                              /-
                                                                🎉 no goals
                                                              -/
  map_comp := by
    -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    /-
      ⊢ ∀ {X Y Z : Type u} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ obj := f …
    -/
                                                              /-
                                                                🎉 no goals
                                                              -/
    intros; ext1; erw [← FreeGroup.map.unique] <;> intros <;> rfl
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- The free-forgetful adjunction for groups.
-/
def adj : free ⊣ forget Grp.{u} :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun _ _ => FreeGroup.lift.symm
      -- Porting note (https://github.com/leanprover-community/mathlib4/pull/11041): used to be just `by intros; ext1; rfl`.
      homEquiv_naturality_left_symm := by
        /-
          ⊢ ∀ {X' X : Type u} {Y : Grp} (f : Quiver.Hom X' X) (g : Quiver.Hom X ((Catego …
        -/
        intros
        /-
          X'✝ X✝ : Type u
          Y✝ : Grp
          f✝ : Quiver.Hom X'✝ X✝
          g✝ : Quiver.Hom X✝ ((CategoryTheory.forget Grp).obj Y✝)
          ⊢ Eq (((fun x x_1 => FreeGroup.lift.symm) X'✝ Y✝).symm (CategoryTheory.Categor …
        -/
        ext1
        /-
          case w
          X'✝ X✝ : Type u
          Y✝ : Grp
          f✝ : Quiver.Hom X'✝ X✝
          g✝ : Quiver.Hom X✝ ((CategoryTheory.forget Grp).obj Y✝)
          x✝ : ↑(Grp.free.obj X'✝)
          ⊢ Eq ((((fun x x_1 => FreeGroup.lift.symm) X'✝ Y✝).symm (CategoryTheory.Catego …
        -/
        simp only [Equiv.symm_symm]
        /-
          case w
          X'✝ X✝ : Type u
          Y✝ : Grp
          f✝ : Quiver.Hom X'✝ X✝
          g✝ : Quiver.Hom X✝ ((CategoryTheory.forget Grp).obj Y✝)
          x✝ : ↑(Grp.free.obj X'✝)
          ⊢ Eq ((FreeGroup.lift (CategoryTheory.CategoryStruct.comp f✝ g✝)) x✝) ((Catego …
        -/
        apply Eq.symm
        /-
          case w.h
          X'✝ X✝ : Type u
          Y✝ : Grp
          f✝ : Quiver.Hom X'✝ X✝
          g✝ : Quiver.Hom X✝ ((CategoryTheory.forget Grp).obj Y✝)
          x✝ : ↑(Grp.free.obj X'✝)
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (Grp.free.map f✝) (FreeGroup.lift g✝ …
        -/
        apply FreeGroup.lift.unique
        /-
          case w.h.hg
          X'✝ X✝ : Type u
          Y✝ : Grp
          f✝ : Quiver.Hom X'✝ X✝
          g✝ : Quiver.Hom X✝ ((CategoryTheory.forget Grp).obj Y✝)
          x✝ : ↑(Grp.free.obj X'✝)
          ⊢ ∀ (x : X'✝), Eq ((CategoryTheory.CategoryStruct.comp (Grp.free.map f✝) (Free …
        -/
        intros
        /-
          case w.h.hg
          X'✝ X✝ : Type u
          Y✝ : Grp
          f✝ : Quiver.Hom X'✝ X✝
          g✝ : Quiver.Hom X✝ ((CategoryTheory.forget Grp).obj Y✝)
          x✝¹ : ↑(Grp.free.obj X'✝)
          x✝ : X'✝
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (Grp.free.map f✝) (FreeGroup.lift g✝ …
        -/
        apply FreeGroup.lift.of }
        /-
          🎉 no goals
        -/


instance : (forget Grp.{u}).IsRightAdjoint  :=
  ⟨_, ⟨adj⟩⟩


/-- The abelianization functor `Group ⥤ CommGroup` sending a group `G` to its abelianization `Gᵃᵇ`.
 -/
def abelianize : Grp.{u} ⥤ CommGrp.{u} where
  obj G := CommGrp.of (Abelianization G)
  map f := Abelianization.lift (Abelianization.of.comp f)
  map_id := by
    /-
      ⊢ ∀ (X : Grp), Eq ({ obj := fun G => CommGrp.of (Abelianization ↑G), map := fu …
    -/
    intros; simp only [coe_id]
    /-
      X✝ : Grp
      ⊢ Eq (Abelianization.lift (Abelianization.of.comp (CategoryTheory.CategoryStru …
    -/
    apply (Equiv.apply_eq_iff_eq_symm_apply Abelianization.lift).mpr; rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  map_comp := by
    /-
      ⊢ ∀ {X Y Z : Grp} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ obj := fun  …
    -/
    intros; simp only [coe_comp]
    /-
      X✝ Y✝ Z✝ : Grp
      f✝ : Quiver.Hom X✝ Y✝
      g✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq (Abelianization.lift (Abelianization.of.comp (CategoryTheory.CategoryStru …
    -/
    apply (Equiv.apply_eq_iff_eq_symm_apply Abelianization.lift).mpr; rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- The abelianization-forgetful adjuction from `Group` to `CommGroup`. -/
def abelianizeAdj : abelianize ⊣ forget₂ CommGrp.{u} Grp.{u} :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun _ _ => Abelianization.lift.symm
      -- Porting note (https://github.com/leanprover-community/mathlib4/pull/11041): used to be just `by intros; ext1; rfl`.
      homEquiv_naturality_left_symm := by
        /-
          ⊢ ∀ {X' X : Grp} {Y : CommGrp} (f : Quiver.Hom X' X) (g : Quiver.Hom X ((Categ …
        -/
        intros
        /-
          X'✝ X✝ : Grp
          Y✝ : CommGrp
          f✝ : Quiver.Hom X'✝ X✝
          g✝ : Quiver.Hom X✝ ((CategoryTheory.forget₂ CommGrp Grp).obj Y✝)
          ⊢ Eq (((fun x x_1 => Abelianization.lift.symm) X'✝ Y✝).symm (CategoryTheory.Ca …
        -/
        ext1
        /-
          case w
          X'✝ X✝ : Grp
          Y✝ : CommGrp
          f✝ : Quiver.Hom X'✝ X✝
          g✝ : Quiver.Hom X✝ ((CategoryTheory.forget₂ CommGrp Grp).obj Y✝)
          x✝ : ↑(Grp.abelianize.obj X'✝)
          ⊢ Eq ((((fun x x_1 => Abelianization.lift.symm) X'✝ Y✝).symm (CategoryTheory.C …
        -/
        simp only [Equiv.symm_symm]
        /-
          case w
          X'✝ X✝ : Grp
          Y✝ : CommGrp
          f✝ : Quiver.Hom X'✝ X✝
          g✝ : Quiver.Hom X✝ ((CategoryTheory.forget₂ CommGrp Grp).obj Y✝)
          x✝ : ↑(Grp.abelianize.obj X'✝)
          ⊢ Eq ((Abelianization.lift (CategoryTheory.CategoryStruct.comp f✝ g✝)) x✝) ((C …
        -/
        apply Eq.symm
        /-
          case w.h
          X'✝ X✝ : Grp
          Y✝ : CommGrp
          f✝ : Quiver.Hom X'✝ X✝
          g✝ : Quiver.Hom X✝ ((CategoryTheory.forget₂ CommGrp Grp).obj Y✝)
          x✝ : ↑(Grp.abelianize.obj X'✝)
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (Grp.abelianize.map f✝) (Abelianizat …
        -/
        apply Abelianization.lift.unique
        /-
          case w.h.hφ
          X'✝ X✝ : Grp
          Y✝ : CommGrp
          f✝ : Quiver.Hom X'✝ X✝
          g✝ : Quiver.Hom X✝ ((CategoryTheory.forget₂ CommGrp Grp).obj Y✝)
          x✝ : ↑(Grp.abelianize.obj X'✝)
          ⊢ ∀ (x : ↑X'✝), Eq ((CategoryTheory.CategoryStruct.comp (Grp.abelianize.map f✝ …
        -/
        intros
        /-
          case w.h.hφ
          X'✝ X✝ : Grp
          Y✝ : CommGrp
          f✝ : Quiver.Hom X'✝ X✝
          g✝ : Quiver.Hom X✝ ((CategoryTheory.forget₂ CommGrp Grp).obj Y✝)
          x✝¹ : ↑(Grp.abelianize.obj X'✝)
          x✝ : ↑X'✝
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (Grp.abelianize.map f✝) (Abelianizat …
        -/
        apply Abelianization.lift.of }
        /-
          🎉 no goals
        -/


/-- The functor taking a monoid to its subgroup of units. -/
@[simps]
def MonCat.units : MonCat.{u} ⥤ Grp.{u} where
  obj R := Grp.of Rˣ
  map f := Grp.ofHom <| Units.map f
  map_id _ := MonoidHom.ext fun _ => Units.ext rfl
  map_comp _ _ := MonoidHom.ext fun _ => Units.ext rfl


/-- The forgetful-units adjunction between `Grp` and `MonCat`. -/
def Grp.forget₂MonAdj : forget₂ Grp MonCat ⊣ MonCat.units.{u} := Adjunction.mk' {
  homEquiv := fun _ Y ↦
    { toFun := fun f => MonoidHom.toHomUnits f
      invFun := fun f => (Units.coeHom Y).comp f
      left_inv := fun _ => MonoidHom.ext fun _ => rfl
      right_inv := fun _ => MonoidHom.ext fun _ => Units.ext rfl }
  unit :=
    { app := fun X => { (@toUnits X _).toMonoidHom with }
      naturality := fun _ _ _ => MonoidHom.ext fun _ => Units.ext rfl }
  counit :=
    { app := fun X => Units.coeHom X
                       /-
                         ⊢ ∀ ⦃X Y : MonCat⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.com …
                       -/
      naturality := by intros; exact MonoidHom.ext fun x => rfl } }
                               /-
                                 🎉 no goals
                               -/


instance : MonCat.units.{u}.IsRightAdjoint :=
  ⟨_, ⟨Grp.forget₂MonAdj⟩⟩


/-- The functor taking a monoid to its subgroup of units. -/
@[simps]
def CommMonCat.units : CommMonCat.{u} ⥤ CommGrp.{u} where
  obj R := CommGrp.of Rˣ
  map f := CommGrp.ofHom <| Units.map f
  map_id _ := MonoidHom.ext fun _ => Units.ext rfl
  map_comp _ _ := MonoidHom.ext fun _ => Units.ext rfl


/-- The forgetful-units adjunction between `CommGrp` and `CommMonCat`. -/
def CommGrp.forget₂CommMonAdj : forget₂ CommGrp CommMonCat ⊣ CommMonCat.units.{u} :=
  Adjunction.mk' {
    homEquiv := fun _ Y ↦
      { toFun := fun f => MonoidHom.toHomUnits f
        invFun := fun f => (Units.coeHom Y).comp f
        left_inv := fun _ => MonoidHom.ext fun _ => rfl
        right_inv := fun _ => MonoidHom.ext fun _ => Units.ext rfl }
    unit :=
      { app := fun X => { (@toUnits X _).toMonoidHom with }
        naturality := fun _ _ _ => MonoidHom.ext fun _ => Units.ext rfl }
    counit :=
      { app := fun X => Units.coeHom X
                         /-
                           ⊢ ∀ ⦃X Y : CommMonCat⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
                         -/
        naturality := by intros; exact MonoidHom.ext fun x => rfl } }
                                 /-
                                   🎉 no goals
                                 -/


instance : CommMonCat.units.{u}.IsRightAdjoint :=
  ⟨_, ⟨CommGrp.forget₂CommMonAdj⟩⟩

