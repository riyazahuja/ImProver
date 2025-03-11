lemma right_adjoint_additive [F.Additive] : G.Additive where
                                                             /-
                                                               C : Type u₁
                                                               D : Type u₂
                                                               inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                               inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                                                               inst✝² : CategoryTheory.Preadditive C
                                                               inst✝¹ : CategoryTheory.Preadditive D
                                                               F : CategoryTheory.Functor C D
                                                               G : CategoryTheory.Functor D C
                                                               adj : CategoryTheory.Adjunction F G
                                                               inst✝ : F.Additive
                                                               X Y : D
                                                               f g : Quiver.Hom X Y
                                                               ⊢ Eq ((adj.homEquiv (G.obj X) Y).symm (G.map (HAdd.hAdd f g))) ((adj.homEquiv  …
                                                             -/
  map_add {X Y} f g := (adj.homEquiv _ _).symm.injective (by simp [homEquiv_counit])
                                                             /-
                                                               🎉 no goals
                                                             -/


lemma left_adjoint_additive [G.Additive] : F.Additive where
                                                        /-
                                                          C : Type u₁
                                                          D : Type u₂
                                                          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                          inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                                                          inst✝² : CategoryTheory.Preadditive C
                                                          inst✝¹ : CategoryTheory.Preadditive D
                                                          F : CategoryTheory.Functor C D
                                                          G : CategoryTheory.Functor D C
                                                          adj : CategoryTheory.Adjunction F G
                                                          inst✝ : G.Additive
                                                          X Y : C
                                                          f g : Quiver.Hom X Y
                                                          ⊢ Eq ((adj.homEquiv X (F.obj Y)) (F.map (HAdd.hAdd f g))) ((adj.homEquiv X (F. …
                                                        -/
  map_add {X Y} f g := (adj.homEquiv _ _).injective (by simp [homEquiv_unit])
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- If we have an adjunction `adj : F ⊣ G` of functors between preadditive categories,
and if `F` is additive, then the hom set equivalence upgrades to an `AddEquiv`.
Note that `F` is additive if and only if `G` is, by `Adjunction.right_adjoint_additive` and
`Adjunction.left_adjoint_additive`.
-/
def homAddEquiv (X : C) (Y : D) : AddEquiv (F.obj X ⟶ Y) (X ⟶ G.obj Y) :=
  { adj.homEquiv _ _ with
    map_add' _ _ := by
      /-
        C : Type u₁
        D : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        inst✝² : CategoryTheory.Preadditive C
        inst✝¹ : CategoryTheory.Preadditive D
        F : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        adj : CategoryTheory.Adjunction F G
        inst✝ : F.Additive
        X : C
        Y : D
        x✝¹ x✝ : Quiver.Hom (F.obj X) Y
        ⊢ Eq (__src✝.toFun (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd (__src✝.toFun x✝¹) (__src✝.t …
      -/
      have := adj.right_adjoint_additive
      /-
        C : Type u₁
        D : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        inst✝² : CategoryTheory.Preadditive C
        inst✝¹ : CategoryTheory.Preadditive D
        F : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        adj : CategoryTheory.Adjunction F G
        inst✝ : F.Additive
        X : C
        Y : D
        x✝¹ x✝ : Quiver.Hom (F.obj X) Y
        this : G.Additive
        ⊢ Eq (__src✝.toFun (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd (__src✝.toFun x✝¹) (__src✝.t …
      -/
      simp [homEquiv_apply] }
      /-
        🎉 no goals
      -/


@[simp]
lemma homAddEquiv_apply (X : C) (Y : D) (f : F.obj X ⟶ Y) :
    adj.homAddEquiv X Y f = adj.homEquiv X Y f := rfl


@[simp]
lemma homAddEquiv_symm_apply (X : C) (Y : D) (f : X ⟶ G.obj Y) :
    (adj.homAddEquiv X Y).symm f = (adj.homEquiv X Y).symm f := rfl


@[simp]
lemma homAddEquiv_zero (X : C) (Y : D) : adj.homEquiv X Y 0 = 0 := map_zero (adj.homAddEquiv X Y)


@[simp]
lemma homAddEquiv_add (X : C) (Y : D) (f f' : F.obj X ⟶ Y) :
    adj.homEquiv X Y (f + f') = adj.homEquiv X Y f + adj.homEquiv X Y f' :=
  map_add (adj.homAddEquiv X Y) _ _


@[simp]
lemma homAddEquiv_sub (X : C) (Y : D) (f f' : F.obj X ⟶ Y) :
    adj.homEquiv X Y (f - f') = adj.homEquiv X Y f - adj.homEquiv X Y f' :=
  map_sub (adj.homAddEquiv X Y) _ _


@[simp]
lemma homAddEquiv_neg (X : C) (Y : D) (f : F.obj X ⟶ Y) :
    adj.homEquiv X Y (- f) = - adj.homEquiv X Y f := map_neg (adj.homAddEquiv X Y) _


@[simp]
lemma homAddEquiv_symm_zero (X : C) (Y : D) :
    (adj.homEquiv X Y).symm 0 = 0 := map_zero (adj.homAddEquiv X Y).symm


@[simp]
lemma homAddEquiv_symm_add (X : C) (Y : D) (f f' : X ⟶ G.obj Y) :
    (adj.homEquiv X Y).symm (f + f') = (adj.homEquiv X Y).symm f + (adj.homEquiv X Y).symm f' :=
  map_add (adj.homAddEquiv X Y).symm _ _


@[simp]
lemma homAddEquiv_symm_sub (X : C) (Y : D) (f f' : X ⟶ G.obj Y) :
    (adj.homEquiv X Y).symm (f - f') = (adj.homEquiv X Y).symm f - (adj.homEquiv X Y).symm f' :=
  map_sub (adj.homAddEquiv X Y).symm _ _


@[simp]
lemma homAddEquiv_symm_neg (X : C) (Y : D) (f : X ⟶ G.obj Y) :
    (adj.homEquiv X Y).symm (- f) = - (adj.homEquiv X Y).symm f :=
  map_neg (adj.homAddEquiv X Y).symm _


open Opposite in
/-- If we have an adjunction `adj : F ⊣ G` of functors between preadditive categories,
and if `F` is additive, then the hom set equivalence upgrades to an isomorphism between
`G ⋙ preadditiveYoneda` and `preadditiveYoneda ⋙ F`, once we throw in the necessary
universe lifting functors.
Note that `F` is additive if and only if `G` is, by `Adjunction.right_adjoint_additive` and
`Adjunction.left_adjoint_additive`.
-/
def compPreadditiveYonedaIso :
    G ⋙ preadditiveYoneda ⋙ (whiskeringRight _ _ _).obj AddCommGrp.uliftFunctor.{max v₁ v₂} ≅
      preadditiveYoneda ⋙ (whiskeringLeft _ _ _).obj F.op ⋙
        (whiskeringRight _ _ _).obj AddCommGrp.uliftFunctor.{max v₁ v₂} :=
  NatIso.ofComponents
    (fun Y ↦ NatIso.ofComponents
      (fun X ↦ (AddEquiv.ulift.trans ((adj.homAddEquiv (unop X) Y).symm.trans
        AddEquiv.ulift.symm)).toAddCommGrpIso)
      (fun g ↦ by
        /-
          C : Type u₁
          D : Type u₂
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.Category.{v₂, u₂} D
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.Preadditive D
          F : CategoryTheory.Functor C D
          G : CategoryTheory.Functor D C
          adj : CategoryTheory.Adjunction F G
          inst✝ : F.Additive
          Y : D
          X✝ Y✝ : Opposite C
          g : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((G.comp (CategoryTheory.preadditive …
        -/
        ext ⟨y⟩
        /-
          case w.up
          C : Type u₁
          D : Type u₂
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.Category.{v₂, u₂} D
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.Preadditive D
          F : CategoryTheory.Functor C D
          G : CategoryTheory.Functor D C
          adj : CategoryTheory.Adjunction F G
          inst✝ : F.Additive
          Y : D
          X✝ Y✝ : Opposite C
          g : Quiver.Hom X✝ Y✝
          y : ↑((CategoryTheory.preadditiveYoneda.obj (G.obj Y)).obj X✝)
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((G.comp (CategoryTheory.preadditiv …
        -/
        exact AddEquiv.ulift.injective (adj.homEquiv_naturality_left_symm g.unop y)))
        /-
          🎉 no goals
        -/
    (fun f ↦ by
      /-
        C : Type u₁
        D : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        inst✝² : CategoryTheory.Preadditive C
        inst✝¹ : CategoryTheory.Preadditive D
        F : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        adj : CategoryTheory.Adjunction F G
        inst✝ : F.Additive
        X✝ Y✝ : D
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.comp (CategoryTheory.preadditiveY …
      -/
      ext _ ⟨x⟩
      /-
        case w.h.w.up
        C : Type u₁
        D : Type u₂
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        inst✝² : CategoryTheory.Preadditive C
        inst✝¹ : CategoryTheory.Preadditive D
        F : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        adj : CategoryTheory.Adjunction F G
        inst✝ : F.Additive
        X✝ Y✝ : D
        f : Quiver.Hom X✝ Y✝
        x✝ : Opposite C
        x : ↑((CategoryTheory.preadditiveYoneda.obj (G.obj X✝)).obj x✝)
        ⊢ Eq (((CategoryTheory.CategoryStruct.comp ((G.comp (CategoryTheory.preadditiv …
      -/
      exact AddEquiv.ulift.injective ((adj.homEquiv_naturality_right_symm x f)))
      /-
        🎉 no goals
      -/


lemma compPreadditiveYonedaIso_hom_app_app_apply (X : Cᵒᵖ) (Y : D)
    (a : ULift.{max v₁ v₂, v₁} (Opposite.unop X ⟶ G.obj Y)) :
      ((adj.compPreadditiveYonedaIso.hom.app Y).app X) a =
        ULift.up ((adj.homEquiv (Opposite.unop X) Y).symm (AddEquiv.ulift a)) := rfl


lemma compPreadditiveYonedaIso_inv_app_app_apply (X : Cᵒᵖ) (Y : D)
    (a : ULift.{max v₁ v₂, v₂} (F.obj (Opposite.unop X) ⟶ Y)) :
      ((adj.compPreadditiveYonedaIso.inv.app Y).app X) a =
        ULift.up ((adj.homEquiv (Opposite.unop X) Y) (AddEquiv.ulift a)) := rfl


