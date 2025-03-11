/-- Given candidate data for an exact pairing,
which is sent by a faithful monoidal functor to an exact pairing,
the equations holds automatically. -/
def exactPairingOfFaithful [F.Faithful] {X Y : C} (eval : Y ⊗ X ⟶ 𝟙_ C)
    (coeval : 𝟙_ C ⟶ X ⊗ Y) [ExactPairing (F.obj X) (F.obj Y)]
    (map_eval : F.map eval = (δ F _ _) ≫ ε_ _ _ ≫ ε F)
    (map_coeval : F.map coeval = (η F) ≫ η_ _ _ ≫ μ F _ _) : ExactPairing X Y where
  evaluation' := eval
  coevaluation' := coeval
  evaluation_coevaluation' :=
    F.map_injective <| by
      simp [map_eval, map_coeval, Functor.Monoidal.map_whiskerLeft,
        Functor.Monoidal.map_whiskerRight]
  coevaluation_evaluation' :=
    F.map_injective <| by
      simp [map_eval, map_coeval, Functor.Monoidal.map_whiskerLeft,
        Functor.Monoidal.map_whiskerRight]


/-- Given a pair of objects which are sent by a fully faithful functor to a pair of objects
with an exact pairing, we get an exact pairing.
-/
def exactPairingOfFullyFaithful [F.Full] [F.Faithful] (X Y : C)
    [ExactPairing (F.obj X) (F.obj Y)] : ExactPairing X Y :=
  exactPairingOfFaithful F (F.preimage (δ F _ _ ≫ ε_ _ _ ≫ (ε F)))
                                              /-
                                                C : Type u_1
                                                D : Type u_2
                                                inst✝⁷ : CategoryTheory.Category.{?u.12019, u_1} C
                                                inst✝⁶ : CategoryTheory.Category.{?u.12023, u_2} D
                                                inst✝⁵ : CategoryTheory.MonoidalCategory C
                                                inst✝⁴ : CategoryTheory.MonoidalCategory D
                                                F : CategoryTheory.Functor C D
                                                inst✝³ : F.Monoidal
                                                inst✝² : F.Full
                                                inst✝¹ : F.Faithful
                                                X Y : C
                                                inst✝ : CategoryTheory.ExactPairing (F.obj X) (F.obj Y)
                                                ⊢ Eq (F.map (F.preimage (CategoryTheory.CategoryStruct.comp (CategoryTheory.Fu …
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
    (F.preimage (η F ≫ η_ _ _ ≫ μ F _ _)) (by simp) (by simp)
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- Pull back a left dual along an equivalence. -/
def hasLeftDualOfEquivalence (X : C) [HasLeftDual (F.obj X)] :
    HasLeftDual X where
  leftDual := G.obj (ᘁ(F.obj X))
  exact := by
    letI := exactPairingCongrLeft (X := F.obj (G.obj ᘁ(F.obj X)))
      (X' := ᘁ(F.obj X)) (Y := F.obj X) (adj.toEquivalence.counitIso.app ᘁ(F.obj X))
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.14496, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.14500, u_2} D
      inst✝⁴ : CategoryTheory.MonoidalCategory C
      inst✝³ : CategoryTheory.MonoidalCategory D
      F : CategoryTheory.Functor C D
      inst✝² : F.Monoidal
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝¹ : F.IsEquivalence
      X : C
      inst✝ : CategoryTheory.HasLeftDual (F.obj X)
      this : CategoryTheory.ExactPairing (F.obj (G.obj (CategoryTheory.HasLeftDual.l …
      ⊢ CategoryTheory.ExactPairing (G.obj (CategoryTheory.HasLeftDual.leftDual (F.o …
    -/
    apply exactPairingOfFullyFaithful F
    /-
      🎉 no goals
    -/


/-- Pull back a right dual along an equivalence. -/
def hasRightDualOfEquivalence (X : C) [HasRightDual (F.obj X)] :
    HasRightDual X where
  rightDual := G.obj ((F.obj X)ᘁ)
  exact := by
    letI := exactPairingCongrRight (X := F.obj X) (Y := F.obj (G.obj (F.obj X)ᘁ))
      (Y' := (F.obj X)ᘁ) (adj.toEquivalence.counitIso.app (F.obj X)ᘁ)
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁶ : CategoryTheory.Category.{?u.19549, u_1} C
      inst✝⁵ : CategoryTheory.Category.{?u.19553, u_2} D
      inst✝⁴ : CategoryTheory.MonoidalCategory C
      inst✝³ : CategoryTheory.MonoidalCategory D
      F : CategoryTheory.Functor C D
      inst✝² : F.Monoidal
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝¹ : F.IsEquivalence
      X : C
      inst✝ : CategoryTheory.HasRightDual (F.obj X)
      this : CategoryTheory.ExactPairing (F.obj X) (F.obj (G.obj (CategoryTheory.Has …
      ⊢ CategoryTheory.ExactPairing X (G.obj (CategoryTheory.HasRightDual.rightDual  …
    -/
    apply exactPairingOfFullyFaithful F
    /-
      🎉 no goals
    -/


/-- Pull back a left rigid structure along an equivalence. -/
def leftRigidCategoryOfEquivalence [LeftRigidCategory D] :
    LeftRigidCategory C where leftDual X := hasLeftDualOfEquivalence adj X


/-- Pull back a right rigid structure along an equivalence. -/
def rightRigidCategoryOfEquivalence [RightRigidCategory D] :
    RightRigidCategory C where rightDual X := hasRightDualOfEquivalence adj X


/-- Pull back a rigid structure along an equivalence. -/
def rigidCategoryOfEquivalence [RigidCategory D] : RigidCategory C where
  leftDual X := hasLeftDualOfEquivalence adj X
  rightDual X := hasRightDualOfEquivalence adj X


