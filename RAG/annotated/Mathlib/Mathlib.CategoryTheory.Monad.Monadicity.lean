/-- The "main pair" for an algebra `(A, α)` is the pair of morphisms `(F α, ε_FA)`. It is always a
reflexive pair, and will be used to construct the left adjoint to the comparison functor and show it
is an equivalence.
-/
instance main_pair_reflexive (A : adj.toMonad.Algebra) :
    IsReflexivePair (F.map A.a) (adj.counit.app (F.obj A.A)) := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₁, u₂} D
    G : CategoryTheory.Functor D C
    F : CategoryTheory.Functor C D
    adj : CategoryTheory.Adjunction F G
    A : adj.toMonad.Algebra
    ⊢ CategoryTheory.IsReflexivePair (F.map A.a) (adj.counit.app (F.obj A.A))
  -/
  apply IsReflexivePair.mk' (F.map (adj.unit.app _)) _ _
    /-
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Category.{v₁, u₂} D
      G : CategoryTheory.Functor D C
      F : CategoryTheory.Functor C D
      adj : CategoryTheory.Adjunction F G
      A : adj.toMonad.Algebra
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (adj.unit.app A.A)) (F.map A.a …
    -/
  · rw [← F.map_comp, ← F.map_id]
    /-
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Category.{v₁, u₂} D
      G : CategoryTheory.Functor D C
      F : CategoryTheory.Functor C D
      adj : CategoryTheory.Adjunction F G
      A : adj.toMonad.Algebra
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (adj.unit.app A.A) A.a)) (F.ma …
    -/
    exact congr_arg F.map A.unit
    /-
      🎉 no goals
    -/
    /-
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Category.{v₁, u₂} D
      G : CategoryTheory.Functor D C
      F : CategoryTheory.Functor C D
      adj : CategoryTheory.Adjunction F G
      A : adj.toMonad.Algebra
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (adj.unit.app A.A)) (adj.couni …
    -/
  · rw [adj.left_triangle_components]
    /-
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Category.{v₁, u₂} D
      G : CategoryTheory.Functor D C
      F : CategoryTheory.Functor C D
      adj : CategoryTheory.Adjunction F G
      A : adj.toMonad.Algebra
      ⊢ Eq (CategoryTheory.CategoryStruct.id (F.obj A.A)) (CategoryTheory.CategorySt …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The "main pair" for an algebra `(A, α)` is the pair of morphisms `(F α, ε_FA)`. It is always a
`G`-split pair, and will be used to construct the left adjoint to the comparison functor and show it
is an equivalence.
-/
instance main_pair_G_split (A : adj.toMonad.Algebra) :
    G.IsSplitPair (F.map A.a)
      (adj.counit.app (F.obj A.A)) where
  splittable := ⟨_, _, ⟨beckSplitCoequalizer A⟩⟩


/-- The object function for the left adjoint to the comparison functor. -/
def comparisonLeftAdjointObj (A : adj.toMonad.Algebra)
    [HasCoequalizer (F.map A.a) (adj.counit.app _)] : D :=
  coequalizer (F.map A.a) (adj.counit.app _)


set_option linter.unusedVariables false in
/--
We have a bijection of homsets which will be used to construct the left adjoint to the comparison
functor.
-/
@[simps!]
def comparisonLeftAdjointHomEquiv (A : adj.toMonad.Algebra) (B : D)
    [HasCoequalizer (F.map A.a) (adj.counit.app (F.obj A.A))] :
    (comparisonLeftAdjointObj adj A ⟶ B) ≃ (A ⟶ (comparison adj).obj B) :=
  calc
    (comparisonLeftAdjointObj adj A ⟶ B) ≃ { f : F.obj A.A ⟶ B // _ } :=
      Cofork.IsColimit.homIso (colimit.isColimit _) B
    _ ≃ { g : A.A ⟶ G.obj B // G.map (F.map g) ≫ G.map (adj.counit.app B) = A.a ≫ g } := by
      /-
        C : Type u₁
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
        G : CategoryTheory.Functor D C
        F : CategoryTheory.Functor C D
        adj : CategoryTheory.Adjunction F G
        A : adj.toMonad.Algebra
        B : D
        inst✝ : CategoryTheory.Limits.HasCoequalizer (F.map A.a) (adj.counit.app (F.ob …
        ⊢ Equiv (Subtype fun f => Eq (CategoryTheory.CategoryStruct.comp (F.map A.a) f …
      -/
      refine (adj.homEquiv _ _).subtypeEquiv ?_
      /-
        C : Type u₁
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
        G : CategoryTheory.Functor D C
        F : CategoryTheory.Functor C D
        adj : CategoryTheory.Adjunction F G
        A : adj.toMonad.Algebra
        B : D
        inst✝ : CategoryTheory.Limits.HasCoequalizer (F.map A.a) (adj.counit.app (F.ob …
        ⊢ ∀ (a : Quiver.Hom (F.obj A.A) B), Iff (Eq (CategoryTheory.CategoryStruct.com …
      -/
      intro f
      rw [← (adj.homEquiv _ _).injective.eq_iff, Adjunction.homEquiv_naturality_left,
        adj.homEquiv_unit, adj.homEquiv_unit, G.map_comp]
      /-
        C : Type u₁
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
        G : CategoryTheory.Functor D C
        F : CategoryTheory.Functor C D
        adj : CategoryTheory.Adjunction F G
        A : adj.toMonad.Algebra
        B : D
        inst✝ : CategoryTheory.Limits.HasCoequalizer (F.map A.a) (adj.counit.app (F.ob …
        f : Quiver.Hom (F.obj A.A) B
        ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp A.a (CategoryTheory.CategoryStru …
      -/
      dsimp
      rw [adj.right_triangle_components_assoc, ← G.map_comp, F.map_comp, Category.assoc,
        adj.counit_naturality, adj.left_triangle_components_assoc]
      /-
        C : Type u₁
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
        G : CategoryTheory.Functor D C
        F : CategoryTheory.Functor C D
        adj : CategoryTheory.Adjunction F G
        A : adj.toMonad.Algebra
        B : D
        inst✝ : CategoryTheory.Limits.HasCoequalizer (F.map A.a) (adj.counit.app (F.ob …
        f : Quiver.Hom (F.obj A.A) B
        ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp A.a (CategoryTheory.CategoryStru …
      -/
      apply eq_comm
      /-
        🎉 no goals
      -/
    _ ≃ (A ⟶ (comparison adj).obj B) :=
      { toFun := fun g =>
          { f := _
            h := g.prop }
        invFun := fun f => ⟨f.f, f.h⟩
                                /-
                                  C : Type u₁
                                  D : Type u₂
                                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                  inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
                                  G : CategoryTheory.Functor D C
                                  F : CategoryTheory.Functor C D
                                  adj : CategoryTheory.Adjunction F G
                                  A : adj.toMonad.Algebra
                                  B : D
                                  inst✝ : CategoryTheory.Limits.HasCoequalizer (F.map A.a) (adj.counit.app (F.ob …
                                  g : Subtype fun g => Eq (CategoryTheory.CategoryStruct.comp (G.map (F.map g))  …
                                  ⊢ Eq ((fun f => ⟨f.f, ⋯⟩) ((fun g => { f := ↑g, h := ⋯ }) g)) g
                                -/
        left_inv := fun g => by ext; rfl
                                     /-
                                       🎉 no goals
                                     -/
                                 /-
                                   C : Type u₁
                                   D : Type u₂
                                   inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                   inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
                                   G : CategoryTheory.Functor D C
                                   F : CategoryTheory.Functor C D
                                   adj : CategoryTheory.Adjunction F G
                                   A : adj.toMonad.Algebra
                                   B : D
                                   inst✝ : CategoryTheory.Limits.HasCoequalizer (F.map A.a) (adj.counit.app (F.ob …
                                   f : Quiver.Hom A ((CategoryTheory.Monad.comparison adj).obj B)
                                   ⊢ Eq ((fun g => { f := ↑g, h := ⋯ }) ((fun f => ⟨f.f, ⋯⟩) f)) f
                                 -/
        right_inv := fun f => by ext; rfl }
                                      /-
                                        🎉 no goals
                                      -/


/-- Construct the adjunction to the comparison functor.
-/
def leftAdjointComparison
    [∀ A : adj.toMonad.Algebra, HasCoequalizer (F.map A.a)
      (adj.counit.app (F.obj A.A))] :
    adj.toMonad.Algebra ⥤ D := by
  refine
    Adjunction.leftAdjointOfEquiv (G := comparison adj)
      (F_obj := fun A => comparisonLeftAdjointObj adj A) (fun A B => ?_) ?_
    /-
      case refine_1
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      G : CategoryTheory.Functor D C
      F : CategoryTheory.Functor C D
      adj : CategoryTheory.Adjunction F G
      inst✝ : ∀ (A : adj.toMonad.Algebra), CategoryTheory.Limits.HasCoequalizer (F.m …
      A : adj.toMonad.Algebra
      B : D
      ⊢ Equiv (Quiver.Hom ((fun A => CategoryTheory.Monad.MonadicityInternal.compari …
    -/
  · apply comparisonLeftAdjointHomEquiv
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      G : CategoryTheory.Functor D C
      F : CategoryTheory.Functor C D
      adj : CategoryTheory.Adjunction F G
      inst✝ : ∀ (A : adj.toMonad.Algebra), CategoryTheory.Limits.HasCoequalizer (F.m …
      ⊢ ∀ (X : adj.toMonad.Algebra) (Y Y' : D) (g : Quiver.Hom Y Y') (h : Quiver.Hom …
    -/
  · intro A B B' g h
    /-
      case refine_2
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      G : CategoryTheory.Functor D C
      F : CategoryTheory.Functor C D
      adj : CategoryTheory.Adjunction F G
      inst✝ : ∀ (A : adj.toMonad.Algebra), CategoryTheory.Limits.HasCoequalizer (F.m …
      A : adj.toMonad.Algebra
      B B' : D
      g : Quiver.Hom B B'
      h : Quiver.Hom ((fun A => CategoryTheory.Monad.MonadicityInternal.comparisonLe …
      ⊢ Eq (((fun A B => CategoryTheory.Monad.MonadicityInternal.comparisonLeftAdjoi …
    -/
    ext1
    -- Porting note: the goal was previously closed by the following, which succeeds until
    -- `Category.assoc`.
    -- dsimp [comparisonLeftAdjointHomEquiv]
    -- rw [← adj.homEquiv_naturality_right, Category.assoc]
    /-
      case refine_2.h
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      G : CategoryTheory.Functor D C
      F : CategoryTheory.Functor C D
      adj : CategoryTheory.Adjunction F G
      inst✝ : ∀ (A : adj.toMonad.Algebra), CategoryTheory.Limits.HasCoequalizer (F.m …
      A : adj.toMonad.Algebra
      B B' : D
      g : Quiver.Hom B B'
      h : Quiver.Hom ((fun A => CategoryTheory.Monad.MonadicityInternal.comparisonLe …
      ⊢ Eq (((fun A B => CategoryTheory.Monad.MonadicityInternal.comparisonLeftAdjoi …
    -/
    simp [Cofork.IsColimit.homIso, Adjunction.homEquiv_unit]
    /-
      🎉 no goals
    -/


/-- Provided we have the appropriate coequalizers, we have an adjunction to the comparison functor.
-/
@[simps! counit]
def comparisonAdjunction
    [∀ A : adj.toMonad.Algebra, HasCoequalizer (F.map A.a)
      (adj.counit.app (F.obj A.A))] :
    leftAdjointComparison adj ⊣ comparison adj :=
  Adjunction.adjunctionOfEquivLeft _ _


theorem comparisonAdjunction_unit_f_aux
    [∀ A : adj.toMonad.Algebra, HasCoequalizer (F.map A.a)
      (adj.counit.app (F.obj A.A))]
    (A : adj.toMonad.Algebra) :
    ((comparisonAdjunction adj).unit.app A).f =
      adj.homEquiv A.A _
        (coequalizer.π (F.map A.a) (adj.counit.app (F.obj A.A))) :=
  congr_arg (adj.homEquiv _ _) (Category.comp_id _)


/-- This is a cofork which is helpful for establishing monadicity: the morphism from the Beck
coequalizer to this cofork is the unit for the adjunction on the comparison functor.
-/
@[simps! pt]
def unitCofork (A : adj.toMonad.Algebra)
    [HasCoequalizer (F.map A.a) (adj.counit.app (F.obj A.A))] :
    Cofork (G.map (F.map A.a)) (G.map (adj.counit.app (F.obj A.A))) :=
  Cofork.ofπ (G.map (coequalizer.π (F.map A.a) (adj.counit.app (F.obj A.A))))
        /-
          C : Type u₁
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
          G : CategoryTheory.Functor D C
          F : CategoryTheory.Functor C D
          adj : CategoryTheory.Adjunction F G
          A : adj.toMonad.Algebra
          inst✝ : CategoryTheory.Limits.HasCoequalizer (F.map A.a) (adj.counit.app (F.ob …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (F.map A.a)) (G.map (CategoryT …
        -/
    (by rw [← G.map_comp, coequalizer.condition, G.map_comp])
        /-
          🎉 no goals
        -/


@[simp]
theorem unitCofork_π (A : adj.toMonad.Algebra)
    [HasCoequalizer (F.map A.a) (adj.counit.app (F.obj A.A))] :
    (unitCofork A).π = G.map (coequalizer.π (F.map A.a) (adj.counit.app (F.obj A.A))) :=
  rfl


theorem comparisonAdjunction_unit_f
    [∀ A : adj.toMonad.Algebra, HasCoequalizer (F.map A.a)
      (adj.counit.app (F.obj A.A))]
    (A : adj.toMonad.Algebra) :
    ((comparisonAdjunction adj).unit.app A).f = (beckCoequalizer A).desc (unitCofork A) := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    G : CategoryTheory.Functor D C
    F : CategoryTheory.Functor C D
    adj : CategoryTheory.Adjunction F G
    inst✝ : ∀ (A : adj.toMonad.Algebra), CategoryTheory.Limits.HasCoequalizer (F.m …
    A : adj.toMonad.Algebra
    ⊢ Eq ((CategoryTheory.Monad.MonadicityInternal.comparisonAdjunction adj).unit. …
  -/
  apply Limits.Cofork.IsColimit.hom_ext (beckCoequalizer A)
  /-
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    G : CategoryTheory.Functor D C
    F : CategoryTheory.Functor C D
    adj : CategoryTheory.Adjunction F G
    inst✝ : ∀ (A : adj.toMonad.Algebra), CategoryTheory.Limits.HasCoequalizer (F.m …
    A : adj.toMonad.Algebra
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Monad.beckCofork A).π …
  -/
  rw [Cofork.IsColimit.π_desc]
  /-
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    G : CategoryTheory.Functor D C
    F : CategoryTheory.Functor C D
    adj : CategoryTheory.Adjunction F G
    inst✝ : ∀ (A : adj.toMonad.Algebra), CategoryTheory.Limits.HasCoequalizer (F.m …
    A : adj.toMonad.Algebra
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Monad.beckCofork A).π …
  -/
  dsimp only [beckCofork_π, unitCofork_π]
  rw [comparisonAdjunction_unit_f_aux, ← adj.homEquiv_naturality_left A.a, coequalizer.condition,
    adj.homEquiv_naturality_right, adj.homEquiv_unit, Category.assoc]
  /-
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    G : CategoryTheory.Functor D C
    F : CategoryTheory.Functor C D
    adj : CategoryTheory.Adjunction F G
    inst✝ : ∀ (A : adj.toMonad.Algebra), CategoryTheory.Limits.HasCoequalizer (F.m …
    A : adj.toMonad.Algebra
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (adj.toMonad.obj A.A))  …
  -/
  apply adj.right_triangle_components_assoc
  /-
    🎉 no goals
  -/


/-- The cofork which describes the counit of the adjunction: the morphism from the coequalizer of
this pair to this morphism is the counit.
-/
@[simps!]
def counitCofork (B : D) :
    Cofork (F.map (G.map (adj.counit.app B)))
      (adj.counit.app (F.obj (G.obj B))) :=
  Cofork.ofπ (adj.counit.app B) (adj.counit_naturality _)


variable {adj} in
/-- The unit cofork is a colimit provided `G` preserves it. -/
def unitColimitOfPreservesCoequalizer (A : adj.toMonad.Algebra)
    [HasCoequalizer (F.map A.a) (adj.counit.app (F.obj A.A))]
    [PreservesColimit (parallelPair (F.map A.a) (adj.counit.app (F.obj A.A))) G] :
    IsColimit (unitCofork (G := G) A) :=
  isColimitOfHasCoequalizerOfPreservesColimit G _ _


/-- The counit cofork is a colimit provided `G` reflects it. -/
def counitCoequalizerOfReflectsCoequalizer (B : D)
    [ReflectsColimit (parallelPair (F.map (G.map (adj.counit.app B)))
      (adj.counit.app (F.obj (G.obj B)))) G] :
    IsColimit (counitCofork (adj := adj) B) :=
  isColimitOfIsColimitCoforkMap G _ (beckCoequalizer ((comparison adj).obj B))

-- Porting note: Lean 3 didn't seem to need this

instance
    [∀ A : adj.toMonad.Algebra, HasCoequalizer (F.map A.a) (adj.counit.app (F.obj A.A))]
    (B : D) : HasColimit (parallelPair
      (F.map (G.map (NatTrans.app adj.counit B)))
      (NatTrans.app adj.counit (F.obj (G.obj B)))) :=
  inferInstanceAs <| HasCoequalizer
    (F.map ((comparison adj).obj B).a)
    (adj.counit.app (F.obj ((comparison adj).obj B).A))


theorem comparisonAdjunction_counit_app
    [∀ A : adj.toMonad.Algebra, HasCoequalizer (F.map A.a) (adj.counit.app (F.obj A.A))] (B : D) :
    (comparisonAdjunction adj).counit.app B = colimit.desc _ (counitCofork adj B) := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    G : CategoryTheory.Functor D C
    F : CategoryTheory.Functor C D
    adj : CategoryTheory.Adjunction F G
    inst✝ : ∀ (A : adj.toMonad.Algebra), CategoryTheory.Limits.HasCoequalizer (F.m …
    B : D
    ⊢ Eq ((CategoryTheory.Monad.MonadicityInternal.comparisonAdjunction adj).couni …
  -/
  apply coequalizer.hom_ext
  change
    coequalizer.π _ _ ≫ coequalizer.desc ((adj.homEquiv _ B).symm (𝟙 _)) _ =
      coequalizer.π _ _ ≫ coequalizer.desc _ _
  /-
    case h
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    G : CategoryTheory.Functor D C
    F : CategoryTheory.Functor C D
    adj : CategoryTheory.Adjunction F G
    inst✝ : ∀ (A : adj.toMonad.Algebra), CategoryTheory.Limits.HasCoequalizer (F.m …
    B : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  simp [Adjunction.homEquiv_counit]
  /-
    🎉 no goals
  -/


variable (G) in
/--
If `G` is monadic, it creates colimits of `G`-split pairs. This is the "boring" direction of Beck's
monadicity theorem, the converse is given in `monadicOfCreatesGSplitCoequalizers`.
-/
def createsGSplitCoequalizersOfMonadic [MonadicRightAdjoint G] ⦃A B⦄ (f g : A ⟶ B)
    [G.IsSplitPair f g] : CreatesColimit (parallelPair f g) G := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₁, u₂} D
    G : CategoryTheory.Functor D C
    F : CategoryTheory.Functor C D
    adj : CategoryTheory.Adjunction F G
    inst✝¹ : CategoryTheory.MonadicRightAdjoint G
    A B : D
    f g : Quiver.Hom A B
    inst✝ : G.IsSplitPair f g
    ⊢ CategoryTheory.CreatesColimit (CategoryTheory.Limits.parallelPair f g) G
  -/
  apply (config := {allowSynthFailures := true}) monadicCreatesColimitOfPreservesColimit
    -- Porting note: oddly (config := {allowSynthFailures := true}) had no effect here and below
  all_goals
    apply @preservesColimit_of_iso_diagram _ _ _ _ _ _ _ _ _ (diagramIsoParallelPair.{v₁} _).symm ?_
    dsimp
    infer_instance


class HasCoequalizerOfIsSplitPair (G : D ⥤ C) : Prop where
  out : ∀ {A B} (f g : A ⟶ B) [G.IsSplitPair f g], HasCoequalizer f g

-- Porting note: cannot find synth order
-- instance {A B} (f g : A ⟶ B) [G.IsSplitPair f g] [HasCoequalizerOfIsSplitPair G] :
--     HasCoequalizer f g := HasCoequalizerOfIsSplitPair.out f g


instance [HasCoequalizerOfIsSplitPair G] : ∀ (A : Algebra adj.toMonad),
    HasCoequalizer (F.map A.a)
      (adj.counit.app (F.obj A.A)) :=
  fun _ => HasCoequalizerOfIsSplitPair.out G _ _

-- Porting note: added these to replace parametric instances https://github.com/leanprover/lean4/issues/2311
-- [∀ ⦃A B⦄ (f g : A ⟶ B) [G.IsSplitPair f g], PreservesColimit (parallelPair f g) G]

class PreservesColimitOfIsSplitPair (G : D ⥤ C) where
  out : ∀ {A B} (f g : A ⟶ B) [G.IsSplitPair f g], PreservesColimit (parallelPair f g) G


instance {A B} (f g : A ⟶ B) [G.IsSplitPair f g] [PreservesColimitOfIsSplitPair G] :
    PreservesColimit (parallelPair f g) G := PreservesColimitOfIsSplitPair.out f g


instance [PreservesColimitOfIsSplitPair G] : ∀ (A : Algebra adj.toMonad),
   PreservesColimit (parallelPair (F.map A.a)
      (NatTrans.app adj.counit (F.obj A.A))) G :=
  fun _ => PreservesColimitOfIsSplitPair.out _ _

-- Porting note: added these to replace parametric instances https://github.com/leanprover/lean4/issues/2311
-- [∀ ⦃A B⦄ (f g : A ⟶ B) [G.IsSplitPair f g], ReflectsColimit (parallelPair f g) G] :

class ReflectsColimitOfIsSplitPair (G : D ⥤ C) where
  out : ∀ {A B} (f g : A ⟶ B) [G.IsSplitPair f g], ReflectsColimit (parallelPair f g) G


instance {A B} (f g : A ⟶ B) [G.IsSplitPair f g] [ReflectsColimitOfIsSplitPair G] :
    ReflectsColimit (parallelPair f g) G := ReflectsColimitOfIsSplitPair.out f g


instance [ReflectsColimitOfIsSplitPair G] : ∀ (A : Algebra adj.toMonad),
    ReflectsColimit (parallelPair (F.map A.a)
      (NatTrans.app adj.counit (F.obj A.A))) G :=
  fun _ => ReflectsColimitOfIsSplitPair.out _ _


/-- To show `G` is a monadic right adjoint, we can show it preserves and reflects `G`-split
coequalizers, and `D` has them.
-/
def monadicOfHasPreservesReflectsGSplitCoequalizers [HasCoequalizerOfIsSplitPair G]
    [PreservesColimitOfIsSplitPair G] [ReflectsColimitOfIsSplitPair G] :
    MonadicRightAdjoint G where
  adj := adj
  eqv := by
    have : ∀ (X : Algebra adj.toMonad), IsIso ((comparisonAdjunction adj).unit.app X) := by
      intro X
      apply @isIso_of_reflects_iso _ _ _ _ _ _ _ (Monad.forget adj.toMonad) ?_ _
      · change IsIso ((comparisonAdjunction adj).unit.app X).f
        rw [comparisonAdjunction_unit_f]
        change
          IsIso
            (IsColimit.coconePointUniqueUpToIso (beckCoequalizer X)
                (unitColimitOfPreservesCoequalizer X)).hom
        exact (IsColimit.coconePointUniqueUpToIso _ _).isIso_hom
    have : ∀ (Y : D), IsIso ((comparisonAdjunction adj).counit.app Y) := by
      intro Y
      rw [comparisonAdjunction_counit_app]
      -- Porting note: passing instances through
      change IsIso (IsColimit.coconePointUniqueUpToIso _ ?_).hom
      infer_instance
      -- Porting note: passing instances through
      apply @counitCoequalizerOfReflectsCoequalizer _ _ _ _ _ _ _ _ ?_
      letI _ :
        G.IsSplitPair (F.map (G.map (adj.counit.app Y)))
          (adj.counit.app (F.obj (G.obj Y))) :=
        MonadicityInternal.main_pair_G_split _ ((comparison adj).obj Y)
      infer_instance
    /-
      C : Type u₁
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.Category.{v₁, u₂} D
      G : CategoryTheory.Functor D C
      F : CategoryTheory.Functor C D
      adj : CategoryTheory.Adjunction F G
      inst✝² : CategoryTheory.Monad.HasCoequalizerOfIsSplitPair G
      inst✝¹ : CategoryTheory.Monad.PreservesColimitOfIsSplitPair G
      inst✝ : CategoryTheory.Monad.ReflectsColimitOfIsSplitPair G
      this✝ : ∀ (X : adj.toMonad.Algebra), CategoryTheory.IsIso ((CategoryTheory.Mon …
      this : ∀ (Y : D), CategoryTheory.IsIso ((CategoryTheory.Monad.MonadicityIntern …
      ⊢ (CategoryTheory.Monad.comparison adj).IsEquivalence
    -/
    exact (comparisonAdjunction adj).toEquivalence.isEquivalence_inverse
    /-
      🎉 no goals
    -/

-- Porting note: added these to replace parametric instances https://github.com/leanprover/lean4/issues/2311
-- [∀ ⦃A B⦄ (f g : A ⟶ B) [G.IsSplitPair f g], CreatesColimit (parallelPair f g) G] :

class CreatesColimitOfIsSplitPair (G : D ⥤ C) where
  out : ∀ {A B} (f g : A ⟶ B) [G.IsSplitPair f g], CreatesColimit (parallelPair f g) G


instance {A B} (f g : A ⟶ B) [G.IsSplitPair f g] [CreatesColimitOfIsSplitPair G] :
    CreatesColimit (parallelPair f g) G := CreatesColimitOfIsSplitPair.out f g


instance [CreatesColimitOfIsSplitPair G] : ∀ (A : Algebra adj.toMonad),
    CreatesColimit (parallelPair (F.map A.a)
      (NatTrans.app adj.counit (F.obj A.A))) G :=
  fun _ => CreatesColimitOfIsSplitPair.out _ _


/--
Beck's monadicity theorem. If `G` has a left adjoint and creates coequalizers of `G`-split pairs,
then it is monadic.
This is the converse of `createsGSplitCoequalizersOfMonadic`.
-/
def monadicOfCreatesGSplitCoequalizers [CreatesColimitOfIsSplitPair G] :
    MonadicRightAdjoint G := by
  let I {A B} (f g : A ⟶ B) [G.IsSplitPair f g] : HasColimit (parallelPair f g ⋙ G) := by
    apply @hasColimitOfIso _ _ _ _ _ _ ?_ (diagramIsoParallelPair.{v₁} _)
    exact inferInstanceAs <| HasCoequalizer (G.map f) (G.map g)
  /-
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    G : CategoryTheory.Functor D C
    F : CategoryTheory.Functor C D
    adj : CategoryTheory.Adjunction F G
    inst✝ : CategoryTheory.Monad.CreatesColimitOfIsSplitPair G
    I : ∀ {A B : D} (f g : Quiver.Hom A B) [inst : G.IsSplitPair f g], CategoryThe …
    ⊢ CategoryTheory.MonadicRightAdjoint G
  -/
  have : HasCoequalizerOfIsSplitPair G := ⟨fun _ _ => hasColimit_of_created (parallelPair _ _) G⟩
  /-
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    G : CategoryTheory.Functor D C
    F : CategoryTheory.Functor C D
    adj : CategoryTheory.Adjunction F G
    inst✝ : CategoryTheory.Monad.CreatesColimitOfIsSplitPair G
    I : ∀ {A B : D} (f g : Quiver.Hom A B) [inst : G.IsSplitPair f g], CategoryThe …
    this : CategoryTheory.Monad.HasCoequalizerOfIsSplitPair G
    ⊢ CategoryTheory.MonadicRightAdjoint G
  -/
  have : PreservesColimitOfIsSplitPair G := ⟨by intros; infer_instance⟩
  /-
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    G : CategoryTheory.Functor D C
    F : CategoryTheory.Functor C D
    adj : CategoryTheory.Adjunction F G
    inst✝ : CategoryTheory.Monad.CreatesColimitOfIsSplitPair G
    I : ∀ {A B : D} (f g : Quiver.Hom A B) [inst : G.IsSplitPair f g], CategoryThe …
    this✝ : CategoryTheory.Monad.HasCoequalizerOfIsSplitPair G
    this : CategoryTheory.Monad.PreservesColimitOfIsSplitPair G
    ⊢ CategoryTheory.MonadicRightAdjoint G
  -/
  have : ReflectsColimitOfIsSplitPair G := ⟨by intros; infer_instance⟩
  /-
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    G : CategoryTheory.Functor D C
    F : CategoryTheory.Functor C D
    adj : CategoryTheory.Adjunction F G
    inst✝ : CategoryTheory.Monad.CreatesColimitOfIsSplitPair G
    I : ∀ {A B : D} (f g : Quiver.Hom A B) [inst : G.IsSplitPair f g], CategoryThe …
    this✝¹ : CategoryTheory.Monad.HasCoequalizerOfIsSplitPair G
    this✝ : CategoryTheory.Monad.PreservesColimitOfIsSplitPair G
    this : CategoryTheory.Monad.ReflectsColimitOfIsSplitPair G
    ⊢ CategoryTheory.MonadicRightAdjoint G
  -/
  exact monadicOfHasPreservesReflectsGSplitCoequalizers adj
  /-
    🎉 no goals
  -/


/-- An alternate version of Beck's monadicity theorem. If `G` reflects isomorphisms, preserves
coequalizers of `G`-split pairs and `C` has coequalizers of `G`-split pairs, then it is monadic.
-/
def monadicOfHasPreservesGSplitCoequalizersOfReflectsIsomorphisms [G.ReflectsIsomorphisms]
    [HasCoequalizerOfIsSplitPair G] [PreservesColimitOfIsSplitPair G] :
    MonadicRightAdjoint G := by
  have : ReflectsColimitOfIsSplitPair G := ⟨fun f g _ => by
    have := HasCoequalizerOfIsSplitPair.out G f g
    apply reflectsColimit_of_reflectsIsomorphisms⟩
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Category.{v₁, u₂} D
    G : CategoryTheory.Functor D C
    F : CategoryTheory.Functor C D
    adj : CategoryTheory.Adjunction F G
    inst✝² : G.ReflectsIsomorphisms
    inst✝¹ : CategoryTheory.Monad.HasCoequalizerOfIsSplitPair G
    inst✝ : CategoryTheory.Monad.PreservesColimitOfIsSplitPair G
    this : CategoryTheory.Monad.ReflectsColimitOfIsSplitPair G
    ⊢ CategoryTheory.MonadicRightAdjoint G
  -/
  apply monadicOfHasPreservesReflectsGSplitCoequalizers adj
  /-
    🎉 no goals
  -/


class PreservesColimitOfIsReflexivePair (G : C ⥤ D) where
  out : ∀ ⦃A B⦄ (f g : A ⟶ B) [IsReflexivePair f g], PreservesColimit (parallelPair f g) G


instance {A B} (f g : A ⟶ B) [IsReflexivePair f g] [PreservesColimitOfIsReflexivePair G] :
  PreservesColimit (parallelPair f g) G := PreservesColimitOfIsReflexivePair.out f g


instance [PreservesColimitOfIsReflexivePair G] : ∀ X : Algebra adj.toMonad,
    PreservesColimit (parallelPair (F.map X.a)
      (NatTrans.app adj.counit (F.obj X.A))) G :=
 fun _ => PreservesColimitOfIsReflexivePair.out _ _


/-- Reflexive (crude) monadicity theorem. If `G` has a right adjoint, `D` has and `G` preserves
reflexive coequalizers and `G` reflects isomorphisms, then `G` is monadic.
-/
def monadicOfHasPreservesReflexiveCoequalizersOfReflectsIsomorphisms : MonadicRightAdjoint G where
  adj := adj
  eqv := by
    have : ∀ (X : Algebra adj.toMonad), IsIso ((comparisonAdjunction adj).unit.app X) := by
      intro X
      apply
        @isIso_of_reflects_iso _ _ _ _ _ _ _ (Monad.forget adj.toMonad) ?_ _
      · change IsIso ((comparisonAdjunction adj).unit.app X).f
        rw [comparisonAdjunction_unit_f]
        exact (IsColimit.coconePointUniqueUpToIso (beckCoequalizer X)
          (unitColimitOfPreservesCoequalizer X)).isIso_hom
    have : ∀ (Y : D), IsIso ((comparisonAdjunction adj).counit.app Y) := by
      intro Y
      rw [comparisonAdjunction_counit_app]
      -- Porting note: passing instances through
      change IsIso (IsColimit.coconePointUniqueUpToIso _ ?_).hom
      infer_instance
      -- Porting note: passing instances through
      apply @counitCoequalizerOfReflectsCoequalizer _ _ _ _ _ _ _ _ ?_
      apply reflectsColimit_of_reflectsIsomorphisms
    /-
      C : Type u₁
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.Category.{v₁, u₂} D
      G : CategoryTheory.Functor D C
      F : CategoryTheory.Functor C D
      adj : CategoryTheory.Adjunction F G
      inst✝² : CategoryTheory.Limits.HasReflexiveCoequalizers D
      inst✝¹ : G.ReflectsIsomorphisms
      inst✝ : CategoryTheory.Monad.PreservesColimitOfIsReflexivePair G
      this✝ : ∀ (X : adj.toMonad.Algebra), CategoryTheory.IsIso ((CategoryTheory.Mon …
      this : ∀ (Y : D), CategoryTheory.IsIso ((CategoryTheory.Monad.MonadicityIntern …
      ⊢ (CategoryTheory.Monad.comparison adj).IsEquivalence
    -/
    exact (comparisonAdjunction adj).toEquivalence.isEquivalence_inverse
    /-
      🎉 no goals
    -/


