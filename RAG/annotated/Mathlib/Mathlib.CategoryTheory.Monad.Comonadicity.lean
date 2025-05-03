/-- The "main pair" for a coalgebra `(A, α)` is the pair of morphisms `(G α, η_GA)`. It is always a
coreflexive pair, and will be used to construct the left adjoint to the comparison functor and show
it is an equivalence.
-/
instance main_pair_coreflexive (A : adj.toComonad.Coalgebra) :
    IsCoreflexivePair (G.map A.a) (adj.unit.app (G.obj A.A)) := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    A : adj.toComonad.Coalgebra
    ⊢ CategoryTheory.IsCoreflexivePair (G.map A.a) (adj.unit.app (G.obj A.A))
  -/
  apply IsCoreflexivePair.mk' (G.map (adj.counit.app _)) _ _
    /-
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      A : adj.toComonad.Coalgebra
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map A.a) (G.map (adj.counit.app A. …
    -/
  · rw [← G.map_comp, ← G.map_id]
    /-
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      A : adj.toComonad.Coalgebra
      ⊢ Eq (G.map (CategoryTheory.CategoryStruct.comp A.a (adj.counit.app A.1))) (G. …
    -/
    exact congr_arg G.map A.counit
    /-
      🎉 no goals
    -/
    /-
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      A : adj.toComonad.Coalgebra
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (G.obj A.A)) (G.map (ad …
    -/
  · rw [adj.right_triangle_components]
    /-
      C : Type u₁
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      A : adj.toComonad.Coalgebra
      ⊢ Eq (CategoryTheory.CategoryStruct.id (G.obj A.A)) (CategoryTheory.CategorySt …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The "main pair" for a coalgebra `(A, α)` is the pair of morphisms `(G α, η_GA)`. It is always a
`G`-cosplit pair, and will be used to construct the right adjoint to the comparison functor and show
it is an equivalence.
-/
instance main_pair_F_cosplit (A : adj.toComonad.Coalgebra) :
    F.IsCosplitPair (G.map A.a)
      (adj.unit.app (G.obj A.A)) where
  splittable := ⟨_, _, ⟨beckSplitEqualizer A⟩⟩


/-- The object function for the right adjoint to the comparison functor. -/
def comparisonRightAdjointObj (A : adj.toComonad.Coalgebra)
    [HasEqualizer (G.map A.a) (adj.unit.app _)] : C :=
  equalizer (G.map A.a) (adj.unit.app _)


/--
We have a bijection of homsets which will be used to construct the right adjoint to the comparison
functor.
-/
@[simps!]
def comparisonRightAdjointHomEquiv (A : adj.toComonad.Coalgebra) (B : C)
    [HasEqualizer (G.map A.a) (adj.unit.app (G.obj A.A))] :
    ((comparison adj).obj B ⟶ A) ≃ (B ⟶ comparisonRightAdjointObj adj A) where
      toFun f := by
        /-
          C : Type u₁
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
          F : CategoryTheory.Functor C D
          G : CategoryTheory.Functor D C
          adj : CategoryTheory.Adjunction F G
          A : adj.toComonad.Coalgebra
          B : C
          inst✝ : CategoryTheory.Limits.HasEqualizer (G.map A.a) (adj.unit.app (G.obj A. …
          f : Quiver.Hom ((CategoryTheory.Comonad.comparison adj).obj B) A
          ⊢ Quiver.Hom B (CategoryTheory.Comonad.ComonadicityInternal.comparisonRightAdj …
        -/
        refine equalizer.lift (adj.homEquiv _ _ f.f) ?_
        simp only [Adjunction.toComonad_coe, Functor.comp_obj, Adjunction.homEquiv_unit,
          Functor.id_obj, Category.assoc, ← G.map_comp, ← f.h, comparison_obj_A, comparison_obj_a]
        rw [Functor.comp_map, Functor.map_comp, Adjunction.unit_naturality_assoc,
          Adjunction.unit_naturality]
      invFun f := by
        /-
          C : Type u₁
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
          F : CategoryTheory.Functor C D
          G : CategoryTheory.Functor D C
          adj : CategoryTheory.Adjunction F G
          A : adj.toComonad.Coalgebra
          B : C
          inst✝ : CategoryTheory.Limits.HasEqualizer (G.map A.a) (adj.unit.app (G.obj A. …
          f : Quiver.Hom B (CategoryTheory.Comonad.ComonadicityInternal.comparisonRightA …
          ⊢ Quiver.Hom ((CategoryTheory.Comonad.comparison adj).obj B) A
        -/
        refine ⟨(adj.homEquiv _ _).symm (f ≫ (equalizer.ι _ _)), (adj.homEquiv _ _).injective ?_⟩
        simp only [Adjunction.toComonad_coe, Functor.comp_obj, comparison_obj_A, comparison_obj_a,
          Adjunction.homEquiv_counit, Functor.id_obj, Functor.map_comp, Category.assoc,
          Functor.comp_map, Adjunction.homEquiv_unit, Adjunction.unit_naturality_assoc,
          Adjunction.unit_naturality, Adjunction.right_triangle_components_assoc]
        /-
          C : Type u₁
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
          F : CategoryTheory.Functor C D
          G : CategoryTheory.Functor D C
          adj : CategoryTheory.Adjunction F G
          A : adj.toComonad.Coalgebra
          B : C
          inst✝ : CategoryTheory.Limits.HasEqualizer (G.map A.a) (adj.unit.app (G.obj A. …
          f : Quiver.Hom B (CategoryTheory.Comonad.ComonadicityInternal.comparisonRightA …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
        -/
        congr 1
        /-
          case e_a
          C : Type u₁
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
          F : CategoryTheory.Functor C D
          G : CategoryTheory.Functor D C
          adj : CategoryTheory.Adjunction F G
          A : adj.toComonad.Coalgebra
          B : C
          inst✝ : CategoryTheory.Limits.HasEqualizer (G.map A.a) (adj.unit.app (G.obj A. …
          f : Quiver.Hom B (CategoryTheory.Comonad.ComonadicityInternal.comparisonRightA …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizer.ι (G …
        -/
        exact (equalizer.condition _ _).symm
        /-
          🎉 no goals
        -/
                       /-
                         C : Type u₁
                         D : Type u₂
                         inst✝² : CategoryTheory.Category.{v₁, u₁} C
                         inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
                         F : CategoryTheory.Functor C D
                         G : CategoryTheory.Functor D C
                         adj : CategoryTheory.Adjunction F G
                         A : adj.toComonad.Coalgebra
                         B : C
                         inst✝ : CategoryTheory.Limits.HasEqualizer (G.map A.a) (adj.unit.app (G.obj A. …
                         f : Quiver.Hom ((CategoryTheory.Comonad.comparison adj).obj B) A
                         ⊢ Eq ((fun f => { f := (adj.homEquiv B A.A).symm (CategoryTheory.CategoryStruc …
                       -/
      left_inv f := by aesop
                       /-
                         🎉 no goals
                       -/
                        /-
                          C : Type u₁
                          D : Type u₂
                          inst✝² : CategoryTheory.Category.{v₁, u₁} C
                          inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
                          F : CategoryTheory.Functor C D
                          G : CategoryTheory.Functor D C
                          adj : CategoryTheory.Adjunction F G
                          A : adj.toComonad.Coalgebra
                          B : C
                          inst✝ : CategoryTheory.Limits.HasEqualizer (G.map A.a) (adj.unit.app (G.obj A. …
                          f : Quiver.Hom B (CategoryTheory.Comonad.ComonadicityInternal.comparisonRightA …
                          ⊢ Eq ((fun f => CategoryTheory.Limits.equalizer.lift ((adj.homEquiv B A.A) f.f …
                        -/
      right_inv f := by apply equalizer.hom_ext; simp
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- Construct the adjunction to the comparison functor.
-/
def rightAdjointComparison
    [∀ A : adj.toComonad.Coalgebra, HasEqualizer (G.map A.a)
      (adj.unit.app (G.obj A.A))] :
    adj.toComonad.Coalgebra ⥤ C := by
  refine
    Adjunction.rightAdjointOfEquiv (F := comparison adj)
      (G_obj := fun A => comparisonRightAdjointObj adj A) (fun A B => ?_) ?_
    /-
      case refine_1
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : ∀ (A : adj.toComonad.Coalgebra), CategoryTheory.Limits.HasEqualizer (G …
      A : C
      B : adj.toComonad.Coalgebra
      ⊢ Equiv (Quiver.Hom ((CategoryTheory.Comonad.comparison adj).obj A) B) (Quiver …
    -/
  · apply comparisonRightAdjointHomEquiv
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : ∀ (A : adj.toComonad.Coalgebra), CategoryTheory.Limits.HasEqualizer (G …
      ⊢ ∀ (X' X : C) (Y : adj.toComonad.Coalgebra) (f : Quiver.Hom X' X) (g : Quiver …
    -/
  · intro A B B' g h
    /-
      case refine_2
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : ∀ (A : adj.toComonad.Coalgebra), CategoryTheory.Limits.HasEqualizer (G …
      A B : C
      B' : adj.toComonad.Coalgebra
      g : Quiver.Hom A B
      h : Quiver.Hom ((CategoryTheory.Comonad.comparison adj).obj B) B'
      ⊢ Eq (((fun A B => CategoryTheory.Comonad.ComonadicityInternal.comparisonRight …
    -/
    apply equalizer.hom_ext
    /-
      case refine_2.h
      C : Type u₁
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝ : ∀ (A : adj.toComonad.Coalgebra), CategoryTheory.Limits.HasEqualizer (G …
      A B : C
      B' : adj.toComonad.Coalgebra
      g : Quiver.Hom A B
      h : Quiver.Hom ((CategoryTheory.Comonad.comparison adj).obj B) B'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun A B => CategoryTheory.Comonad. …
    -/
    simp [Adjunction.homEquiv_unit]
    /-
      🎉 no goals
    -/


/-- Provided we have the appropriate equalizers, we have an adjunction to the comparison functor.
-/
@[simps! counit]
def comparisonAdjunction
    [∀ A : adj.toComonad.Coalgebra, HasEqualizer (G.map A.a)
      (adj.unit.app (G.obj A.A))] :
    comparison adj ⊣ rightAdjointComparison adj :=
  Adjunction.adjunctionOfEquivRight _ _


theorem comparisonAdjunction_counit_f_aux
    [∀ A : adj.toComonad.Coalgebra, HasEqualizer (G.map A.a)
      (adj.unit.app (G.obj A.A))]
    (A : adj.toComonad.Coalgebra) :
    ((comparisonAdjunction adj).counit.app A).f =
      (adj.homEquiv _ A.A).symm (equalizer.ι (G.map A.a) (adj.unit.app (G.obj A.A))) :=
  congr_arg (adj.homEquiv _ _).symm (Category.id_comp _)


/-- This is a fork which is helpful for establishing comonadicity: the morphism from this fork to
the Beck equalizer is the counit for the adjunction on the comparison functor.
-/
@[simps! pt]
def counitFork (A : adj.toComonad.Coalgebra)
    [HasEqualizer (G.map A.a) (adj.unit.app (G.obj A.A))] :
    Fork (F.map (G.map A.a)) (F.map (adj.unit.app (G.obj A.A))) :=
  Fork.ofι (F.map (equalizer.ι (G.map A.a) (adj.unit.app (G.obj A.A))))
        /-
          C : Type u₁
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
          F : CategoryTheory.Functor C D
          G : CategoryTheory.Functor D C
          adj : CategoryTheory.Adjunction F G
          A : adj.toComonad.Coalgebra
          inst✝ : CategoryTheory.Limits.HasEqualizer (G.map A.a) (adj.unit.app (G.obj A. …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.equaliz …
        -/
    (by rw [← F.map_comp, equalizer.condition, F.map_comp])
        /-
          🎉 no goals
        -/


@[simp]
theorem unitFork_ι (A : adj.toComonad.Coalgebra)
    [HasEqualizer (G.map A.a) (adj.unit.app (G.obj A.A))] :
    (counitFork A).ι = F.map (equalizer.ι (G.map A.a) (adj.unit.app (G.obj A.A))) :=
  rfl


theorem comparisonAdjunction_counit_f
    [∀ A : adj.toComonad.Coalgebra, HasEqualizer (G.map A.a)
      (adj.unit.app (G.obj A.A))]
    (A : adj.toComonad.Coalgebra) :
    ((comparisonAdjunction adj).counit.app A).f = (beckEqualizer A).lift (counitFork A) := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝ : ∀ (A : adj.toComonad.Coalgebra), CategoryTheory.Limits.HasEqualizer (G …
    A : adj.toComonad.Coalgebra
    ⊢ Eq ((CategoryTheory.Comonad.ComonadicityInternal.comparisonAdjunction adj).c …
  -/
  simp [Adjunction.homEquiv_counit]
  /-
    🎉 no goals
  -/


/-- The fork which describes the unit of the adjunction: the morphism from this fork to the
the equalizer of this pair is the unit.
-/
@[simps!]
def unitFork (B : C) :
    Fork (G.map (F.map (adj.unit.app B)))
      (adj.unit.app (G.obj (F.obj B))) :=
  Fork.ofι (adj.unit.app B) (adj.unit_naturality _)


variable {adj} in
/-- The counit fork is a limit provided `F` preserves it. -/
def counitLimitOfPreservesEqualizer (A : adj.toComonad.Coalgebra)
    [HasEqualizer (G.map A.a) (adj.unit.app (G.obj A.A))]
    [PreservesLimit (parallelPair (G.map A.a) (adj.unit.app (G.obj A.A))) F] :
    IsLimit (counitFork (G := G) A) :=
  isLimitOfHasEqualizerOfPreservesLimit F _ _


/-- The unit fork is a limit provided `F` coreflects it. -/
def unitEqualizerOfCoreflectsEqualizer (B : C)
    [ReflectsLimit (parallelPair (G.map (F.map (adj.unit.app B)))
      (adj.unit.app (G.obj (F.obj B)))) F] :
    IsLimit (unitFork (adj := adj) B) :=
  isLimitOfIsLimitForkMap F _ (beckEqualizer ((comparison adj).obj B))


instance
    [∀ A : adj.toComonad.Coalgebra, HasEqualizer (G.map A.a) (adj.unit.app (G.obj A.A))]
    (B : C) : HasLimit (parallelPair
      (G.map (F.map (NatTrans.app adj.unit B)))
      (NatTrans.app adj.unit (G.obj (F.obj B)))) :=
  inferInstanceAs <| HasEqualizer
    (G.map ((comparison adj).obj B).a)
    (adj.unit.app (G.obj ((comparison adj).obj B).A))


theorem comparisonAdjunction_unit_app
    [∀ A : adj.toComonad.Coalgebra, HasEqualizer (G.map A.a) (adj.unit.app (G.obj A.A))] (B : C) :
    (comparisonAdjunction adj).unit.app B = limit.lift _ (unitFork adj B) := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝ : ∀ (A : adj.toComonad.Coalgebra), CategoryTheory.Limits.HasEqualizer (G …
    B : C
    ⊢ Eq ((CategoryTheory.Comonad.ComonadicityInternal.comparisonAdjunction adj).u …
  -/
  apply equalizer.hom_ext
  change
    equalizer.lift ((adj.homEquiv B _) (𝟙 _)) _ ≫ equalizer.ι _ _ =
      equalizer.lift _ _ ≫ equalizer.ι _ _
  /-
    case h
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝ : ∀ (A : adj.toComonad.Coalgebra), CategoryTheory.Limits.HasEqualizer (G …
    B : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizer.lift …
  -/
  simp [Adjunction.homEquiv_unit]
  /-
    🎉 no goals
  -/


variable (G) in
/--
If `F` is comonadic, it creates limits of `F`-cosplit pairs. This is the "boring" direction of
Beck's comonadicity theorem, the converse is given in `comonadicOfCreatesFSplitEqualizers`.
-/
def createsFSplitEqualizersOfComonadic [ComonadicLeftAdjoint F] ⦃A B⦄ (f g : A ⟶ B)
    [F.IsCosplitPair f g] : CreatesLimit (parallelPair f g) F := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝¹ : CategoryTheory.ComonadicLeftAdjoint F
    A B : C
    f g : Quiver.Hom A B
    inst✝ : F.IsCosplitPair f g
    ⊢ CategoryTheory.CreatesLimit (CategoryTheory.Limits.parallelPair f g) F
  -/
  apply (config := {allowSynthFailures := true}) comonadicCreatesLimitOfPreservesLimit
  all_goals
    apply @preservesLimit_of_iso_diagram _ _ _ _ _ _ _ _ _ (diagramIsoParallelPair.{v₁} _).symm ?_
    dsimp
    infer_instance


/-- Dual to `Monad.HasCoequalizerOfIsSplitPair`. -/
class HasEqualizerOfIsCosplitPair (F : C ⥤ D) : Prop where
  /-- If `f, g` is an `F`-cosplit pair, then they have an equalizer. -/
  out : ∀ {A B} (f g : A ⟶ B) [F.IsCosplitPair f g], HasEqualizer f g


instance [HasEqualizerOfIsCosplitPair F] : ∀ (A : Coalgebra adj.toComonad),
    HasEqualizer (G.map A.a)
      (adj.unit.app (G.obj A.A)) :=
  fun _ => HasEqualizerOfIsCosplitPair.out F _ _


/-- Dual to `Monad.PreservesColimitOfIsSplitPair`. -/
class PreservesLimitOfIsCosplitPair (F : C ⥤ D) where
  /-- If `f, g` is an `F`-cosplit pair, then `F` preserves limits of `parallelPair f g`. -/
  out : ∀ {A B} (f g : A ⟶ B) [F.IsCosplitPair f g], PreservesLimit (parallelPair f g) F


instance {A B} (f g : A ⟶ B) [F.IsCosplitPair f g] [PreservesLimitOfIsCosplitPair F] :
    PreservesLimit (parallelPair f g) F := PreservesLimitOfIsCosplitPair.out f g


instance [PreservesLimitOfIsCosplitPair F] : ∀ (A : Coalgebra adj.toComonad),
   PreservesLimit (parallelPair (G.map A.a)
      (NatTrans.app adj.unit (G.obj A.A))) F :=
  fun _ => PreservesLimitOfIsCosplitPair.out _ _


/-- Dual to `Monad.ReflectsColimitOfIsSplitPair`. -/
class ReflectsLimitOfIsCosplitPair (F : C ⥤ D) where
  /-- If `f, g` is an `F`-cosplit pair, then `F` reflects limits for `parallelPair f g`. -/
  out : ∀ {A B} (f g : A ⟶ B) [F.IsCosplitPair f g], ReflectsLimit (parallelPair f g) F


instance {A B} (f g : A ⟶ B) [F.IsCosplitPair f g] [ReflectsLimitOfIsCosplitPair F] :
    ReflectsLimit (parallelPair f g) F := ReflectsLimitOfIsCosplitPair.out f g


instance [ReflectsLimitOfIsCosplitPair F] : ∀ (A : Coalgebra adj.toComonad),
    ReflectsLimit (parallelPair (G.map A.a)
      (NatTrans.app adj.unit (G.obj A.A))) F :=
  fun _ => ReflectsLimitOfIsCosplitPair.out _ _


/-- To show `F` is a comonadic left adjoint, we can show it preserves and reflects `F`-split
equalizers, and `C` has them.
-/
def comonadicOfHasPreservesReflectsFSplitEqualizers [HasEqualizerOfIsCosplitPair F]
    [PreservesLimitOfIsCosplitPair F] [ReflectsLimitOfIsCosplitPair F] :
    ComonadicLeftAdjoint F where
  adj := adj
  eqv := by
    have : ∀ (X : Coalgebra adj.toComonad), IsIso ((comparisonAdjunction adj).counit.app X) := by
      intro X
      apply @isIso_of_reflects_iso _ _ _ _ _ _ _ (Comonad.forget adj.toComonad) ?_ _
      · change IsIso ((comparisonAdjunction adj).counit.app X).f
        rw [comparisonAdjunction_counit_f]
        change
          IsIso
            (IsLimit.conePointUniqueUpToIso (beckEqualizer X)
                (counitLimitOfPreservesEqualizer X)).inv
        exact (IsLimit.conePointUniqueUpToIso _ _).isIso_inv
    have : ∀ (Y : C), IsIso ((comparisonAdjunction adj).unit.app Y) := by
      intro Y
      rw [comparisonAdjunction_unit_app]
      change IsIso (IsLimit.conePointUniqueUpToIso _ ?_).inv
      infer_instance
      apply @unitEqualizerOfCoreflectsEqualizer _ _ _ _ _ _ _ _ ?_
      letI _ :
        F.IsCosplitPair (G.map (F.map (adj.unit.app Y)))
          (adj.unit.app (G.obj (F.obj Y))) :=
        ComonadicityInternal.main_pair_F_cosplit _ ((comparison adj).obj Y)
      infer_instance
    /-
      C : Type u₁
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝² : CategoryTheory.Comonad.HasEqualizerOfIsCosplitPair F
      inst✝¹ : CategoryTheory.Comonad.PreservesLimitOfIsCosplitPair F
      inst✝ : CategoryTheory.Comonad.ReflectsLimitOfIsCosplitPair F
      this✝ : ∀ (X : adj.toComonad.Coalgebra), CategoryTheory.IsIso ((CategoryTheory …
      this : ∀ (Y : C), CategoryTheory.IsIso ((CategoryTheory.Comonad.ComonadicityIn …
      ⊢ (CategoryTheory.Comonad.comparison adj).IsEquivalence
    -/
    exact (comparisonAdjunction adj).toEquivalence.symm.isEquivalence_inverse
    /-
      🎉 no goals
    -/


/-- Dual to `Monad.CreatesColimitOfIsSplitPair`. -/
class CreatesLimitOfIsCosplitPair (F : C ⥤ D) where
  /-- If `f, g` is an `F`-cosplit pair, then `F` creates limits of `parallelPair f g`. -/
  out : ∀ {A B} (f g : A ⟶ B) [F.IsCosplitPair f g], CreatesLimit (parallelPair f g) F


instance {A B} (f g : A ⟶ B) [F.IsCosplitPair f g] [CreatesLimitOfIsCosplitPair F] :
    CreatesLimit (parallelPair f g) F := CreatesLimitOfIsCosplitPair.out f g


instance [CreatesLimitOfIsCosplitPair F] : ∀ (A : Coalgebra adj.toComonad),
    CreatesLimit (parallelPair (G.map A.a)
      (NatTrans.app adj.unit (G.obj A.A))) F :=
  fun _ => CreatesLimitOfIsCosplitPair.out _ _


/--
Beck's comonadicity theorem. If `F` has a right adjoint and creates equalizers of `F`-cosplit pairs,
then it is comonadic.
This is the converse of `createsFSplitEqualizersOfComonadic`.
-/
def comonadicOfCreatesFSplitEqualizers [CreatesLimitOfIsCosplitPair F] :
    ComonadicLeftAdjoint F := by
  let I {A B} (f g : A ⟶ B) [F.IsCosplitPair f g] : HasLimit (parallelPair f g ⋙ F) := by
    apply @hasLimitOfIso _ _ _ _ _ _ ?_ (diagramIsoParallelPair.{v₁} _).symm
    exact inferInstanceAs <| HasEqualizer (F.map f) (F.map g)
  /-
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝ : CategoryTheory.Comonad.CreatesLimitOfIsCosplitPair F
    I : ∀ {A B : C} (f g : Quiver.Hom A B) [inst : F.IsCosplitPair f g], CategoryT …
    ⊢ CategoryTheory.ComonadicLeftAdjoint F
  -/
  have : HasEqualizerOfIsCosplitPair F := ⟨fun _ _ => hasLimit_of_created (parallelPair _ _) F⟩
  /-
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝ : CategoryTheory.Comonad.CreatesLimitOfIsCosplitPair F
    I : ∀ {A B : C} (f g : Quiver.Hom A B) [inst : F.IsCosplitPair f g], CategoryT …
    this : CategoryTheory.Comonad.HasEqualizerOfIsCosplitPair F
    ⊢ CategoryTheory.ComonadicLeftAdjoint F
  -/
  have : PreservesLimitOfIsCosplitPair F := ⟨by intros; infer_instance⟩
  /-
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝ : CategoryTheory.Comonad.CreatesLimitOfIsCosplitPair F
    I : ∀ {A B : C} (f g : Quiver.Hom A B) [inst : F.IsCosplitPair f g], CategoryT …
    this✝ : CategoryTheory.Comonad.HasEqualizerOfIsCosplitPair F
    this : CategoryTheory.Comonad.PreservesLimitOfIsCosplitPair F
    ⊢ CategoryTheory.ComonadicLeftAdjoint F
  -/
  have : ReflectsLimitOfIsCosplitPair F := ⟨by intros; infer_instance⟩
  /-
    C : Type u₁
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝ : CategoryTheory.Comonad.CreatesLimitOfIsCosplitPair F
    I : ∀ {A B : C} (f g : Quiver.Hom A B) [inst : F.IsCosplitPair f g], CategoryT …
    this✝¹ : CategoryTheory.Comonad.HasEqualizerOfIsCosplitPair F
    this✝ : CategoryTheory.Comonad.PreservesLimitOfIsCosplitPair F
    this : CategoryTheory.Comonad.ReflectsLimitOfIsCosplitPair F
    ⊢ CategoryTheory.ComonadicLeftAdjoint F
  -/
  exact comonadicOfHasPreservesReflectsFSplitEqualizers adj
  /-
    🎉 no goals
  -/


/-- An alternate version of Beck's comonadicity theorem. If `F` reflects isomorphisms, preserves
equalizers of `F`-cosplit pairs and `C` has equalizers of `F`-cosplit pairs, then it is comonadic.
-/
def comonadicOfHasPreservesFSplitEqualizersOfReflectsIsomorphisms [F.ReflectsIsomorphisms]
    [HasEqualizerOfIsCosplitPair F] [PreservesLimitOfIsCosplitPair F] :
    ComonadicLeftAdjoint F := by
  have : ReflectsLimitOfIsCosplitPair F := ⟨fun f g _ => by
    have := HasEqualizerOfIsCosplitPair.out F f g
    apply reflectsLimit_of_reflectsIsomorphisms⟩
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Category.{v₁, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    inst✝² : F.ReflectsIsomorphisms
    inst✝¹ : CategoryTheory.Comonad.HasEqualizerOfIsCosplitPair F
    inst✝ : CategoryTheory.Comonad.PreservesLimitOfIsCosplitPair F
    this : CategoryTheory.Comonad.ReflectsLimitOfIsCosplitPair F
    ⊢ CategoryTheory.ComonadicLeftAdjoint F
  -/
  apply comonadicOfHasPreservesReflectsFSplitEqualizers adj
  /-
    🎉 no goals
  -/


/-- Dual to `Monad.PreservesColimitOfIsReflexivePair`. -/
class PreservesLimitOfIsCoreflexivePair (F : C ⥤ D) where
  /-- `f, g` is a coreflexive pair, then `F` preserves limits of `parallelPair f g`. -/
  out : ∀ ⦃A B⦄ (f g : A ⟶ B) [IsCoreflexivePair f g], PreservesLimit (parallelPair f g) F


instance {A B} (f g : A ⟶ B) [IsCoreflexivePair f g] [PreservesLimitOfIsCoreflexivePair F] :
  PreservesLimit (parallelPair f g) F := PreservesLimitOfIsCoreflexivePair.out f g


instance [PreservesLimitOfIsCoreflexivePair F] : ∀ X : Coalgebra adj.toComonad,
    PreservesLimit (parallelPair (G.map X.a)
      (NatTrans.app adj.unit (G.obj X.A))) F :=
 fun _ => PreservesLimitOfIsCoreflexivePair.out _ _


/-- Coreflexive (crude) comonadicity theorem. If `F` has a right adjoint, `C` has and `F` preserves
coreflexive equalizers and `F` reflects isomorphisms, then `F` is comonadic.
-/
def comonadicOfHasPreservesCoreflexiveEqualizersOfReflectsIsomorphisms :
    ComonadicLeftAdjoint F where
  adj := adj
  eqv := by
    have : ∀ (X : adj.toComonad.Coalgebra), IsIso ((comparisonAdjunction adj).counit.app X) := by
      intro X
      apply
        @isIso_of_reflects_iso _ _ _ _ _ _ _ (Comonad.forget adj.toComonad) ?_ _
      · change IsIso ((comparisonAdjunction adj).counit.app X).f
        rw [comparisonAdjunction_counit_f]
        exact (IsLimit.conePointUniqueUpToIso (beckEqualizer X)
          (counitLimitOfPreservesEqualizer X)).isIso_inv
    have : ∀ (Y : C), IsIso ((comparisonAdjunction adj).unit.app Y) := by
      intro Y
      rw [comparisonAdjunction_unit_app]
      change IsIso (IsLimit.conePointUniqueUpToIso _ ?_).inv
      infer_instance
      have : IsCoreflexivePair (G.map (F.map (adj.unit.app Y)))
          (adj.unit.app (G.obj (F.obj Y))) := by
        apply IsCoreflexivePair.mk' (G.map (adj.counit.app _)) _ _
        · rw [← G.map_comp, ← G.map_id]
          exact congr_arg G.map (adj.left_triangle_components Y)
        · rw [← G.map_id]
          simp
      apply @unitEqualizerOfCoreflectsEqualizer _ _ _ _ _ _ _ _ ?_
      apply reflectsLimit_of_reflectsIsomorphisms
    /-
      C : Type u₁
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.Category.{v₁, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝² : CategoryTheory.Limits.HasCoreflexiveEqualizers C
      inst✝¹ : F.ReflectsIsomorphisms
      inst✝ : CategoryTheory.Comonad.PreservesLimitOfIsCoreflexivePair F
      this✝ : ∀ (X : adj.toComonad.Coalgebra), CategoryTheory.IsIso ((CategoryTheory …
      this : ∀ (Y : C), CategoryTheory.IsIso ((CategoryTheory.Comonad.ComonadicityIn …
      ⊢ (CategoryTheory.Comonad.comparison adj).IsEquivalence
    -/
    exact (comparisonAdjunction adj).toEquivalence.symm.isEquivalence_inverse
    /-
      🎉 no goals
    -/


