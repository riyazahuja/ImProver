/-- The subcategory `D` of `C` expressed as an inclusion functor is an *exponential ideal* if
`B ∈ D` implies `A ⟹ B ∈ D` for all `A`.
-/
class ExponentialIdeal : Prop where
  exp_closed : ∀ {B}, B ∈ i.essImage → ∀ A, (A ⟹ B) ∈ i.essImage

/-- To show `i` is an exponential ideal it suffices to show that `A ⟹ iB` is "in" `D` for any `A` in
`C` and `B` in `D`.
-/
theorem ExponentialIdeal.mk' (h : ∀ (B : D) (A : C), (A ⟹ i.obj B) ∈ i.essImage) :
    ExponentialIdeal i :=
  ⟨fun hB A => by
    /-
      C : Type u₁
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Category.{v₁, u₂} D
      i : CategoryTheory.Functor D C
      inst✝¹ : CategoryTheory.ChosenFiniteProducts C
      inst✝ : CategoryTheory.CartesianClosed C
      h : ∀ (B : D) (A : C), Membership.mem i.essImage ((CategoryTheory.exp A).obj ( …
      B✝ : C
      hB : Membership.mem i.essImage B✝
      A : C
      ⊢ Membership.mem i.essImage ((CategoryTheory.exp A).obj B✝)
    -/
    rcases hB with ⟨B', ⟨iB'⟩⟩
    /-
      case intro.intro
      C : Type u₁
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Category.{v₁, u₂} D
      i : CategoryTheory.Functor D C
      inst✝¹ : CategoryTheory.ChosenFiniteProducts C
      inst✝ : CategoryTheory.CartesianClosed C
      h : ∀ (B : D) (A : C), Membership.mem i.essImage ((CategoryTheory.exp A).obj ( …
      B✝ A : C
      B' : D
      iB' : CategoryTheory.Iso (i.obj B') B✝
      ⊢ Membership.mem i.essImage ((CategoryTheory.exp A).obj B✝)
    -/
    exact Functor.essImage.ofIso ((exp A).mapIso iB') (h B' A)⟩
    /-
      🎉 no goals
    -/


/-- The entire category viewed as a subcategory is an exponential ideal. -/
instance : ExponentialIdeal (𝟭 C) :=
  ExponentialIdeal.mk' _ fun _ _ => ⟨_, ⟨Iso.refl _⟩⟩


/-- The subcategory of subterminal objects is an exponential ideal. -/
instance : ExponentialIdeal (subterminalInclusion C) := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts C
    inst✝ : CategoryTheory.CartesianClosed C
    ⊢ CategoryTheory.ExponentialIdeal (CategoryTheory.subterminalInclusion C)
  -/
  apply ExponentialIdeal.mk'
  /-
    case h
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts C
    inst✝ : CategoryTheory.CartesianClosed C
    ⊢ ∀ (B : CategoryTheory.Subterminals C) (A : C), Membership.mem (CategoryTheor …
  -/
  intro B A
  /-
    case h
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts C
    inst✝ : CategoryTheory.CartesianClosed C
    B : CategoryTheory.Subterminals C
    A : C
    ⊢ Membership.mem (CategoryTheory.subterminalInclusion C).essImage ((CategoryTh …
  -/
  refine ⟨⟨A ⟹ B.1, fun Z g h => ?_⟩, ⟨Iso.refl _⟩⟩
  /-
    case h
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts C
    inst✝ : CategoryTheory.CartesianClosed C
    B : CategoryTheory.Subterminals C
    A Z : C
    g h : Quiver.Hom Z ((CategoryTheory.exp A).obj B.obj)
    ⊢ Eq g h
  -/
  exact uncurry_injective (B.2 (CartesianClosed.uncurry g) (CartesianClosed.uncurry h))
  /-
    🎉 no goals
  -/


/-- If `D` is a reflective subcategory, the property of being an exponential ideal is equivalent to
the presence of a natural isomorphism `i ⋙ exp A ⋙ leftAdjoint i ⋙ i ≅ i ⋙ exp A`, that is:
`(A ⟹ iB) ≅ i L (A ⟹ iB)`, naturally in `B`.
The converse is given in `ExponentialIdeal.mk_of_iso`.
-/
def exponentialIdealReflective (A : C) [Reflective i] [ExponentialIdeal i] :
    i ⋙ exp A ⋙ reflector i ⋙ i ≅ i ⋙ exp A := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝³ : CategoryTheory.ChosenFiniteProducts C
    inst✝² : CategoryTheory.CartesianClosed C
    A : C
    inst✝¹ : CategoryTheory.Reflective i
    inst✝ : CategoryTheory.ExponentialIdeal i
    ⊢ CategoryTheory.Iso (i.comp ((CategoryTheory.exp A).comp ((CategoryTheory.ref …
  -/
  symm
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝³ : CategoryTheory.ChosenFiniteProducts C
    inst✝² : CategoryTheory.CartesianClosed C
    A : C
    inst✝¹ : CategoryTheory.Reflective i
    inst✝ : CategoryTheory.ExponentialIdeal i
    ⊢ CategoryTheory.Iso (i.comp (CategoryTheory.exp A)) (i.comp ((CategoryTheory. …
  -/
  apply NatIso.ofComponents _ _
    /-
      C : Type u₁
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁴ : CategoryTheory.Category.{v₁, u₂} D
      i : CategoryTheory.Functor D C
      inst✝³ : CategoryTheory.ChosenFiniteProducts C
      inst✝² : CategoryTheory.CartesianClosed C
      A : C
      inst✝¹ : CategoryTheory.Reflective i
      inst✝ : CategoryTheory.ExponentialIdeal i
      ⊢ (X : D) → CategoryTheory.Iso ((i.comp (CategoryTheory.exp A)).obj X) ((i.com …
    -/
  · intro X
    /-
      C : Type u₁
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁴ : CategoryTheory.Category.{v₁, u₂} D
      i : CategoryTheory.Functor D C
      inst✝³ : CategoryTheory.ChosenFiniteProducts C
      inst✝² : CategoryTheory.CartesianClosed C
      A : C
      inst✝¹ : CategoryTheory.Reflective i
      inst✝ : CategoryTheory.ExponentialIdeal i
      X : D
      ⊢ CategoryTheory.Iso ((i.comp (CategoryTheory.exp A)).obj X) ((i.comp ((Catego …
    -/
    haveI := Functor.essImage.unit_isIso (ExponentialIdeal.exp_closed (i.obj_mem_essImage X) A)
    /-
      C : Type u₁
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁴ : CategoryTheory.Category.{v₁, u₂} D
      i : CategoryTheory.Functor D C
      inst✝³ : CategoryTheory.ChosenFiniteProducts C
      inst✝² : CategoryTheory.CartesianClosed C
      A : C
      inst✝¹ : CategoryTheory.Reflective i
      inst✝ : CategoryTheory.ExponentialIdeal i
      X : D
      this : CategoryTheory.IsIso ((CategoryTheory.reflectorAdjunction i).unit.app ( …
      ⊢ CategoryTheory.Iso ((i.comp (CategoryTheory.exp A)).obj X) ((i.comp ((Catego …
    -/
    apply asIso ((reflectorAdjunction i).unit.app (A ⟹ i.obj X))
    /-
      🎉 no goals
    -/
    /-
      C : Type u₁
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁴ : CategoryTheory.Category.{v₁, u₂} D
      i : CategoryTheory.Functor D C
      inst✝³ : CategoryTheory.ChosenFiniteProducts C
      inst✝² : CategoryTheory.CartesianClosed C
      A : C
      inst✝¹ : CategoryTheory.Reflective i
      inst✝ : CategoryTheory.ExponentialIdeal i
      ⊢ ∀ {X Y : D} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((i …
    -/
  · simp [asIso]
    /-
      🎉 no goals
    -/


/-- Given a natural isomorphism `i ⋙ exp A ⋙ leftAdjoint i ⋙ i ≅ i ⋙ exp A`, we can show `i`
is an exponential ideal.
-/
theorem ExponentialIdeal.mk_of_iso [Reflective i]
    (h : ∀ A : C, i ⋙ exp A ⋙ reflector i ⋙ i ≅ i ⋙ exp A) : ExponentialIdeal i := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.ChosenFiniteProducts C
    inst✝¹ : CategoryTheory.CartesianClosed C
    inst✝ : CategoryTheory.Reflective i
    h : (A : C) → CategoryTheory.Iso (i.comp ((CategoryTheory.exp A).comp ((Catego …
    ⊢ CategoryTheory.ExponentialIdeal i
  -/
  apply ExponentialIdeal.mk'
  /-
    case h
    C : Type u₁
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.ChosenFiniteProducts C
    inst✝¹ : CategoryTheory.CartesianClosed C
    inst✝ : CategoryTheory.Reflective i
    h : (A : C) → CategoryTheory.Iso (i.comp ((CategoryTheory.exp A).comp ((Catego …
    ⊢ ∀ (B : D) (A : C), Membership.mem i.essImage ((CategoryTheory.exp A).obj (i. …
  -/
  intro B A
  /-
    case h
    C : Type u₁
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝² : CategoryTheory.ChosenFiniteProducts C
    inst✝¹ : CategoryTheory.CartesianClosed C
    inst✝ : CategoryTheory.Reflective i
    h : (A : C) → CategoryTheory.Iso (i.comp ((CategoryTheory.exp A).comp ((Catego …
    B : D
    A : C
    ⊢ Membership.mem i.essImage ((CategoryTheory.exp A).obj (i.obj B))
  -/
  exact ⟨_, ⟨(h A).app B⟩⟩
  /-
    🎉 no goals
  -/


theorem reflective_products [Limits.HasFiniteProducts C] [Reflective i] :
    Limits.HasFiniteProducts D := ⟨fun _ => hasLimitsOfShape_of_reflective i⟩


open Limits in
/-- Given a reflective subcategory `D` of a category with chosen finite products `C`, `D` admits
finite chosen products. -/
-- Note: This is not an instance as one might already have a (different) `ChosenFiniteProducts`
-- instance on `D` (as for example with sheaves).
def reflectiveChosenFiniteProducts [ChosenFiniteProducts C] [Reflective i] :
    ChosenFiniteProducts D where
  product X Y :=
    { cone := BinaryFan.mk
        ((reflector i).map (fst (i.obj X) (i.obj Y)) ≫ (reflectorAdjunction i).counit.app _)
        ((reflector i).map (snd (i.obj X) (i.obj Y)) ≫ (reflectorAdjunction i).counit.app _)
      isLimit := by
        /-
          C : Type u₁
          D : Type u₂
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Category.{v₁, u₂} D
          i : CategoryTheory.Functor D C
          inst✝¹ : CategoryTheory.ChosenFiniteProducts C
          inst✝ : CategoryTheory.Reflective i
          X Y : D
          ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.BinaryFan.mk (CategoryT …
        -/
        apply isLimitOfReflects i
        apply IsLimit.equivOfNatIsoOfIso (pairComp X Y _) _ _ _|>.invFun
          (product (i.obj X) (i.obj Y)).isLimit
        /-
          C : Type u₁
          D : Type u₂
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Category.{v₁, u₂} D
          i : CategoryTheory.Functor D C
          inst✝¹ : CategoryTheory.ChosenFiniteProducts C
          inst✝ : CategoryTheory.Reflective i
          X Y : D
          ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.Cones.postcompose (CategoryTheory …
        -/
        fapply BinaryFan.ext
          /-
            case e
            C : Type u₁
            D : Type u₂
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.Category.{v₁, u₂} D
            i : CategoryTheory.Functor D C
            inst✝¹ : CategoryTheory.ChosenFiniteProducts C
            inst✝ : CategoryTheory.Reflective i
            X Y : D
            ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.Cones.postcompose (CategoryTheory …
          -/
        · change (reflector i ⋙ i).obj (i.obj X ⊗ i.obj Y) ≅ (𝟭 C).obj (i.obj X ⊗ i.obj Y)
          letI : IsIso ((reflectorAdjunction i).unit.app (i.obj X ⊗ i.obj Y)) := by
            apply Functor.essImage.unit_isIso
            haveI := reflective_products i
            use Limits.prod X Y
            constructor
            apply Limits.PreservesLimitPair.iso i _ _|>.trans
            refine Limits.IsLimit.conePointUniqueUpToIso (limit.isLimit (pair (i.obj X) (i.obj Y)))
              (ChosenFiniteProducts.product _ _).isLimit
          /-
            case e
            C : Type u₁
            D : Type u₂
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.Category.{v₁, u₂} D
            i : CategoryTheory.Functor D C
            inst✝¹ : CategoryTheory.ChosenFiniteProducts C
            inst✝ : CategoryTheory.Reflective i
            X Y : D
            this : CategoryTheory.IsIso ((CategoryTheory.reflectorAdjunction i).unit.app ( …
            ⊢ CategoryTheory.Iso (((CategoryTheory.reflector i).comp i).obj (CategoryTheor …
          -/
          exact asIso ((reflectorAdjunction i).unit.app (i.obj X ⊗ i.obj Y))|>.symm
          /-
            🎉 no goals
          -/
          /-
            case h₁
            C : Type u₁
            D : Type u₂
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.Category.{v₁, u₂} D
            i : CategoryTheory.Functor D C
            inst✝¹ : CategoryTheory.ChosenFiniteProducts C
            inst✝ : CategoryTheory.Reflective i
            X Y : D
            ⊢ Eq (CategoryTheory.Limits.BinaryFan.fst ((CategoryTheory.Limits.Cones.postco …
          -/
        · simp only [BinaryFan.fst, Cones.postcompose, pairComp]
          /-
            case h₁
            C : Type u₁
            D : Type u₂
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.Category.{v₁, u₂} D
            i : CategoryTheory.Functor D C
            inst✝¹ : CategoryTheory.ChosenFiniteProducts C
            inst✝ : CategoryTheory.Reflective i
            X Y : D
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp (i.mapCone (CategoryTheory.Limits.Bi …
          -/
          simp [← Functor.comp_map, ← NatTrans.naturality_assoc, fst]
          /-
            🎉 no goals
          -/
          /-
            case h₂
            C : Type u₁
            D : Type u₂
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.Category.{v₁, u₂} D
            i : CategoryTheory.Functor D C
            inst✝¹ : CategoryTheory.ChosenFiniteProducts C
            inst✝ : CategoryTheory.Reflective i
            X Y : D
            ⊢ Eq (CategoryTheory.Limits.BinaryFan.snd ((CategoryTheory.Limits.Cones.postco …
          -/
        · simp only [BinaryFan.snd, Cones.postcompose, pairComp]
          /-
            case h₂
            C : Type u₁
            D : Type u₂
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.Category.{v₁, u₂} D
            i : CategoryTheory.Functor D C
            inst✝¹ : CategoryTheory.ChosenFiniteProducts C
            inst✝ : CategoryTheory.Reflective i
            X Y : D
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp (i.mapCone (CategoryTheory.Limits.Bi …
          -/
          simp [← Functor.comp_map, ← NatTrans.naturality_assoc, snd] }
          /-
            🎉 no goals
          -/
  terminal :=
    { cone := Limits.asEmptyCone <| (reflector i).obj (𝟙_ C)
      isLimit := by
        /-
          C : Type u₁
          D : Type u₂
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Category.{v₁, u₂} D
          i : CategoryTheory.Functor D C
          inst✝¹ : CategoryTheory.ChosenFiniteProducts C
          inst✝ : CategoryTheory.Reflective i
          ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.asEmptyCone ((CategoryT …
        -/
        apply isLimitOfReflects i
        /-
          case t
          C : Type u₁
          D : Type u₂
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Category.{v₁, u₂} D
          i : CategoryTheory.Functor D C
          inst✝¹ : CategoryTheory.ChosenFiniteProducts C
          inst✝ : CategoryTheory.Reflective i
          ⊢ CategoryTheory.Limits.IsLimit (i.mapCone (CategoryTheory.Limits.asEmptyCone  …
        -/
        apply isLimitChangeEmptyCone _ ChosenFiniteProducts.terminal.isLimit
        letI : IsIso ((reflectorAdjunction i).unit.app (𝟙_ C)) := by
          apply Functor.essImage.unit_isIso
          haveI := reflective_products i
          use Limits.terminal D
          constructor
          apply Limits.PreservesTerminal.iso i|>.trans
          refine Limits.IsLimit.conePointUniqueUpToIso (limit.isLimit _)
            (ChosenFiniteProducts.terminal).isLimit
        /-
          case t.hi
          C : Type u₁
          D : Type u₂
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.Category.{v₁, u₂} D
          i : CategoryTheory.Functor D C
          inst✝¹ : CategoryTheory.ChosenFiniteProducts C
          inst✝ : CategoryTheory.Reflective i
          this : CategoryTheory.IsIso ((CategoryTheory.reflectorAdjunction i).unit.app C …
          ⊢ CategoryTheory.Iso CategoryTheory.ChosenFiniteProducts.terminal.cone.pt (i.m …
        -/
        exact asIso ((reflectorAdjunction i).unit.app (𝟙_ C)) }
        /-
          🎉 no goals
        -/


/-- If the reflector preserves binary products, the subcategory is an exponential ideal.
This is the converse of `preservesBinaryProductsOfExponentialIdeal`.
-/
instance (priority := 10) exponentialIdeal_of_preservesBinaryProducts
    [Limits.PreservesLimitsOfShape (Discrete Limits.WalkingPair) (reflector i)] :
    ExponentialIdeal i := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.Reflective i
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    ⊢ CategoryTheory.ExponentialIdeal i
  -/
  let ir := reflectorAdjunction i
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.Reflective i
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    ir : CategoryTheory.Adjunction (CategoryTheory.reflector i) i := CategoryTheor …
    ⊢ CategoryTheory.ExponentialIdeal i
  -/
  let L : C ⥤ D := reflector i
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.Reflective i
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    ir : CategoryTheory.Adjunction (CategoryTheory.reflector i) i := CategoryTheor …
    L : CategoryTheory.Functor C D := CategoryTheory.reflector i
    ⊢ CategoryTheory.ExponentialIdeal i
  -/
  let η : 𝟭 C ⟶ L ⋙ i := ir.unit
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.Reflective i
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    ir : CategoryTheory.Adjunction (CategoryTheory.reflector i) i := CategoryTheor …
    L : CategoryTheory.Functor C D := CategoryTheory.reflector i
    η : Quiver.Hom (CategoryTheory.Functor.id C) (L.comp i) := ir.unit
    ⊢ CategoryTheory.ExponentialIdeal i
  -/
  let ε : i ⋙ L ⟶ 𝟭 D := ir.counit
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.Reflective i
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    ir : CategoryTheory.Adjunction (CategoryTheory.reflector i) i := CategoryTheor …
    L : CategoryTheory.Functor C D := CategoryTheory.reflector i
    η : Quiver.Hom (CategoryTheory.Functor.id C) (L.comp i) := ir.unit
    ε : Quiver.Hom (i.comp L) (CategoryTheory.Functor.id D) := ir.counit
    ⊢ CategoryTheory.ExponentialIdeal i
  -/
  apply ExponentialIdeal.mk'
  /-
    case h
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.Reflective i
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    ir : CategoryTheory.Adjunction (CategoryTheory.reflector i) i := CategoryTheor …
    L : CategoryTheory.Functor C D := CategoryTheory.reflector i
    η : Quiver.Hom (CategoryTheory.Functor.id C) (L.comp i) := ir.unit
    ε : Quiver.Hom (i.comp L) (CategoryTheory.Functor.id D) := ir.counit
    ⊢ ∀ (B : D) (A : C), Membership.mem i.essImage ((CategoryTheory.exp A).obj (i. …
  -/
  intro B A
  let q : i.obj (L.obj (A ⟹ i.obj B)) ⟶ A ⟹ i.obj B := by
    apply CartesianClosed.curry (ir.homEquiv _ _ _)
    apply _ ≫ (ir.homEquiv _ _).symm ((exp.ev A).app (i.obj B))
    exact prodComparison L A _ ≫ (_ ◁ (ε.app _)) ≫ inv (prodComparison _ _ _)
  have : η.app (A ⟹ i.obj B) ≫ q = 𝟙 (A ⟹ i.obj B) := by
    dsimp
    rw [← curry_natural_left, curry_eq_iff, uncurry_id_eq_ev, ← ir.homEquiv_naturality_left,
      ir.homEquiv_apply_eq, assoc, assoc, prodComparison_natural_whiskerLeft_assoc,
      ← MonoidalCategory.whiskerLeft_comp_assoc,
      ir.left_triangle_components, MonoidalCategory.whiskerLeft_id, id_comp]
    apply IsIso.hom_inv_id_assoc
  /-
    case h
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.Reflective i
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    ir : CategoryTheory.Adjunction (CategoryTheory.reflector i) i := CategoryTheor …
    L : CategoryTheory.Functor C D := CategoryTheory.reflector i
    η : Quiver.Hom (CategoryTheory.Functor.id C) (L.comp i) := ir.unit
    ε : Quiver.Hom (i.comp L) (CategoryTheory.Functor.id D) := ir.counit
    B : D
    A : C
    q : Quiver.Hom (i.obj (L.obj ((CategoryTheory.exp A).obj (i.obj B)))) ((Catego …
    this : Eq (CategoryTheory.CategoryStruct.comp (η.app ((CategoryTheory.exp A).o …
    ⊢ Membership.mem i.essImage ((CategoryTheory.exp A).obj (i.obj B))
  -/
  haveI : IsSplitMono (η.app (A ⟹ i.obj B)) := IsSplitMono.mk' ⟨_, this⟩
  /-
    case h
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.Reflective i
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    ir : CategoryTheory.Adjunction (CategoryTheory.reflector i) i := CategoryTheor …
    L : CategoryTheory.Functor C D := CategoryTheory.reflector i
    η : Quiver.Hom (CategoryTheory.Functor.id C) (L.comp i) := ir.unit
    ε : Quiver.Hom (i.comp L) (CategoryTheory.Functor.id D) := ir.counit
    B : D
    A : C
    q : Quiver.Hom (i.obj (L.obj ((CategoryTheory.exp A).obj (i.obj B)))) ((Catego …
    this✝ : Eq (CategoryTheory.CategoryStruct.comp (η.app ((CategoryTheory.exp A). …
    this : CategoryTheory.IsSplitMono (η.app ((CategoryTheory.exp A).obj (i.obj B)))
    ⊢ Membership.mem i.essImage ((CategoryTheory.exp A).obj (i.obj B))
  -/
  apply mem_essImage_of_unit_isSplitMono
  /-
    🎉 no goals
  -/


/-- If `i` witnesses that `D` is a reflective subcategory and an exponential ideal, then `D` is
itself cartesian closed.
-/
def cartesianClosedOfReflective : CartesianClosed D where
  closed := fun B =>
    { rightAdj := i ⋙ exp (i.obj B) ⋙ reflector i
      adj := by
        apply (exp.adjunction (i.obj B)).restrictFullyFaithful i.fullyFaithfulOfReflective
          i.fullyFaithfulOfReflective
          /-
            case comm1
            C : Type u₁
            D : Type u₂
            inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
            inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
            i : CategoryTheory.Functor D C
            inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
            inst✝³ : CategoryTheory.Reflective i
            inst✝² : CategoryTheory.CartesianClosed C
            inst✝¹ : CategoryTheory.ChosenFiniteProducts D
            inst✝ : CategoryTheory.ExponentialIdeal i
            B : D
            ⊢ CategoryTheory.Iso (i.comp (CategoryTheory.MonoidalCategory.tensorLeft (i.ob …
          -/
        · symm
          /-
            case comm1
            C : Type u₁
            D : Type u₂
            inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
            inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
            i : CategoryTheory.Functor D C
            inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
            inst✝³ : CategoryTheory.Reflective i
            inst✝² : CategoryTheory.CartesianClosed C
            inst✝¹ : CategoryTheory.ChosenFiniteProducts D
            inst✝ : CategoryTheory.ExponentialIdeal i
            B : D
            ⊢ CategoryTheory.Iso ((CategoryTheory.MonoidalCategory.tensorLeft B).comp i) ( …
          -/
          refine NatIso.ofComponents (fun X => ?_) (fun f => ?_)
          · haveI :=
              Adjunction.rightAdjoint_preservesLimits.{0, 0} (reflectorAdjunction i)
            /-
              case comm1.refine_1
              C : Type u₁
              D : Type u₂
              inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
              inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
              i : CategoryTheory.Functor D C
              inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
              inst✝³ : CategoryTheory.Reflective i
              inst✝² : CategoryTheory.CartesianClosed C
              inst✝¹ : CategoryTheory.ChosenFiniteProducts D
              inst✝ : CategoryTheory.ExponentialIdeal i
              B X : D
              this : CategoryTheory.Limits.PreservesLimitsOfSize.{0, 0, v₁, v₁, u₂, u₁} i
              ⊢ CategoryTheory.Iso (((CategoryTheory.MonoidalCategory.tensorLeft B).comp i). …
            -/
            apply asIso (prodComparison i B X)
            /-
              🎉 no goals
            -/
            /-
              case comm1.refine_2
              C : Type u₁
              D : Type u₂
              inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
              inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
              i : CategoryTheory.Functor D C
              inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
              inst✝³ : CategoryTheory.Reflective i
              inst✝² : CategoryTheory.CartesianClosed C
              inst✝¹ : CategoryTheory.ChosenFiniteProducts D
              inst✝ : CategoryTheory.ExponentialIdeal i
              B X✝ Y✝ : D
              f : Quiver.Hom X✝ Y✝
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.MonoidalCategory.te …
            -/
          · dsimp [asIso]
            /-
              case comm1.refine_2
              C : Type u₁
              D : Type u₂
              inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
              inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
              i : CategoryTheory.Functor D C
              inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
              inst✝³ : CategoryTheory.Reflective i
              inst✝² : CategoryTheory.CartesianClosed C
              inst✝¹ : CategoryTheory.ChosenFiniteProducts D
              inst✝ : CategoryTheory.ExponentialIdeal i
              B X✝ Y✝ : D
              f : Quiver.Hom X✝ Y✝
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (i.map (CategoryTheory.MonoidalCatego …
            -/
            rw [prodComparison_natural_whiskerLeft]
            /-
              🎉 no goals
            -/
          /-
            case comm2
            C : Type u₁
            D : Type u₂
            inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
            inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
            i : CategoryTheory.Functor D C
            inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
            inst✝³ : CategoryTheory.Reflective i
            inst✝² : CategoryTheory.CartesianClosed C
            inst✝¹ : CategoryTheory.ChosenFiniteProducts D
            inst✝ : CategoryTheory.ExponentialIdeal i
            B : D
            ⊢ CategoryTheory.Iso (i.comp (CategoryTheory.exp (i.obj B))) ((i.comp ((Catego …
          -/
        · apply (exponentialIdealReflective i _).symm }
          /-
            🎉 no goals
          -/

-- It's annoying that I need to do this.

/-- We construct a bijection between morphisms `L(A ⊗ B) ⟶ X` and morphisms `LA ⊗ LB ⟶ X`.
This bijection has two key properties:
* It is natural in `X`: See `bijection_natural`.
* When `X = LA ⨯ LB`, then the backwards direction sends the identity morphism to the product
  comparison morphism: See `bijection_symm_apply_id`.

Together these help show that `L` preserves binary products. This should be considered
*internal implementation* towards `preservesBinaryProductsOfExponentialIdeal`.
-/
noncomputable def bijection (A B : C) (X : D) :
    ((reflector i).obj (A ⊗ B) ⟶ X) ≃ ((reflector i).obj A ⊗ (reflector i).obj B ⟶ X) :=
  calc
    _ ≃ (A ⊗ B ⟶ i.obj X) := (reflectorAdjunction i).homEquiv _ _
    _ ≃ (B ⊗ A ⟶ i.obj X) := (β_ _ _).homCongr (Iso.refl _)
    _ ≃ (A ⟶ B ⟹ i.obj X) := (exp.adjunction _).homEquiv _ _
    _ ≃ (i.obj ((reflector i).obj A) ⟶ B ⟹ i.obj X) :=
      (unitCompPartialBijective _ (ExponentialIdeal.exp_closed (i.obj_mem_essImage _) _))
    _ ≃ (B ⊗ i.obj ((reflector i).obj A) ⟶ i.obj X) := ((exp.adjunction _).homEquiv _ _).symm
    _ ≃ (i.obj ((reflector i).obj A) ⊗ B ⟶ i.obj X) :=
      ((β_ _ _).homCongr (Iso.refl _))
    _ ≃ (B ⟶ i.obj ((reflector i).obj A) ⟹ i.obj X) := (exp.adjunction _).homEquiv _ _
    _ ≃ (i.obj ((reflector i).obj B) ⟶ i.obj ((reflector i).obj A) ⟹ i.obj X) :=
      (unitCompPartialBijective _ (ExponentialIdeal.exp_closed (i.obj_mem_essImage _) _))
    _ ≃ (i.obj ((reflector i).obj A) ⊗ i.obj ((reflector i).obj B) ⟶ i.obj X) :=
      ((exp.adjunction _).homEquiv _ _).symm
    _ ≃ (i.obj ((reflector i).obj A ⊗ (reflector i).obj B) ⟶ i.obj X) :=
      haveI : Limits.PreservesLimits i := (reflectorAdjunction i).rightAdjoint_preservesLimits
      haveI := Limits.preservesSmallestLimits_of_preservesLimits i
      Iso.homCongr (prodComparisonIso _ _ _).symm (Iso.refl (i.obj X))
    _ ≃ ((reflector i).obj A ⊗ (reflector i).obj B ⟶ X) :=
      i.fullyFaithfulOfReflective.homEquiv.symm


theorem bijection_symm_apply_id (A B : C) :
    (bijection i A B _).symm (𝟙 _) = prodComparison _ _ _ := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.Reflective i
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    inst✝ : CategoryTheory.ExponentialIdeal i
    A B : C
    ⊢ Eq ((CategoryTheory.bijection i A B (CategoryTheory.MonoidalCategoryStruct.t …
  -/
  dsimp [bijection]
  -- Porting note: added
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.Reflective i
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    inst✝ : CategoryTheory.ExponentialIdeal i
    A B : C
    ⊢ Eq (((CategoryTheory.reflectorAdjunction i).homEquiv (CategoryTheory.Monoida …
  -/
  erw [homEquiv_symm_apply_eq, homEquiv_symm_apply_eq, homEquiv_apply_eq, homEquiv_apply_eq]
  rw [comp_id, comp_id, comp_id, i.map_id, comp_id, unitCompPartialBijective_symm_apply,
    unitCompPartialBijective_symm_apply, uncurry_natural_left, uncurry_curry,
    uncurry_natural_left, uncurry_curry, ← BraidedCategory.braiding_naturality_left_assoc]
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.Reflective i
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    inst✝ : CategoryTheory.ExponentialIdeal i
    A B : C
    ⊢ Eq (((CategoryTheory.reflectorAdjunction i).homEquiv (CategoryTheory.Monoida …
  -/
  erw [SymmetricCategory.symmetry_assoc, ← MonoidalCategory.whisker_exchange_assoc]
  -- Porting note: added
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.Reflective i
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    inst✝ : CategoryTheory.ExponentialIdeal i
    A B : C
    ⊢ Eq (((CategoryTheory.reflectorAdjunction i).homEquiv (CategoryTheory.Monoida …
  -/
  dsimp only [Functor.comp_obj]
  rw [← tensorHom_def'_assoc, Adjunction.homEquiv_symm_apply,
    ← Adjunction.eq_unit_comp_map_iff, Iso.comp_inv_eq, assoc]
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.Reflective i
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    inst✝ : CategoryTheory.ExponentialIdeal i
    A B : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom ((CategoryTheory.reflect …
  -/
  rw [prodComparisonIso_hom i ((reflector i).obj A) ((reflector i).obj B)]
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.Reflective i
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    inst✝ : CategoryTheory.ExponentialIdeal i
    A B : C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorHom ((CategoryTheory.reflect …
  -/
  apply hom_ext
  · rw [tensorHom_fst, assoc, assoc, prodComparison_fst, ← i.map_comp,
    prodComparison_fst]
    /-
      case h_fst
      C : Type u₁
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
      i : CategoryTheory.Functor D C
      inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
      inst✝³ : CategoryTheory.Reflective i
      inst✝² : CategoryTheory.CartesianClosed C
      inst✝¹ : CategoryTheory.ChosenFiniteProducts D
      inst✝ : CategoryTheory.ExponentialIdeal i
      A B : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts. …
    -/
    apply (reflectorAdjunction i).unit.naturality
    /-
      🎉 no goals
    -/
  · rw [tensorHom_snd, assoc, assoc, prodComparison_snd, ← i.map_comp,
    prodComparison_snd]
    /-
      case h_snd
      C : Type u₁
      D : Type u₂
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
      i : CategoryTheory.Functor D C
      inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
      inst✝³ : CategoryTheory.Reflective i
      inst✝² : CategoryTheory.CartesianClosed C
      inst✝¹ : CategoryTheory.ChosenFiniteProducts D
      inst✝ : CategoryTheory.ExponentialIdeal i
      A B : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts. …
    -/
    apply (reflectorAdjunction i).unit.naturality
    /-
      🎉 no goals
    -/


theorem bijection_natural (A B : C) (X X' : D) (f : (reflector i).obj (A ⊗ B) ⟶ X) (g : X ⟶ X') :
    bijection i _ _ _ (f ≫ g) = bijection i _ _ _ f ≫ g := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.Reflective i
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    inst✝ : CategoryTheory.ExponentialIdeal i
    A B : C
    X X' : D
    f : Quiver.Hom ((CategoryTheory.reflector i).obj (CategoryTheory.MonoidalCateg …
    g : Quiver.Hom X X'
    ⊢ Eq ((CategoryTheory.bijection i A B X') (CategoryTheory.CategoryStruct.comp  …
  -/
  dsimp [bijection]
  -- Porting note: added
  erw [homEquiv_symm_apply_eq, homEquiv_symm_apply_eq, homEquiv_apply_eq, homEquiv_apply_eq,
    homEquiv_symm_apply_eq, homEquiv_symm_apply_eq, homEquiv_apply_eq, homEquiv_apply_eq]
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.Reflective i
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    inst✝ : CategoryTheory.ExponentialIdeal i
    A B : C
    X X' : D
    f : Quiver.Hom ((CategoryTheory.reflector i).obj (CategoryTheory.MonoidalCateg …
    g : Quiver.Hom X X'
    ⊢ Eq (i.fullyFaithfulOfReflective.preimage (CategoryTheory.CategoryStruct.comp …
  -/
  apply i.map_injective
  rw [Functor.FullyFaithful.map_preimage, i.map_comp,
    Adjunction.homEquiv_unit, Adjunction.homEquiv_unit]
  /-
    case a
    C : Type u₁
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
    inst✝³ : CategoryTheory.Reflective i
    inst✝² : CategoryTheory.CartesianClosed C
    inst✝¹ : CategoryTheory.ChosenFiniteProducts D
    inst✝ : CategoryTheory.ExponentialIdeal i
    A B : C
    X X' : D
    f : Quiver.Hom ((CategoryTheory.reflector i).obj (CategoryTheory.MonoidalCateg …
    g : Quiver.Hom X X'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ChosenFiniteProducts. …
  -/
  simp only [comp_id, Functor.map_comp, Functor.FullyFaithful.map_preimage, assoc]
  rw [← assoc, ← assoc, curry_natural_right _ (i.map g),
    unitCompPartialBijective_natural, uncurry_natural_right, ← assoc, curry_natural_right,
    unitCompPartialBijective_natural, uncurry_natural_right, assoc]


/--
The bijection allows us to show that `prodComparison L A B` is an isomorphism, where the inverse
is the forward map of the identity morphism.
-/
theorem prodComparison_iso (A B : C) : IsIso
    (prodComparison (reflector i) A B) :=
  ⟨⟨bijection i _ _ _ (𝟙 _), by
      rw [← (bijection i _ _ _).injective.eq_iff, bijection_natural, ← bijection_symm_apply_id,
        Equiv.apply_symm_apply, id_comp],
         /-
           C : Type u₁
           D : Type u₂
           inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
           inst✝⁵ : CategoryTheory.Category.{v₁, u₂} D
           i : CategoryTheory.Functor D C
           inst✝⁴ : CategoryTheory.ChosenFiniteProducts C
           inst✝³ : CategoryTheory.Reflective i
           inst✝² : CategoryTheory.CartesianClosed C
           inst✝¹ : CategoryTheory.ChosenFiniteProducts D
           inst✝ : CategoryTheory.ExponentialIdeal i
           A B : C
           ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.bijection i A B ((Ca …
         -/
      by rw [← bijection_natural, id_comp, ← bijection_symm_apply_id, Equiv.apply_symm_apply]⟩⟩
         /-
           🎉 no goals
         -/


/--
If a reflective subcategory is an exponential ideal, then the reflector preserves binary products.
This is the converse of `exponentialIdeal_of_preserves_binary_products`.
-/
lemma preservesBinaryProducts_of_exponentialIdeal :
    PreservesLimitsOfShape (Discrete WalkingPair) (reflector i) where
  preservesLimit {K} :=
    letI := preservesLimit_pair_of_isIso_prodComparison
      (reflector i) (K.obj ⟨WalkingPair.left⟩) (K.obj ⟨WalkingPair.right⟩)
    Limits.preservesLimit_of_iso_diagram _ (diagramIsoPair K).symm


/--
If a reflective subcategory is an exponential ideal, then the reflector preserves finite products.
-/
lemma preservesFiniteProducts_of_exponentialIdeal (J : Type) [Fintype J] :
    PreservesLimitsOfShape (Discrete J) (reflector i) := by
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁵ : CategoryTheory.ChosenFiniteProducts C
    inst✝⁴ : CategoryTheory.Reflective i
    inst✝³ : CategoryTheory.CartesianClosed C
    inst✝² : CategoryTheory.ChosenFiniteProducts D
    inst✝¹ : CategoryTheory.ExponentialIdeal i
    J : Type
    inst✝ : Fintype J
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete J) (Ca …
  -/
  letI := preservesBinaryProducts_of_exponentialIdeal i
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁵ : CategoryTheory.ChosenFiniteProducts C
    inst✝⁴ : CategoryTheory.Reflective i
    inst✝³ : CategoryTheory.CartesianClosed C
    inst✝² : CategoryTheory.ChosenFiniteProducts D
    inst✝¹ : CategoryTheory.ExponentialIdeal i
    J : Type
    inst✝ : Fintype J
    this : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete C …
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete J) (Ca …
  -/
  letI : PreservesLimitsOfShape _ (reflector i) := leftAdjoint_preservesTerminal_of_reflective.{0} i
  /-
    C : Type u₁
    D : Type u₂
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.Category.{v₁, u₂} D
    i : CategoryTheory.Functor D C
    inst✝⁵ : CategoryTheory.ChosenFiniteProducts C
    inst✝⁴ : CategoryTheory.Reflective i
    inst✝³ : CategoryTheory.CartesianClosed C
    inst✝² : CategoryTheory.ChosenFiniteProducts D
    inst✝¹ : CategoryTheory.ExponentialIdeal i
    J : Type
    inst✝ : Fintype J
    this✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    this : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete P …
    ⊢ CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete J) (Ca …
  -/
  apply preservesFiniteProducts_of_preserves_binary_and_terminal (reflector i) J
  /-
    🎉 no goals
  -/


