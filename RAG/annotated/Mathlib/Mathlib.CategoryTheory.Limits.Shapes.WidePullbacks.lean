/-- A wide pullback shape for any type `J` can be written simply as `Option J`. -/
def WidePullbackShape := Option J

-- Porting note: strangely this could be synthesized

instance : Inhabited (WidePullbackShape J) where
  default := none


/-- A wide pushout shape for any type `J` can be written simply as `Option J`. -/
def WidePushoutShape := Option J


instance : Inhabited (WidePushoutShape J) where
  default := none


/-- The type of arrows for the shape indexing a wide pullback. -/
inductive Hom : WidePullbackShape J → WidePullbackShape J → Type w
  | id : ∀ X, Hom X X
  | term : ∀ j : J, Hom (some j) none
  deriving DecidableEq

-- This is relying on an automatically generated instance name, generated in a `deriving` handler.
-- See https://github.com/leanprover/lean4/issues/2343

instance struct : CategoryStruct (WidePullbackShape J) where
  Hom := Hom
  id j := Hom.id j
  comp f g := by
    /-
      J : Type w
      X✝ Y✝ Z✝ : CategoryTheory.Limits.WidePullbackShape J
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Quiver.Hom X✝ Z✝
    -/
    cases f
      /-
        case id
        J : Type w
        X✝ Z✝ : CategoryTheory.Limits.WidePullbackShape J
        g : Quiver.Hom X✝ Z✝
        ⊢ Quiver.Hom X✝ Z✝
      -/
    · exact g
      /-
        🎉 no goals
      -/
    /-
      case term
      J : Type w
      Z✝ : CategoryTheory.Limits.WidePullbackShape J
      j✝ : J
      g : Quiver.Hom Option.none Z✝
      ⊢ Quiver.Hom (Option.some j✝) Z✝
    -/
    cases g
    /-
      case term.id
      J : Type w
      j✝ : J
      ⊢ Quiver.Hom (Option.some j✝) Option.none
    -/
    apply Hom.term _
    /-
      🎉 no goals
    -/


instance Hom.inhabited : Inhabited (Hom (none : WidePullbackShape J) none) :=
  ⟨Hom.id (none : WidePullbackShape J)⟩


/-- An aesop tactic for bulk cases on morphisms in `WidePushoutShape` -/
def evalCasesBash : TacticM Unit := do
  evalTactic
    (← `(tactic| casesm* WidePullbackShape _,
      (_ : WidePullbackShape _) ⟶ (_ : WidePullbackShape _) ))


instance subsingleton_hom : Quiver.IsThin (WidePullbackShape J) := fun _ _ => by
  /-
    J : Type w
    x✝¹ x✝ : CategoryTheory.Limits.WidePullbackShape J
    ⊢ Subsingleton (Quiver.Hom x✝¹ x✝)
  -/
  constructor
  /-
    case allEq
    J : Type w
    x✝¹ x✝ : CategoryTheory.Limits.WidePullbackShape J
    ⊢ ∀ (a b : Quiver.Hom x✝¹ x✝), Eq a b
  -/
  intro a b
  /-
    case allEq
    J : Type w
    x✝¹ x✝ : CategoryTheory.Limits.WidePullbackShape J
    a b : Quiver.Hom x✝¹ x✝
    ⊢ Eq a b
  -/
  casesm* WidePullbackShape _, (_ : WidePullbackShape _) ⟶ (_ : WidePullbackShape _)
    /-
      case allEq.none.none.id.id
      J : Type w
      ⊢ Eq (CategoryTheory.Limits.WidePullbackShape.Hom.id Option.none) (CategoryThe …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case allEq.some.none.term.term
      J : Type w
      val✝ : J
      ⊢ Eq (CategoryTheory.Limits.WidePullbackShape.Hom.term val✝) (CategoryTheory.L …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case allEq.some.some.id.id
      J : Type w
      val✝ : J
      ⊢ Eq (CategoryTheory.Limits.WidePullbackShape.Hom.id (Option.some val✝)) (Cate …
    -/
  · rfl
    /-
      🎉 no goals
    -/


instance category : SmallCategory (WidePullbackShape J) :=
  thin_category


@[simp]
theorem hom_id (X : WidePullbackShape J) : Hom.id X = 𝟙 X :=
  rfl

/- Porting note: we get a warning that we should change LHS to `sizeOf (𝟙 X)` but Lean cannot
find the category instance on `WidePullbackShape J` in that case. Once supplied in the proof,
the proposed proof of `simp [only WidePullbackShape.hom_id]` does not work -/

/-- Construct a functor out of the wide pullback shape given a J-indexed collection of arrows to a
fixed object.
-/
@[simps]
def wideCospan (B : C) (objs : J → C) (arrows : ∀ j : J, objs j ⟶ B) : WidePullbackShape J ⥤ C where
  obj j := Option.casesOn j B objs
  map f := by
    /-
      J : Type w
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      B : C
      objs : J → C
      arrows : (j : J) → Quiver.Hom (objs j) B
      X✝ Y✝ : CategoryTheory.Limits.WidePullbackShape J
      f : Quiver.Hom X✝ Y✝
      ⊢ Quiver.Hom ((fun j => Option.casesOn j B objs) X✝) ((fun j => Option.casesOn …
    -/
    cases' f with _ j
      /-
        case id
        J : Type w
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        B : C
        objs : J → C
        arrows : (j : J) → Quiver.Hom (objs j) B
        X✝ : CategoryTheory.Limits.WidePullbackShape J
        ⊢ Quiver.Hom ((fun j => Option.casesOn j B objs) X✝) ((fun j => Option.casesOn …
      -/
    · apply 𝟙 _
      /-
        🎉 no goals
      -/
      /-
        case term
        J : Type w
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        B : C
        objs : J → C
        arrows : (j : J) → Quiver.Hom (objs j) B
        j : J
        ⊢ Quiver.Hom ((fun j => Option.casesOn j B objs) (Option.some j)) ((fun j => O …
      -/
    · exact arrows j
      /-
        🎉 no goals
      -/


/-- Every diagram is naturally isomorphic (actually, equal) to a `wideCospan` -/
def diagramIsoWideCospan (F : WidePullbackShape J ⥤ C) :
    F ≅ wideCospan (F.obj none) (fun j => F.obj (some j)) fun j => F.map (Hom.term j) :=
                                             /-
                                               J : Type w
                                               C : Type u
                                               inst✝ : CategoryTheory.Category.{v, u} C
                                               F : CategoryTheory.Functor (CategoryTheory.Limits.WidePullbackShape J) C
                                               j : CategoryTheory.Limits.WidePullbackShape J
                                               ⊢ Eq (F.obj j) ((CategoryTheory.Limits.WidePullbackShape.wideCospan (F.obj Opt …
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
  NatIso.ofComponents fun j => eqToIso <| by aesop_cat
  /-
    🎉 no goals
  -/


/-- Construct a cone over a wide cospan. -/
@[simps]
def mkCone {F : WidePullbackShape J ⥤ C} {X : C} (f : X ⟶ F.obj none) (π : ∀ j, X ⟶ F.obj (some j))
    (w : ∀ j, π j ≫ F.map (Hom.term j) = f) : Cone F :=
  { pt := X
    π :=
      { app := fun j =>
          match j with
          | none => f
          | some j => π j
        naturality := fun j j' f => by
          /-
            J : Type w
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            F : CategoryTheory.Functor (CategoryTheory.Limits.WidePullbackShape J) C
            X : C
            f✝ : Quiver.Hom X (F.obj Option.none)
            π : (j : J) → Quiver.Hom X (F.obj (Option.some j))
            w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (π j) (F.map (CategoryTh …
            j j' : CategoryTheory.Limits.WidePullbackShape J
            f : Quiver.Hom j j'
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Cate …
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
          cases j <;> cases j' <;> cases f <;> dsimp <;> simp [w] } }
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- Wide pullback diagrams of equivalent index types are equivalent. -/
def equivalenceOfEquiv (J' : Type w') (h : J ≃ J') :
    WidePullbackShape J ≌ WidePullbackShape J' where
  functor := wideCospan none (fun j => some (h j)) fun j => Hom.term (h j)
  inverse := wideCospan none (fun j => some (h.invFun j)) fun j => Hom.term (h.invFun j)
                                              /-
                                                J : Type w
                                                C : Type u
                                                inst✝ : CategoryTheory.Category.{v, u} C
                                                J' : Type w'
                                                h : Equiv J J'
                                                j : CategoryTheory.Limits.WidePullbackShape J
                                                ⊢ CategoryTheory.Iso ((CategoryTheory.Functor.id (CategoryTheory.Limits.WidePu …
                                              -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  unitIso := NatIso.ofComponents (fun j => by cases j <;> exact eqToIso (by simp))
             /-
               🎉 no goals
             -/
                                                /-
                                                  J : Type w
                                                  C : Type u
                                                  inst✝ : CategoryTheory.Category.{v, u} C
                                                  J' : Type w'
                                                  h : Equiv J J'
                                                  j : CategoryTheory.Limits.WidePullbackShape J'
                                                  ⊢ CategoryTheory.Iso (((CategoryTheory.Limits.WidePullbackShape.wideCospan Opt …
                                                -/
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  counitIso := NatIso.ofComponents (fun j => by cases j <;> exact eqToIso (by simp))
               /-
                 🎉 no goals
               -/


/-- Lifting universe and morphism levels preserves wide pullback diagrams. -/
def uliftEquivalence :
    ULiftHom.{w'} (ULift.{w'} (WidePullbackShape J)) ≌ WidePullbackShape (ULift J) :=
  (ULiftHomULiftCategory.equiv.{w', w', w, w} (WidePullbackShape J)).symm.trans
    (equivalenceOfEquiv _ (Equiv.ulift.{w', w}.symm : J ≃ ULift.{w'} J))


/-- The type of arrows for the shape indexing a wide pushout. -/
inductive Hom : WidePushoutShape J → WidePushoutShape J → Type w
  | id : ∀ X, Hom X X
  | init : ∀ j : J, Hom none (some j)
  deriving DecidableEq

-- This is relying on an automatically generated instance name, generated in a `deriving` handler.
-- See https://github.com/leanprover/lean4/issues/2343

instance struct : CategoryStruct (WidePushoutShape J) where
  Hom := Hom
  id j := Hom.id j
  comp f g := by
    /-
      J : Type w
      X✝ Y✝ Z✝ : CategoryTheory.Limits.WidePushoutShape J
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Quiver.Hom X✝ Z✝
    -/
    cases f
      /-
        case id
        J : Type w
        X✝ Z✝ : CategoryTheory.Limits.WidePushoutShape J
        g : Quiver.Hom X✝ Z✝
        ⊢ Quiver.Hom X✝ Z✝
      -/
    · exact g
      /-
        🎉 no goals
      -/
    /-
      case init
      J : Type w
      Z✝ : CategoryTheory.Limits.WidePushoutShape J
      j✝ : J
      g : Quiver.Hom (Option.some j✝) Z✝
      ⊢ Quiver.Hom Option.none Z✝
    -/
    cases g
    /-
      case init.id
      J : Type w
      j✝ : J
      ⊢ Quiver.Hom Option.none (Option.some j✝)
    -/
    apply Hom.init _
    /-
      🎉 no goals
    -/


instance Hom.inhabited : Inhabited (Hom (none : WidePushoutShape J) none) :=
  ⟨Hom.id (none : WidePushoutShape J)⟩


/-- An aesop tactic for bulk cases on morphisms in `WidePushoutShape` -/
def evalCasesBash' : TacticM Unit := do
  evalTactic
    (← `(tactic| casesm* WidePushoutShape _,
      (_ : WidePushoutShape _) ⟶ (_ : WidePushoutShape _) ))


instance subsingleton_hom : Quiver.IsThin (WidePushoutShape J) := fun _ _ => by
  /-
    J : Type w
    x✝¹ x✝ : CategoryTheory.Limits.WidePushoutShape J
    ⊢ Subsingleton (Quiver.Hom x✝¹ x✝)
  -/
  constructor
  /-
    case allEq
    J : Type w
    x✝¹ x✝ : CategoryTheory.Limits.WidePushoutShape J
    ⊢ ∀ (a b : Quiver.Hom x✝¹ x✝), Eq a b
  -/
  intro a b
  /-
    case allEq
    J : Type w
    x✝¹ x✝ : CategoryTheory.Limits.WidePushoutShape J
    a b : Quiver.Hom x✝¹ x✝
    ⊢ Eq a b
  -/
  casesm* WidePushoutShape _, (_ : WidePushoutShape _) ⟶ (_ : WidePushoutShape _)
  /-
    case allEq.none.none.id.id
    J : Type w
    ⊢ Eq (CategoryTheory.Limits.WidePushoutShape.Hom.id Option.none) (CategoryTheo …
  -/
  repeat rfl
  /-
    🎉 no goals
  -/


instance category : SmallCategory (WidePushoutShape J) :=
  thin_category


@[simp]
theorem hom_id (X : WidePushoutShape J) : Hom.id X = 𝟙 X :=
  rfl

/- Porting note: we get a warning that we should change LHS to `sizeOf (𝟙 X)` but Lean cannot
find the category instance on `WidePushoutShape J` in that case. Once supplied in the proof,
the proposed proof of `simp [only WidePushoutShape.hom_id]` does not work -/

/-- Construct a functor out of the wide pushout shape given a J-indexed collection of arrows from a
fixed object.
-/
@[simps]
def wideSpan (B : C) (objs : J → C) (arrows : ∀ j : J, B ⟶ objs j) : WidePushoutShape J ⥤ C where
  obj j := Option.casesOn j B objs
  map f := by
    /-
      J : Type w
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      B : C
      objs : J → C
      arrows : (j : J) → Quiver.Hom B (objs j)
      X✝ Y✝ : CategoryTheory.Limits.WidePushoutShape J
      f : Quiver.Hom X✝ Y✝
      ⊢ Quiver.Hom ((fun j => Option.casesOn j B objs) X✝) ((fun j => Option.casesOn …
    -/
    cases' f with _ j
      /-
        case id
        J : Type w
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        B : C
        objs : J → C
        arrows : (j : J) → Quiver.Hom B (objs j)
        X✝ : CategoryTheory.Limits.WidePushoutShape J
        ⊢ Quiver.Hom ((fun j => Option.casesOn j B objs) X✝) ((fun j => Option.casesOn …
      -/
    · apply 𝟙 _
      /-
        🎉 no goals
      -/
      /-
        case init
        J : Type w
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        B : C
        objs : J → C
        arrows : (j : J) → Quiver.Hom B (objs j)
        j : J
        ⊢ Quiver.Hom ((fun j => Option.casesOn j B objs) Option.none) ((fun j => Optio …
      -/
    · exact arrows j
      /-
        🎉 no goals
      -/
  map_comp := fun f g => by
    /-
      J : Type w
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      B : C
      objs : J → C
      arrows : (j : J) → Quiver.Hom B (objs j)
      X✝ Y✝ Z✝ : CategoryTheory.Limits.WidePushoutShape J
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun j => Option.casesOn j B objs, map := fun {X Y} f => Categor …
    -/
    cases f
      /-
        case id
        J : Type w
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        B : C
        objs : J → C
        arrows : (j : J) → Quiver.Hom B (objs j)
        X✝ Z✝ : CategoryTheory.Limits.WidePushoutShape J
        g : Quiver.Hom X✝ Z✝
        ⊢ Eq ({ obj := fun j => Option.casesOn j B objs, map := fun {X Y} f => Categor …
      -/
    · simp only [Eq.ndrec, hom_id, eq_rec_constant, Category.id_comp]; congr
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
      /-
        case init
        J : Type w
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        B : C
        objs : J → C
        arrows : (j : J) → Quiver.Hom B (objs j)
        Z✝ : CategoryTheory.Limits.WidePushoutShape J
        j✝ : J
        g : Quiver.Hom (Option.some j✝) Z✝
        ⊢ Eq ({ obj := fun j => Option.casesOn j B objs, map := fun {X Y} f => Categor …
      -/
    · cases g
      /-
        case init.id
        J : Type w
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        B : C
        objs : J → C
        arrows : (j : J) → Quiver.Hom B (objs j)
        j✝ : J
        ⊢ Eq ({ obj := fun j => Option.casesOn j B objs, map := fun {X Y} f => Categor …
      -/
      simp only [Eq.ndrec, hom_id, eq_rec_constant, Category.comp_id]; congr
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


/-- Every diagram is naturally isomorphic (actually, equal) to a `wideSpan` -/
def diagramIsoWideSpan (F : WidePushoutShape J ⥤ C) :
    F ≅ wideSpan (F.obj none) (fun j => F.obj (some j)) fun j => F.map (Hom.init j) :=
                                             /-
                                               J : Type w
                                               C : Type u
                                               inst✝ : CategoryTheory.Category.{v, u} C
                                               F : CategoryTheory.Functor (CategoryTheory.Limits.WidePushoutShape J) C
                                               j : CategoryTheory.Limits.WidePushoutShape J
                                               ⊢ Eq (F.obj j) ((CategoryTheory.Limits.WidePushoutShape.wideSpan (F.obj Option …
                                             -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  NatIso.ofComponents fun j => eqToIso <| by cases j; repeat rfl
  /-
    🎉 no goals
  -/


/-- Construct a cocone over a wide span. -/
@[simps]
def mkCocone {F : WidePushoutShape J ⥤ C} {X : C} (f : F.obj none ⟶ X) (ι : ∀ j, F.obj (some j) ⟶ X)
    (w : ∀ j, F.map (Hom.init j) ≫ ι j = f) : Cocone F :=
  { pt := X
    ι :=
      { app := fun j =>
          match j with
          | none => f
          | some j => ι j
        naturality := fun j j' f => by
          /-
            J : Type w
            C : Type u
            inst✝ : CategoryTheory.Category.{v, u} C
            F : CategoryTheory.Functor (CategoryTheory.Limits.WidePushoutShape J) C
            X : C
            f✝ : Quiver.Hom (F.obj Option.none) X
            ι : (j : J) → Quiver.Hom (F.obj (Option.some j)) X
            w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.L …
            j j' : CategoryTheory.Limits.WidePushoutShape J
            f : Quiver.Hom j j'
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun j => CategoryTheory.L …
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
          cases j <;> cases j' <;> cases f <;> dsimp <;> simp [w] } }
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- Wide pushout diagrams of equivalent index types are equivalent. -/
def equivalenceOfEquiv (J' : Type w') (h : J ≃ J') : WidePushoutShape J ≌ WidePushoutShape J' where
  functor := wideSpan none (fun j => some (h j)) fun j => Hom.init (h j)
  inverse := wideSpan none (fun j => some (h.invFun j)) fun j => Hom.init (h.invFun j)
                                              /-
                                                J : Type w
                                                C : Type u
                                                inst✝ : CategoryTheory.Category.{v, u} C
                                                J' : Type w'
                                                h : Equiv J J'
                                                j : CategoryTheory.Limits.WidePushoutShape J
                                                ⊢ CategoryTheory.Iso ((CategoryTheory.Functor.id (CategoryTheory.Limits.WidePu …
                                              -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  unitIso := NatIso.ofComponents (fun j => by cases j <;> exact eqToIso (by simp))
             /-
               🎉 no goals
             -/
                                                /-
                                                  J : Type w
                                                  C : Type u
                                                  inst✝ : CategoryTheory.Category.{v, u} C
                                                  J' : Type w'
                                                  h : Equiv J J'
                                                  j : CategoryTheory.Limits.WidePushoutShape J'
                                                  ⊢ CategoryTheory.Iso (((CategoryTheory.Limits.WidePushoutShape.wideSpan Option …
                                                -/
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  counitIso := NatIso.ofComponents (fun j => by cases j <;> exact eqToIso (by simp))
               /-
                 🎉 no goals
               -/


/-- Lifting universe and morphism levels preserves wide pushout diagrams. -/
def uliftEquivalence :
    ULiftHom.{w'} (ULift.{w'} (WidePushoutShape J)) ≌ WidePushoutShape (ULift J) :=
  (ULiftHomULiftCategory.equiv.{w', w', w, w} (WidePushoutShape J)).symm.trans
    (equivalenceOfEquiv _ (Equiv.ulift.{w', w}.symm : J ≃ ULift.{w'} J))


/-- `HasWidePullbacks` represents a choice of wide pullback for every collection of morphisms -/
abbrev HasWidePullbacks : Prop :=
  ∀ J : Type w, HasLimitsOfShape (WidePullbackShape J) C


/-- `HasWidePushouts` represents a choice of wide pushout for every collection of morphisms -/
abbrev HasWidePushouts : Prop :=
  ∀ J : Type w, HasColimitsOfShape (WidePushoutShape J) C


/-- `HasWidePullback B objs arrows` means that `wideCospan B objs arrows` has a limit. -/
abbrev HasWidePullback (B : C) (objs : J → C) (arrows : ∀ j : J, objs j ⟶ B) : Prop :=
  HasLimit (WidePullbackShape.wideCospan B objs arrows)


/-- `HasWidePushout B objs arrows` means that `wideSpan B objs arrows` has a colimit. -/
abbrev HasWidePushout (B : C) (objs : J → C) (arrows : ∀ j : J, B ⟶ objs j) : Prop :=
  HasColimit (WidePushoutShape.wideSpan B objs arrows)


/-- A choice of wide pullback. -/
noncomputable abbrev widePullback (B : C) (objs : J → C) (arrows : ∀ j : J, objs j ⟶ B)
    [HasWidePullback B objs arrows] : C :=
  limit (WidePullbackShape.wideCospan B objs arrows)


/-- A choice of wide pushout. -/
noncomputable abbrev widePushout (B : C) (objs : J → C) (arrows : ∀ j : J, B ⟶ objs j)
    [HasWidePushout B objs arrows] : C :=
  colimit (WidePushoutShape.wideSpan B objs arrows)


/-- The `j`-th projection from the pullback. -/
noncomputable abbrev π (j : J) : widePullback _ _ arrows ⟶ objs j :=
  limit.π (WidePullbackShape.wideCospan _ _ _) (Option.some j)



/-- The unique map to the base from the pullback. -/
noncomputable abbrev base : widePullback _ _ arrows ⟶ B :=
  limit.π (WidePullbackShape.wideCospan _ _ _) Option.none


@[reassoc (attr := simp)]
theorem π_arrow (j : J) : π arrows j ≫ arrows _ = base arrows := by
  /-
    J : Type w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    B : C
    objs : J → C
    arrows : (j : J) → Quiver.Hom (objs j) B
    inst✝ : CategoryTheory.Limits.HasWidePullback B objs arrows
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePullback.π …
  -/
  apply limit.w (WidePullbackShape.wideCospan _ _ _) (WidePullbackShape.Hom.term j)
  /-
    🎉 no goals
  -/


/-- Lift a collection of morphisms to a morphism to the pullback. -/
noncomputable abbrev lift {X : C} (f : X ⟶ B) (fs : ∀ j : J, X ⟶ objs j)
    (w : ∀ j, fs j ≫ arrows j = f) : X ⟶ widePullback _ _ arrows :=
  limit.lift (WidePullbackShape.wideCospan _ _ _) (WidePullbackShape.mkCone f fs <| w)


@[reassoc]
theorem lift_π (j : J) : lift f fs w ≫ π arrows j = fs _ := by
  /-
    J : Type w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    B : C
    objs : J → C
    arrows : (j : J) → Quiver.Hom (objs j) B
    inst✝ : CategoryTheory.Limits.HasWidePullback B objs arrows
    X : C
    f : Quiver.Hom X B
    fs : (j : J) → Quiver.Hom X (objs j)
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (fs j) (arrows j)) f
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePullback.l …
  -/
  simp only [limit.lift_π, WidePullbackShape.mkCone_pt, WidePullbackShape.mkCone_π_app]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem lift_base : lift f fs w ≫ base arrows = f := by
  /-
    J : Type w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    B : C
    objs : J → C
    arrows : (j : J) → Quiver.Hom (objs j) B
    inst✝ : CategoryTheory.Limits.HasWidePullback B objs arrows
    X : C
    f : Quiver.Hom X B
    fs : (j : J) → Quiver.Hom X (objs j)
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (fs j) (arrows j)) f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePullback.l …
  -/
  simp only [limit.lift_π, WidePullbackShape.mkCone_pt, WidePullbackShape.mkCone_π_app]
  /-
    🎉 no goals
  -/


theorem eq_lift_of_comp_eq (g : X ⟶ widePullback _ _ arrows) :
    (∀ j : J, g ≫ π arrows j = fs j) → g ≫ base arrows = f → g = lift f fs w := by
  /-
    J : Type w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    B : C
    objs : J → C
    arrows : (j : J) → Quiver.Hom (objs j) B
    inst✝ : CategoryTheory.Limits.HasWidePullback B objs arrows
    X : C
    f : Quiver.Hom X B
    fs : (j : J) → Quiver.Hom X (objs j)
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (fs j) (arrows j)) f
    g : Quiver.Hom X (CategoryTheory.Limits.widePullback B objs arrows)
    ⊢ (∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Limits. …
  -/
  intro h1 h2
  apply
    (limit.isLimit (WidePullbackShape.wideCospan B objs arrows)).uniq
      (WidePullbackShape.mkCone f fs <| w)
  /-
    case x
    J : Type w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    B : C
    objs : J → C
    arrows : (j : J) → Quiver.Hom (objs j) B
    inst✝ : CategoryTheory.Limits.HasWidePullback B objs arrows
    X : C
    f : Quiver.Hom X B
    fs : (j : J) → Quiver.Hom X (objs j)
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (fs j) (arrows j)) f
    g : Quiver.Hom X (CategoryTheory.Limits.widePullback B objs arrows)
    h1 : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Limit …
    h2 : Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Limits.WidePullb …
    ⊢ ∀ (j : CategoryTheory.Limits.WidePullbackShape J), Eq (CategoryTheory.Catego …
  -/
  rintro (_ | _)
    /-
      case x.none
      J : Type w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      B : C
      objs : J → C
      arrows : (j : J) → Quiver.Hom (objs j) B
      inst✝ : CategoryTheory.Limits.HasWidePullback B objs arrows
      X : C
      f : Quiver.Hom X B
      fs : (j : J) → Quiver.Hom X (objs j)
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (fs j) (arrows j)) f
      g : Quiver.Hom X (CategoryTheory.Limits.widePullback B objs arrows)
      h1 : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Limit …
      h2 : Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Limits.WidePullb …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g ((CategoryTheory.Limits.limit.cone  …
    -/
  · apply h2
    /-
      🎉 no goals
    -/
    /-
      case x.some
      J : Type w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      B : C
      objs : J → C
      arrows : (j : J) → Quiver.Hom (objs j) B
      inst✝ : CategoryTheory.Limits.HasWidePullback B objs arrows
      X : C
      f : Quiver.Hom X B
      fs : (j : J) → Quiver.Hom X (objs j)
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (fs j) (arrows j)) f
      g : Quiver.Hom X (CategoryTheory.Limits.widePullback B objs arrows)
      h1 : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Limit …
      h2 : Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Limits.WidePullb …
      val✝ : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g ((CategoryTheory.Limits.limit.cone  …
    -/
  · apply h1
    /-
      🎉 no goals
    -/


theorem hom_eq_lift (g : X ⟶ widePullback _ _ arrows) :
                                                             /-
                                                               J : Type w
                                                               C✝ : Type u
                                                               inst✝² : CategoryTheory.Category.{v, u} C✝
                                                               C : Type u
                                                               inst✝¹ : CategoryTheory.Category.{v, u} C
                                                               B : C
                                                               objs : J → C
                                                               arrows : (j : J) → Quiver.Hom (objs j) B
                                                               inst✝ : CategoryTheory.Limits.HasWidePullback B objs arrows
                                                               X : C
                                                               f : Quiver.Hom X B
                                                               fs : (j : J) → Quiver.Hom X (objs j)
                                                               w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (fs j) (arrows j)) f
                                                               g : Quiver.Hom X (CategoryTheory.Limits.widePullback B objs arrows)
                                                               ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((fun j => CategoryTheory. …
                                                             -/
    g = lift (g ≫ base arrows) (fun j => g ≫ π arrows j) (by aesop_cat) := by
                                                             /-
                                                               🎉 no goals
                                                             -/
  /-
    J : Type w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    B : C
    objs : J → C
    arrows : (j : J) → Quiver.Hom (objs j) B
    inst✝ : CategoryTheory.Limits.HasWidePullback B objs arrows
    X : C
    g : Quiver.Hom X (CategoryTheory.Limits.widePullback B objs arrows)
    ⊢ Eq g (CategoryTheory.Limits.WidePullback.lift (CategoryTheory.CategoryStruct …
  -/
  apply eq_lift_of_comp_eq
    /-
      case a
      J : Type w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      B : C
      objs : J → C
      arrows : (j : J) → Quiver.Hom (objs j) B
      inst✝ : CategoryTheory.Limits.HasWidePullback B objs arrows
      X : C
      g : Quiver.Hom X (CategoryTheory.Limits.widePullback B objs arrows)
      ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Limits.W …
    -/
  · aesop_cat
    /-
      🎉 no goals
    -/
    /-
      case a
      J : Type w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      B : C
      objs : J → C
      arrows : (j : J) → Quiver.Hom (objs j) B
      inst✝ : CategoryTheory.Limits.HasWidePullback B objs arrows
      X : C
      g : Quiver.Hom X (CategoryTheory.Limits.widePullback B objs arrows)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Limits.WidePullback …
    -/
  · rfl  -- Porting note: quite a few missing refl's in aesop_cat now
    /-
      🎉 no goals
    -/


@[ext 1100]
theorem hom_ext (g1 g2 : X ⟶ widePullback _ _ arrows) : (∀ j : J,
    g1 ≫ π arrows j = g2 ≫ π arrows j) → g1 ≫ base arrows = g2 ≫ base arrows → g1 = g2 := by
  /-
    J : Type w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    B : C
    objs : J → C
    arrows : (j : J) → Quiver.Hom (objs j) B
    inst✝ : CategoryTheory.Limits.HasWidePullback B objs arrows
    X : C
    g1 g2 : Quiver.Hom X (CategoryTheory.Limits.widePullback B objs arrows)
    ⊢ (∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp g1 (CategoryTheory.Limits …
  -/
  intro h1 h2
  /-
    J : Type w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    B : C
    objs : J → C
    arrows : (j : J) → Quiver.Hom (objs j) B
    inst✝ : CategoryTheory.Limits.HasWidePullback B objs arrows
    X : C
    g1 g2 : Quiver.Hom X (CategoryTheory.Limits.widePullback B objs arrows)
    h1 : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp g1 (CategoryTheory.Limi …
    h2 : Eq (CategoryTheory.CategoryStruct.comp g1 (CategoryTheory.Limits.WidePull …
    ⊢ Eq g1 g2
  -/
  apply limit.hom_ext
  /-
    case w
    J : Type w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    B : C
    objs : J → C
    arrows : (j : J) → Quiver.Hom (objs j) B
    inst✝ : CategoryTheory.Limits.HasWidePullback B objs arrows
    X : C
    g1 g2 : Quiver.Hom X (CategoryTheory.Limits.widePullback B objs arrows)
    h1 : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp g1 (CategoryTheory.Limi …
    h2 : Eq (CategoryTheory.CategoryStruct.comp g1 (CategoryTheory.Limits.WidePull …
    ⊢ ∀ (j : CategoryTheory.Limits.WidePullbackShape J), Eq (CategoryTheory.Catego …
  -/
  rintro (_ | _)
    /-
      case w.none
      J : Type w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      B : C
      objs : J → C
      arrows : (j : J) → Quiver.Hom (objs j) B
      inst✝ : CategoryTheory.Limits.HasWidePullback B objs arrows
      X : C
      g1 g2 : Quiver.Hom X (CategoryTheory.Limits.widePullback B objs arrows)
      h1 : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp g1 (CategoryTheory.Limi …
      h2 : Eq (CategoryTheory.CategoryStruct.comp g1 (CategoryTheory.Limits.WidePull …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g1 (CategoryTheory.Limits.limit.π (Ca …
    -/
  · apply h2
    /-
      🎉 no goals
    -/
    /-
      case w.some
      J : Type w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      B : C
      objs : J → C
      arrows : (j : J) → Quiver.Hom (objs j) B
      inst✝ : CategoryTheory.Limits.HasWidePullback B objs arrows
      X : C
      g1 g2 : Quiver.Hom X (CategoryTheory.Limits.widePullback B objs arrows)
      h1 : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp g1 (CategoryTheory.Limi …
      h2 : Eq (CategoryTheory.CategoryStruct.comp g1 (CategoryTheory.Limits.WidePull …
      val✝ : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g1 (CategoryTheory.Limits.limit.π (Ca …
    -/
  · apply h1
    /-
      🎉 no goals
    -/


/-- The `j`-th inclusion to the pushout. -/
noncomputable abbrev ι (j : J) : objs j ⟶ widePushout _ _ arrows :=
  colimit.ι (WidePushoutShape.wideSpan _ _ _) (Option.some j)


/-- The unique map from the head to the pushout. -/
noncomputable abbrev head : B ⟶ widePushout B objs arrows :=
  colimit.ι (WidePushoutShape.wideSpan _ _ _) Option.none


@[reassoc (attr := simp)]
theorem arrow_ι (j : J) : arrows j ≫ ι arrows j = head arrows := by
  /-
    J : Type w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    B : C
    objs : J → C
    arrows : (j : J) → Quiver.Hom B (objs j)
    inst✝ : CategoryTheory.Limits.HasWidePushout B objs arrows
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (arrows j) (CategoryTheory.Limits.Wid …
  -/
  apply colimit.w (WidePushoutShape.wideSpan _ _ _) (WidePushoutShape.Hom.init j)
  /-
    🎉 no goals
  -/

-- Porting note: this can simplify itself

/-- Descend a collection of morphisms to a morphism from the pushout. -/
noncomputable abbrev desc {X : C} (f : B ⟶ X) (fs : ∀ j : J, objs j ⟶ X)
    (w : ∀ j, arrows j ≫ fs j = f) : widePushout _ _ arrows ⟶ X :=
  colimit.desc (WidePushoutShape.wideSpan B objs arrows) (WidePushoutShape.mkCocone f fs <| w)


@[reassoc]
theorem ι_desc (j : J) : ι arrows j ≫ desc f fs w = fs _ := by
  /-
    J : Type w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    B : C
    objs : J → C
    arrows : (j : J) → Quiver.Hom B (objs j)
    inst✝ : CategoryTheory.Limits.HasWidePushout B objs arrows
    X : C
    f : Quiver.Hom B X
    fs : (j : J) → Quiver.Hom (objs j) X
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (arrows j) (fs j)) f
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout.ι  …
  -/
  simp only [colimit.ι_desc, WidePushoutShape.mkCocone_pt, WidePushoutShape.mkCocone_ι_app]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem head_desc : head arrows ≫ desc f fs w = f := by
  /-
    J : Type w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    B : C
    objs : J → C
    arrows : (j : J) → Quiver.Hom B (objs j)
    inst✝ : CategoryTheory.Limits.HasWidePushout B objs arrows
    X : C
    f : Quiver.Hom B X
    fs : (j : J) → Quiver.Hom (objs j) X
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (arrows j) (fs j)) f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout.he …
  -/
  simp only [colimit.ι_desc, WidePushoutShape.mkCocone_pt, WidePushoutShape.mkCocone_ι_app]
  /-
    🎉 no goals
  -/


theorem eq_desc_of_comp_eq (g : widePushout _ _ arrows ⟶ X) :
    (∀ j : J, ι arrows j ≫ g = fs j) → head arrows ≫ g = f → g = desc f fs w := by
  /-
    J : Type w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    B : C
    objs : J → C
    arrows : (j : J) → Quiver.Hom B (objs j)
    inst✝ : CategoryTheory.Limits.HasWidePushout B objs arrows
    X : C
    f : Quiver.Hom B X
    fs : (j : J) → Quiver.Hom (objs j) X
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (arrows j) (fs j)) f
    g : Quiver.Hom (CategoryTheory.Limits.widePushout B objs arrows) X
    ⊢ (∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Wi …
  -/
  intro h1 h2
  apply
    (colimit.isColimit (WidePushoutShape.wideSpan B objs arrows)).uniq
      (WidePushoutShape.mkCocone f fs <| w)
  /-
    case x
    J : Type w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    B : C
    objs : J → C
    arrows : (j : J) → Quiver.Hom B (objs j)
    inst✝ : CategoryTheory.Limits.HasWidePushout B objs arrows
    X : C
    f : Quiver.Hom B X
    fs : (j : J) → Quiver.Hom (objs j) X
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (arrows j) (fs j)) f
    g : Quiver.Hom (CategoryTheory.Limits.widePushout B objs arrows) X
    h1 : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits. …
    h2 : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout …
    ⊢ ∀ (j : CategoryTheory.Limits.WidePushoutShape J), Eq (CategoryTheory.Categor …
  -/
  rintro (_ | _)
    /-
      case x.none
      J : Type w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      B : C
      objs : J → C
      arrows : (j : J) → Quiver.Hom B (objs j)
      inst✝ : CategoryTheory.Limits.HasWidePushout B objs arrows
      X : C
      f : Quiver.Hom B X
      fs : (j : J) → Quiver.Hom (objs j) X
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (arrows j) (fs j)) f
      g : Quiver.Hom (CategoryTheory.Limits.widePushout B objs arrows) X
      h1 : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits. …
      h2 : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.colimit.cocon …
    -/
  · apply h2
    /-
      🎉 no goals
    -/
    /-
      case x.some
      J : Type w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      B : C
      objs : J → C
      arrows : (j : J) → Quiver.Hom B (objs j)
      inst✝ : CategoryTheory.Limits.HasWidePushout B objs arrows
      X : C
      f : Quiver.Hom B X
      fs : (j : J) → Quiver.Hom (objs j) X
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (arrows j) (fs j)) f
      g : Quiver.Hom (CategoryTheory.Limits.widePushout B objs arrows) X
      h1 : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits. …
      h2 : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout …
      val✝ : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.colimit.cocon …
    -/
  · apply h1
    /-
      🎉 no goals
    -/


theorem hom_eq_desc (g : widePushout _ _ arrows ⟶ X) :
    g =
      desc (head arrows ≫ g) (fun j => ι arrows j ≫ g) fun j => by
        /-
          J : Type w
          C✝ : Type u
          inst✝² : CategoryTheory.Category.{v, u} C✝
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          B : C
          objs : J → C
          arrows : (j : J) → Quiver.Hom B (objs j)
          inst✝ : CategoryTheory.Limits.HasWidePushout B objs arrows
          X : C
          f : Quiver.Hom B X
          fs : (j : J) → Quiver.Hom (objs j) X
          w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (arrows j) (fs j)) f
          g : Quiver.Hom (CategoryTheory.Limits.widePushout B objs arrows) X
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (arrows j) ((fun j => CategoryTheory. …
        -/
        rw [← Category.assoc]
        /-
          J : Type w
          C✝ : Type u
          inst✝² : CategoryTheory.Category.{v, u} C✝
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          B : C
          objs : J → C
          arrows : (j : J) → Quiver.Hom B (objs j)
          inst✝ : CategoryTheory.Limits.HasWidePushout B objs arrows
          X : C
          f : Quiver.Hom B X
          fs : (j : J) → Quiver.Hom (objs j) X
          w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (arrows j) (fs j)) f
          g : Quiver.Hom (CategoryTheory.Limits.widePushout B objs arrows) X
          j : J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        simp := by
        /-
          🎉 no goals
        -/
  /-
    J : Type w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    B : C
    objs : J → C
    arrows : (j : J) → Quiver.Hom B (objs j)
    inst✝ : CategoryTheory.Limits.HasWidePushout B objs arrows
    X : C
    g : Quiver.Hom (CategoryTheory.Limits.widePushout B objs arrows) X
    ⊢ Eq g (CategoryTheory.Limits.WidePushout.desc (CategoryTheory.CategoryStruct. …
  -/
  apply eq_desc_of_comp_eq
    /-
      case a
      J : Type w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      B : C
      objs : J → C
      arrows : (j : J) → Quiver.Hom B (objs j)
      inst✝ : CategoryTheory.Limits.HasWidePushout B objs arrows
      X : C
      g : Quiver.Hom (CategoryTheory.Limits.widePushout B objs arrows) X
      ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Wid …
    -/
  · aesop_cat
    /-
      🎉 no goals
    -/
    /-
      case a
      J : Type w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      B : C
      objs : J → C
      arrows : (j : J) → Quiver.Hom B (objs j)
      inst✝ : CategoryTheory.Limits.HasWidePushout B objs arrows
      X : C
      g : Quiver.Hom (CategoryTheory.Limits.widePushout B objs arrows) X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout.he …
    -/
  · rfl -- Porting note: another missing rfl
    /-
      🎉 no goals
    -/


@[ext 1100]
theorem hom_ext (g1 g2 : widePushout _ _ arrows ⟶ X) : (∀ j : J,
    ι arrows j ≫ g1 = ι arrows j ≫ g2) → head arrows ≫ g1 = head arrows ≫ g2 → g1 = g2 := by
  /-
    J : Type w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    B : C
    objs : J → C
    arrows : (j : J) → Quiver.Hom B (objs j)
    inst✝ : CategoryTheory.Limits.HasWidePushout B objs arrows
    X : C
    g1 g2 : Quiver.Hom (CategoryTheory.Limits.widePushout B objs arrows) X
    ⊢ (∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Wi …
  -/
  intro h1 h2
  /-
    J : Type w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    B : C
    objs : J → C
    arrows : (j : J) → Quiver.Hom B (objs j)
    inst✝ : CategoryTheory.Limits.HasWidePushout B objs arrows
    X : C
    g1 g2 : Quiver.Hom (CategoryTheory.Limits.widePushout B objs arrows) X
    h1 : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits. …
    h2 : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout …
    ⊢ Eq g1 g2
  -/
  apply colimit.hom_ext
  /-
    case w
    J : Type w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    B : C
    objs : J → C
    arrows : (j : J) → Quiver.Hom B (objs j)
    inst✝ : CategoryTheory.Limits.HasWidePushout B objs arrows
    X : C
    g1 g2 : Quiver.Hom (CategoryTheory.Limits.widePushout B objs arrows) X
    h1 : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits. …
    h2 : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout …
    ⊢ ∀ (j : CategoryTheory.Limits.WidePushoutShape J), Eq (CategoryTheory.Categor …
  -/
  rintro (_ | _)
    /-
      case w.none
      J : Type w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      B : C
      objs : J → C
      arrows : (j : J) → Quiver.Hom B (objs j)
      inst✝ : CategoryTheory.Limits.HasWidePushout B objs arrows
      X : C
      g1 g2 : Quiver.Hom (CategoryTheory.Limits.widePushout B objs arrows) X
      h1 : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits. …
      h2 : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
    -/
  · apply h2
    /-
      🎉 no goals
    -/
    /-
      case w.some
      J : Type w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      B : C
      objs : J → C
      arrows : (j : J) → Quiver.Hom B (objs j)
      inst✝ : CategoryTheory.Limits.HasWidePushout B objs arrows
      X : C
      g1 g2 : Quiver.Hom (CategoryTheory.Limits.widePushout B objs arrows) X
      h1 : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits. …
      h2 : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.WidePushout …
      val✝ : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
    -/
  · apply h1
    /-
      🎉 no goals
    -/


/-- The action on morphisms of the obvious functor
  `WidePullbackShape_op : WidePullbackShape J ⥤ (WidePushoutShape J)ᵒᵖ`-/
def widePullbackShapeOpMap :
    ∀ X Y : WidePullbackShape J,
      (X ⟶ Y) → ((op X : (WidePushoutShape J)ᵒᵖ) ⟶ (op Y : (WidePushoutShape J)ᵒᵖ))
  | _, _, WidePullbackShape.Hom.id X => Quiver.Hom.op (WidePushoutShape.Hom.id _)
  | _, _, WidePullbackShape.Hom.term _ => Quiver.Hom.op (WidePushoutShape.Hom.init _)


/-- The obvious functor `WidePullbackShape J ⥤ (WidePushoutShape J)ᵒᵖ` -/
@[simps]
def widePullbackShapeOp : WidePullbackShape J ⥤ (WidePushoutShape J)ᵒᵖ where
  obj X := op X
  map {X₁} {X₂} := widePullbackShapeOpMap J X₁ X₂


/-- The action on morphisms of the obvious functor
`widePushoutShapeOp : WidePushoutShape J ⥤ (WidePullbackShape J)ᵒᵖ` -/
def widePushoutShapeOpMap :
    ∀ X Y : WidePushoutShape J,
      (X ⟶ Y) → ((op X : (WidePullbackShape J)ᵒᵖ) ⟶ (op Y : (WidePullbackShape J)ᵒᵖ))
  | _, _, WidePushoutShape.Hom.id X => Quiver.Hom.op (WidePullbackShape.Hom.id _)
  | _, _, WidePushoutShape.Hom.init _ => Quiver.Hom.op (WidePullbackShape.Hom.term _)


/-- The obvious functor `WidePushoutShape J ⥤ (WidePullbackShape J)ᵒᵖ` -/
@[simps]
def widePushoutShapeOp : WidePushoutShape J ⥤ (WidePullbackShape J)ᵒᵖ where
  obj X := op X
  map := fun {X} {Y} => widePushoutShapeOpMap J X Y


/-- The obvious functor `(WidePullbackShape J)ᵒᵖ ⥤ WidePushoutShape J`-/
@[simps!]
def widePullbackShapeUnop : (WidePullbackShape J)ᵒᵖ ⥤ WidePushoutShape J :=
  (widePullbackShapeOp J).leftOp


/-- The obvious functor `(WidePushoutShape J)ᵒᵖ ⥤ WidePullbackShape J` -/
@[simps!]
def widePushoutShapeUnop : (WidePushoutShape J)ᵒᵖ ⥤ WidePullbackShape J :=
  (widePushoutShapeOp J).leftOp


/-- The inverse of the unit isomorphism of the equivalence
`widePushoutShapeOpEquiv : (WidePushoutShape J)ᵒᵖ ≌ WidePullbackShape J` -/
def widePushoutShapeOpUnop : widePushoutShapeUnop J ⋙ widePullbackShapeOp J ≅ 𝟭 _ :=
  /-
    J : Type w
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ ∀ {X Y : Opposite (CategoryTheory.Limits.WidePushoutShape J)} (f : Quiver.Ho …
  -/
  NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


/-- The counit isomorphism of the equivalence
`widePullbackShapeOpEquiv : (WidePullbackShape J)ᵒᵖ ≌ WidePushoutShape J` -/
def widePushoutShapeUnopOp : widePushoutShapeOp J ⋙ widePullbackShapeUnop J ≅ 𝟭 _ :=
  /-
    J : Type w
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ ∀ {X Y : CategoryTheory.Limits.WidePushoutShape J} (f : Quiver.Hom X Y), Eq  …
  -/
  NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


/-- The inverse of the unit isomorphism of the equivalence
`widePullbackShapeOpEquiv : (WidePullbackShape J)ᵒᵖ ≌ WidePushoutShape J` -/
def widePullbackShapeOpUnop : widePullbackShapeUnop J ⋙ widePushoutShapeOp J ≅ 𝟭 _ :=
  /-
    J : Type w
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ ∀ {X Y : Opposite (CategoryTheory.Limits.WidePullbackShape J)} (f : Quiver.H …
  -/
  NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


/-- The counit isomorphism of the equivalence
`widePushoutShapeOpEquiv : (WidePushoutShape J)ᵒᵖ ≌ WidePullbackShape J` -/
def widePullbackShapeUnopOp : widePullbackShapeOp J ⋙ widePushoutShapeUnop J ≅ 𝟭 _ :=
  /-
    J : Type w
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ ∀ {X Y : CategoryTheory.Limits.WidePullbackShape J} (f : Quiver.Hom X Y), Eq …
  -/
  NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


/-- The duality equivalence `(WidePushoutShape J)ᵒᵖ ≌ WidePullbackShape J` -/
@[simps]
def widePushoutShapeOpEquiv : (WidePushoutShape J)ᵒᵖ ≌ WidePullbackShape J where
  functor := widePushoutShapeUnop J
  inverse := widePullbackShapeOp J
  unitIso := (widePushoutShapeOpUnop J).symm
  counitIso := widePullbackShapeUnopOp J


/-- The duality equivalence `(WidePullbackShape J)ᵒᵖ ≌ WidePushoutShape J` -/
@[simps]
def widePullbackShapeOpEquiv : (WidePullbackShape J)ᵒᵖ ≌ WidePushoutShape J where
  functor := widePullbackShapeUnop J
  inverse := widePushoutShapeOp J
  unitIso := (widePullbackShapeOpUnop J).symm
  counitIso := widePushoutShapeUnopOp J


/-- If a category has wide pushouts on a higher universe level it also has wide pushouts
on a lower universe level. -/
theorem hasWidePushouts_shrink [HasWidePushouts.{max w w'} C] : HasWidePushouts.{w} C := fun _ =>
  hasColimitsOfShape_of_equivalence (WidePushoutShape.equivalenceOfEquiv _ Equiv.ulift.{w'})


/-- If a category has wide pullbacks on a higher universe level it also has wide pullbacks
on a lower universe level. -/
theorem hasWidePullbacks_shrink [HasWidePullbacks.{max w w'} C] : HasWidePullbacks.{w} C := fun _ =>
  hasLimitsOfShape_of_equivalence (WidePullbackShape.equivalenceOfEquiv _ Equiv.ulift.{w'})


