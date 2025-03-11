/-- `IsFreeGroupoid.Generators G` is a type synonym for `G`. We think of this as
the vertices of the generating quiver of `G` when `G` is free. We can't use `G` directly,
since `G` already has a quiver instance from being a groupoid. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): @[nolint has_nonempty_instance]
@[nolint unusedArguments]
def IsFreeGroupoid.Generators (G) [Groupoid G] :=
  G


/-- A groupoid `G` is free when we have the following data:
 - a quiver on `IsFreeGroupoid.Generators G` (a type synonym for `G`)
 - a function `of` taking a generating arrow to a morphism in `G`
 - such that a functor from `G` to any group `X` is uniquely determined
   by assigning labels in `X` to the generating arrows.

   This definition is nonstandard. Normally one would require that functors `G ⥤ X`
   to any _groupoid_ `X` are given by graph homomorphisms from `generators`. -/
class IsFreeGroupoid (G) [Groupoid.{v} G] where
  quiverGenerators : Quiver.{v + 1} (IsFreeGroupoid.Generators G)
  of : ∀ {a b : IsFreeGroupoid.Generators G}, (a ⟶ b) → ((show G from a) ⟶ b)
  unique_lift :
    ∀ {X : Type v} [Group X] (f : Labelling (IsFreeGroupoid.Generators G) X),
      ∃! F : G ⥤ CategoryTheory.SingleObj X, ∀ (a b) (g : a ⟶ b), F.map (of g) = f g


/-- Two functors from a free groupoid to a group are equal when they agree on the generating
quiver. -/
@[ext]
theorem ext_functor {G} [Groupoid.{v} G] [IsFreeGroupoid G] {X : Type v} [Group X]
    (f g : G ⥤ CategoryTheory.SingleObj X) (h : ∀ (a b) (e : a ⟶ b), f.map (of e) = g.map (of e)) :
    f = g :=
  let ⟨_, _, u⟩ := @unique_lift G _ _ X _ fun (a b : Generators G) (e : a ⟶ b) => g.map (of e)
  _root_.trans (u _ h) (u _ fun _ _ _ => rfl).symm


set_option linter.unusedVariables false in
/-- An action groupoid over a free group is free. More generally, one could show that the groupoid
of elements over a free groupoid is free, but this version is easier to prove and suffices for our
purposes.

Analogous to the fact that a covering space of a graph is a graph. (A free groupoid is like a graph,
and a groupoid of elements is like a covering space.) -/
instance actionGroupoidIsFree {G A : Type u} [Group G] [IsFreeGroup G] [MulAction G A] :
    IsFreeGroupoid (ActionCategory G A) where
  quiverGenerators :=
    ⟨fun a b => { e : IsFreeGroup.Generators G // IsFreeGroup.of e • a.back = b.back }⟩
  of := fun (e : { e // _ }) => ⟨IsFreeGroup.of e, e.property⟩
  unique_lift := by
    /-
      G A : Type u
      inst✝² : Group G
      inst✝¹ : IsFreeGroup G
      inst✝ : MulAction G A
      ⊢ ∀ {X : Type u} [inst : Group X] (f : Quiver.Labelling (IsFreeGroupoid.Genera …
    -/
    intro X _ f
    let f' : IsFreeGroup.Generators G → (A → X) ⋊[mulAutArrow] G := fun e =>
      ⟨fun b => @f ⟨(), _⟩ ⟨(), b⟩ ⟨e, smul_inv_smul _ b⟩, IsFreeGroup.of e⟩
    /-
      G A : Type u
      inst✝³ : Group G
      inst✝² : IsFreeGroup G
      inst✝¹ : MulAction G A
      X : Type u
      inst✝ : Group X
      f : Quiver.Labelling (IsFreeGroupoid.Generators (CategoryTheory.ActionCategory …
      f' : IsFreeGroup.Generators G → SemidirectProduct (A → X) G mulAutArrow := fun …
      ⊢ ExistsUnique fun F => ∀ (a b : IsFreeGroupoid.Generators (CategoryTheory.Act …
    -/
    rcases IsFreeGroup.unique_lift f' with ⟨F', hF', uF'⟩
    /-
      case intro.intro
      G A : Type u
      inst✝³ : Group G
      inst✝² : IsFreeGroup G
      inst✝¹ : MulAction G A
      X : Type u
      inst✝ : Group X
      f : Quiver.Labelling (IsFreeGroupoid.Generators (CategoryTheory.ActionCategory …
      f' : IsFreeGroup.Generators G → SemidirectProduct (A → X) G mulAutArrow := fun …
      F' : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)
      hF' : ∀ (a : IsFreeGroup.Generators G), Eq (F' (IsFreeGroup.of a)) (f' a)
      uF' : ∀ (y : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)), (fun F => …
      ⊢ ExistsUnique fun F => ∀ (a b : IsFreeGroupoid.Generators (CategoryTheory.Act …
    -/
    refine ⟨uncurry F' ?_, ?_, ?_⟩
    · suffices SemidirectProduct.rightHom.comp F' = MonoidHom.id _ by
        exact DFunLike.ext_iff.mp this
      /-
        case intro.intro.refine_1
        G A : Type u
        inst✝³ : Group G
        inst✝² : IsFreeGroup G
        inst✝¹ : MulAction G A
        X : Type u
        inst✝ : Group X
        f : Quiver.Labelling (IsFreeGroupoid.Generators (CategoryTheory.ActionCategory …
        f' : IsFreeGroup.Generators G → SemidirectProduct (A → X) G mulAutArrow := fun …
        F' : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)
        hF' : ∀ (a : IsFreeGroup.Generators G), Eq (F' (IsFreeGroup.of a)) (f' a)
        uF' : ∀ (y : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)), (fun F => …
        ⊢ Eq (SemidirectProduct.rightHom.comp F') (MonoidHom.id G)
      -/
      apply IsFreeGroup.ext_hom (fun x ↦ ?_)
      /-
        G A : Type u
        inst✝³ : Group G
        inst✝² : IsFreeGroup G
        inst✝¹ : MulAction G A
        X : Type u
        inst✝ : Group X
        f : Quiver.Labelling (IsFreeGroupoid.Generators (CategoryTheory.ActionCategory …
        f' : IsFreeGroup.Generators G → SemidirectProduct (A → X) G mulAutArrow := fun …
        F' : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)
        hF' : ∀ (a : IsFreeGroup.Generators G), Eq (F' (IsFreeGroup.of a)) (f' a)
        uF' : ∀ (y : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)), (fun F => …
        x : IsFreeGroup.Generators G
        ⊢ Eq ((SemidirectProduct.rightHom.comp F') (IsFreeGroup.of x)) ((MonoidHom.id  …
      -/
      rw [MonoidHom.comp_apply, hF']
      /-
        G A : Type u
        inst✝³ : Group G
        inst✝² : IsFreeGroup G
        inst✝¹ : MulAction G A
        X : Type u
        inst✝ : Group X
        f : Quiver.Labelling (IsFreeGroupoid.Generators (CategoryTheory.ActionCategory …
        f' : IsFreeGroup.Generators G → SemidirectProduct (A → X) G mulAutArrow := fun …
        F' : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)
        hF' : ∀ (a : IsFreeGroup.Generators G), Eq (F' (IsFreeGroup.of a)) (f' a)
        uF' : ∀ (y : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)), (fun F => …
        x : IsFreeGroup.Generators G
        ⊢ Eq (SemidirectProduct.rightHom (f' x)) ((MonoidHom.id G) (IsFreeGroup.of x))
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.refine_2
        G A : Type u
        inst✝³ : Group G
        inst✝² : IsFreeGroup G
        inst✝¹ : MulAction G A
        X : Type u
        inst✝ : Group X
        f : Quiver.Labelling (IsFreeGroupoid.Generators (CategoryTheory.ActionCategory …
        f' : IsFreeGroup.Generators G → SemidirectProduct (A → X) G mulAutArrow := fun …
        F' : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)
        hF' : ∀ (a : IsFreeGroup.Generators G), Eq (F' (IsFreeGroup.of a)) (f' a)
        uF' : ∀ (y : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)), (fun F => …
        ⊢ (fun F => ∀ (a b : IsFreeGroupoid.Generators (CategoryTheory.ActionCategory  …
      -/
    · rintro ⟨⟨⟩, a : A⟩ ⟨⟨⟩, b⟩ ⟨e, h : IsFreeGroup.of e • a = b⟩
      /-
        case intro.intro.refine_2.mk.unit.mk.unit.mk
        G A : Type u
        inst✝³ : Group G
        inst✝² : IsFreeGroup G
        inst✝¹ : MulAction G A
        X : Type u
        inst✝ : Group X
        f : Quiver.Labelling (IsFreeGroupoid.Generators (CategoryTheory.ActionCategory …
        f' : IsFreeGroup.Generators G → SemidirectProduct (A → X) G mulAutArrow := fun …
        F' : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)
        hF' : ∀ (a : IsFreeGroup.Generators G), Eq (F' (IsFreeGroup.of a)) (f' a)
        uF' : ∀ (y : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)), (fun F => …
        a : A
        b : (CategoryTheory.actionAsFunctor G A).obj PUnit.unit
        e : IsFreeGroup.Generators G
        h : Eq (HSMul.hSMul (IsFreeGroup.of e) a) b
        ⊢ Eq ((CategoryTheory.ActionCategory.uncurry F' ⋯).map ((fun {a b} e => ⟨IsFre …
      -/
      change (F' (IsFreeGroup.of _)).left _ = _
      /-
        case intro.intro.refine_2.mk.unit.mk.unit.mk
        G A : Type u
        inst✝³ : Group G
        inst✝² : IsFreeGroup G
        inst✝¹ : MulAction G A
        X : Type u
        inst✝ : Group X
        f : Quiver.Labelling (IsFreeGroupoid.Generators (CategoryTheory.ActionCategory …
        f' : IsFreeGroup.Generators G → SemidirectProduct (A → X) G mulAutArrow := fun …
        F' : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)
        hF' : ∀ (a : IsFreeGroup.Generators G), Eq (F' (IsFreeGroup.of a)) (f' a)
        uF' : ∀ (y : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)), (fun F => …
        a : A
        b : (CategoryTheory.actionAsFunctor G A).obj PUnit.unit
        e : IsFreeGroup.Generators G
        h : Eq (HSMul.hSMul (IsFreeGroup.of e) a) b
        ⊢ Eq ((F' (IsFreeGroup.of ↑⟨e, h⟩)).left (CategoryTheory.ActionCategory.back ⟨ …
      -/
      rw [hF']
      /-
        case intro.intro.refine_2.mk.unit.mk.unit.mk
        G A : Type u
        inst✝³ : Group G
        inst✝² : IsFreeGroup G
        inst✝¹ : MulAction G A
        X : Type u
        inst✝ : Group X
        f : Quiver.Labelling (IsFreeGroupoid.Generators (CategoryTheory.ActionCategory …
        f' : IsFreeGroup.Generators G → SemidirectProduct (A → X) G mulAutArrow := fun …
        F' : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)
        hF' : ∀ (a : IsFreeGroup.Generators G), Eq (F' (IsFreeGroup.of a)) (f' a)
        uF' : ∀ (y : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)), (fun F => …
        a : A
        b : (CategoryTheory.actionAsFunctor G A).obj PUnit.unit
        e : IsFreeGroup.Generators G
        h : Eq (HSMul.hSMul (IsFreeGroup.of e) a) b
        ⊢ Eq ((f' ↑⟨e, h⟩).left (CategoryTheory.ActionCategory.back ⟨PUnit.unit, b⟩))  …
      -/
      cases inv_smul_eq_iff.mpr h.symm
      /-
        case intro.intro.refine_2.mk.unit.mk.unit.mk.refl
        G A : Type u
        inst✝³ : Group G
        inst✝² : IsFreeGroup G
        inst✝¹ : MulAction G A
        X : Type u
        inst✝ : Group X
        f : Quiver.Labelling (IsFreeGroupoid.Generators (CategoryTheory.ActionCategory …
        f' : IsFreeGroup.Generators G → SemidirectProduct (A → X) G mulAutArrow := fun …
        F' : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)
        hF' : ∀ (a : IsFreeGroup.Generators G), Eq (F' (IsFreeGroup.of a)) (f' a)
        uF' : ∀ (y : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)), (fun F => …
        b : (CategoryTheory.actionAsFunctor G A).obj PUnit.unit
        e : IsFreeGroup.Generators G
        h : Eq (HSMul.hSMul (IsFreeGroup.of e) (HSMul.hSMul (Inv.inv (IsFreeGroup.of e …
        ⊢ Eq ((f' ↑⟨e, h⟩).left (CategoryTheory.ActionCategory.back ⟨PUnit.unit, b⟩))  …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.refine_3
        G A : Type u
        inst✝³ : Group G
        inst✝² : IsFreeGroup G
        inst✝¹ : MulAction G A
        X : Type u
        inst✝ : Group X
        f : Quiver.Labelling (IsFreeGroupoid.Generators (CategoryTheory.ActionCategory …
        f' : IsFreeGroup.Generators G → SemidirectProduct (A → X) G mulAutArrow := fun …
        F' : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)
        hF' : ∀ (a : IsFreeGroup.Generators G), Eq (F' (IsFreeGroup.of a)) (f' a)
        uF' : ∀ (y : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)), (fun F => …
        ⊢ ∀ (y : CategoryTheory.Functor (CategoryTheory.ActionCategory G A) (CategoryT …
      -/
    · intro E hE
      have : curry E = F' := by
        apply uF'
        intro e
        ext
        · convert hE _ _ _
          rfl
        · rfl
      /-
        case intro.intro.refine_3
        G A : Type u
        inst✝³ : Group G
        inst✝² : IsFreeGroup G
        inst✝¹ : MulAction G A
        X : Type u
        inst✝ : Group X
        f : Quiver.Labelling (IsFreeGroupoid.Generators (CategoryTheory.ActionCategory …
        f' : IsFreeGroup.Generators G → SemidirectProduct (A → X) G mulAutArrow := fun …
        F' : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)
        hF' : ∀ (a : IsFreeGroup.Generators G), Eq (F' (IsFreeGroup.of a)) (f' a)
        uF' : ∀ (y : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)), (fun F => …
        E : CategoryTheory.Functor (CategoryTheory.ActionCategory G A) (CategoryTheory …
        hE : ∀ (a b : IsFreeGroupoid.Generators (CategoryTheory.ActionCategory G A)) ( …
        this : Eq (CategoryTheory.ActionCategory.curry E) F'
        ⊢ Eq E (CategoryTheory.ActionCategory.uncurry F' ⋯)
      -/
      apply Functor.hext
        /-
          case intro.intro.refine_3.h_obj
          G A : Type u
          inst✝³ : Group G
          inst✝² : IsFreeGroup G
          inst✝¹ : MulAction G A
          X : Type u
          inst✝ : Group X
          f : Quiver.Labelling (IsFreeGroupoid.Generators (CategoryTheory.ActionCategory …
          f' : IsFreeGroup.Generators G → SemidirectProduct (A → X) G mulAutArrow := fun …
          F' : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)
          hF' : ∀ (a : IsFreeGroup.Generators G), Eq (F' (IsFreeGroup.of a)) (f' a)
          uF' : ∀ (y : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)), (fun F => …
          E : CategoryTheory.Functor (CategoryTheory.ActionCategory G A) (CategoryTheory …
          hE : ∀ (a b : IsFreeGroupoid.Generators (CategoryTheory.ActionCategory G A)) ( …
          this : Eq (CategoryTheory.ActionCategory.curry E) F'
          ⊢ ∀ (X_1 : CategoryTheory.ActionCategory G A), Eq (E.obj X_1) ((CategoryTheory …
        -/
      · intro
        /-
          case intro.intro.refine_3.h_obj
          G A : Type u
          inst✝³ : Group G
          inst✝² : IsFreeGroup G
          inst✝¹ : MulAction G A
          X : Type u
          inst✝ : Group X
          f : Quiver.Labelling (IsFreeGroupoid.Generators (CategoryTheory.ActionCategory …
          f' : IsFreeGroup.Generators G → SemidirectProduct (A → X) G mulAutArrow := fun …
          F' : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)
          hF' : ∀ (a : IsFreeGroup.Generators G), Eq (F' (IsFreeGroup.of a)) (f' a)
          uF' : ∀ (y : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)), (fun F => …
          E : CategoryTheory.Functor (CategoryTheory.ActionCategory G A) (CategoryTheory …
          hE : ∀ (a b : IsFreeGroupoid.Generators (CategoryTheory.ActionCategory G A)) ( …
          this : Eq (CategoryTheory.ActionCategory.curry E) F'
          X✝ : CategoryTheory.ActionCategory G A
          ⊢ Eq (E.obj X✝) ((CategoryTheory.ActionCategory.uncurry F' ⋯).obj X✝)
        -/
        apply Unit.ext
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.refine_3.h_map
          G A : Type u
          inst✝³ : Group G
          inst✝² : IsFreeGroup G
          inst✝¹ : MulAction G A
          X : Type u
          inst✝ : Group X
          f : Quiver.Labelling (IsFreeGroupoid.Generators (CategoryTheory.ActionCategory …
          f' : IsFreeGroup.Generators G → SemidirectProduct (A → X) G mulAutArrow := fun …
          F' : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)
          hF' : ∀ (a : IsFreeGroup.Generators G), Eq (F' (IsFreeGroup.of a)) (f' a)
          uF' : ∀ (y : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)), (fun F => …
          E : CategoryTheory.Functor (CategoryTheory.ActionCategory G A) (CategoryTheory …
          hE : ∀ (a b : IsFreeGroupoid.Generators (CategoryTheory.ActionCategory G A)) ( …
          this : Eq (CategoryTheory.ActionCategory.curry E) F'
          ⊢ ∀ (X_1 Y : CategoryTheory.ActionCategory G A) (f : Quiver.Hom X_1 Y), HEq (E …
        -/
      · refine ActionCategory.cases ?_
        /-
          case intro.intro.refine_3.h_map
          G A : Type u
          inst✝³ : Group G
          inst✝² : IsFreeGroup G
          inst✝¹ : MulAction G A
          X : Type u
          inst✝ : Group X
          f : Quiver.Labelling (IsFreeGroupoid.Generators (CategoryTheory.ActionCategory …
          f' : IsFreeGroup.Generators G → SemidirectProduct (A → X) G mulAutArrow := fun …
          F' : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)
          hF' : ∀ (a : IsFreeGroup.Generators G), Eq (F' (IsFreeGroup.of a)) (f' a)
          uF' : ∀ (y : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)), (fun F => …
          E : CategoryTheory.Functor (CategoryTheory.ActionCategory G A) (CategoryTheory …
          hE : ∀ (a b : IsFreeGroupoid.Generators (CategoryTheory.ActionCategory G A)) ( …
          this : Eq (CategoryTheory.ActionCategory.curry E) F'
          ⊢ ∀ (t : A) (g : G), HEq (E.map (CategoryTheory.ActionCategory.homOfPair t g)) …
        -/
        intros
        /-
          case intro.intro.refine_3.h_map
          G A : Type u
          inst✝³ : Group G
          inst✝² : IsFreeGroup G
          inst✝¹ : MulAction G A
          X : Type u
          inst✝ : Group X
          f : Quiver.Labelling (IsFreeGroupoid.Generators (CategoryTheory.ActionCategory …
          f' : IsFreeGroup.Generators G → SemidirectProduct (A → X) G mulAutArrow := fun …
          F' : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)
          hF' : ∀ (a : IsFreeGroup.Generators G), Eq (F' (IsFreeGroup.of a)) (f' a)
          uF' : ∀ (y : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)), (fun F => …
          E : CategoryTheory.Functor (CategoryTheory.ActionCategory G A) (CategoryTheory …
          hE : ∀ (a b : IsFreeGroupoid.Generators (CategoryTheory.ActionCategory G A)) ( …
          this : Eq (CategoryTheory.ActionCategory.curry E) F'
          t✝ : A
          g✝ : G
          ⊢ HEq (E.map (CategoryTheory.ActionCategory.homOfPair t✝ g✝)) ((CategoryTheory …
        -/
        simp only [← this, uncurry_map, curry_apply_left, coe_back, homOfPair.val]
        /-
          case intro.intro.refine_3.h_map
          G A : Type u
          inst✝³ : Group G
          inst✝² : IsFreeGroup G
          inst✝¹ : MulAction G A
          X : Type u
          inst✝ : Group X
          f : Quiver.Labelling (IsFreeGroupoid.Generators (CategoryTheory.ActionCategory …
          f' : IsFreeGroup.Generators G → SemidirectProduct (A → X) G mulAutArrow := fun …
          F' : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)
          hF' : ∀ (a : IsFreeGroup.Generators G), Eq (F' (IsFreeGroup.of a)) (f' a)
          uF' : ∀ (y : MonoidHom G (SemidirectProduct (A → X) G mulAutArrow)), (fun F => …
          E : CategoryTheory.Functor (CategoryTheory.ActionCategory G A) (CategoryTheory …
          hE : ∀ (a b : IsFreeGroupoid.Generators (CategoryTheory.ActionCategory G A)) ( …
          this : Eq (CategoryTheory.ActionCategory.curry E) F'
          t✝ : A
          g✝ : G
          ⊢ HEq (E.map (CategoryTheory.ActionCategory.homOfPair t✝ g✝)) (E.map (Category …
        -/
        rfl
        /-
          🎉 no goals
        -/


/-- The root of `T`, except its type is `G` instead of the type synonym `T`. -/
private def root' : G :=
  show T from root T

-- this has to be marked noncomputable, see issue https://github.com/leanprover-community/mathlib4/pull/451.
-- It might be nicer to define this in terms of `composePath`

/-- A path in the tree gives a hom, by composition. -/
-- Porting note: removed noncomputable. This is already declared at the beginning of the section.
def homOfPath : ∀ {a : G}, Path (root T) a → (root' T ⟶ a)
  | _, Path.nil => 𝟙 _
  | _, Path.cons p f => homOfPath p ≫ Sum.recOn f.val (fun e => of e) fun e => inv (of e)


/-- For every vertex `a`, there is a canonical hom from the root, given by the path in the tree. -/
def treeHom (a : G) : root' T ⟶ a :=
  homOfPath T default


/-- Any path to `a` gives `treeHom T a`, since paths in the tree are unique. -/
theorem treeHom_eq {a : G} (p : Path (root T) a) : treeHom T a = homOfPath T p := by
  /-
    G : Type u
    inst✝² : CategoryTheory.Groupoid G
    inst✝¹ : IsFreeGroupoid G
    T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
    inst✝ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeGr …
    a : G
    p : Quiver.Path (Quiver.root (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
    ⊢ Eq (IsFreeGroupoid.SpanningTree.treeHom T a) (IsFreeGroupoid.SpanningTree.ho …
  -/
  rw [treeHom, Unique.default_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem treeHom_root : treeHom T (root' T) = 𝟙 _ :=
  -- this should just be `treeHom_eq T Path.nil`, but Lean treats `homOfPath` with suspicion.
    _root_.trans
    (treeHom_eq T Path.nil) rfl


/-- Any hom in `G` can be made into a loop, by conjugating with `treeHom`s. -/
def loopOfHom {a b : G} (p : a ⟶ b) : End (root' T) :=
  treeHom T a ≫ p ≫ inv (treeHom T b)


/-- Turning an edge in the spanning tree into a loop gives the identity loop. -/
theorem loopOfHom_eq_id {a b : Generators G} (e) (H : e ∈ wideSubquiverSymmetrify T a b) :
    loopOfHom T (of e) = 𝟙 (root' T) := by
  /-
    G : Type u
    inst✝² : CategoryTheory.Groupoid G
    inst✝¹ : IsFreeGroupoid G
    T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
    inst✝ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeGr …
    a b : IsFreeGroupoid.Generators G
    e : Quiver.Hom a b
    H : Membership.mem (Quiver.wideSubquiverSymmetrify T a b) e
    ⊢ Eq (IsFreeGroupoid.SpanningTree.loopOfHom T (IsFreeGroupoid.of e)) (Category …
  -/
  rw [loopOfHom, ← Category.assoc, IsIso.comp_inv_eq, Category.id_comp]
  /-
    G : Type u
    inst✝² : CategoryTheory.Groupoid G
    inst✝¹ : IsFreeGroupoid G
    T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
    inst✝ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeGr …
    a b : IsFreeGroupoid.Generators G
    e : Quiver.Hom a b
    H : Membership.mem (Quiver.wideSubquiverSymmetrify T a b) e
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (IsFreeGroupoid.SpanningTree.treeHom  …
  -/
  cases' H with H H
    /-
      case inl
      G : Type u
      inst✝² : CategoryTheory.Groupoid G
      inst✝¹ : IsFreeGroupoid G
      T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
      inst✝ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeGr …
      a b : IsFreeGroupoid.Generators G
      e : Quiver.Hom a b
      H : T a b (Sum.inl e)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (IsFreeGroupoid.SpanningTree.treeHom  …
    -/
  · rw [treeHom_eq T (Path.cons default ⟨Sum.inl e, H⟩), homOfPath]
    /-
      case inl
      G : Type u
      inst✝² : CategoryTheory.Groupoid G
      inst✝¹ : IsFreeGroupoid G
      T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
      inst✝ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeGr …
      a b : IsFreeGroupoid.Generators G
      e : Quiver.Hom a b
      H : T a b (Sum.inl e)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (IsFreeGroupoid.SpanningTree.treeHom  …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case inr
      G : Type u
      inst✝² : CategoryTheory.Groupoid G
      inst✝¹ : IsFreeGroupoid G
      T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
      inst✝ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeGr …
      a b : IsFreeGroupoid.Generators G
      e : Quiver.Hom a b
      H : T b a (Sum.inr e)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (IsFreeGroupoid.SpanningTree.treeHom  …
    -/
  · rw [treeHom_eq T (Path.cons default ⟨Sum.inr e, H⟩), homOfPath]
    /-
      case inr
      G : Type u
      inst✝² : CategoryTheory.Groupoid G
      inst✝¹ : IsFreeGroupoid G
      T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
      inst✝ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeGr …
      a b : IsFreeGroupoid.Generators G
      e : Quiver.Hom a b
      H : T b a (Sum.inr e)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [IsIso.inv_hom_id, Category.comp_id, Category.assoc, treeHom]
    /-
      🎉 no goals
    -/


/-- Since a hom gives a loop, any homomorphism from the vertex group at the root
    extends to a functor on the whole groupoid. -/
@[simps]
def functorOfMonoidHom {X} [Monoid X] (f : End (root' T) →* X) :
    G ⥤ CategoryTheory.SingleObj X where
  obj _ := ()
  map p := f (loopOfHom T p)
  map_id := by
    /-
      G : Type u
      inst✝³ : CategoryTheory.Groupoid G
      inst✝² : IsFreeGroupoid G
      T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
      inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
      X : Type ?u.23251
      inst✝ : Monoid X
      f : MonoidHom (CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)) X
      ⊢ ∀ (X_1 : G), Eq ({ obj := fun x => Unit.unit, map := fun {X_2 Y} p => f (IsF …
    -/
    intro a
    /-
      G : Type u
      inst✝³ : CategoryTheory.Groupoid G
      inst✝² : IsFreeGroupoid G
      T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
      inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
      X : Type ?u.23251
      inst✝ : Monoid X
      f : MonoidHom (CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)) X
      a : G
      ⊢ Eq ({ obj := fun x => Unit.unit, map := fun {X_1 Y} p => f (IsFreeGroupoid.S …
    -/
    dsimp only [loopOfHom]
    /-
      G : Type u
      inst✝³ : CategoryTheory.Groupoid G
      inst✝² : IsFreeGroupoid G
      T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
      inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
      X : Type ?u.23251
      inst✝ : Monoid X
      f : MonoidHom (CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)) X
      a : G
      ⊢ Eq (f (CategoryTheory.CategoryStruct.comp (IsFreeGroupoid.SpanningTree.treeH …
    -/
    rw [Category.id_comp, IsIso.hom_inv_id, ← End.one_def, f.map_one, id_as_one]
    /-
      🎉 no goals
    -/
  map_comp := by
    /-
      G : Type u
      inst✝³ : CategoryTheory.Groupoid G
      inst✝² : IsFreeGroupoid G
      T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
      inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
      X : Type ?u.23251
      inst✝ : Monoid X
      f : MonoidHom (CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)) X
      ⊢ ∀ {X_1 Y Z : G} (f_1 : Quiver.Hom X_1 Y) (g : Quiver.Hom Y Z), Eq ({ obj :=  …
    -/
    intros
    /-
      G : Type u
      inst✝³ : CategoryTheory.Groupoid G
      inst✝² : IsFreeGroupoid G
      T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
      inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
      X : Type ?u.23251
      inst✝ : Monoid X
      f : MonoidHom (CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)) X
      X✝ Y✝ Z✝ : G
      f✝ : Quiver.Hom X✝ Y✝
      g✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun x => Unit.unit, map := fun {X_1 Y} p => f (IsFreeGroupoid.S …
    -/
    rw [comp_as_mul, ← f.map_mul]
    /-
      G : Type u
      inst✝³ : CategoryTheory.Groupoid G
      inst✝² : IsFreeGroupoid G
      T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
      inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
      X : Type ?u.23251
      inst✝ : Monoid X
      f : MonoidHom (CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)) X
      X✝ Y✝ Z✝ : G
      f✝ : Quiver.Hom X✝ Y✝
      g✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun x => Unit.unit, map := fun {X_1 Y} p => f (IsFreeGroupoid.S …
    -/
    simp only [IsIso.inv_hom_id_assoc, loopOfHom, End.mul_def, Category.assoc]
    /-
      🎉 no goals
    -/


/-- Given a free groupoid and an arborescence of its generating quiver, the vertex
    group at the root is freely generated by loops coming from generating arrows
    in the complement of the tree. -/
lemma endIsFree : IsFreeGroup (End (root' T)) :=
  IsFreeGroup.ofUniqueLift ((wideSubquiverEquivSetTotal <| wideSubquiverSymmetrify T)ᶜ : Set _)
    (fun e => loopOfHom T (of e.val.hom))
    (by
      /-
        G : Type u
        inst✝² : CategoryTheory.Groupoid G
        inst✝¹ : IsFreeGroupoid G
        T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
        inst✝ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeGr …
        ⊢ ∀ {H : Type u} [inst : Group H] (f : ↑(HasCompl.compl (Quiver.wideSubquiverE …
      -/
      intro X _ f
      let f' : Labelling (Generators G) X := fun a b e =>
        if h : e ∈ wideSubquiverSymmetrify T a b then 1 else f ⟨⟨a, b, e⟩, h⟩
      /-
        G : Type u
        inst✝³ : CategoryTheory.Groupoid G
        inst✝² : IsFreeGroupoid G
        T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
        inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
        X : Type u
        inst✝ : Group X
        f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
        f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
        ⊢ ExistsUnique fun F => ∀ (a : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetT …
      -/
      rcases unique_lift f' with ⟨F', hF', uF'⟩
      /-
        case intro.intro
        G : Type u
        inst✝³ : CategoryTheory.Groupoid G
        inst✝² : IsFreeGroupoid G
        T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
        inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
        X : Type u
        inst✝ : Group X
        f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
        f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
        F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
        hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
        uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
        ⊢ ExistsUnique fun F => ∀ (a : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetT …
      -/
      refine ⟨F'.mapEnd _, ?_, ?_⟩
      · suffices ∀ {x y} (q : x ⟶ y), F'.map (loopOfHom T q) = (F'.map q : X) by
          rintro ⟨⟨a, b, e⟩, h⟩
          erw [Functor.mapEnd_apply, this, hF']
          exact dif_neg h
        /-
          case intro.intro.refine_1
          G : Type u
          inst✝³ : CategoryTheory.Groupoid G
          inst✝² : IsFreeGroupoid G
          T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
          inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
          X : Type u
          inst✝ : Group X
          f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
          f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
          F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
          hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
          uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
          ⊢ ∀ {x y : G} (q : Quiver.Hom x y), Eq (F'.map (IsFreeGroupoid.SpanningTree.lo …
        -/
        intros x y q
        suffices ∀ {a} (p : Path (root T) a), F'.map (homOfPath T p) = 1 by
          simp only [this, treeHom, comp_as_mul, inv_as_inv, loopOfHom, inv_one, mul_one,
            one_mul, Functor.map_inv, Functor.map_comp]
        /-
          case intro.intro.refine_1
          G : Type u
          inst✝³ : CategoryTheory.Groupoid G
          inst✝² : IsFreeGroupoid G
          T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
          inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
          X : Type u
          inst✝ : Group X
          f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
          f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
          F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
          hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
          uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
          x y : G
          q : Quiver.Hom x y
          ⊢ ∀ {a : WideSubquiver.toType (Quiver.Symmetrify (IsFreeGroupoid.Generators G) …
        -/
        intro a p
        /-
          case intro.intro.refine_1
          G : Type u
          inst✝³ : CategoryTheory.Groupoid G
          inst✝² : IsFreeGroupoid G
          T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
          inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
          X : Type u
          inst✝ : Group X
          f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
          f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
          F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
          hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
          uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
          x y : G
          q : Quiver.Hom x y
          a : WideSubquiver.toType (Quiver.Symmetrify (IsFreeGroupoid.Generators G)) T
          p : Quiver.Path (Quiver.root (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
          ⊢ Eq (F'.map (IsFreeGroupoid.SpanningTree.homOfPath T p)) 1
        -/
        induction' p with b c p e ih
          /-
            case intro.intro.refine_1.nil
            G : Type u
            inst✝³ : CategoryTheory.Groupoid G
            inst✝² : IsFreeGroupoid G
            T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
            inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
            X : Type u
            inst✝ : Group X
            f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
            f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
            F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
            hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
            uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
            x y : G
            q : Quiver.Hom x y
            a : WideSubquiver.toType (Quiver.Symmetrify (IsFreeGroupoid.Generators G)) T
            ⊢ Eq (F'.map (IsFreeGroupoid.SpanningTree.homOfPath T Quiver.Path.nil)) 1
          -/
        · rw [homOfPath, F'.map_id, id_as_one]
          /-
            🎉 no goals
          -/
        /-
          case intro.intro.refine_1.cons
          G : Type u
          inst✝³ : CategoryTheory.Groupoid G
          inst✝² : IsFreeGroupoid G
          T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
          inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
          X : Type u
          inst✝ : Group X
          f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
          f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
          F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
          hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
          uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
          x y : G
          q : Quiver.Hom x y
          a b c : WideSubquiver.toType (Quiver.Symmetrify (IsFreeGroupoid.Generators G)) T
          p : Quiver.Path (Quiver.root (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
          e : Quiver.Hom b c
          ih : Eq (F'.map (IsFreeGroupoid.SpanningTree.homOfPath T p)) 1
          ⊢ Eq (F'.map (IsFreeGroupoid.SpanningTree.homOfPath T (p.cons e))) 1
        -/
        rw [homOfPath, F'.map_comp, comp_as_mul, ih, mul_one]
        /-
          case intro.intro.refine_1.cons
          G : Type u
          inst✝³ : CategoryTheory.Groupoid G
          inst✝² : IsFreeGroupoid G
          T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
          inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
          X : Type u
          inst✝ : Group X
          f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
          f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
          F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
          hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
          uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
          x y : G
          q : Quiver.Hom x y
          a b c : WideSubquiver.toType (Quiver.Symmetrify (IsFreeGroupoid.Generators G)) T
          p : Quiver.Path (Quiver.root (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
          e : Quiver.Hom b c
          ih : Eq (F'.map (IsFreeGroupoid.SpanningTree.homOfPath T p)) 1
          ⊢ Eq (F'.map (Sum.recOn (↑e) (fun e => IsFreeGroupoid.of e) fun e => CategoryT …
        -/
        rcases e with ⟨e | e, eT⟩
          /-
            case intro.intro.refine_1.cons.mk.inl
            G : Type u
            inst✝³ : CategoryTheory.Groupoid G
            inst✝² : IsFreeGroupoid G
            T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
            inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
            X : Type u
            inst✝ : Group X
            f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
            f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
            F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
            hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
            uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
            x y : G
            q : Quiver.Hom x y
            a b c : WideSubquiver.toType (Quiver.Symmetrify (IsFreeGroupoid.Generators G)) T
            p : Quiver.Path (Quiver.root (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
            ih : Eq (F'.map (IsFreeGroupoid.SpanningTree.homOfPath T p)) 1
            e : Quiver.Hom b c
            eT : Membership.mem (T b c) (Sum.inl e)
            ⊢ Eq (F'.map (Sum.recOn (↑⟨Sum.inl e, eT⟩) (fun e => IsFreeGroupoid.of e) fun  …
          -/
        · rw [hF']
          /-
            case intro.intro.refine_1.cons.mk.inl
            G : Type u
            inst✝³ : CategoryTheory.Groupoid G
            inst✝² : IsFreeGroupoid G
            T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
            inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
            X : Type u
            inst✝ : Group X
            f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
            f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
            F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
            hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
            uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
            x y : G
            q : Quiver.Hom x y
            a b c : WideSubquiver.toType (Quiver.Symmetrify (IsFreeGroupoid.Generators G)) T
            p : Quiver.Path (Quiver.root (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
            ih : Eq (F'.map (IsFreeGroupoid.SpanningTree.homOfPath T p)) 1
            e : Quiver.Hom b c
            eT : Membership.mem (T b c) (Sum.inl e)
            ⊢ Eq (f' e) 1
          -/
          exact dif_pos (Or.inl eT)
          /-
            🎉 no goals
          -/
          /-
            case intro.intro.refine_1.cons.mk.inr
            G : Type u
            inst✝³ : CategoryTheory.Groupoid G
            inst✝² : IsFreeGroupoid G
            T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
            inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
            X : Type u
            inst✝ : Group X
            f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
            f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
            F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
            hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
            uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
            x y : G
            q : Quiver.Hom x y
            a b c : WideSubquiver.toType (Quiver.Symmetrify (IsFreeGroupoid.Generators G)) T
            p : Quiver.Path (Quiver.root (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
            ih : Eq (F'.map (IsFreeGroupoid.SpanningTree.homOfPath T p)) 1
            e : Quiver.Hom c b
            eT : Membership.mem (T b c) (Sum.inr e)
            ⊢ Eq (F'.map (Sum.recOn (↑⟨Sum.inr e, eT⟩) (fun e => IsFreeGroupoid.of e) fun  …
          -/
        · rw [F'.map_inv, inv_as_inv, inv_eq_one, hF']
          /-
            case intro.intro.refine_1.cons.mk.inr
            G : Type u
            inst✝³ : CategoryTheory.Groupoid G
            inst✝² : IsFreeGroupoid G
            T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
            inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
            X : Type u
            inst✝ : Group X
            f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
            f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
            F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
            hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
            uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
            x y : G
            q : Quiver.Hom x y
            a b c : WideSubquiver.toType (Quiver.Symmetrify (IsFreeGroupoid.Generators G)) T
            p : Quiver.Path (Quiver.root (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
            ih : Eq (F'.map (IsFreeGroupoid.SpanningTree.homOfPath T p)) 1
            e : Quiver.Hom c b
            eT : Membership.mem (T b c) (Sum.inr e)
            ⊢ Eq (f' e) 1
          -/
          exact dif_pos (Or.inr eT)
          /-
            🎉 no goals
          -/
        /-
          case intro.intro.refine_2
          G : Type u
          inst✝³ : CategoryTheory.Groupoid G
          inst✝² : IsFreeGroupoid G
          T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
          inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
          X : Type u
          inst✝ : Group X
          f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
          f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
          F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
          hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
          uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
          ⊢ ∀ (y : MonoidHom (CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T))  …
        -/
      · intro E hE
        /-
          case intro.intro.refine_2
          G : Type u
          inst✝³ : CategoryTheory.Groupoid G
          inst✝² : IsFreeGroupoid G
          T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
          inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
          X : Type u
          inst✝ : Group X
          f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
          f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
          F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
          hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
          uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
          E : MonoidHom (CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)) X
          hE : ∀ (a : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSu …
          ⊢ Eq E (CategoryTheory.Functor.mapEnd (IsFreeGroupoid.SpanningTree.root' T) F')
        -/
        ext x
        suffices (functorOfMonoidHom T E).map x = F'.map x by
          simpa only [loopOfHom, functorOfMonoidHom, IsIso.inv_id, treeHom_root,
            Category.id_comp, Category.comp_id] using this
        /-
          case intro.intro.refine_2.h
          G : Type u
          inst✝³ : CategoryTheory.Groupoid G
          inst✝² : IsFreeGroupoid G
          T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
          inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
          X : Type u
          inst✝ : Group X
          f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
          f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
          F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
          hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
          uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
          E : MonoidHom (CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)) X
          hE : ∀ (a : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSu …
          x : CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)
          ⊢ Eq ((IsFreeGroupoid.SpanningTree.functorOfMonoidHom T E).map x) (F'.map x)
        -/
        congr
        /-
          case intro.intro.refine_2.h.h.e_5.h.e_self
          G : Type u
          inst✝³ : CategoryTheory.Groupoid G
          inst✝² : IsFreeGroupoid G
          T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
          inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
          X : Type u
          inst✝ : Group X
          f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
          f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
          F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
          hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
          uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
          E : MonoidHom (CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)) X
          hE : ∀ (a : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSu …
          x : CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)
          ⊢ Eq (IsFreeGroupoid.SpanningTree.functorOfMonoidHom T E) F'
        -/
        apply uF'
        /-
          case intro.intro.refine_2.h.h.e_5.h.e_self.a
          G : Type u
          inst✝³ : CategoryTheory.Groupoid G
          inst✝² : IsFreeGroupoid G
          T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
          inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
          X : Type u
          inst✝ : Group X
          f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
          f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
          F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
          hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
          uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
          E : MonoidHom (CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)) X
          hE : ∀ (a : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSu …
          x : CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)
          ⊢ ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq ((IsFreeGroup …
        -/
        intro a b e
        /-
          case intro.intro.refine_2.h.h.e_5.h.e_self.a
          G : Type u
          inst✝³ : CategoryTheory.Groupoid G
          inst✝² : IsFreeGroupoid G
          T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
          inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
          X : Type u
          inst✝ : Group X
          f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
          f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
          F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
          hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
          uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
          E : MonoidHom (CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)) X
          hE : ∀ (a : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSu …
          x : CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)
          a b : IsFreeGroupoid.Generators G
          e : Quiver.Hom a b
          ⊢ Eq ((IsFreeGroupoid.SpanningTree.functorOfMonoidHom T E).map (IsFreeGroupoid …
        -/
        change E (loopOfHom T _) = dite _ _ _
        /-
          case intro.intro.refine_2.h.h.e_5.h.e_self.a
          G : Type u
          inst✝³ : CategoryTheory.Groupoid G
          inst✝² : IsFreeGroupoid G
          T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
          inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
          X : Type u
          inst✝ : Group X
          f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
          f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
          F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
          hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
          uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
          E : MonoidHom (CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)) X
          hE : ∀ (a : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSu …
          x : CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)
          a b : IsFreeGroupoid.Generators G
          e : Quiver.Hom a b
          ⊢ Eq (E (IsFreeGroupoid.SpanningTree.loopOfHom T (IsFreeGroupoid.of e))) (dite …
        -/
        split_ifs with h
          /-
            case pos
            G : Type u
            inst✝³ : CategoryTheory.Groupoid G
            inst✝² : IsFreeGroupoid G
            T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
            inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
            X : Type u
            inst✝ : Group X
            f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
            f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
            F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
            hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
            uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
            E : MonoidHom (CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)) X
            hE : ∀ (a : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSu …
            x : CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)
            a b : IsFreeGroupoid.Generators G
            e : Quiver.Hom a b
            h : Membership.mem (Quiver.wideSubquiverSymmetrify T a b) e
            ⊢ Eq (E (IsFreeGroupoid.SpanningTree.loopOfHom T (IsFreeGroupoid.of e))) 1
          -/
        · rw [loopOfHom_eq_id T e h, ← End.one_def, E.map_one]
          /-
            🎉 no goals
          -/
          /-
            case neg
            G : Type u
            inst✝³ : CategoryTheory.Groupoid G
            inst✝² : IsFreeGroupoid G
            T : WideSubquiver (Quiver.Symmetrify (IsFreeGroupoid.Generators G))
            inst✝¹ : Quiver.Arborescence (WideSubquiver.toType (Quiver.Symmetrify (IsFreeG …
            X : Type u
            inst✝ : Group X
            f : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSubquiverS …
            f' : Quiver.Labelling (IsFreeGroupoid.Generators G) X := fun a b e => dite (Me …
            F' : CategoryTheory.Functor G (CategoryTheory.SingleObj X)
            hF' : ∀ (a b : IsFreeGroupoid.Generators G) (g : Quiver.Hom a b), Eq (F'.map ( …
            uF' : ∀ (y : CategoryTheory.Functor G (CategoryTheory.SingleObj X)), (fun F => …
            E : MonoidHom (CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)) X
            hE : ∀ (a : ↑(HasCompl.compl (Quiver.wideSubquiverEquivSetTotal (Quiver.wideSu …
            x : CategoryTheory.End (IsFreeGroupoid.SpanningTree.root' T)
            a b : IsFreeGroupoid.Generators G
            e : Quiver.Hom a b
            h : Not (Membership.mem (Quiver.wideSubquiverSymmetrify T a b) e)
            ⊢ Eq (E (IsFreeGroupoid.SpanningTree.loopOfHom T (IsFreeGroupoid.of e))) (f ⟨{ …
          -/
        · exact hE ⟨⟨a, b, e⟩, h⟩)
          /-
            🎉 no goals
          -/


/-- Another name for the identity function `G → G`, to help type checking. -/
private def symgen {G : Type u} [Groupoid.{v} G] [IsFreeGroupoid G] :
    G → Symmetrify (Generators G) :=
  id


/-- If there exists a morphism `a → b` in a free groupoid, then there also exists a zigzag
from `a` to `b` in the generating quiver. -/
theorem path_nonempty_of_hom {G} [Groupoid.{u, u} G] [IsFreeGroupoid G] {a b : G} :
    Nonempty (a ⟶ b) → Nonempty (Path (symgen a) (symgen b)) := by
  /-
    G : Type u
    inst✝¹ : CategoryTheory.Groupoid G
    inst✝ : IsFreeGroupoid G
    a b : G
    ⊢ Nonempty (Quiver.Hom a b) → Nonempty (Quiver.Path (IsFreeGroupoid.symgen a)  …
  -/
  rintro ⟨p⟩
  rw [← @WeaklyConnectedComponent.eq (Generators G), eq_comm, ← FreeGroup.of_injective.eq_iff, ←
    mul_inv_eq_one]
  /-
    case intro
    G : Type u
    inst✝¹ : CategoryTheory.Groupoid G
    inst✝ : IsFreeGroupoid G
    a b : G
    p : Quiver.Hom a b
    ⊢ Eq (HMul.hMul (FreeGroup.of (Quiver.WeaklyConnectedComponent.mk (IsFreeGroup …
  -/
  let X := FreeGroup (WeaklyConnectedComponent <| Generators G)
  /-
    case intro
    G : Type u
    inst✝¹ : CategoryTheory.Groupoid G
    inst✝ : IsFreeGroupoid G
    a b : G
    p : Quiver.Hom a b
    X : Type u := FreeGroup (Quiver.WeaklyConnectedComponent (IsFreeGroupoid.Gener …
    ⊢ Eq (HMul.hMul (FreeGroup.of (Quiver.WeaklyConnectedComponent.mk (IsFreeGroup …
  -/
  let f : G → X := fun g => FreeGroup.of (WeaklyConnectedComponent.mk g)
  /-
    case intro
    G : Type u
    inst✝¹ : CategoryTheory.Groupoid G
    inst✝ : IsFreeGroupoid G
    a b : G
    p : Quiver.Hom a b
    X : Type u := FreeGroup (Quiver.WeaklyConnectedComponent (IsFreeGroupoid.Gener …
    f : G → X := fun g => FreeGroup.of (Quiver.WeaklyConnectedComponent.mk g)
    ⊢ Eq (HMul.hMul (FreeGroup.of (Quiver.WeaklyConnectedComponent.mk (IsFreeGroup …
  -/
  let F : G ⥤ CategoryTheory.SingleObj.{u} (X : Type u) := SingleObj.differenceFunctor f
  /-
    case intro
    G : Type u
    inst✝¹ : CategoryTheory.Groupoid G
    inst✝ : IsFreeGroupoid G
    a b : G
    p : Quiver.Hom a b
    X : Type u := FreeGroup (Quiver.WeaklyConnectedComponent (IsFreeGroupoid.Gener …
    f : G → X := fun g => FreeGroup.of (Quiver.WeaklyConnectedComponent.mk g)
    F : CategoryTheory.Functor G (CategoryTheory.SingleObj X) := CategoryTheory.Si …
    ⊢ Eq (HMul.hMul (FreeGroup.of (Quiver.WeaklyConnectedComponent.mk (IsFreeGroup …
  -/
  change (F.map p) = ((@CategoryTheory.Functor.const G _ _ (SingleObj.category X)).obj ()).map p
  /-
    case intro
    G : Type u
    inst✝¹ : CategoryTheory.Groupoid G
    inst✝ : IsFreeGroupoid G
    a b : G
    p : Quiver.Hom a b
    X : Type u := FreeGroup (Quiver.WeaklyConnectedComponent (IsFreeGroupoid.Gener …
    f : G → X := fun g => FreeGroup.of (Quiver.WeaklyConnectedComponent.mk g)
    F : CategoryTheory.Functor G (CategoryTheory.SingleObj X) := CategoryTheory.Si …
    ⊢ Eq (F.map p) (((CategoryTheory.Functor.const G).obj Unit.unit).map p)
  -/
  congr; ext
  /-
    case intro.h.e_5.h.e_self.h
    G : Type u
    inst✝¹ : CategoryTheory.Groupoid G
    inst✝ : IsFreeGroupoid G
    a b : G
    p : Quiver.Hom a b
    X : Type u := FreeGroup (Quiver.WeaklyConnectedComponent (IsFreeGroupoid.Gener …
    f : G → X := fun g => FreeGroup.of (Quiver.WeaklyConnectedComponent.mk g)
    F : CategoryTheory.Functor G (CategoryTheory.SingleObj X) := CategoryTheory.Si …
    a✝ b✝ : IsFreeGroupoid.Generators G
    e✝ : Quiver.Hom a✝ b✝
    ⊢ Eq (F.map (IsFreeGroupoid.of e✝)) (((CategoryTheory.Functor.const G).obj Uni …
  -/
  rw [Functor.const_obj_map, id_as_one, differenceFunctor_map, @mul_inv_eq_one _ _ (f _)]
  /-
    case intro.h.e_5.h.e_self.h
    G : Type u
    inst✝¹ : CategoryTheory.Groupoid G
    inst✝ : IsFreeGroupoid G
    a b : G
    p : Quiver.Hom a b
    X : Type u := FreeGroup (Quiver.WeaklyConnectedComponent (IsFreeGroupoid.Gener …
    f : G → X := fun g => FreeGroup.of (Quiver.WeaklyConnectedComponent.mk g)
    F : CategoryTheory.Functor G (CategoryTheory.SingleObj X) := CategoryTheory.Si …
    a✝ b✝ : IsFreeGroupoid.Generators G
    e✝ : Quiver.Hom a✝ b✝
    ⊢ Eq (f b✝) (f (letFun a✝ fun this => this))
  -/
  apply congr_arg FreeGroup.of
  /-
    case intro.h.e_5.h.e_self.h
    G : Type u
    inst✝¹ : CategoryTheory.Groupoid G
    inst✝ : IsFreeGroupoid G
    a b : G
    p : Quiver.Hom a b
    X : Type u := FreeGroup (Quiver.WeaklyConnectedComponent (IsFreeGroupoid.Gener …
    f : G → X := fun g => FreeGroup.of (Quiver.WeaklyConnectedComponent.mk g)
    F : CategoryTheory.Functor G (CategoryTheory.SingleObj X) := CategoryTheory.Si …
    a✝ b✝ : IsFreeGroupoid.Generators G
    e✝ : Quiver.Hom a✝ b✝
    ⊢ Eq (Quiver.WeaklyConnectedComponent.mk b✝) (Quiver.WeaklyConnectedComponent. …
  -/
  apply (WeaklyConnectedComponent.eq _ _).mpr
  /-
    case intro.h.e_5.h.e_self.h
    G : Type u
    inst✝¹ : CategoryTheory.Groupoid G
    inst✝ : IsFreeGroupoid G
    a b : G
    p : Quiver.Hom a b
    X : Type u := FreeGroup (Quiver.WeaklyConnectedComponent (IsFreeGroupoid.Gener …
    f : G → X := fun g => FreeGroup.of (Quiver.WeaklyConnectedComponent.mk g)
    F : CategoryTheory.Functor G (CategoryTheory.SingleObj X) := CategoryTheory.Si …
    a✝ b✝ : IsFreeGroupoid.Generators G
    e✝ : Quiver.Hom a✝ b✝
    ⊢ Nonempty (Quiver.Path b✝ (letFun a✝ fun this => this))
  -/
  exact ⟨Hom.toPath (Sum.inr (by assumption))⟩
  /-
    🎉 no goals
  -/


/-- Given a connected free groupoid, its generating quiver is rooted-connected. -/
instance generators_connected (G) [Groupoid.{u, u} G] [IsConnected G] [IsFreeGroupoid G] (r : G) :
    RootedConnected (symgen r) :=
  ⟨fun b => path_nonempty_of_hom (CategoryTheory.nonempty_hom_of_preconnected_groupoid r b)⟩


/-- A vertex group in a free connected groupoid is free. With some work one could drop the
connectedness assumption, by looking at connected components. -/
instance endIsFreeOfConnectedFree
    {G : Type u} [Groupoid G] [IsConnected G] [IsFreeGroupoid G] (r : G) :
    IsFreeGroup.{u} (End r) :=
  SpanningTree.endIsFree <| geodesicSubtree (symgen r)


/-- The Nielsen-Schreier theorem: a subgroup of a free group is free. -/
instance subgroupIsFreeOfIsFree {G : Type u} [Group G] [IsFreeGroup G] (H : Subgroup G) :
    IsFreeGroup H :=
  IsFreeGroup.ofMulEquiv (endMulEquivSubgroup H)

