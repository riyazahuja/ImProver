/--
The prelocal predicate on functions into the stalks, asserting that the function is equal to a germ.
-/
def isGerm : PrelocalPredicate fun x => F.stalk x where
  pred {U} f := ∃ g : F.obj (op U), ∀ x : U, f x = F.germ U x.1 x.2 g
  res := fun i _ ⟨g, p⟩ => ⟨F.map i.op g, fun x ↦ (p (i x)).trans (F.germ_res_apply i x x.2 g).symm⟩


/-- The local predicate on functions into the stalks,
asserting that the function is locally equal to a germ.
-/
def isLocallyGerm : LocalPredicate fun x => F.stalk x :=
  (isGerm F).sheafify


/-- The sheafification of a `Type` valued presheaf, defined as the functions into the stalks which
are locally equal to germs.
-/
def sheafify : Sheaf (Type v) X :=
  subsheafToTypes (Sheafify.isLocallyGerm F)


/-- The morphism from a presheaf to its sheafification,
sending each section to its germs.
(This forms the unit of the adjunction.)
-/
def toSheafify : F ⟶ F.sheafify.1 where
  app U f := ⟨fun x => F.germ _ x x.2 f, PrelocalPredicate.sheafifyOf ⟨f, fun x => rfl⟩⟩
  naturality U U' f := by
    /-
      X : TopCat
      F : TopCat.Presheaf (Type v) X
      U U' : Opposite (TopologicalSpace.Opens ↑X)
      f : Quiver.Hom U U'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun U f => ⟨fun x => F.ge …
    -/
    ext x
    /-
      case h
      X : TopCat
      F : TopCat.Presheaf (Type v) X
      U U' : Opposite (TopologicalSpace.Opens ↑X)
      f : Quiver.Hom U U'
      x : F.obj U
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun U f => ⟨fun x => F.ge …
    -/
    apply Subtype.ext -- Porting note: Added `apply`
    /-
      case h.a
      X : TopCat
      F : TopCat.Presheaf (Type v) X
      U U' : Opposite (TopologicalSpace.Opens ↑X)
      f : Quiver.Hom U U'
      x : F.obj U
      ⊢ Eq ↑(CategoryTheory.CategoryStruct.comp (F.map f) ((fun U f => ⟨fun x => F.g …
    -/
    ext ⟨u, m⟩
    /-
      case h.a.h.mk
      X : TopCat
      F : TopCat.Presheaf (Type v) X
      U U' : Opposite (TopologicalSpace.Opens ↑X)
      f : Quiver.Hom U U'
      x : F.obj U
      u : ↑X
      m : Membership.mem (Opposite.unop U') u
      ⊢ Eq (↑(CategoryTheory.CategoryStruct.comp (F.map f) ((fun U f => ⟨fun x => F. …
    -/
    exact germ_res_apply F f.unop u m x
    /-
      🎉 no goals
    -/


/-- The natural morphism from the stalk of the sheafification to the original stalk.
In `sheafifyStalkIso` we show this is an isomorphism.
-/
def stalkToFiber (x : X) : F.sheafify.presheaf.stalk x ⟶ F.stalk x :=
  TopCat.stalkToFiber (Sheafify.isLocallyGerm F) x


theorem stalkToFiber_surjective (x : X) : Function.Surjective (F.stalkToFiber x) := by
  /-
    X : TopCat
    F : TopCat.Presheaf (Type v) X
    x : ↑X
    ⊢ Function.Surjective (F.stalkToFiber x)
  -/
  apply TopCat.stalkToFiber_surjective
  /-
    case w
    X : TopCat
    F : TopCat.Presheaf (Type v) X
    x : ↑X
    ⊢ ∀ (t : F.stalk x), Exists fun U => Exists fun f => Exists fun x_1 => Eq (f ⟨ …
  -/
  intro t
  /-
    case w
    X : TopCat
    F : TopCat.Presheaf (Type v) X
    x : ↑X
    t : F.stalk x
    ⊢ Exists fun U => Exists fun f => Exists fun x_1 => Eq (f ⟨x, ⋯⟩) t
  -/
  obtain ⟨U, m, s, rfl⟩ := F.germ_exist _ t
  /-
    case w.intro.intro.intro
    X : TopCat
    F : TopCat.Presheaf (Type v) X
    x : ↑X
    U : TopologicalSpace.Opens ↑X
    m : Membership.mem U x
    s : (CategoryTheory.forget (Type v)).obj (F.obj { unop := U })
    ⊢ Exists fun U_1 => Exists fun f => Exists fun x_1 => Eq (f ⟨x, ⋯⟩) ((F.germ U …
  -/
  use ⟨U, m⟩
  /-
    case h
    X : TopCat
    F : TopCat.Presheaf (Type v) X
    x : ↑X
    U : TopologicalSpace.Opens ↑X
    m : Membership.mem U x
    s : (CategoryTheory.forget (Type v)).obj (F.obj { unop := U })
    ⊢ Exists fun f => Exists fun x_1 => Eq (f ⟨x, ⋯⟩) ((F.germ U x m) s)
  -/
  fconstructor
    /-
      case h.w
      X : TopCat
      F : TopCat.Presheaf (Type v) X
      x : ↑X
      U : TopologicalSpace.Opens ↑X
      m : Membership.mem U x
      s : (CategoryTheory.forget (Type v)).obj (F.obj { unop := U })
      ⊢ (y : Subtype fun x_1 => Membership.mem { obj := U, property := m }.obj x_1)  …
    -/
  · exact fun y => F.germ _ _ y.2 s
    /-
      🎉 no goals
    -/
    /-
      case h.h
      X : TopCat
      F : TopCat.Presheaf (Type v) X
      x : ↑X
      U : TopologicalSpace.Opens ↑X
      m : Membership.mem U x
      s : (CategoryTheory.forget (Type v)).obj (F.obj { unop := U })
      ⊢ Exists fun x_1 => Eq ((fun y => F.germ { obj := U, property := m }.obj ↑y ⋯  …
    -/
  · exact ⟨PrelocalPredicate.sheafifyOf ⟨s, fun _ => rfl⟩, rfl⟩
    /-
      🎉 no goals
    -/


theorem stalkToFiber_injective (x : X) : Function.Injective (F.stalkToFiber x) := by
  /-
    X : TopCat
    F : TopCat.Presheaf (Type v) X
    x : ↑X
    ⊢ Function.Injective (F.stalkToFiber x)
  -/
  apply TopCat.stalkToFiber_injective
  /-
    case w
    X : TopCat
    F : TopCat.Presheaf (Type v) X
    x : ↑X
    ⊢ ∀ (U V : TopologicalSpace.OpenNhds x) (fU : (y : Subtype fun x_1 => Membersh …
  -/
  intro U V fU hU fV hV e
  /-
    case w
    X : TopCat
    F : TopCat.Presheaf (Type v) X
    x : ↑X
    U V : TopologicalSpace.OpenNhds x
    fU : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → F.stalk ↑y
    hU : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fU
    fV : (y : Subtype fun x_1 => Membership.mem V.obj x_1) → F.stalk ↑y
    hV : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fV
    e : Eq (fU ⟨x, ⋯⟩) (fV ⟨x, ⋯⟩)
    ⊢ Exists fun W => Exists fun iU => Exists fun iV => ∀ (w : Subtype fun x_1 =>  …
  -/
  rcases hU ⟨x, U.2⟩ with ⟨U', mU, iU, gU, wU⟩
  /-
    case w.intro.intro.intro.intro
    X : TopCat
    F : TopCat.Presheaf (Type v) X
    x : ↑X
    U V : TopologicalSpace.OpenNhds x
    fU : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → F.stalk ↑y
    hU : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fU
    fV : (y : Subtype fun x_1 => Membership.mem V.obj x_1) → F.stalk ↑y
    hV : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fV
    e : Eq (fU ⟨x, ⋯⟩) (fV ⟨x, ⋯⟩)
    U' : TopologicalSpace.Opens ↑X
    mU : Membership.mem U' ↑⟨x, ⋯⟩
    iU : Quiver.Hom U' U.obj
    gU : F.obj { unop := U' }
    wU : ∀ (x_1 : Subtype fun x => Membership.mem U' x), Eq ((fun x_2 => fU ((fun  …
    ⊢ Exists fun W => Exists fun iU => Exists fun iV => ∀ (w : Subtype fun x_1 =>  …
  -/
  rcases hV ⟨x, V.2⟩ with ⟨V', mV, iV, gV, wV⟩
  /-
    case w.intro.intro.intro.intro.intro.intro.intro.intro
    X : TopCat
    F : TopCat.Presheaf (Type v) X
    x : ↑X
    U V : TopologicalSpace.OpenNhds x
    fU : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → F.stalk ↑y
    hU : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fU
    fV : (y : Subtype fun x_1 => Membership.mem V.obj x_1) → F.stalk ↑y
    hV : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fV
    e : Eq (fU ⟨x, ⋯⟩) (fV ⟨x, ⋯⟩)
    U' : TopologicalSpace.Opens ↑X
    mU : Membership.mem U' ↑⟨x, ⋯⟩
    iU : Quiver.Hom U' U.obj
    gU : F.obj { unop := U' }
    wU : ∀ (x_1 : Subtype fun x => Membership.mem U' x), Eq ((fun x_2 => fU ((fun  …
    V' : TopologicalSpace.Opens ↑X
    mV : Membership.mem V' ↑⟨x, ⋯⟩
    iV : Quiver.Hom V' V.obj
    gV : F.obj { unop := V' }
    wV : ∀ (x_1 : Subtype fun x => Membership.mem V' x), Eq ((fun x_2 => fV ((fun  …
    ⊢ Exists fun W => Exists fun iU => Exists fun iV => ∀ (w : Subtype fun x_1 =>  …
  -/
  have wUx := wU ⟨x, mU⟩
  /-
    case w.intro.intro.intro.intro.intro.intro.intro.intro
    X : TopCat
    F : TopCat.Presheaf (Type v) X
    x : ↑X
    U V : TopologicalSpace.OpenNhds x
    fU : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → F.stalk ↑y
    hU : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fU
    fV : (y : Subtype fun x_1 => Membership.mem V.obj x_1) → F.stalk ↑y
    hV : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fV
    e : Eq (fU ⟨x, ⋯⟩) (fV ⟨x, ⋯⟩)
    U' : TopologicalSpace.Opens ↑X
    mU : Membership.mem U' ↑⟨x, ⋯⟩
    iU : Quiver.Hom U' U.obj
    gU : F.obj { unop := U' }
    wU : ∀ (x_1 : Subtype fun x => Membership.mem U' x), Eq ((fun x_2 => fU ((fun  …
    V' : TopologicalSpace.Opens ↑X
    mV : Membership.mem V' ↑⟨x, ⋯⟩
    iV : Quiver.Hom V' V.obj
    gV : F.obj { unop := V' }
    wV : ∀ (x_1 : Subtype fun x => Membership.mem V' x), Eq ((fun x_2 => fV ((fun  …
    wUx : Eq ((fun x_1 => fU ((fun x_2 => ⟨↑x_2, ⋯⟩) x_1)) ⟨x, mU⟩) (F.germ U' ↑⟨x …
    ⊢ Exists fun W => Exists fun iU => Exists fun iV => ∀ (w : Subtype fun x_1 =>  …
  -/
  dsimp at wUx; rw [wUx] at e; clear wUx
  /-
    case w.intro.intro.intro.intro.intro.intro.intro.intro
    X : TopCat
    F : TopCat.Presheaf (Type v) X
    x : ↑X
    U V : TopologicalSpace.OpenNhds x
    fU : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → F.stalk ↑y
    hU : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fU
    fV : (y : Subtype fun x_1 => Membership.mem V.obj x_1) → F.stalk ↑y
    hV : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fV
    U' : TopologicalSpace.Opens ↑X
    mU : Membership.mem U' ↑⟨x, ⋯⟩
    iU : Quiver.Hom U' U.obj
    gU : F.obj { unop := U' }
    e : Eq (F.germ U' x ⋯ gU) (fV ⟨x, ⋯⟩)
    wU : ∀ (x_1 : Subtype fun x => Membership.mem U' x), Eq ((fun x_2 => fU ((fun  …
    V' : TopologicalSpace.Opens ↑X
    mV : Membership.mem V' ↑⟨x, ⋯⟩
    iV : Quiver.Hom V' V.obj
    gV : F.obj { unop := V' }
    wV : ∀ (x_1 : Subtype fun x => Membership.mem V' x), Eq ((fun x_2 => fV ((fun  …
    ⊢ Exists fun W => Exists fun iU => Exists fun iV => ∀ (w : Subtype fun x_1 =>  …
  -/
  have wVx := wV ⟨x, mV⟩
  /-
    case w.intro.intro.intro.intro.intro.intro.intro.intro
    X : TopCat
    F : TopCat.Presheaf (Type v) X
    x : ↑X
    U V : TopologicalSpace.OpenNhds x
    fU : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → F.stalk ↑y
    hU : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fU
    fV : (y : Subtype fun x_1 => Membership.mem V.obj x_1) → F.stalk ↑y
    hV : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fV
    U' : TopologicalSpace.Opens ↑X
    mU : Membership.mem U' ↑⟨x, ⋯⟩
    iU : Quiver.Hom U' U.obj
    gU : F.obj { unop := U' }
    e : Eq (F.germ U' x ⋯ gU) (fV ⟨x, ⋯⟩)
    wU : ∀ (x_1 : Subtype fun x => Membership.mem U' x), Eq ((fun x_2 => fU ((fun  …
    V' : TopologicalSpace.Opens ↑X
    mV : Membership.mem V' ↑⟨x, ⋯⟩
    iV : Quiver.Hom V' V.obj
    gV : F.obj { unop := V' }
    wV : ∀ (x_1 : Subtype fun x => Membership.mem V' x), Eq ((fun x_2 => fV ((fun  …
    wVx : Eq ((fun x_1 => fV ((fun x_2 => ⟨↑x_2, ⋯⟩) x_1)) ⟨x, mV⟩) (F.germ V' ↑⟨x …
    ⊢ Exists fun W => Exists fun iU => Exists fun iV => ∀ (w : Subtype fun x_1 =>  …
  -/
  dsimp at wVx; rw [wVx] at e; clear wVx
  /-
    case w.intro.intro.intro.intro.intro.intro.intro.intro
    X : TopCat
    F : TopCat.Presheaf (Type v) X
    x : ↑X
    U V : TopologicalSpace.OpenNhds x
    fU : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → F.stalk ↑y
    hU : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fU
    fV : (y : Subtype fun x_1 => Membership.mem V.obj x_1) → F.stalk ↑y
    hV : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fV
    U' : TopologicalSpace.Opens ↑X
    mU : Membership.mem U' ↑⟨x, ⋯⟩
    iU : Quiver.Hom U' U.obj
    gU : F.obj { unop := U' }
    wU : ∀ (x_1 : Subtype fun x => Membership.mem U' x), Eq ((fun x_2 => fU ((fun  …
    V' : TopologicalSpace.Opens ↑X
    mV : Membership.mem V' ↑⟨x, ⋯⟩
    iV : Quiver.Hom V' V.obj
    gV : F.obj { unop := V' }
    e : Eq (F.germ U' x ⋯ gU) (F.germ V' x ⋯ gV)
    wV : ∀ (x_1 : Subtype fun x => Membership.mem V' x), Eq ((fun x_2 => fV ((fun  …
    ⊢ Exists fun W => Exists fun iU => Exists fun iV => ∀ (w : Subtype fun x_1 =>  …
  -/
  rcases F.germ_eq x mU mV gU gV e with ⟨W, mW, iU', iV', (e' : F.map iU'.op gU = F.map iV'.op gV)⟩
  /-
    case w.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    X : TopCat
    F : TopCat.Presheaf (Type v) X
    x : ↑X
    U V : TopologicalSpace.OpenNhds x
    fU : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → F.stalk ↑y
    hU : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fU
    fV : (y : Subtype fun x_1 => Membership.mem V.obj x_1) → F.stalk ↑y
    hV : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fV
    U' : TopologicalSpace.Opens ↑X
    mU : Membership.mem U' ↑⟨x, ⋯⟩
    iU : Quiver.Hom U' U.obj
    gU : F.obj { unop := U' }
    wU : ∀ (x_1 : Subtype fun x => Membership.mem U' x), Eq ((fun x_2 => fU ((fun  …
    V' : TopologicalSpace.Opens ↑X
    mV : Membership.mem V' ↑⟨x, ⋯⟩
    iV : Quiver.Hom V' V.obj
    gV : F.obj { unop := V' }
    e : Eq (F.germ U' x ⋯ gU) (F.germ V' x ⋯ gV)
    wV : ∀ (x_1 : Subtype fun x => Membership.mem V' x), Eq ((fun x_2 => fV ((fun  …
    W : TopologicalSpace.Opens ↑X
    mW : Membership.mem W x
    iU' : Quiver.Hom W U'
    iV' : Quiver.Hom W V'
    e' : Eq (F.map iU'.op gU) (F.map iV'.op gV)
    ⊢ Exists fun W => Exists fun iU => Exists fun iV => ∀ (w : Subtype fun x_1 =>  …
  -/
  use ⟨W ⊓ (U' ⊓ V'), ⟨mW, mU, mV⟩⟩
  /-
    case h
    X : TopCat
    F : TopCat.Presheaf (Type v) X
    x : ↑X
    U V : TopologicalSpace.OpenNhds x
    fU : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → F.stalk ↑y
    hU : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fU
    fV : (y : Subtype fun x_1 => Membership.mem V.obj x_1) → F.stalk ↑y
    hV : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fV
    U' : TopologicalSpace.Opens ↑X
    mU : Membership.mem U' ↑⟨x, ⋯⟩
    iU : Quiver.Hom U' U.obj
    gU : F.obj { unop := U' }
    wU : ∀ (x_1 : Subtype fun x => Membership.mem U' x), Eq ((fun x_2 => fU ((fun  …
    V' : TopologicalSpace.Opens ↑X
    mV : Membership.mem V' ↑⟨x, ⋯⟩
    iV : Quiver.Hom V' V.obj
    gV : F.obj { unop := V' }
    e : Eq (F.germ U' x ⋯ gU) (F.germ V' x ⋯ gV)
    wV : ∀ (x_1 : Subtype fun x => Membership.mem V' x), Eq ((fun x_2 => fV ((fun  …
    W : TopologicalSpace.Opens ↑X
    mW : Membership.mem W x
    iU' : Quiver.Hom W U'
    iV' : Quiver.Hom W V'
    e' : Eq (F.map iU'.op gU) (F.map iV'.op gV)
    ⊢ Exists fun iU => Exists fun iV => ∀ (w : Subtype fun x_1 => Membership.mem { …
  -/
  refine ⟨?_, ?_, ?_⟩
    /-
      case h.refine_1
      X : TopCat
      F : TopCat.Presheaf (Type v) X
      x : ↑X
      U V : TopologicalSpace.OpenNhds x
      fU : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → F.stalk ↑y
      hU : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fU
      fV : (y : Subtype fun x_1 => Membership.mem V.obj x_1) → F.stalk ↑y
      hV : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fV
      U' : TopologicalSpace.Opens ↑X
      mU : Membership.mem U' ↑⟨x, ⋯⟩
      iU : Quiver.Hom U' U.obj
      gU : F.obj { unop := U' }
      wU : ∀ (x_1 : Subtype fun x => Membership.mem U' x), Eq ((fun x_2 => fU ((fun  …
      V' : TopologicalSpace.Opens ↑X
      mV : Membership.mem V' ↑⟨x, ⋯⟩
      iV : Quiver.Hom V' V.obj
      gV : F.obj { unop := V' }
      e : Eq (F.germ U' x ⋯ gU) (F.germ V' x ⋯ gV)
      wV : ∀ (x_1 : Subtype fun x => Membership.mem V' x), Eq ((fun x_2 => fV ((fun  …
      W : TopologicalSpace.Opens ↑X
      mW : Membership.mem W x
      iU' : Quiver.Hom W U'
      iV' : Quiver.Hom W V'
      e' : Eq (F.map iU'.op gU) (F.map iV'.op gV)
      ⊢ Quiver.Hom { obj := Min.min W (Min.min U' V'), property := ⋯ } U
    -/
  · change W ⊓ (U' ⊓ V') ⟶ U.obj
    /-
      case h.refine_1
      X : TopCat
      F : TopCat.Presheaf (Type v) X
      x : ↑X
      U V : TopologicalSpace.OpenNhds x
      fU : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → F.stalk ↑y
      hU : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fU
      fV : (y : Subtype fun x_1 => Membership.mem V.obj x_1) → F.stalk ↑y
      hV : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fV
      U' : TopologicalSpace.Opens ↑X
      mU : Membership.mem U' ↑⟨x, ⋯⟩
      iU : Quiver.Hom U' U.obj
      gU : F.obj { unop := U' }
      wU : ∀ (x_1 : Subtype fun x => Membership.mem U' x), Eq ((fun x_2 => fU ((fun  …
      V' : TopologicalSpace.Opens ↑X
      mV : Membership.mem V' ↑⟨x, ⋯⟩
      iV : Quiver.Hom V' V.obj
      gV : F.obj { unop := V' }
      e : Eq (F.germ U' x ⋯ gU) (F.germ V' x ⋯ gV)
      wV : ∀ (x_1 : Subtype fun x => Membership.mem V' x), Eq ((fun x_2 => fV ((fun  …
      W : TopologicalSpace.Opens ↑X
      mW : Membership.mem W x
      iU' : Quiver.Hom W U'
      iV' : Quiver.Hom W V'
      e' : Eq (F.map iU'.op gU) (F.map iV'.op gV)
      ⊢ Quiver.Hom (Min.min W (Min.min U' V')) U.obj
    -/
    exact Opens.infLERight _ _ ≫ Opens.infLELeft _ _ ≫ iU
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      X : TopCat
      F : TopCat.Presheaf (Type v) X
      x : ↑X
      U V : TopologicalSpace.OpenNhds x
      fU : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → F.stalk ↑y
      hU : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fU
      fV : (y : Subtype fun x_1 => Membership.mem V.obj x_1) → F.stalk ↑y
      hV : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fV
      U' : TopologicalSpace.Opens ↑X
      mU : Membership.mem U' ↑⟨x, ⋯⟩
      iU : Quiver.Hom U' U.obj
      gU : F.obj { unop := U' }
      wU : ∀ (x_1 : Subtype fun x => Membership.mem U' x), Eq ((fun x_2 => fU ((fun  …
      V' : TopologicalSpace.Opens ↑X
      mV : Membership.mem V' ↑⟨x, ⋯⟩
      iV : Quiver.Hom V' V.obj
      gV : F.obj { unop := V' }
      e : Eq (F.germ U' x ⋯ gU) (F.germ V' x ⋯ gV)
      wV : ∀ (x_1 : Subtype fun x => Membership.mem V' x), Eq ((fun x_2 => fV ((fun  …
      W : TopologicalSpace.Opens ↑X
      mW : Membership.mem W x
      iU' : Quiver.Hom W U'
      iV' : Quiver.Hom W V'
      e' : Eq (F.map iU'.op gU) (F.map iV'.op gV)
      ⊢ Quiver.Hom { obj := Min.min W (Min.min U' V'), property := ⋯ } V
    -/
  · change W ⊓ (U' ⊓ V') ⟶ V.obj
    /-
      case h.refine_2
      X : TopCat
      F : TopCat.Presheaf (Type v) X
      x : ↑X
      U V : TopologicalSpace.OpenNhds x
      fU : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → F.stalk ↑y
      hU : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fU
      fV : (y : Subtype fun x_1 => Membership.mem V.obj x_1) → F.stalk ↑y
      hV : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fV
      U' : TopologicalSpace.Opens ↑X
      mU : Membership.mem U' ↑⟨x, ⋯⟩
      iU : Quiver.Hom U' U.obj
      gU : F.obj { unop := U' }
      wU : ∀ (x_1 : Subtype fun x => Membership.mem U' x), Eq ((fun x_2 => fU ((fun  …
      V' : TopologicalSpace.Opens ↑X
      mV : Membership.mem V' ↑⟨x, ⋯⟩
      iV : Quiver.Hom V' V.obj
      gV : F.obj { unop := V' }
      e : Eq (F.germ U' x ⋯ gU) (F.germ V' x ⋯ gV)
      wV : ∀ (x_1 : Subtype fun x => Membership.mem V' x), Eq ((fun x_2 => fV ((fun  …
      W : TopologicalSpace.Opens ↑X
      mW : Membership.mem W x
      iU' : Quiver.Hom W U'
      iV' : Quiver.Hom W V'
      e' : Eq (F.map iU'.op gU) (F.map iV'.op gV)
      ⊢ Quiver.Hom (Min.min W (Min.min U' V')) V.obj
    -/
    exact Opens.infLERight _ _ ≫ Opens.infLERight _ _ ≫ iV
    /-
      🎉 no goals
    -/
    /-
      case h.refine_3
      X : TopCat
      F : TopCat.Presheaf (Type v) X
      x : ↑X
      U V : TopologicalSpace.OpenNhds x
      fU : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → F.stalk ↑y
      hU : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fU
      fV : (y : Subtype fun x_1 => Membership.mem V.obj x_1) → F.stalk ↑y
      hV : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fV
      U' : TopologicalSpace.Opens ↑X
      mU : Membership.mem U' ↑⟨x, ⋯⟩
      iU : Quiver.Hom U' U.obj
      gU : F.obj { unop := U' }
      wU : ∀ (x_1 : Subtype fun x => Membership.mem U' x), Eq ((fun x_2 => fU ((fun  …
      V' : TopologicalSpace.Opens ↑X
      mV : Membership.mem V' ↑⟨x, ⋯⟩
      iV : Quiver.Hom V' V.obj
      gV : F.obj { unop := V' }
      e : Eq (F.germ U' x ⋯ gU) (F.germ V' x ⋯ gV)
      wV : ∀ (x_1 : Subtype fun x => Membership.mem V' x), Eq ((fun x_2 => fV ((fun  …
      W : TopologicalSpace.Opens ↑X
      mW : Membership.mem W x
      iU' : Quiver.Hom W U'
      iV' : Quiver.Hom W V'
      e' : Eq (F.map iU'.op gU) (F.map iV'.op gV)
      ⊢ ∀ (w : Subtype fun x_1 => Membership.mem { obj := Min.min W (Min.min U' V'), …
    -/
  · intro w
    /-
      case h.refine_3
      X : TopCat
      F : TopCat.Presheaf (Type v) X
      x : ↑X
      U V : TopologicalSpace.OpenNhds x
      fU : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → F.stalk ↑y
      hU : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fU
      fV : (y : Subtype fun x_1 => Membership.mem V.obj x_1) → F.stalk ↑y
      hV : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fV
      U' : TopologicalSpace.Opens ↑X
      mU : Membership.mem U' ↑⟨x, ⋯⟩
      iU : Quiver.Hom U' U.obj
      gU : F.obj { unop := U' }
      wU : ∀ (x_1 : Subtype fun x => Membership.mem U' x), Eq ((fun x_2 => fU ((fun  …
      V' : TopologicalSpace.Opens ↑X
      mV : Membership.mem V' ↑⟨x, ⋯⟩
      iV : Quiver.Hom V' V.obj
      gV : F.obj { unop := V' }
      e : Eq (F.germ U' x ⋯ gU) (F.germ V' x ⋯ gV)
      wV : ∀ (x_1 : Subtype fun x => Membership.mem V' x), Eq ((fun x_2 => fV ((fun  …
      W : TopologicalSpace.Opens ↑X
      mW : Membership.mem W x
      iU' : Quiver.Hom W U'
      iV' : Quiver.Hom W V'
      e' : Eq (F.map iU'.op gU) (F.map iV'.op gV)
      w : Subtype fun x_1 => Membership.mem { obj := Min.min W (Min.min U' V'), prop …
      ⊢ Eq (fU ((fun x_1 => ⟨↑x_1, ⋯⟩) w)) (fV ((fun x_1 => ⟨↑x_1, ⋯⟩) w))
    -/
    specialize wU ⟨w.1, w.2.2.1⟩
    /-
      case h.refine_3
      X : TopCat
      F : TopCat.Presheaf (Type v) X
      x : ↑X
      U V : TopologicalSpace.OpenNhds x
      fU : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → F.stalk ↑y
      hU : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fU
      fV : (y : Subtype fun x_1 => Membership.mem V.obj x_1) → F.stalk ↑y
      hV : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fV
      U' : TopologicalSpace.Opens ↑X
      mU : Membership.mem U' ↑⟨x, ⋯⟩
      iU : Quiver.Hom U' U.obj
      gU : F.obj { unop := U' }
      V' : TopologicalSpace.Opens ↑X
      mV : Membership.mem V' ↑⟨x, ⋯⟩
      iV : Quiver.Hom V' V.obj
      gV : F.obj { unop := V' }
      e : Eq (F.germ U' x ⋯ gU) (F.germ V' x ⋯ gV)
      wV : ∀ (x_1 : Subtype fun x => Membership.mem V' x), Eq ((fun x_2 => fV ((fun  …
      W : TopologicalSpace.Opens ↑X
      mW : Membership.mem W x
      iU' : Quiver.Hom W U'
      iV' : Quiver.Hom W V'
      e' : Eq (F.map iU'.op gU) (F.map iV'.op gV)
      w : Subtype fun x_1 => Membership.mem { obj := Min.min W (Min.min U' V'), prop …
      wU : Eq ((fun x_1 => fU ((fun x_2 => ⟨↑x_2, ⋯⟩) x_1)) ⟨↑w, ⋯⟩) (F.germ U' ↑⟨↑w …
      ⊢ Eq (fU ((fun x_1 => ⟨↑x_1, ⋯⟩) w)) (fV ((fun x_1 => ⟨↑x_1, ⋯⟩) w))
    -/
    specialize wV ⟨w.1, w.2.2.2⟩
    /-
      case h.refine_3
      X : TopCat
      F : TopCat.Presheaf (Type v) X
      x : ↑X
      U V : TopologicalSpace.OpenNhds x
      fU : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → F.stalk ↑y
      hU : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fU
      fV : (y : Subtype fun x_1 => Membership.mem V.obj x_1) → F.stalk ↑y
      hV : (TopCat.Presheaf.Sheafify.isLocallyGerm F).pred fV
      U' : TopologicalSpace.Opens ↑X
      mU : Membership.mem U' ↑⟨x, ⋯⟩
      iU : Quiver.Hom U' U.obj
      gU : F.obj { unop := U' }
      V' : TopologicalSpace.Opens ↑X
      mV : Membership.mem V' ↑⟨x, ⋯⟩
      iV : Quiver.Hom V' V.obj
      gV : F.obj { unop := V' }
      e : Eq (F.germ U' x ⋯ gU) (F.germ V' x ⋯ gV)
      W : TopologicalSpace.Opens ↑X
      mW : Membership.mem W x
      iU' : Quiver.Hom W U'
      iV' : Quiver.Hom W V'
      e' : Eq (F.map iU'.op gU) (F.map iV'.op gV)
      w : Subtype fun x_1 => Membership.mem { obj := Min.min W (Min.min U' V'), prop …
      wU : Eq ((fun x_1 => fU ((fun x_2 => ⟨↑x_2, ⋯⟩) x_1)) ⟨↑w, ⋯⟩) (F.germ U' ↑⟨↑w …
      wV : Eq ((fun x_1 => fV ((fun x_2 => ⟨↑x_2, ⋯⟩) x_1)) ⟨↑w, ⋯⟩) (F.germ V' ↑⟨↑w …
      ⊢ Eq (fU ((fun x_1 => ⟨↑x_1, ⋯⟩) w)) (fV ((fun x_1 => ⟨↑x_1, ⋯⟩) w))
    -/
    dsimp at wU wV ⊢
    rw [wU, ← F.germ_res iU' w w.2.1, wV, ← F.germ_res iV' w w.2.1,
      CategoryTheory.types_comp_apply, CategoryTheory.types_comp_apply, e']


/-- The isomorphism between a stalk of the sheafification and the original stalk.
-/
def sheafifyStalkIso (x : X) : F.sheafify.presheaf.stalk x ≅ F.stalk x :=
  (Equiv.ofBijective _ ⟨stalkToFiber_injective _ _, stalkToFiber_surjective _ _⟩).toIso

-- PROJECT functoriality, and that sheafification is the left adjoint of the forgetful functor.

