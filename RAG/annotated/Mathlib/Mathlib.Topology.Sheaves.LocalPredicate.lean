/-- Given a topological space `X : TopCat` and a type family `T : X → Type`,
a `P : PrelocalPredicate T` consists of:
* a family of predicates `P.pred`, one for each `U : Opens X`, of the form `(Π x : U, T x) → Prop`
* a proof that if `f : Π x : V, T x` satisfies the predicate on `V : Opens X`, then
  the restriction of `f` to any open subset `U` also satisfies the predicate.
-/
structure PrelocalPredicate where
  /-- The underlying predicate of a prelocal predicate -/
  pred : ∀ {U : Opens X}, (∀ x : U, T x) → Prop
  /-- The underlying predicate should be invariant under restriction -/
  res : ∀ {U V : Opens X} (i : U ⟶ V) (f : ∀ x : V, T x) (_ : pred f), pred fun x : U => f (i x)


/-- Continuity is a "prelocal" predicate on functions to a fixed topological space `T`.
-/
@[simps!]
def continuousPrelocal (T : TopCat.{v}) : PrelocalPredicate fun _ : X => T where
  pred {_} f := Continuous f
  res {_ _} i _ h := Continuous.comp h (Opens.isOpenEmbedding_of_le i.le).continuous


/-- Satisfying the inhabited linter. -/
instance inhabitedPrelocalPredicate (T : TopCat.{v}) :
    Inhabited (PrelocalPredicate fun _ : X => T) :=
  ⟨continuousPrelocal X T⟩


/-- Given a topological space `X : TopCat` and a type family `T : X → Type`,
a `P : LocalPredicate T` consists of:
* a family of predicates `P.pred`, one for each `U : Opens X`, of the form `(Π x : U, T x) → Prop`
* a proof that if `f : Π x : V, T x` satisfies the predicate on `V : Opens X`, then
  the restriction of `f` to any open subset `U` also satisfies the predicate, and
* a proof that given some `f : Π x : U, T x`,
  if for every `x : U` we can find an open set `x ∈ V ≤ U`
  so that the restriction of `f` to `V` satisfies the predicate,
  then `f` itself satisfies the predicate.
-/
structure LocalPredicate extends PrelocalPredicate T where
  /-- A local predicate must be local --- provided that it is locally satisfied, it is also globally
    satisfied -/
  locality :
    ∀ {U : Opens X} (f : ∀ x : U, T x)
      (_ : ∀ x : U, ∃ (V : Opens X) (_ : x.1 ∈ V) (i : V ⟶ U),
        pred fun x : V => f (i x : U)), pred f


/-- Continuity is a "local" predicate on functions to a fixed topological space `T`.
-/
def continuousLocal (T : TopCat.{v}) : LocalPredicate fun _ : X => T :=
  { continuousPrelocal X T with
    locality := fun {U} f w => by
      /-
        X : TopCat
        T✝ : ↑X → Type v
        T : TopCat
        U : TopologicalSpace.Opens ↑X
        f : (Subtype fun x => Membership.mem U x) → ↑T
        w : ∀ (x : Subtype fun x => Membership.mem U x), Exists fun V => Exists fun x  …
        ⊢ __src✝.pred f
      -/
      apply continuous_iff_continuousAt.2
      /-
        X : TopCat
        T✝ : ↑X → Type v
        T : TopCat
        U : TopologicalSpace.Opens ↑X
        f : (Subtype fun x => Membership.mem U x) → ↑T
        w : ∀ (x : Subtype fun x => Membership.mem U x), Exists fun V => Exists fun x  …
        ⊢ ∀ (x : Subtype fun x => Membership.mem U x), ContinuousAt f x
      -/
      intro x
      /-
        X : TopCat
        T✝ : ↑X → Type v
        T : TopCat
        U : TopologicalSpace.Opens ↑X
        f : (Subtype fun x => Membership.mem U x) → ↑T
        w : ∀ (x : Subtype fun x => Membership.mem U x), Exists fun V => Exists fun x  …
        x : Subtype fun x => Membership.mem U x
        ⊢ ContinuousAt f x
      -/
      specialize w x
      /-
        X : TopCat
        T✝ : ↑X → Type v
        T : TopCat
        U : TopologicalSpace.Opens ↑X
        f : (Subtype fun x => Membership.mem U x) → ↑T
        x : Subtype fun x => Membership.mem U x
        w : Exists fun V => Exists fun x => Exists fun i => __src✝.pred fun x => f ((f …
        ⊢ ContinuousAt f x
      -/
      rcases w with ⟨V, m, i, w⟩
      /-
        case intro.intro.intro
        X : TopCat
        T✝ : ↑X → Type v
        T : TopCat
        U : TopologicalSpace.Opens ↑X
        f : (Subtype fun x => Membership.mem U x) → ↑T
        x : Subtype fun x => Membership.mem U x
        V : TopologicalSpace.Opens ↑X
        m : Membership.mem V ↑x
        i : Quiver.Hom V U
        w : __src✝.pred fun x => f ((fun x => ⟨↑x, ⋯⟩) x)
        ⊢ ContinuousAt f x
      -/
      dsimp at w
      /-
        case intro.intro.intro
        X : TopCat
        T✝ : ↑X → Type v
        T : TopCat
        U : TopologicalSpace.Opens ↑X
        f : (Subtype fun x => Membership.mem U x) → ↑T
        x : Subtype fun x => Membership.mem U x
        V : TopologicalSpace.Opens ↑X
        m : Membership.mem V ↑x
        i : Quiver.Hom V U
        w : Continuous fun x => f ⟨↑x, ⋯⟩
        ⊢ ContinuousAt f x
      -/
      rw [continuous_iff_continuousAt] at w
      /-
        case intro.intro.intro
        X : TopCat
        T✝ : ↑X → Type v
        T : TopCat
        U : TopologicalSpace.Opens ↑X
        f : (Subtype fun x => Membership.mem U x) → ↑T
        x : Subtype fun x => Membership.mem U x
        V : TopologicalSpace.Opens ↑X
        m : Membership.mem V ↑x
        i : Quiver.Hom V U
        w : ∀ (x : Subtype fun x => Membership.mem V x), ContinuousAt (fun x => f ⟨↑x, …
        ⊢ ContinuousAt f x
      -/
      specialize w ⟨x, m⟩
      /-
        case intro.intro.intro
        X : TopCat
        T✝ : ↑X → Type v
        T : TopCat
        U : TopologicalSpace.Opens ↑X
        f : (Subtype fun x => Membership.mem U x) → ↑T
        x : Subtype fun x => Membership.mem U x
        V : TopologicalSpace.Opens ↑X
        m : Membership.mem V ↑x
        i : Quiver.Hom V U
        w : ContinuousAt (fun x => f ⟨↑x, ⋯⟩) ⟨↑x, m⟩
        ⊢ ContinuousAt f x
      -/
      simpa using (Opens.isOpenEmbedding_of_le i.le).continuousAt_iff.1 w }
      /-
        🎉 no goals
      -/


/-- Satisfying the inhabited linter. -/
instance inhabitedLocalPredicate (T : TopCat.{v}) : Inhabited (LocalPredicate fun _ : X => T) :=
  ⟨continuousLocal X T⟩


/-- Given a `P : PrelocalPredicate`, we can always construct a `LocalPredicate`
by asking that the condition from `P` holds locally near every point.
-/
def PrelocalPredicate.sheafify {T : X → Type v} (P : PrelocalPredicate T) : LocalPredicate T where
  pred {U} f := ∀ x : U, ∃ (V : Opens X) (_ : x.1 ∈ V) (i : V ⟶ U), P.pred fun x : V => f (i x : U)
  res {V U} i f w x := by
    /-
      X : TopCat
      T✝ T : ↑X → Type v
      P : TopCat.PrelocalPredicate T
      V U : TopologicalSpace.Opens ↑X
      i : Quiver.Hom V U
      f : (x : Subtype fun x => Membership.mem U x) → T ↑x
      w : (fun {U} f => ∀ (x : Subtype fun x => Membership.mem U x), Exists fun V => …
      x : Subtype fun x => Membership.mem V x
      ⊢ Exists fun V_1 => Exists fun x => Exists fun i_1 => P.pred fun x => (fun x = …
    -/
    specialize w (i x)
    /-
      X : TopCat
      T✝ T : ↑X → Type v
      P : TopCat.PrelocalPredicate T
      V U : TopologicalSpace.Opens ↑X
      i : Quiver.Hom V U
      f : (x : Subtype fun x => Membership.mem U x) → T ↑x
      x : Subtype fun x => Membership.mem V x
      w : Exists fun V_1 => Exists fun x => Exists fun i => P.pred fun x => f ((fun  …
      ⊢ Exists fun V_1 => Exists fun x => Exists fun i_1 => P.pred fun x => (fun x = …
    -/
    rcases w with ⟨V', m', i', p⟩
    /-
      case intro.intro.intro
      X : TopCat
      T✝ T : ↑X → Type v
      P : TopCat.PrelocalPredicate T
      V U : TopologicalSpace.Opens ↑X
      i : Quiver.Hom V U
      f : (x : Subtype fun x => Membership.mem U x) → T ↑x
      x : Subtype fun x => Membership.mem V x
      V' : TopologicalSpace.Opens ↑X
      m' : Membership.mem V' ↑((fun x => ⟨↑x, ⋯⟩) x)
      i' : Quiver.Hom V' U
      p : P.pred fun x => f ((fun x => ⟨↑x, ⋯⟩) x)
      ⊢ Exists fun V_1 => Exists fun x => Exists fun i_1 => P.pred fun x => (fun x = …
    -/
    refine ⟨V ⊓ V', ⟨x.2, m'⟩, Opens.infLELeft _ _, ?_⟩
    /-
      case intro.intro.intro
      X : TopCat
      T✝ T : ↑X → Type v
      P : TopCat.PrelocalPredicate T
      V U : TopologicalSpace.Opens ↑X
      i : Quiver.Hom V U
      f : (x : Subtype fun x => Membership.mem U x) → T ↑x
      x : Subtype fun x => Membership.mem V x
      V' : TopologicalSpace.Opens ↑X
      m' : Membership.mem V' ↑((fun x => ⟨↑x, ⋯⟩) x)
      i' : Quiver.Hom V' U
      p : P.pred fun x => f ((fun x => ⟨↑x, ⋯⟩) x)
      ⊢ P.pred fun x => (fun x => f ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) x)
    -/
    convert P.res (Opens.infLERight V V') _ p
    /-
      🎉 no goals
    -/
  locality {U} f w x := by
    /-
      X : TopCat
      T✝ T : ↑X → Type v
      P : TopCat.PrelocalPredicate T
      U : TopologicalSpace.Opens ↑X
      f : (x : Subtype fun x => Membership.mem U x) → T ↑x
      w : ∀ (x : Subtype fun x => Membership.mem U x), Exists fun V => Exists fun x  …
      x : Subtype fun x => Membership.mem U x
      ⊢ Exists fun V => Exists fun x => Exists fun i => P.pred fun x => f ((fun x => …
    -/
    specialize w x
    /-
      X : TopCat
      T✝ T : ↑X → Type v
      P : TopCat.PrelocalPredicate T
      U : TopologicalSpace.Opens ↑X
      f : (x : Subtype fun x => Membership.mem U x) → T ↑x
      x : Subtype fun x => Membership.mem U x
      w : Exists fun V => Exists fun x => Exists fun i => { pred := fun {U} f => ∀ ( …
      ⊢ Exists fun V => Exists fun x => Exists fun i => P.pred fun x => f ((fun x => …
    -/
    rcases w with ⟨V, m, i, p⟩
    /-
      case intro.intro.intro
      X : TopCat
      T✝ T : ↑X → Type v
      P : TopCat.PrelocalPredicate T
      U : TopologicalSpace.Opens ↑X
      f : (x : Subtype fun x => Membership.mem U x) → T ↑x
      x : Subtype fun x => Membership.mem U x
      V : TopologicalSpace.Opens ↑X
      m : Membership.mem V ↑x
      i : Quiver.Hom V U
      p : { pred := fun {U} f => ∀ (x : Subtype fun x => Membership.mem U x), Exists …
      ⊢ Exists fun V => Exists fun x => Exists fun i => P.pred fun x => f ((fun x => …
    -/
    specialize p ⟨x.1, m⟩
    /-
      case intro.intro.intro
      X : TopCat
      T✝ T : ↑X → Type v
      P : TopCat.PrelocalPredicate T
      U : TopologicalSpace.Opens ↑X
      f : (x : Subtype fun x => Membership.mem U x) → T ↑x
      x : Subtype fun x => Membership.mem U x
      V : TopologicalSpace.Opens ↑X
      m : Membership.mem V ↑x
      i : Quiver.Hom V U
      p : Exists fun V_1 => Exists fun x => Exists fun i_1 => P.pred fun x => (fun x …
      ⊢ Exists fun V => Exists fun x => Exists fun i => P.pred fun x => f ((fun x => …
    -/
    rcases p with ⟨V', m', i', p'⟩
    /-
      case intro.intro.intro.intro.intro.intro
      X : TopCat
      T✝ T : ↑X → Type v
      P : TopCat.PrelocalPredicate T
      U : TopologicalSpace.Opens ↑X
      f : (x : Subtype fun x => Membership.mem U x) → T ↑x
      x : Subtype fun x => Membership.mem U x
      V : TopologicalSpace.Opens ↑X
      m : Membership.mem V ↑x
      i : Quiver.Hom V U
      V' : TopologicalSpace.Opens ↑X
      m' : Membership.mem V' ↑⟨↑x, m⟩
      i' : Quiver.Hom V' V
      p' : P.pred fun x => (fun x => f ((fun x => ⟨↑x, ⋯⟩) x)) ((fun x => ⟨↑x, ⋯⟩) x)
      ⊢ Exists fun V => Exists fun x => Exists fun i => P.pred fun x => f ((fun x => …
    -/
    exact ⟨V', m', i' ≫ i, p'⟩
    /-
      🎉 no goals
    -/


theorem PrelocalPredicate.sheafifyOf {T : X → Type v} {P : PrelocalPredicate T} {U : Opens X}
    {f : ∀ x : U, T x} (h : P.pred f) : P.sheafify.pred f := fun x =>
                   /-
                     X : TopCat
                     T : ↑X → Type v
                     P : TopCat.PrelocalPredicate T
                     U : TopologicalSpace.Opens ↑X
                     f : (x : Subtype fun x => Membership.mem U x) → T ↑x
                     h : P.pred f
                     x : Subtype fun x => Membership.mem U x
                     ⊢ P.pred fun x => f ((fun x => ⟨↑x, ⋯⟩) x)
                   -/
  ⟨U, x.2, 𝟙 _, by convert h⟩
                   /-
                     🎉 no goals
                   -/


/-- The subpresheaf of dependent functions on `X` satisfying the "pre-local" predicate `P`.
-/
@[simps!]
def subpresheafToTypes (P : PrelocalPredicate T) : Presheaf (Type v) X where
  obj U := { f : ∀ x : U.unop , T x // P.pred f }
  map {_ _} i f := ⟨fun x => f.1 (i.unop x), P.res i.unop f.1 f.2⟩


/-- The natural transformation including the subpresheaf of functions satisfying a local predicate
into the presheaf of all functions.
-/
def subtype : subpresheafToTypes P ⟶ presheafToTypes X T where app _ f := f.1


/-- The functions satisfying a local predicate satisfy the sheaf condition.
-/
theorem isSheaf (P : LocalPredicate T) : (subpresheafToTypes P.toPrelocalPredicate).IsSheaf :=
  Presheaf.isSheaf_of_isSheafUniqueGluing_types.{v} _ fun ι U sf sf_comp => by
    -- We show the sheaf condition in terms of unique gluing.
    -- First we obtain a family of sections for the underlying sheaf of functions,
    -- by forgetting that the predicate holds
    /-
      X : TopCat
      T : ↑X → Type v
      P : TopCat.LocalPredicate T
      ι : Type v
      U : ι → TopologicalSpace.Opens ↑X
      sf : (i : ι) → (CategoryTheory.forget (Type v)).obj ((TopCat.subpresheafToType …
      sf_comp : (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsCompatible U sf
      ⊢ ExistsUnique fun s => (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsGl …
    -/
    let sf' : ∀ i : ι, (presheafToTypes X T).obj (op (U i)) := fun i => (sf i).val
    -- Since our original family is compatible, this one is as well
    have sf'_comp : (presheafToTypes X T).IsCompatible U sf' := fun i j =>
      congr_arg Subtype.val (sf_comp i j)
    -- So, we can obtain a unique gluing
    /-
      X : TopCat
      T : ↑X → Type v
      P : TopCat.LocalPredicate T
      ι : Type v
      U : ι → TopologicalSpace.Opens ↑X
      sf : (i : ι) → (CategoryTheory.forget (Type v)).obj ((TopCat.subpresheafToType …
      sf_comp : (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsCompatible U sf
      sf' : (i : ι) → (X.presheafToTypes T).obj { unop := U i } := fun i => ↑(sf i)
      sf'_comp : (X.presheafToTypes T).IsCompatible U sf'
      ⊢ ExistsUnique fun s => (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsGl …
    -/
    obtain ⟨gl, gl_spec, gl_uniq⟩ := (sheafToTypes X T).existsUnique_gluing U sf' sf'_comp
    /-
      case intro.intro
      X : TopCat
      T : ↑X → Type v
      P : TopCat.LocalPredicate T
      ι : Type v
      U : ι → TopologicalSpace.Opens ↑X
      sf : (i : ι) → (CategoryTheory.forget (Type v)).obj ((TopCat.subpresheafToType …
      sf_comp : (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsCompatible U sf
      sf' : (i : ι) → (X.presheafToTypes T).obj { unop := U i } := fun i => ↑(sf i)
      sf'_comp : (X.presheafToTypes T).IsCompatible U sf'
      gl : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val.obj { unop : …
      gl_spec : TopCat.Presheaf.IsGluing (X.sheafToTypes T).val U sf' gl
      gl_uniq : ∀ (y : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val. …
      ⊢ ExistsUnique fun s => (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsGl …
    -/
    refine ⟨⟨gl, ?_⟩, ?_, ?_⟩
    · -- Our first goal is to show that this chosen gluing satisfies the
      -- predicate. Of course, we use locality of the predicate.
      /-
        case intro.intro.refine_1
        X : TopCat
        T : ↑X → Type v
        P : TopCat.LocalPredicate T
        ι : Type v
        U : ι → TopologicalSpace.Opens ↑X
        sf : (i : ι) → (CategoryTheory.forget (Type v)).obj ((TopCat.subpresheafToType …
        sf_comp : (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsCompatible U sf
        sf' : (i : ι) → (X.presheafToTypes T).obj { unop := U i } := fun i => ↑(sf i)
        sf'_comp : (X.presheafToTypes T).IsCompatible U sf'
        gl : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val.obj { unop : …
        gl_spec : TopCat.Presheaf.IsGluing (X.sheafToTypes T).val U sf' gl
        gl_uniq : ∀ (y : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val. …
        ⊢ P.pred gl
      -/
      apply P.locality
      /-
        case intro.intro.refine_1.x
        X : TopCat
        T : ↑X → Type v
        P : TopCat.LocalPredicate T
        ι : Type v
        U : ι → TopologicalSpace.Opens ↑X
        sf : (i : ι) → (CategoryTheory.forget (Type v)).obj ((TopCat.subpresheafToType …
        sf_comp : (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsCompatible U sf
        sf' : (i : ι) → (X.presheafToTypes T).obj { unop := U i } := fun i => ↑(sf i)
        sf'_comp : (X.presheafToTypes T).IsCompatible U sf'
        gl : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val.obj { unop : …
        gl_spec : TopCat.Presheaf.IsGluing (X.sheafToTypes T).val U sf' gl
        gl_uniq : ∀ (y : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val. …
        ⊢ ∀ (x : Subtype fun x => Membership.mem (Opposite.unop { unop := iSup U }) x) …
      -/
      rintro ⟨x, mem⟩
      -- Once we're at a particular point `x`, we can select some open set `x ∈ U i`.
      /-
        case intro.intro.refine_1.x.mk
        X : TopCat
        T : ↑X → Type v
        P : TopCat.LocalPredicate T
        ι : Type v
        U : ι → TopologicalSpace.Opens ↑X
        sf : (i : ι) → (CategoryTheory.forget (Type v)).obj ((TopCat.subpresheafToType …
        sf_comp : (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsCompatible U sf
        sf' : (i : ι) → (X.presheafToTypes T).obj { unop := U i } := fun i => ↑(sf i)
        sf'_comp : (X.presheafToTypes T).IsCompatible U sf'
        gl : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val.obj { unop : …
        gl_spec : TopCat.Presheaf.IsGluing (X.sheafToTypes T).val U sf' gl
        gl_uniq : ∀ (y : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val. …
        x : ↑X
        mem : Membership.mem (Opposite.unop { unop := iSup U }) x
        ⊢ Exists fun V => Exists fun x => Exists fun i => P.pred fun x => gl ((fun x = …
      -/
      choose i hi using Opens.mem_iSup.mp mem
      -- We claim that the predicate holds in `U i`
      /-
        case intro.intro.refine_1.x.mk
        X : TopCat
        T : ↑X → Type v
        P : TopCat.LocalPredicate T
        ι : Type v
        U : ι → TopologicalSpace.Opens ↑X
        sf : (i : ι) → (CategoryTheory.forget (Type v)).obj ((TopCat.subpresheafToType …
        sf_comp : (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsCompatible U sf
        sf' : (i : ι) → (X.presheafToTypes T).obj { unop := U i } := fun i => ↑(sf i)
        sf'_comp : (X.presheafToTypes T).IsCompatible U sf'
        gl : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val.obj { unop : …
        gl_spec : TopCat.Presheaf.IsGluing (X.sheafToTypes T).val U sf' gl
        gl_uniq : ∀ (y : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val. …
        x : ↑X
        mem : Membership.mem (Opposite.unop { unop := iSup U }) x
        i : ι
        hi : Membership.mem (U i) x
        ⊢ Exists fun V => Exists fun x => Exists fun i => P.pred fun x => gl ((fun x = …
      -/
      use U i, hi, Opens.leSupr U i
      -- This follows, since our original family `sf` satisfies the predicate
      /-
        case h
        X : TopCat
        T : ↑X → Type v
        P : TopCat.LocalPredicate T
        ι : Type v
        U : ι → TopologicalSpace.Opens ↑X
        sf : (i : ι) → (CategoryTheory.forget (Type v)).obj ((TopCat.subpresheafToType …
        sf_comp : (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsCompatible U sf
        sf' : (i : ι) → (X.presheafToTypes T).obj { unop := U i } := fun i => ↑(sf i)
        sf'_comp : (X.presheafToTypes T).IsCompatible U sf'
        gl : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val.obj { unop : …
        gl_spec : TopCat.Presheaf.IsGluing (X.sheafToTypes T).val U sf' gl
        gl_uniq : ∀ (y : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val. …
        x : ↑X
        mem : Membership.mem (Opposite.unop { unop := iSup U }) x
        i : ι
        hi : Membership.mem (U i) x
        ⊢ P.pred fun x => gl ((fun x => ⟨↑x, ⋯⟩) x)
      -/
      convert (sf i).property using 1
      /-
        case h.e'_5
        X : TopCat
        T : ↑X → Type v
        P : TopCat.LocalPredicate T
        ι : Type v
        U : ι → TopologicalSpace.Opens ↑X
        sf : (i : ι) → (CategoryTheory.forget (Type v)).obj ((TopCat.subpresheafToType …
        sf_comp : (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsCompatible U sf
        sf' : (i : ι) → (X.presheafToTypes T).obj { unop := U i } := fun i => ↑(sf i)
        sf'_comp : (X.presheafToTypes T).IsCompatible U sf'
        gl : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val.obj { unop : …
        gl_spec : TopCat.Presheaf.IsGluing (X.sheafToTypes T).val U sf' gl
        gl_uniq : ∀ (y : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val. …
        x : ↑X
        mem : Membership.mem (Opposite.unop { unop := iSup U }) x
        i : ι
        hi : Membership.mem (U i) x
        ⊢ Eq (fun x => gl ((fun x => ⟨↑x, ⋯⟩) x)) ↑(sf i)
      -/
      exact gl_spec i
      /-
        🎉 no goals
      -/

    -- It remains to show that the chosen lift is really a gluing for the subsheaf and
    -- that it is unique. Both of which follow immediately from the corresponding facts
    -- in the sheaf of functions without the local predicate.
      /-
        case intro.intro.refine_2
        X : TopCat
        T : ↑X → Type v
        P : TopCat.LocalPredicate T
        ι : Type v
        U : ι → TopologicalSpace.Opens ↑X
        sf : (i : ι) → (CategoryTheory.forget (Type v)).obj ((TopCat.subpresheafToType …
        sf_comp : (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsCompatible U sf
        sf' : (i : ι) → (X.presheafToTypes T).obj { unop := U i } := fun i => ↑(sf i)
        sf'_comp : (X.presheafToTypes T).IsCompatible U sf'
        gl : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val.obj { unop : …
        gl_spec : TopCat.Presheaf.IsGluing (X.sheafToTypes T).val U sf' gl
        gl_uniq : ∀ (y : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val. …
        ⊢ (fun s => (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsGluing U sf s) …
      -/
    · exact fun i => Subtype.ext (gl_spec i)
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.refine_3
        X : TopCat
        T : ↑X → Type v
        P : TopCat.LocalPredicate T
        ι : Type v
        U : ι → TopologicalSpace.Opens ↑X
        sf : (i : ι) → (CategoryTheory.forget (Type v)).obj ((TopCat.subpresheafToType …
        sf_comp : (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsCompatible U sf
        sf' : (i : ι) → (X.presheafToTypes T).obj { unop := U i } := fun i => ↑(sf i)
        sf'_comp : (X.presheafToTypes T).IsCompatible U sf'
        gl : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val.obj { unop : …
        gl_spec : TopCat.Presheaf.IsGluing (X.sheafToTypes T).val U sf' gl
        gl_uniq : ∀ (y : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val. …
        ⊢ ∀ (y : (CategoryTheory.forget (Type v)).obj ((TopCat.subpresheafToTypes P.to …
      -/
    · intro gl' hgl'
      /-
        case intro.intro.refine_3
        X : TopCat
        T : ↑X → Type v
        P : TopCat.LocalPredicate T
        ι : Type v
        U : ι → TopologicalSpace.Opens ↑X
        sf : (i : ι) → (CategoryTheory.forget (Type v)).obj ((TopCat.subpresheafToType …
        sf_comp : (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsCompatible U sf
        sf' : (i : ι) → (X.presheafToTypes T).obj { unop := U i } := fun i => ↑(sf i)
        sf'_comp : (X.presheafToTypes T).IsCompatible U sf'
        gl : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val.obj { unop : …
        gl_spec : TopCat.Presheaf.IsGluing (X.sheafToTypes T).val U sf' gl
        gl_uniq : ∀ (y : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val. …
        gl' : (CategoryTheory.forget (Type v)).obj ((TopCat.subpresheafToTypes P.toPre …
        hgl' : (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsGluing U sf gl'
        ⊢ Eq gl' ⟨gl, ⋯⟩
      -/
      refine Subtype.ext ?_
      /-
        case intro.intro.refine_3
        X : TopCat
        T : ↑X → Type v
        P : TopCat.LocalPredicate T
        ι : Type v
        U : ι → TopologicalSpace.Opens ↑X
        sf : (i : ι) → (CategoryTheory.forget (Type v)).obj ((TopCat.subpresheafToType …
        sf_comp : (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsCompatible U sf
        sf' : (i : ι) → (X.presheafToTypes T).obj { unop := U i } := fun i => ↑(sf i)
        sf'_comp : (X.presheafToTypes T).IsCompatible U sf'
        gl : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val.obj { unop : …
        gl_spec : TopCat.Presheaf.IsGluing (X.sheafToTypes T).val U sf' gl
        gl_uniq : ∀ (y : (CategoryTheory.forget (Type v)).obj ((X.sheafToTypes T).val. …
        gl' : (CategoryTheory.forget (Type v)).obj ((TopCat.subpresheafToTypes P.toPre …
        hgl' : (TopCat.subpresheafToTypes P.toPrelocalPredicate).IsGluing U sf gl'
        ⊢ Eq ↑gl' ↑⟨gl, ⋯⟩
      -/
      exact gl_uniq gl'.1 fun i => congr_arg Subtype.val (hgl' i)
      /-
        🎉 no goals
      -/


/-- The subsheaf of the sheaf of all dependently typed functions satisfying the local predicate `P`.
-/
@[simps]
def subsheafToTypes (P : LocalPredicate T) : Sheaf (Type v) X :=
  ⟨subpresheafToTypes P.toPrelocalPredicate, subpresheafToTypes.isSheaf P⟩


/-- There is a canonical map from the stalk to the original fiber, given by evaluating sections.
-/
def stalkToFiber (P : LocalPredicate T) (x : X) : (subsheafToTypes P).presheaf.stalk x ⟶ T x := by
  refine
    colimit.desc _
      { pt := T x
        ι :=
          { app := fun U f => ?_
            naturality := ?_ } }
    /-
      case refine_1
      X : TopCat
      T : ↑X → Type v
      P : TopCat.LocalPredicate T
      x : ↑X
      U : Opposite (TopologicalSpace.OpenNhds x)
      f : (((CategoryTheory.whiskeringLeft (Opposite (TopologicalSpace.OpenNhds x))  …
      ⊢ ((CategoryTheory.Functor.const (Opposite (TopologicalSpace.OpenNhds x))).obj …
    -/
  · exact f.1 ⟨x, (unop U).2⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : TopCat
      T : ↑X → Type v
      P : TopCat.LocalPredicate T
      x : ↑X
      ⊢ ∀ ⦃X_1 Y : Opposite (TopologicalSpace.OpenNhds x)⦄ (f : Quiver.Hom X_1 Y), E …
    -/
  · aesop
    /-
      🎉 no goals
    -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): removed `simp` attribute,
-- due to left hand side is not in simple normal form.

theorem stalkToFiber_germ (P : LocalPredicate T) (U : Opens X) (x : X) (hx : x ∈ U) (f) :
    stalkToFiber P x ((subsheafToTypes P).presheaf.germ U x hx f) = f.1 ⟨x, hx⟩ := by
  /-
    X : TopCat
    T : ↑X → Type v
    P : TopCat.LocalPredicate T
    U : TopologicalSpace.Opens ↑X
    x : ↑X
    hx : Membership.mem U x
    f : (TopCat.subsheafToTypes P).presheaf.obj { unop := U }
    ⊢ Eq (TopCat.stalkToFiber P x ((TopCat.subsheafToTypes P).presheaf.germ U x hx …
  -/
  simp [Presheaf.germ, stalkToFiber]
  /-
    🎉 no goals
  -/


/-- The `stalkToFiber` map is surjective at `x` if
every point in the fiber `T x` has an allowed section passing through it.
-/
theorem stalkToFiber_surjective (P : LocalPredicate T) (x : X)
    (w : ∀ t : T x, ∃ (U : OpenNhds x) (f : ∀ y : U.1, T y) (_ : P.pred f), f ⟨x, U.2⟩ = t) :
    Function.Surjective (stalkToFiber P x) := fun t => by
  /-
    X : TopCat
    T : ↑X → Type v
    P : TopCat.LocalPredicate T
    x : ↑X
    w : ∀ (t : T x), Exists fun U => Exists fun f => Exists fun x_1 => Eq (f ⟨x, ⋯ …
    t : T x
    ⊢ Exists fun a => Eq (TopCat.stalkToFiber P x a) t
  -/
  rcases w t with ⟨U, f, h, rfl⟩
  /-
    case intro.intro.intro
    X : TopCat
    T : ↑X → Type v
    P : TopCat.LocalPredicate T
    x : ↑X
    w : ∀ (t : T x), Exists fun U => Exists fun f => Exists fun x_1 => Eq (f ⟨x, ⋯ …
    U : TopologicalSpace.OpenNhds x
    f : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → T ↑y
    h : P.pred f
    ⊢ Exists fun a => Eq (TopCat.stalkToFiber P x a) (f ⟨x, ⋯⟩)
  -/
  fconstructor
    /-
      case intro.intro.intro.w
      X : TopCat
      T : ↑X → Type v
      P : TopCat.LocalPredicate T
      x : ↑X
      w : ∀ (t : T x), Exists fun U => Exists fun f => Exists fun x_1 => Eq (f ⟨x, ⋯ …
      U : TopologicalSpace.OpenNhds x
      f : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → T ↑y
      h : P.pred f
      ⊢ (TopCat.subsheafToTypes P).presheaf.stalk x
    -/
  · exact (subsheafToTypes P).presheaf.germ _ x U.2 ⟨f, h⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.h
      X : TopCat
      T : ↑X → Type v
      P : TopCat.LocalPredicate T
      x : ↑X
      w : ∀ (t : T x), Exists fun U => Exists fun f => Exists fun x_1 => Eq (f ⟨x, ⋯ …
      U : TopologicalSpace.OpenNhds x
      f : (y : Subtype fun x_1 => Membership.mem U.obj x_1) → T ↑y
      h : P.pred f
      ⊢ Eq (TopCat.stalkToFiber P x ((TopCat.subsheafToTypes P).presheaf.germ U.obj  …
    -/
  · exact stalkToFiber_germ P U.1 x U.2 ⟨f, h⟩
    /-
      🎉 no goals
    -/


/-- The `stalkToFiber` map is injective at `x` if any two allowed sections which agree at `x`
agree on some neighborhood of `x`.
-/
theorem stalkToFiber_injective (P : LocalPredicate T) (x : X)
    (w :
      ∀ (U V : OpenNhds x) (fU : ∀ y : U.1, T y) (_ : P.pred fU) (fV : ∀ y : V.1, T y)
        (_ : P.pred fV) (_ : fU ⟨x, U.2⟩ = fV ⟨x, V.2⟩),
        ∃ (W : OpenNhds x) (iU : W ⟶ U) (iV : W ⟶ V), ∀ w : W.1,
          fU (iU w : U.1) = fV (iV w : V.1)) :
    Function.Injective (stalkToFiber P x) := fun tU tV h => by
  -- We promise to provide all the ingredients of the proof later:
  let Q :
    ∃ (W : (OpenNhds x)ᵒᵖ) (s : ∀ w : (unop W).1, T w) (hW : P.pred s),
      tU = (subsheafToTypes P).presheaf.germ _ x (unop W).2 ⟨s, hW⟩ ∧
        tV = (subsheafToTypes P).presheaf.germ _ x (unop W).2 ⟨s, hW⟩ :=
    ?_
    /-
      case refine_2
      X : TopCat
      T : ↑X → Type v
      P : TopCat.LocalPredicate T
      x : ↑X
      w : ∀ (U V : TopologicalSpace.OpenNhds x) (fU : (y : Subtype fun x_1 => Member …
      tU tV : (TopCat.subsheafToTypes P).presheaf.stalk x
      h : Eq (TopCat.stalkToFiber P x tU) (TopCat.stalkToFiber P x tV)
      Q : Exists fun W => Exists fun s => Exists fun hW => And (Eq tU ((TopCat.subsh …
      ⊢ Eq tU tV
    -/
  · choose W s hW e using Q
    /-
      case refine_2
      X : TopCat
      T : ↑X → Type v
      P : TopCat.LocalPredicate T
      x : ↑X
      w : ∀ (U V : TopologicalSpace.OpenNhds x) (fU : (y : Subtype fun x_1 => Member …
      tU tV : (TopCat.subsheafToTypes P).presheaf.stalk x
      h : Eq (TopCat.stalkToFiber P x tU) (TopCat.stalkToFiber P x tV)
      W : Opposite (TopologicalSpace.OpenNhds x)
      s : (w : Subtype fun x_1 => Membership.mem (Opposite.unop W).obj x_1) → T ↑w
      hW : P.pred s
      e : And (Eq tU ((TopCat.subsheafToTypes P).presheaf.germ (Opposite.unop W).obj …
      ⊢ Eq tU tV
    -/
    exact e.1.trans e.2.symm
    /-
      🎉 no goals
    -/
  -- Then use induction to pick particular representatives of `tU tV : stalk x`
  /-
    case refine_1
    X : TopCat
    T : ↑X → Type v
    P : TopCat.LocalPredicate T
    x : ↑X
    w : ∀ (U V : TopologicalSpace.OpenNhds x) (fU : (y : Subtype fun x_1 => Member …
    tU tV : (TopCat.subsheafToTypes P).presheaf.stalk x
    h : Eq (TopCat.stalkToFiber P x tU) (TopCat.stalkToFiber P x tV)
    ⊢ Exists fun W => Exists fun s => Exists fun hW => And (Eq tU ((TopCat.subshea …
  -/
  obtain ⟨U, ⟨fU, hU⟩, rfl⟩ := jointly_surjective'.{v, v} tU
  /-
    case refine_1.intro.intro.mk
    X : TopCat
    T : ↑X → Type v
    P : TopCat.LocalPredicate T
    x : ↑X
    w : ∀ (U V : TopologicalSpace.OpenNhds x) (fU : (y : Subtype fun x_1 => Member …
    tV : (TopCat.subsheafToTypes P).presheaf.stalk x
    U : Opposite (TopologicalSpace.OpenNhds x)
    fU : (x_1 : Subtype fun x_1 => Membership.mem (Opposite.unop ((TopologicalSpac …
    hU : P.pred fU
    h : Eq (TopCat.stalkToFiber P x (CategoryTheory.Limits.colimit.ι (((CategoryTh …
    ⊢ Exists fun W => Exists fun s => Exists fun hW => And (Eq (CategoryTheory.Lim …
  -/
  obtain ⟨V, ⟨fV, hV⟩, rfl⟩ := jointly_surjective'.{v, v} tV
  -- Decompose everything into its constituent parts:
  /-
    case refine_1.intro.intro.mk.intro.intro.mk
    X : TopCat
    T : ↑X → Type v
    P : TopCat.LocalPredicate T
    x : ↑X
    w : ∀ (U V : TopologicalSpace.OpenNhds x) (fU : (y : Subtype fun x_1 => Member …
    U : Opposite (TopologicalSpace.OpenNhds x)
    fU : (x_1 : Subtype fun x_1 => Membership.mem (Opposite.unop ((TopologicalSpac …
    hU : P.pred fU
    V : Opposite (TopologicalSpace.OpenNhds x)
    fV : (x_1 : Subtype fun x_1 => Membership.mem (Opposite.unop ((TopologicalSpac …
    hV : P.pred fV
    h : Eq (TopCat.stalkToFiber P x (CategoryTheory.Limits.colimit.ι (((CategoryTh …
    ⊢ Exists fun W => Exists fun s => Exists fun hW => And (Eq (CategoryTheory.Lim …
  -/
  dsimp
  /-
    case refine_1.intro.intro.mk.intro.intro.mk
    X : TopCat
    T : ↑X → Type v
    P : TopCat.LocalPredicate T
    x : ↑X
    w : ∀ (U V : TopologicalSpace.OpenNhds x) (fU : (y : Subtype fun x_1 => Member …
    U : Opposite (TopologicalSpace.OpenNhds x)
    fU : (x_1 : Subtype fun x_1 => Membership.mem (Opposite.unop ((TopologicalSpac …
    hU : P.pred fU
    V : Opposite (TopologicalSpace.OpenNhds x)
    fV : (x_1 : Subtype fun x_1 => Membership.mem (Opposite.unop ((TopologicalSpac …
    hV : P.pred fV
    h : Eq (TopCat.stalkToFiber P x (CategoryTheory.Limits.colimit.ι (((CategoryTh …
    ⊢ Exists fun W => Exists fun s => Exists fun hW => And (Eq (CategoryTheory.Lim …
  -/
  simp only [stalkToFiber, Types.Colimit.ι_desc_apply'] at h
  /-
    case refine_1.intro.intro.mk.intro.intro.mk
    X : TopCat
    T : ↑X → Type v
    P : TopCat.LocalPredicate T
    x : ↑X
    w : ∀ (U V : TopologicalSpace.OpenNhds x) (fU : (y : Subtype fun x_1 => Member …
    U : Opposite (TopologicalSpace.OpenNhds x)
    fU : (x_1 : Subtype fun x_1 => Membership.mem (Opposite.unop ((TopologicalSpac …
    hU : P.pred fU
    V : Opposite (TopologicalSpace.OpenNhds x)
    fV : (x_1 : Subtype fun x_1 => Membership.mem (Opposite.unop ((TopologicalSpac …
    hV : P.pred fV
    h : Eq (fU ⟨x, ⋯⟩) (fV ⟨x, ⋯⟩)
    ⊢ Exists fun W => Exists fun s => Exists fun hW => And (Eq (CategoryTheory.Lim …
  -/
  specialize w (unop U) (unop V) fU hU fV hV h
  /-
    case refine_1.intro.intro.mk.intro.intro.mk
    X : TopCat
    T : ↑X → Type v
    P : TopCat.LocalPredicate T
    x : ↑X
    U : Opposite (TopologicalSpace.OpenNhds x)
    fU : (x_1 : Subtype fun x_1 => Membership.mem (Opposite.unop ((TopologicalSpac …
    hU : P.pred fU
    V : Opposite (TopologicalSpace.OpenNhds x)
    fV : (x_1 : Subtype fun x_1 => Membership.mem (Opposite.unop ((TopologicalSpac …
    hV : P.pred fV
    h : Eq (fU ⟨x, ⋯⟩) (fV ⟨x, ⋯⟩)
    w : Exists fun W => Exists fun iU => Exists fun iV => ∀ (w : Subtype fun x_1 = …
    ⊢ Exists fun W => Exists fun s => Exists fun hW => And (Eq (CategoryTheory.Lim …
  -/
  rcases w with ⟨W, iU, iV, w⟩
  -- and put it back together again in the correct order.
  /-
    case refine_1.intro.intro.mk.intro.intro.mk.intro.intro.intro
    X : TopCat
    T : ↑X → Type v
    P : TopCat.LocalPredicate T
    x : ↑X
    U : Opposite (TopologicalSpace.OpenNhds x)
    fU : (x_1 : Subtype fun x_1 => Membership.mem (Opposite.unop ((TopologicalSpac …
    hU : P.pred fU
    V : Opposite (TopologicalSpace.OpenNhds x)
    fV : (x_1 : Subtype fun x_1 => Membership.mem (Opposite.unop ((TopologicalSpac …
    hV : P.pred fV
    h : Eq (fU ⟨x, ⋯⟩) (fV ⟨x, ⋯⟩)
    W : TopologicalSpace.OpenNhds x
    iU : Quiver.Hom W (Opposite.unop U)
    iV : Quiver.Hom W (Opposite.unop V)
    w : ∀ (w : Subtype fun x_1 => Membership.mem W.obj x_1), Eq (fU ((fun x_1 => ⟨ …
    ⊢ Exists fun W => Exists fun s => Exists fun hW => And (Eq (CategoryTheory.Lim …
  -/
  refine ⟨op W, fun w => fU (iU w : (unop U).1), P.res ?_ _ hU, ?_⟩
    /-
      case refine_1.intro.intro.mk.intro.intro.mk.intro.intro.intro.refine_1
      X : TopCat
      T : ↑X → Type v
      P : TopCat.LocalPredicate T
      x : ↑X
      U : Opposite (TopologicalSpace.OpenNhds x)
      fU : (x_1 : Subtype fun x_1 => Membership.mem (Opposite.unop ((TopologicalSpac …
      hU : P.pred fU
      V : Opposite (TopologicalSpace.OpenNhds x)
      fV : (x_1 : Subtype fun x_1 => Membership.mem (Opposite.unop ((TopologicalSpac …
      hV : P.pred fV
      h : Eq (fU ⟨x, ⋯⟩) (fV ⟨x, ⋯⟩)
      W : TopologicalSpace.OpenNhds x
      iU : Quiver.Hom W (Opposite.unop U)
      iV : Quiver.Hom W (Opposite.unop V)
      w : ∀ (w : Subtype fun x_1 => Membership.mem W.obj x_1), Eq (fU ((fun x_1 => ⟨ …
      ⊢ Quiver.Hom (Opposite.unop { unop := W }).obj (Opposite.unop U).obj
    -/
  · rcases W with ⟨W, m⟩
    /-
      case refine_1.intro.intro.mk.intro.intro.mk.intro.intro.intro.refine_1.mk
      X : TopCat
      T : ↑X → Type v
      P : TopCat.LocalPredicate T
      x : ↑X
      U : Opposite (TopologicalSpace.OpenNhds x)
      fU : (x_1 : Subtype fun x_1 => Membership.mem (Opposite.unop ((TopologicalSpac …
      hU : P.pred fU
      V : Opposite (TopologicalSpace.OpenNhds x)
      fV : (x_1 : Subtype fun x_1 => Membership.mem (Opposite.unop ((TopologicalSpac …
      hV : P.pred fV
      h : Eq (fU ⟨x, ⋯⟩) (fV ⟨x, ⋯⟩)
      W : TopologicalSpace.Opens ↑X
      m : Membership.mem W x
      iU : Quiver.Hom { obj := W, property := m } (Opposite.unop U)
      iV : Quiver.Hom { obj := W, property := m } (Opposite.unop V)
      w : ∀ (w : Subtype fun x_1 => Membership.mem { obj := W, property := m }.obj x …
      ⊢ Quiver.Hom (Opposite.unop { unop := { obj := W, property := m } }).obj (Oppo …
    -/
    exact iU
    /-
      🎉 no goals
    -/
    /-
      case refine_1.intro.intro.mk.intro.intro.mk.intro.intro.intro.refine_2
      X : TopCat
      T : ↑X → Type v
      P : TopCat.LocalPredicate T
      x : ↑X
      U : Opposite (TopologicalSpace.OpenNhds x)
      fU : (x_1 : Subtype fun x_1 => Membership.mem (Opposite.unop ((TopologicalSpac …
      hU : P.pred fU
      V : Opposite (TopologicalSpace.OpenNhds x)
      fV : (x_1 : Subtype fun x_1 => Membership.mem (Opposite.unop ((TopologicalSpac …
      hV : P.pred fV
      h : Eq (fU ⟨x, ⋯⟩) (fV ⟨x, ⋯⟩)
      W : TopologicalSpace.OpenNhds x
      iU : Quiver.Hom W (Opposite.unop U)
      iV : Quiver.Hom W (Opposite.unop V)
      w : ∀ (w : Subtype fun x_1 => Membership.mem W.obj x_1), Eq (fU ((fun x_1 => ⟨ …
      ⊢ And (Eq (CategoryTheory.Limits.colimit.ι ((TopologicalSpace.OpenNhds.inclusi …
    -/
  · exact ⟨colimit_sound iU.op (Subtype.eq rfl), colimit_sound iV.op (Subtype.eq (funext w).symm)⟩
    /-
      🎉 no goals
    -/


/-- Some repackaging:
the presheaf of functions satisfying `continuousPrelocal` is just the same thing as
the presheaf of continuous functions.
-/
def subpresheafContinuousPrelocalIsoPresheafToTop (T : TopCat.{v}) :
    subpresheafToTypes (continuousPrelocal X T) ≅ presheafToTop X T :=
  /-
    X : TopCat
    T✝ : ↑X → Type v
    T : TopCat
    ⊢ ∀ {X_1 Y : Opposite (TopologicalSpace.Opens ↑X)} (f : Quiver.Hom X_1 Y), Eq  …
  -/
                /-
                  X✝ : TopCat
                  T✝ : ↑X✝ → Type v
                  T : TopCat
                  X : Opposite (TopologicalSpace.Opens ↑X✝)
                  ⊢ Quiver.Hom ((TopCat.subpresheafToTypes (X✝.continuousPrelocal T)).obj X) ((X …
                -/
  NatIso.ofComponents fun X =>
                               /-
                                 🎉 no goals
                               -/
                /-
                  X✝ : TopCat
                  T✝ : ↑X✝ → Type v
                  T : TopCat
                  X : Opposite (TopologicalSpace.Opens ↑X✝)
                  ⊢ Quiver.Hom ((X✝.presheafToTop T).obj X) ((TopCat.subpresheafToTypes (X✝.cont …
                -/
  /-
    🎉 no goals
  -/
                               /-
                                 🎉 no goals
                               -/
    { hom := by rintro ⟨f, c⟩; exact ⟨f, c⟩
      inv := by rintro ⟨f, c⟩; exact ⟨f, c⟩ }


/-- The sheaf of continuous functions on `X` with values in a space `T`.
-/
def sheafToTop (T : TopCat.{v}) : Sheaf (Type v) X :=
  ⟨presheafToTop X T,
    Presheaf.isSheaf_of_iso (subpresheafContinuousPrelocalIsoPresheafToTop T)
      (subpresheafToTypes.isSheaf (continuousLocal X T))⟩


