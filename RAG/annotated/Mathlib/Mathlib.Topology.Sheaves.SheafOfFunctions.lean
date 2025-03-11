/-- We show that the presheaf of functions to a type `T`
(no continuity assumptions, just plain functions)
form a sheaf.

In fact, the proof is identical when we do this for dependent functions to a type family `T`,
so we do the more general case.
-/
theorem toTypes_isSheaf (T : X → Type u) : (presheafToTypes X T).IsSheaf :=
  isSheaf_of_isSheafUniqueGluing_types.{u} _ fun ι U sf hsf => by
  -- We use the sheaf condition in terms of unique gluing
  -- U is a family of open sets, indexed by `ι` and `sf` is a compatible family of sections.
  -- In the informal comments below, I'll just write `U` to represent the union.
    -- Our first goal is to define a function "lifted" to all of `U`.
    -- We do this one point at a time. Using the axiom of choice, we can pick for each
    -- `x : ↑(iSup U)` an index `i : ι` such that `x` lies in `U i`
    /-
      X : TopCat
      T : ↑X → Type u
      ι : Type u
      U : ι → TopologicalSpace.Opens ↑X
      sf : (i : ι) → (CategoryTheory.forget (Type u)).obj ((X.presheafToTypes T).obj …
      hsf : (X.presheafToTypes T).IsCompatible U sf
      ⊢ ExistsUnique fun s => (X.presheafToTypes T).IsGluing U sf s
    -/
    choose index index_spec using fun x : ↑(iSup U) => Opens.mem_iSup.mp x.2
    -- Using this data, we can glue our functions together to a single section
    /-
      X : TopCat
      T : ↑X → Type u
      ι : Type u
      U : ι → TopologicalSpace.Opens ↑X
      sf : (i : ι) → (CategoryTheory.forget (Type u)).obj ((X.presheafToTypes T).obj …
      hsf : (X.presheafToTypes T).IsCompatible U sf
      index : (Subtype fun x => Membership.mem (iSup U) x) → ι
      index_spec : ∀ (x : Subtype fun x => Membership.mem (iSup U) x), Membership.me …
      ⊢ ExistsUnique fun s => (X.presheafToTypes T).IsGluing U sf s
    -/
    let s : ∀ x : ↑(iSup U), T x := fun x => sf (index x) ⟨x.1, index_spec x⟩
    /-
      X : TopCat
      T : ↑X → Type u
      ι : Type u
      U : ι → TopologicalSpace.Opens ↑X
      sf : (i : ι) → (CategoryTheory.forget (Type u)).obj ((X.presheafToTypes T).obj …
      hsf : (X.presheafToTypes T).IsCompatible U sf
      index : (Subtype fun x => Membership.mem (iSup U) x) → ι
      index_spec : ∀ (x : Subtype fun x => Membership.mem (iSup U) x), Membership.me …
      s : (x : Subtype fun x => Membership.mem (iSup U) x) → T ↑x := fun x => sf (in …
      ⊢ ExistsUnique fun s => (X.presheafToTypes T).IsGluing U sf s
    -/
    refine ⟨s, ?_, ?_⟩
      /-
        case refine_1
        X : TopCat
        T : ↑X → Type u
        ι : Type u
        U : ι → TopologicalSpace.Opens ↑X
        sf : (i : ι) → (CategoryTheory.forget (Type u)).obj ((X.presheafToTypes T).obj …
        hsf : (X.presheafToTypes T).IsCompatible U sf
        index : (Subtype fun x => Membership.mem (iSup U) x) → ι
        index_spec : ∀ (x : Subtype fun x => Membership.mem (iSup U) x), Membership.me …
        s : (x : Subtype fun x => Membership.mem (iSup U) x) → T ↑x := fun x => sf (in …
        ⊢ (fun s => (X.presheafToTypes T).IsGluing U sf s) s
      -/
    · intro i
      /-
        case refine_1
        X : TopCat
        T : ↑X → Type u
        ι : Type u
        U : ι → TopologicalSpace.Opens ↑X
        sf : (i : ι) → (CategoryTheory.forget (Type u)).obj ((X.presheafToTypes T).obj …
        hsf : (X.presheafToTypes T).IsCompatible U sf
        index : (Subtype fun x => Membership.mem (iSup U) x) → ι
        index_spec : ∀ (x : Subtype fun x => Membership.mem (iSup U) x), Membership.me …
        s : (x : Subtype fun x => Membership.mem (iSup U) x) → T ↑x := fun x => sf (in …
        i : ι
        ⊢ Eq (((X.presheafToTypes T).map (TopologicalSpace.Opens.leSupr U i).op) s) (s …
      -/
      funext x
      -- Now we need to verify that this lifted function restricts correctly to each set `U i`.
      -- Of course, the difficulty is that at any given point `x ∈ U i`,
      -- we may have used the axiom of choice to pick a different `j` with `x ∈ U j`
      -- when defining the function.
      -- Thus we'll need to use the fact that the restrictions are compatible.
      /-
        case refine_1.h
        X : TopCat
        T : ↑X → Type u
        ι : Type u
        U : ι → TopologicalSpace.Opens ↑X
        sf : (i : ι) → (CategoryTheory.forget (Type u)).obj ((X.presheafToTypes T).obj …
        hsf : (X.presheafToTypes T).IsCompatible U sf
        index : (Subtype fun x => Membership.mem (iSup U) x) → ι
        index_spec : ∀ (x : Subtype fun x => Membership.mem (iSup U) x), Membership.me …
        s : (x : Subtype fun x => Membership.mem (iSup U) x) → T ↑x := fun x => sf (in …
        i : ι
        x : Subtype fun x => Membership.mem (Opposite.unop { unop := U i }) x
        ⊢ Eq (((X.presheafToTypes T).map (TopologicalSpace.Opens.leSupr U i).op) s x)  …
      -/
      exact congr_fun (hsf (index ⟨x, _⟩) i) ⟨x, ⟨index_spec ⟨x.1, _⟩, x.2⟩⟩
      /-
        🎉 no goals
      -/
    · -- Now we just need to check that the lift we picked was the only possible one.
      -- So we suppose we had some other gluing `t` of our sections
      /-
        case refine_2
        X : TopCat
        T : ↑X → Type u
        ι : Type u
        U : ι → TopologicalSpace.Opens ↑X
        sf : (i : ι) → (CategoryTheory.forget (Type u)).obj ((X.presheafToTypes T).obj …
        hsf : (X.presheafToTypes T).IsCompatible U sf
        index : (Subtype fun x => Membership.mem (iSup U) x) → ι
        index_spec : ∀ (x : Subtype fun x => Membership.mem (iSup U) x), Membership.me …
        s : (x : Subtype fun x => Membership.mem (iSup U) x) → T ↑x := fun x => sf (in …
        ⊢ ∀ (y : (CategoryTheory.forget (Type u)).obj ((X.presheafToTypes T).obj { uno …
      -/
      intro t ht
      -- and observe that we need to check that it agrees with our choice
      -- for each `x ∈ ↑(iSup U)`.
      /-
        case refine_2
        X : TopCat
        T : ↑X → Type u
        ι : Type u
        U : ι → TopologicalSpace.Opens ↑X
        sf : (i : ι) → (CategoryTheory.forget (Type u)).obj ((X.presheafToTypes T).obj …
        hsf : (X.presheafToTypes T).IsCompatible U sf
        index : (Subtype fun x => Membership.mem (iSup U) x) → ι
        index_spec : ∀ (x : Subtype fun x => Membership.mem (iSup U) x), Membership.me …
        s : (x : Subtype fun x => Membership.mem (iSup U) x) → T ↑x := fun x => sf (in …
        t : (CategoryTheory.forget (Type u)).obj ((X.presheafToTypes T).obj { unop :=  …
        ht : (X.presheafToTypes T).IsGluing U sf t
        ⊢ Eq t s
      -/
      funext x
      /-
        case refine_2.h
        X : TopCat
        T : ↑X → Type u
        ι : Type u
        U : ι → TopologicalSpace.Opens ↑X
        sf : (i : ι) → (CategoryTheory.forget (Type u)).obj ((X.presheafToTypes T).obj …
        hsf : (X.presheafToTypes T).IsCompatible U sf
        index : (Subtype fun x => Membership.mem (iSup U) x) → ι
        index_spec : ∀ (x : Subtype fun x => Membership.mem (iSup U) x), Membership.me …
        s : (x : Subtype fun x => Membership.mem (iSup U) x) → T ↑x := fun x => sf (in …
        t : (CategoryTheory.forget (Type u)).obj ((X.presheafToTypes T).obj { unop :=  …
        ht : (X.presheafToTypes T).IsGluing U sf t
        x : Subtype fun x => Membership.mem (Opposite.unop { unop := iSup U }) x
        ⊢ Eq (t x) (s x)
      -/
      exact congr_fun (ht (index x)) ⟨x.1, index_spec x⟩
      /-
        🎉 no goals
      -/

-- We verify that the non-dependent version is an immediate consequence:

/-- The presheaf of not-necessarily-continuous functions to
a target type `T` satisfies the sheaf condition.
-/
theorem toType_isSheaf (T : Type u) : (presheafToType X T).IsSheaf :=
  toTypes_isSheaf X fun _ => T


/-- The sheaf of not-necessarily-continuous functions on `X` with values in type family
`T : X → Type u`.
-/
def sheafToTypes (T : X → Type u) : Sheaf (Type u) X :=
  ⟨presheafToTypes X T, Presheaf.toTypes_isSheaf _ _⟩


/-- The sheaf of not-necessarily-continuous functions on `X` with values in a type `T`.
-/
def sheafToType (T : Type u) : Sheaf (Type u) X :=
  ⟨presheafToType X T, Presheaf.toType_isSheaf _ _⟩


