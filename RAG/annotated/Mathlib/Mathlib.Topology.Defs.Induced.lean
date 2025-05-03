/-- Given `f : X → Y` and a topology on `Y`,
  the induced topology on `X` is the collection of sets
  that are preimages of some open set in `Y`.
  This is the coarsest topology that makes `f` continuous. -/
def induced (f : X → Y) (t : TopologicalSpace Y) : TopologicalSpace X where
  IsOpen s := ∃ t, IsOpen t ∧ f ⁻¹' t = s
  isOpen_univ := ⟨univ, isOpen_univ, preimage_univ⟩
  isOpen_inter := by
    /-
      X : Type u_1
      Y : Type u_2
      f : X → Y
      t : TopologicalSpace Y
      ⊢ ∀ (s t_1 : Set X), (fun s => Exists fun t_2 => And (IsOpen t_2) (Eq (Set.pre …
    -/
    rintro s₁ s₂ ⟨s'₁, hs₁, rfl⟩ ⟨s'₂, hs₂, rfl⟩
    /-
      case intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      f : X → Y
      t : TopologicalSpace Y
      s'₁ : Set Y
      hs₁ : IsOpen s'₁
      s'₂ : Set Y
      hs₂ : IsOpen s'₂
      ⊢ Exists fun t_1 => And (IsOpen t_1) (Eq (Set.preimage f t_1) (Inter.inter (Se …
    -/
    exact ⟨s'₁ ∩ s'₂, hs₁.inter hs₂, preimage_inter⟩
    /-
      🎉 no goals
    -/
  isOpen_sUnion S h := by
    /-
      X : Type u_1
      Y : Type u_2
      f : X → Y
      t : TopologicalSpace Y
      S : Set (Set X)
      h : ∀ (t_1 : Set X), Membership.mem S t_1 → (fun s => Exists fun t_2 => And (I …
      ⊢ (fun s => Exists fun t_1 => And (IsOpen t_1) (Eq (Set.preimage f t_1) s)) S. …
    -/
    choose! g hgo hfg using h
    /-
      X : Type u_1
      Y : Type u_2
      f : X → Y
      t : TopologicalSpace Y
      S : Set (Set X)
      g : Set X → Set Y
      hgo : ∀ (t_1 : Set X), Membership.mem S t_1 → IsOpen (g t_1)
      hfg : ∀ (t : Set X), Membership.mem S t → Eq (Set.preimage f (g t)) t
      ⊢ Exists fun t_1 => And (IsOpen t_1) (Eq (Set.preimage f t_1) S.sUnion)
    -/
    refine ⟨⋃₀ (g '' S), isOpen_sUnion <| forall_mem_image.2 hgo, ?_⟩
    /-
      X : Type u_1
      Y : Type u_2
      f : X → Y
      t : TopologicalSpace Y
      S : Set (Set X)
      g : Set X → Set Y
      hgo : ∀ (t_1 : Set X), Membership.mem S t_1 → IsOpen (g t_1)
      hfg : ∀ (t : Set X), Membership.mem S t → Eq (Set.preimage f (g t)) t
      ⊢ Eq (Set.preimage f (Set.image g S).sUnion) S.sUnion
    -/
    rw [preimage_sUnion, biUnion_image, sUnion_eq_biUnion]
    /-
      X : Type u_1
      Y : Type u_2
      f : X → Y
      t : TopologicalSpace Y
      S : Set (Set X)
      g : Set X → Set Y
      hgo : ∀ (t_1 : Set X), Membership.mem S t_1 → IsOpen (g t_1)
      hfg : ∀ (t : Set X), Membership.mem S t → Eq (Set.preimage f (g t)) t
      ⊢ Eq (Set.iUnion fun y => Set.iUnion fun h => Set.preimage f (g y)) (Set.iUnio …
    -/
    exact iUnion₂_congr hfg
    /-
      🎉 no goals
    -/


instance _root_.instTopologicalSpaceSubtype {p : X → Prop} [t : TopologicalSpace X] :
    TopologicalSpace (Subtype p) :=
  induced (↑) t


/-- Given `f : X → Y` and a topology on `X`,
  the coinduced topology on `Y` is defined such that
  `s : Set Y` is open if the preimage of `s` is open.
  This is the finest topology that makes `f` continuous. -/
def coinduced (f : X → Y) (t : TopologicalSpace X) : TopologicalSpace Y where
  IsOpen s := IsOpen (f ⁻¹' s)
  isOpen_univ := t.isOpen_univ
  isOpen_inter _ _ h₁ h₂ := h₁.inter h₂
                          /-
                            X : Type u_1
                            Y : Type u_2
                            f : X → Y
                            t : TopologicalSpace X
                            s : Set (Set Y)
                            h : ∀ (t_1 : Set Y), Membership.mem s t_1 → (fun s => IsOpen (Set.preimage f s …
                            ⊢ (fun s => IsOpen (Set.preimage f s)) s.sUnion
                          -/
  isOpen_sUnion s h := by simpa only [preimage_sUnion] using isOpen_biUnion h
                          /-
                            🎉 no goals
                          -/


/-- We say that restrictions of the topology on `X` to sets from a family `S`
generates the original topology,
if either of the following equivalent conditions hold:

- a set which is relatively open in each `s ∈ S` is open;
- a set which is relatively closed in each `s ∈ S` is closed;
- for any topological space `Y`, a function `f : X → Y` is continuous
  provided that it is continuous on each `s ∈ S`.
-/
structure RestrictGenTopology (S : Set (Set X)) : Prop where
  isOpen_of_forall_induced (u : Set X) : (∀ s ∈ S, IsOpen ((↑) ⁻¹' u : Set s)) → IsOpen u


/-- A function `f : X → Y` between topological spaces is inducing if the topology on `X` is induced
by the topology on `Y` through `f`, meaning that a set `s : Set X` is open iff it is the preimage
under `f` of some open set `t : Set Y`. -/
@[mk_iff]
structure IsInducing (f : X → Y) : Prop where
  /-- The topology on the domain is equal to the induced topology. -/
  eq_induced : tX = tY.induced f


@[deprecated (since := "2024-10-28")] alias Inducing := IsInducing


/-- A function between topological spaces is an embedding if it is injective,
  and for all `s : Set X`, `s` is open iff it is the preimage of an open set. -/
@[mk_iff]
structure IsEmbedding (f : X → Y) extends IsInducing f : Prop where
  /-- A topological embedding is injective. -/
  injective : Function.Injective f


@[deprecated (since := "2024-10-26")]
alias Embedding := IsEmbedding


/-- An open embedding is an embedding with open range. -/
@[mk_iff]
structure IsOpenEmbedding (f : X → Y) extends IsEmbedding f : Prop where
  /-- The range of an open embedding is an open set. -/
  isOpen_range : IsOpen <| range f


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding := IsOpenEmbedding


/-- A closed embedding is an embedding with closed image. -/
@[mk_iff]
structure IsClosedEmbedding (f : X → Y) extends IsEmbedding f : Prop where
  /-- The range of a closed embedding is a closed set. -/
  isClosed_range : IsClosed <| range f


@[deprecated (since := "2024-10-20")]
alias ClosedEmbedding := IsClosedEmbedding


/-- A function between topological spaces is a quotient map if it is surjective,
  and for all `s : Set Y`, `s` is open iff its preimage is an open set. -/
@[mk_iff isQuotientMap_iff']
structure IsQuotientMap {X : Type*} {Y : Type*} [tX : TopologicalSpace X] [tY : TopologicalSpace Y]
    (f : X → Y) : Prop where
  surjective : Function.Surjective f
  eq_coinduced : tY = tX.coinduced f


@[deprecated (since := "2024-10-22")]
alias QuotientMap := IsQuotientMap


