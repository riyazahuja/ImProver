/-- The identity relation, or the graph of the identity function -/
def idRel {α : Type*} :=
  { p : α × α | p.1 = p.2 }


@[simp]
theorem mem_idRel {a b : α} : (a, b) ∈ @idRel α ↔ a = b :=
  Iff.rfl


@[simp]
theorem idRel_subset {s : Set (α × α)} : idRel ⊆ s ↔ ∀ a, (a, a) ∈ s := by
  /-
    α : Type ua
    s : Set (Prod α α)
    ⊢ Iff (HasSubset.Subset idRel s) (∀ (a : α), Membership.mem s { fst := a, snd  …
  -/
  simp [subset_def]
  /-
    🎉 no goals
  -/


theorem eq_singleton_left_of_prod_subset_idRel {X : Type*} {S T : Set X} (hS : S.Nonempty)
    (hT : T.Nonempty) (h_diag : S ×ˢ T ⊆ idRel) : ∃ x, S = {x} := by
  /-
    X : Type u_2
    S T : Set X
    hS : S.Nonempty
    hT : T.Nonempty
    h_diag : HasSubset.Subset (SProd.sprod S T) idRel
    ⊢ Exists fun x => Eq S (Singleton.singleton x)
  -/
  rcases hS, hT with ⟨⟨s, hs⟩, ⟨t, ht⟩⟩
  /-
    case intro.intro
    X : Type u_2
    S T : Set X
    h_diag : HasSubset.Subset (SProd.sprod S T) idRel
    s : X
    hs : Membership.mem S s
    t : X
    ht : Membership.mem T t
    ⊢ Exists fun x => Eq S (Singleton.singleton x)
  -/
  refine ⟨s, eq_singleton_iff_nonempty_unique_mem.mpr ⟨⟨s, hs⟩, fun x hx ↦ ?_⟩⟩
  /-
    case intro.intro
    X : Type u_2
    S T : Set X
    h_diag : HasSubset.Subset (SProd.sprod S T) idRel
    s : X
    hs : Membership.mem S s
    t : X
    ht : Membership.mem T t
    x : X
    hx : Membership.mem S x
    ⊢ Eq x s
  -/
  rw [prod_subset_iff] at h_diag
  /-
    case intro.intro
    X : Type u_2
    S T : Set X
    h_diag : ∀ (x : X), Membership.mem S x → ∀ (y : X), Membership.mem T y → Membe …
    s : X
    hs : Membership.mem S s
    t : X
    ht : Membership.mem T t
    x : X
    hx : Membership.mem S x
    ⊢ Eq x s
  -/
  replace hs := h_diag s hs t ht
  /-
    case intro.intro
    X : Type u_2
    S T : Set X
    h_diag : ∀ (x : X), Membership.mem S x → ∀ (y : X), Membership.mem T y → Membe …
    s t : X
    ht : Membership.mem T t
    x : X
    hx : Membership.mem S x
    hs : Membership.mem idRel { fst := s, snd := t }
    ⊢ Eq x s
  -/
  replace hx := h_diag x hx t ht
  /-
    case intro.intro
    X : Type u_2
    S T : Set X
    h_diag : ∀ (x : X), Membership.mem S x → ∀ (y : X), Membership.mem T y → Membe …
    s t : X
    ht : Membership.mem T t
    x : X
    hs : Membership.mem idRel { fst := s, snd := t }
    hx : Membership.mem idRel { fst := x, snd := t }
    ⊢ Eq x s
  -/
  simp only [idRel, mem_setOf_eq] at hx hs
  /-
    case intro.intro
    X : Type u_2
    S T : Set X
    h_diag : ∀ (x : X), Membership.mem S x → ∀ (y : X), Membership.mem T y → Membe …
    s t : X
    ht : Membership.mem T t
    x : X
    hs : Eq s t
    hx : Eq x t
    ⊢ Eq x s
  -/
  rwa [← hs] at hx
  /-
    🎉 no goals
  -/


theorem eq_singleton_right_prod_subset_idRel {X : Type*} {S T : Set X} (hS : S.Nonempty)
    (hT : T.Nonempty) (h_diag : S ×ˢ T ⊆ idRel) : ∃ x, T = {x} := by
  /-
    X : Type u_2
    S T : Set X
    hS : S.Nonempty
    hT : T.Nonempty
    h_diag : HasSubset.Subset (SProd.sprod S T) idRel
    ⊢ Exists fun x => Eq T (Singleton.singleton x)
  -/
  rw [Set.prod_subset_iff] at h_diag
  /-
    X : Type u_2
    S T : Set X
    hS : S.Nonempty
    hT : T.Nonempty
    h_diag : ∀ (x : X), Membership.mem S x → ∀ (y : X), Membership.mem T y → Membe …
    ⊢ Exists fun x => Eq T (Singleton.singleton x)
  -/
  replace h_diag := fun x hx y hy => (h_diag y hy x hx).symm
  /-
    X : Type u_2
    S T : Set X
    hS : S.Nonempty
    hT : T.Nonempty
    h_diag : ∀ (x : X), Membership.mem T x → ∀ (y : X), Membership.mem S y → Eq {  …
    ⊢ Exists fun x => Eq T (Singleton.singleton x)
  -/
  exact eq_singleton_left_of_prod_subset_idRel hT hS (prod_subset_iff.mpr h_diag)
  /-
    🎉 no goals
  -/


theorem eq_singleton_prod_subset_idRel {X : Type*} {S T : Set X} (hS : S.Nonempty)
    (hT : T.Nonempty) (h_diag : S ×ˢ T ⊆ idRel) : ∃ x, S = {x} ∧ T = {x} := by
  obtain ⟨⟨x, hx⟩, ⟨y, hy⟩⟩ := eq_singleton_left_of_prod_subset_idRel hS hT h_diag,
    eq_singleton_right_prod_subset_idRel hS hT h_diag
  /-
    case intro.intro
    X : Type u_2
    S T : Set X
    hS : S.Nonempty
    hT : T.Nonempty
    h_diag : HasSubset.Subset (SProd.sprod S T) idRel
    x : X
    hx : Eq S (Singleton.singleton x)
    y : X
    hy : Eq T (Singleton.singleton y)
    ⊢ Exists fun x => And (Eq S (Singleton.singleton x)) (Eq T (Singleton.singleto …
  -/
  refine ⟨x, ⟨hx, ?_⟩⟩
  /-
    case intro.intro
    X : Type u_2
    S T : Set X
    hS : S.Nonempty
    hT : T.Nonempty
    h_diag : HasSubset.Subset (SProd.sprod S T) idRel
    x : X
    hx : Eq S (Singleton.singleton x)
    y : X
    hy : Eq T (Singleton.singleton y)
    ⊢ Eq T (Singleton.singleton x)
  -/
  rw [hy, Set.singleton_eq_singleton_iff]
  exact (Set.prod_subset_iff.mp h_diag x (by simp only [hx, Set.mem_singleton]) y
    (by simp only [hy, Set.mem_singleton])).symm


/-- The composition of relations -/
def compRel (r₁ r₂ : Set (α × α)) :=
  { p : α × α | ∃ z : α, (p.1, z) ∈ r₁ ∧ (z, p.2) ∈ r₂ }


@[inherit_doc]
scoped[Uniformity] infixl:62 " ○ " => compRel

@[simp]
theorem mem_compRel {α : Type u} {r₁ r₂ : Set (α × α)} {x y : α} :
    (x, y) ∈ r₁ ○ r₂ ↔ ∃ z, (x, z) ∈ r₁ ∧ (z, y) ∈ r₂ :=
  Iff.rfl


@[simp]
theorem swap_idRel : Prod.swap '' idRel = @idRel α :=
                           /-
                             α : Type ua
                             x✝ : Prod α α
                             a b : α
                             ⊢ Iff (Membership.mem (Set.image Prod.swap idRel) { fst := a, snd := b }) (Mem …
                           -/
  Set.ext fun ⟨a, b⟩ => by simpa [image_swap_eq_preimage_swap] using eq_comm
                           /-
                             🎉 no goals
                           -/


theorem Monotone.compRel [Preorder β] {f g : β → Set (α × α)} (hf : Monotone f) (hg : Monotone g) :
    Monotone fun x => f x ○ g x := fun _ _ h _ ⟨z, h₁, h₂⟩ => ⟨z, hf h h₁, hg h h₂⟩


@[mono, gcongr]
theorem compRel_mono {f g h k : Set (α × α)} (h₁ : f ⊆ h) (h₂ : g ⊆ k) : f ○ g ⊆ h ○ k :=
  fun _ ⟨z, h, h'⟩ => ⟨z, h₁ h, h₂ h'⟩


theorem prod_mk_mem_compRel {a b c : α} {s t : Set (α × α)} (h₁ : (a, c) ∈ s) (h₂ : (c, b) ∈ t) :
    (a, b) ∈ s ○ t :=
  ⟨c, h₁, h₂⟩


@[simp]
theorem id_compRel {r : Set (α × α)} : idRel ○ r = r :=
                           /-
                             α : Type ua
                             r : Set (Prod α α)
                             x✝ : Prod α α
                             a b : α
                             ⊢ Iff (Membership.mem (compRel idRel r) { fst := a, snd := b }) (Membership.me …
                           -/
  Set.ext fun ⟨a, b⟩ => by simp
                           /-
                             🎉 no goals
                           -/


theorem compRel_assoc {r s t : Set (α × α)} : r ○ s ○ t = r ○ (s ○ t) := by
  /-
    α : Type ua
    r s t : Set (Prod α α)
    ⊢ Eq (compRel (compRel r s) t) (compRel r (compRel s t))
  -/
  ext ⟨a, b⟩; simp only [mem_compRel]; tauto
                                       /-
                                         🎉 no goals
                                       -/


theorem left_subset_compRel {s t : Set (α × α)} (h : idRel ⊆ t) : s ⊆ s ○ t := fun ⟨_x, y⟩ xy_in =>
  ⟨y, xy_in, h <| rfl⟩


theorem right_subset_compRel {s t : Set (α × α)} (h : idRel ⊆ s) : t ⊆ s ○ t := fun ⟨x, _y⟩ xy_in =>
  ⟨x, h <| rfl, xy_in⟩


theorem subset_comp_self {s : Set (α × α)} (h : idRel ⊆ s) : s ⊆ s ○ s :=
  left_subset_compRel h


theorem subset_iterate_compRel {s t : Set (α × α)} (h : idRel ⊆ s) (n : ℕ) :
    t ⊆ (s ○ ·)^[n] t := by
  /-
    α : Type ua
    s t : Set (Prod α α)
    h : HasSubset.Subset idRel s
    n : Nat
    ⊢ HasSubset.Subset t (Nat.iterate (fun x => compRel s x) n t)
  -/
  induction' n with n ihn generalizing t
  /-
    case zero
    α : Type ua
    s : Set (Prod α α)
    h : HasSubset.Subset idRel s
    t : Set (Prod α α)
    ⊢ HasSubset.Subset t (Nat.iterate (fun x => compRel s x) 0 t)
  -/
  exacts [Subset.rfl, (right_subset_compRel h).trans ihn]
  /-
    🎉 no goals
  -/


/-- The relation is invariant under swapping factors. -/
def SymmetricRel (V : Set (α × α)) : Prop :=
  Prod.swap ⁻¹' V = V


/-- The maximal symmetric relation contained in a given relation. -/
def symmetrizeRel (V : Set (α × α)) : Set (α × α) :=
  V ∩ Prod.swap ⁻¹' V


theorem symmetric_symmetrizeRel (V : Set (α × α)) : SymmetricRel (symmetrizeRel V) := by
  /-
    α : Type ua
    V : Set (Prod α α)
    ⊢ SymmetricRel (symmetrizeRel V)
  -/
  simp [SymmetricRel, symmetrizeRel, preimage_inter, inter_comm, ← preimage_comp]
  /-
    🎉 no goals
  -/


theorem symmetrizeRel_subset_self (V : Set (α × α)) : symmetrizeRel V ⊆ V :=
  sep_subset _ _


@[mono]
theorem symmetrize_mono {V W : Set (α × α)} (h : V ⊆ W) : symmetrizeRel V ⊆ symmetrizeRel W :=
  inter_subset_inter h <| preimage_mono h


theorem SymmetricRel.mk_mem_comm {V : Set (α × α)} (hV : SymmetricRel V) {x y : α} :
    (x, y) ∈ V ↔ (y, x) ∈ V :=
  Set.ext_iff.1 hV (y, x)


theorem SymmetricRel.eq {U : Set (α × α)} (hU : SymmetricRel U) : Prod.swap ⁻¹' U = U :=
  hU


theorem SymmetricRel.inter {U V : Set (α × α)} (hU : SymmetricRel U) (hV : SymmetricRel V) :
                               /-
                                 α : Type ua
                                 U V : Set (Prod α α)
                                 hU : SymmetricRel U
                                 hV : SymmetricRel V
                                 ⊢ SymmetricRel (Inter.inter U V)
                               -/
    SymmetricRel (U ∩ V) := by rw [SymmetricRel, preimage_inter, hU.eq, hV.eq]
                               /-
                                 🎉 no goals
                               -/


/-- This core description of a uniform space is outside of the type class hierarchy. It is useful
  for constructions of uniform spaces, when the topology is derived from the uniform space. -/
structure UniformSpace.Core (α : Type u) where
  /-- The uniformity filter. Once `UniformSpace` is defined, `𝓤 α` (`_root_.uniformity`) becomes the
  normal form. -/
  uniformity : Filter (α × α)
  /-- Every set in the uniformity filter includes the diagonal. -/
  refl : 𝓟 idRel ≤ uniformity
  /-- If `s ∈ uniformity`, then `Prod.swap ⁻¹' s ∈ uniformity`. -/
  symm : Tendsto Prod.swap uniformity uniformity
  /-- For every set `u ∈ uniformity`, there exists `v ∈ uniformity` such that `v ○ v ⊆ u`. -/
  comp : (uniformity.lift' fun s => s ○ s) ≤ uniformity


protected theorem UniformSpace.Core.comp_mem_uniformity_sets {c : Core α} {s : Set (α × α)}
    (hs : s ∈ c.uniformity) : ∃ t ∈ c.uniformity, t ○ t ⊆ s :=
  (mem_lift'_sets <| monotone_id.compRel monotone_id).mp <| c.comp hs


/-- An alternative constructor for `UniformSpace.Core`. This version unfolds various
`Filter`-related definitions. -/
def UniformSpace.Core.mk' {α : Type u} (U : Filter (α × α)) (refl : ∀ r ∈ U, ∀ (x), (x, x) ∈ r)
    (symm : ∀ r ∈ U, Prod.swap ⁻¹' r ∈ U) (comp : ∀ r ∈ U, ∃ t ∈ U, t ○ t ⊆ r) :
    UniformSpace.Core α :=
  ⟨U, fun _r ru => idRel_subset.2 (refl _ ru), symm, fun _r ru =>
    let ⟨_s, hs, hsr⟩ := comp _ ru
    mem_of_superset (mem_lift' hs) hsr⟩


/-- Defining a `UniformSpace.Core` from a filter basis satisfying some uniformity-like axioms. -/
def UniformSpace.Core.mkOfBasis {α : Type u} (B : FilterBasis (α × α))
    (refl : ∀ r ∈ B, ∀ (x), (x, x) ∈ r) (symm : ∀ r ∈ B, ∃ t ∈ B, t ⊆ Prod.swap ⁻¹' r)
    (comp : ∀ r ∈ B, ∃ t ∈ B, t ○ t ⊆ r) : UniformSpace.Core α where
  uniformity := B.filter
  refl := B.hasBasis.ge_iff.mpr fun _r ru => idRel_subset.2 <| refl _ ru
  symm := (B.hasBasis.tendsto_iff B.hasBasis).mpr symm
  comp := (HasBasis.le_basis_iff (B.hasBasis.lift' (monotone_id.compRel monotone_id))
    B.hasBasis).2 comp


/-- A uniform space generates a topological space -/
def UniformSpace.Core.toTopologicalSpace {α : Type u} (u : UniformSpace.Core α) :
    TopologicalSpace α :=
  .mkOfNhds fun x ↦ .comap (Prod.mk x) u.uniformity


theorem UniformSpace.Core.ext :
    ∀ {u₁ u₂ : UniformSpace.Core α}, u₁.uniformity = u₂.uniformity → u₁ = u₂
  | ⟨_, _, _, _⟩, ⟨_, _, _, _⟩, rfl => rfl


theorem UniformSpace.Core.nhds_toTopologicalSpace {α : Type u} (u : Core α) (x : α) :
    @nhds α u.toTopologicalSpace x = comap (Prod.mk x) u.uniformity := by
  /-
    α : Type u
    u : UniformSpace.Core α
    x : α
    ⊢ Eq (nhds x) (Filter.comap (Prod.mk x) u.uniformity)
  -/
  apply TopologicalSpace.nhds_mkOfNhds_of_hasBasis (fun _ ↦ (basis_sets _).comap _)
    /-
      case hpure
      α : Type u
      u : UniformSpace.Core α
      x : α
      ⊢ ∀ (a : α) (i : Set (Prod α α)), Membership.mem u.uniformity i → Membership.m …
    -/
  · exact fun a U hU ↦ u.refl hU rfl
    /-
      🎉 no goals
    -/
    /-
      case hopen
      α : Type u
      u : UniformSpace.Core α
      x : α
      ⊢ ∀ (a : α) (i : Set (Prod α α)), Membership.mem u.uniformity i → Filter.Event …
    -/
  · intro a U hU
    /-
      case hopen
      α : Type u
      u : UniformSpace.Core α
      x a : α
      U : Set (Prod α α)
      hU : Membership.mem u.uniformity U
      ⊢ Filter.Eventually (fun x => Membership.mem (Filter.comap (Prod.mk x) u.unifo …
    -/
    rcases u.comp_mem_uniformity_sets hU with ⟨V, hV, hVU⟩
    /-
      case hopen.intro.intro
      α : Type u
      u : UniformSpace.Core α
      x a : α
      U : Set (Prod α α)
      hU : Membership.mem u.uniformity U
      V : Set (Prod α α)
      hV : Membership.mem u.uniformity V
      hVU : HasSubset.Subset (compRel V V) U
      ⊢ Filter.Eventually (fun x => Membership.mem (Filter.comap (Prod.mk x) u.unifo …
    -/
    filter_upwards [preimage_mem_comap hV] with b hb
    /-
      case h
      α : Type u
      u : UniformSpace.Core α
      x a : α
      U : Set (Prod α α)
      hU : Membership.mem u.uniformity U
      V : Set (Prod α α)
      hV : Membership.mem u.uniformity V
      hVU : HasSubset.Subset (compRel V V) U
      b : α
      hb : Membership.mem (Set.preimage (Prod.mk a) V) b
      ⊢ Membership.mem (Filter.comap (Prod.mk b) u.uniformity) (Set.preimage (Prod.m …
    -/
    filter_upwards [preimage_mem_comap hV] with c hc
    /-
      case h
      α : Type u
      u : UniformSpace.Core α
      x a : α
      U : Set (Prod α α)
      hU : Membership.mem u.uniformity U
      V : Set (Prod α α)
      hV : Membership.mem u.uniformity V
      hVU : HasSubset.Subset (compRel V V) U
      b : α
      hb : Membership.mem (Set.preimage (Prod.mk a) V) b
      c : α
      hc : Membership.mem (Set.preimage (Prod.mk b) V) c
      ⊢ Membership.mem (Set.preimage (Prod.mk a) (id U)) c
    -/
    exact hVU ⟨b, hb, hc⟩
    /-
      🎉 no goals
    -/

-- the topological structure is embedded in the uniform structure
-- to avoid instance diamond issues. See Note [forgetful inheritance].

/-- A uniform space is a generalization of the "uniform" topological aspects of a
  metric space. It consists of a filter on `α × α` called the "uniformity", which
  satisfies properties analogous to the reflexivity, symmetry, and triangle properties
  of a metric.

  A metric space has a natural uniformity, and a uniform space has a natural topology.
  A topological group also has a natural uniformity, even when it is not metrizable. -/
class UniformSpace (α : Type u) extends TopologicalSpace α where
  /-- The uniformity filter. -/
  protected uniformity : Filter (α × α)
  /-- If `s ∈ uniformity`, then `Prod.swap ⁻¹' s ∈ uniformity`. -/
  protected symm : Tendsto Prod.swap uniformity uniformity
  /-- For every set `u ∈ uniformity`, there exists `v ∈ uniformity` such that `v ○ v ⊆ u`. -/
  protected comp : (uniformity.lift' fun s => s ○ s) ≤ uniformity
  /-- The uniformity agrees with the topology: the neighborhoods filter of each point `x`
  is equal to `Filter.comap (Prod.mk x) (𝓤 α)`. -/
  protected nhds_eq_comap_uniformity (x : α) : 𝓝 x = comap (Prod.mk x) uniformity


/-- The uniformity is a filter on α × α (inferred from an ambient uniform space
  structure on α). -/
def uniformity (α : Type u) [UniformSpace α] : Filter (α × α) :=
  @UniformSpace.uniformity α _


/-- Notation for the uniformity filter with respect to a non-standard `UniformSpace` instance. -/
scoped[Uniformity] notation "𝓤[" u "]" => @uniformity _ u


@[inherit_doc] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: should we drop the `uniformity` def?
scoped[Uniformity] notation "𝓤" => uniformity


/-- Construct a `UniformSpace` from a `u : UniformSpace.Core` and a `TopologicalSpace` structure
that is equal to `u.toTopologicalSpace`. -/
abbrev UniformSpace.ofCoreEq {α : Type u} (u : UniformSpace.Core α) (t : TopologicalSpace α)
    (h : t = u.toTopologicalSpace) : UniformSpace α where
  __ := u
  toTopologicalSpace := t
                                   /-
                                     α✝ : Type ua
                                     β : Type ub
                                     γ : Type uc
                                     δ : Type ud
                                     ι : Sort u_1
                                     α : Type u
                                     u : UniformSpace.Core α
                                     t : TopologicalSpace α
                                     h : Eq t u.toTopologicalSpace
                                     x : α
                                     ⊢ Eq (nhds x) (Filter.comap (Prod.mk x) __spread✝⁻⁰.uniformity)
                                   -/
  nhds_eq_comap_uniformity x := by rw [h, u.nhds_toTopologicalSpace]
                                   /-
                                     🎉 no goals
                                   -/


/-- Construct a `UniformSpace` from a `UniformSpace.Core`. -/
abbrev UniformSpace.ofCore {α : Type u} (u : UniformSpace.Core α) : UniformSpace α :=
  .ofCoreEq u _ rfl


/-- Construct a `UniformSpace.Core` from a `UniformSpace`. -/
abbrev UniformSpace.toCore (u : UniformSpace α) : UniformSpace.Core α where
  __ := u
  refl := by
    /-
      α : Type ua
      β : Type ub
      γ : Type uc
      δ : Type ud
      ι : Sort u_1
      u : UniformSpace α
      ⊢ LE.le (Filter.principal idRel) UniformSpace.uniformity
    -/
    rintro U hU ⟨x, y⟩ (rfl : x = y)
    have : Prod.mk x ⁻¹' U ∈ 𝓝 x := by
      rw [UniformSpace.nhds_eq_comap_uniformity]
      exact preimage_mem_comap hU
    /-
      case mk
      α : Type ua
      β : Type ub
      γ : Type uc
      δ : Type ud
      ι : Sort u_1
      u : UniformSpace α
      U : Set (Prod α α)
      hU : Membership.mem UniformSpace.uniformity U
      x : α
      this : Membership.mem (nhds x) (Set.preimage (Prod.mk x) U)
      ⊢ Membership.mem U { fst := x, snd := x }
    -/
    convert mem_of_mem_nhds this
    /-
      🎉 no goals
    -/


theorem UniformSpace.toCore_toTopologicalSpace (u : UniformSpace α) :
    u.toCore.toTopologicalSpace = u.toTopologicalSpace :=
  TopologicalSpace.ext_nhds fun a ↦ by
    /-
      α : Type ua
      u : UniformSpace α
      a : α
      ⊢ Eq (nhds a) (nhds a)
    -/
    rw [u.nhds_eq_comap_uniformity, u.toCore.nhds_toTopologicalSpace]
    /-
      🎉 no goals
    -/


/-- Build a `UniformSpace` from a `UniformSpace.Core` and a compatible topology.
Use `UniformSpace.mk` instead to avoid proving
the unnecessary assumption `UniformSpace.Core.refl`.

The main constructor used to use a different compatibility assumption.
This definition was created as a step towards porting to a new definition.
Now the main definition is ported,
so this constructor will be removed in a few months. -/
@[deprecated UniformSpace.mk (since := "2024-03-20")]
def UniformSpace.ofNhdsEqComap (u : UniformSpace.Core α) (_t : TopologicalSpace α)
    (h : ∀ x, 𝓝 x = u.uniformity.comap (Prod.mk x)) : UniformSpace α where
  __ := u
  nhds_eq_comap_uniformity := h


@[ext (iff := false)]
protected theorem UniformSpace.ext {u₁ u₂ : UniformSpace α} (h : 𝓤[u₁] = 𝓤[u₂]) : u₁ = u₂ := by
  have : u₁.toTopologicalSpace = u₂.toTopologicalSpace := TopologicalSpace.ext_nhds fun x ↦ by
    rw [u₁.nhds_eq_comap_uniformity, u₂.nhds_eq_comap_uniformity]
    exact congr_arg (comap _) h
  /-
    α : Type ua
    u₁ u₂ : UniformSpace α
    h : Eq (uniformity α) (uniformity α)
    this : Eq UniformSpace.toTopologicalSpace UniformSpace.toTopologicalSpace
    ⊢ Eq u₁ u₂
  -/
  cases u₁; cases u₂; congr
                      /-
                        🎉 no goals
                      -/


protected theorem UniformSpace.ext_iff {u₁ u₂ : UniformSpace α} :
    u₁ = u₂ ↔ ∀ s, s ∈ 𝓤[u₁] ↔ s ∈ 𝓤[u₂] :=
                                       /-
                                         α : Type ua
                                         u₁ u₂ : UniformSpace α
                                         h : ∀ (s : Set (Prod α α)), Iff (Membership.mem (uniformity α) s) (Membership. …
                                         ⊢ Eq u₁ u₂
                                       -/
  ⟨fun h _ => h ▸ Iff.rfl, fun h => by ext; exact h _⟩
                                            /-
                                              🎉 no goals
                                            -/


theorem UniformSpace.ofCoreEq_toCore (u : UniformSpace α) (t : TopologicalSpace α)
    (h : t = u.toCore.toTopologicalSpace) : .ofCoreEq u.toCore t h = u :=
  UniformSpace.ext rfl


/-- Replace topology in a `UniformSpace` instance with a propositionally (but possibly not
definitionally) equal one. -/
abbrev UniformSpace.replaceTopology {α : Type*} [i : TopologicalSpace α] (u : UniformSpace α)
    (h : i = u.toTopologicalSpace) : UniformSpace α where
  __ := u
  toTopologicalSpace := i
                                   /-
                                     α✝ : Type ua
                                     β : Type ub
                                     γ : Type uc
                                     δ : Type ud
                                     ι : Sort u_1
                                     α : Type u_2
                                     i : TopologicalSpace α
                                     u : UniformSpace α
                                     h : Eq i UniformSpace.toTopologicalSpace
                                     x : α
                                     ⊢ Eq (nhds x) (Filter.comap (Prod.mk x) UniformSpace.uniformity)
                                   -/
  nhds_eq_comap_uniformity x := by rw [h, u.nhds_eq_comap_uniformity]
                                   /-
                                     🎉 no goals
                                   -/


theorem UniformSpace.replaceTopology_eq {α : Type*} [i : TopologicalSpace α] (u : UniformSpace α)
    (h : i = u.toTopologicalSpace) : u.replaceTopology h = u :=
  UniformSpace.ext rfl


theorem nhds_eq_comap_uniformity {x : α} : 𝓝 x = (𝓤 α).comap (Prod.mk x) :=
  UniformSpace.nhds_eq_comap_uniformity x


theorem isOpen_uniformity {s : Set α} :
    IsOpen s ↔ ∀ x ∈ s, { p : α × α | p.1 = x → p.2 ∈ s } ∈ 𝓤 α := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    s : Set α
    ⊢ Iff (IsOpen s) (∀ (x : α), Membership.mem s x → Membership.mem (uniformity α …
  -/
  simp only [isOpen_iff_mem_nhds, nhds_eq_comap_uniformity, mem_comap_prod_mk]
  /-
    🎉 no goals
  -/


theorem refl_le_uniformity : 𝓟 idRel ≤ 𝓤 α :=
  (@UniformSpace.toCore α _).refl


instance uniformity.neBot [Nonempty α] : NeBot (𝓤 α) :=
  diagonal_nonempty.principal_neBot.mono refl_le_uniformity


theorem refl_mem_uniformity {x : α} {s : Set (α × α)} (h : s ∈ 𝓤 α) : (x, x) ∈ s :=
  refl_le_uniformity h rfl


theorem mem_uniformity_of_eq {x y : α} {s : Set (α × α)} (h : s ∈ 𝓤 α) (hx : x = y) : (x, y) ∈ s :=
  refl_le_uniformity h hx


theorem symm_le_uniformity : map (@Prod.swap α α) (𝓤 _) ≤ 𝓤 _ :=
  UniformSpace.symm


theorem comp_le_uniformity : ((𝓤 α).lift' fun s : Set (α × α) => s ○ s) ≤ 𝓤 α :=
  UniformSpace.comp


theorem lift'_comp_uniformity : ((𝓤 α).lift' fun s : Set (α × α) => s ○ s) = 𝓤 α :=
  comp_le_uniformity.antisymm <| le_lift'.2 fun _s hs ↦ mem_of_superset hs <|
    subset_comp_self <| idRel_subset.2 fun _ ↦ refl_mem_uniformity hs


theorem tendsto_swap_uniformity : Tendsto (@Prod.swap α α) (𝓤 α) (𝓤 α) :=
  symm_le_uniformity


theorem comp_mem_uniformity_sets {s : Set (α × α)} (hs : s ∈ 𝓤 α) : ∃ t ∈ 𝓤 α, t ○ t ⊆ s :=
  (mem_lift'_sets <| monotone_id.compRel monotone_id).mp <| comp_le_uniformity hs


/-- If `s ∈ 𝓤 α`, then for any natural `n`, for a subset `t` of a sufficiently small set in `𝓤 α`,
we have `t ○ t ○ ... ○ t ⊆ s` (`n` compositions). -/
theorem eventually_uniformity_iterate_comp_subset {s : Set (α × α)} (hs : s ∈ 𝓤 α) (n : ℕ) :
    ∀ᶠ t in (𝓤 α).smallSets, (t ○ ·)^[n] t ⊆ s := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    n : Nat
    ⊢ Filter.Eventually (fun t => HasSubset.Subset (Nat.iterate (fun x => compRel  …
  -/
  suffices ∀ᶠ t in (𝓤 α).smallSets, t ⊆ s ∧ (t ○ ·)^[n] t ⊆ s from (eventually_and.1 this).2
  /-
    α : Type ua
    inst✝ : UniformSpace α
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    n : Nat
    ⊢ Filter.Eventually (fun t => And (HasSubset.Subset t s) (HasSubset.Subset (Na …
  -/
  induction' n with n ihn generalizing s
    /-
      case zero
      α : Type ua
      inst✝ : UniformSpace α
      s : Set (Prod α α)
      hs : Membership.mem (uniformity α) s
      ⊢ Filter.Eventually (fun t => And (HasSubset.Subset t s) (HasSubset.Subset (Na …
    -/
  · simpa
    /-
      🎉 no goals
    -/
  /-
    case succ
    α : Type ua
    inst✝ : UniformSpace α
    n : Nat
    ihn : ∀ {s : Set (Prod α α)}, Membership.mem (uniformity α) s → Filter.Eventua …
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    ⊢ Filter.Eventually (fun t => And (HasSubset.Subset t s) (HasSubset.Subset (Na …
  -/
  rcases comp_mem_uniformity_sets hs with ⟨t, htU, hts⟩
  /-
    case succ.intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    n : Nat
    ihn : ∀ {s : Set (Prod α α)}, Membership.mem (uniformity α) s → Filter.Eventua …
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    t : Set (Prod α α)
    htU : Membership.mem (uniformity α) t
    hts : HasSubset.Subset (compRel t t) s
    ⊢ Filter.Eventually (fun t => And (HasSubset.Subset t s) (HasSubset.Subset (Na …
  -/
  refine (ihn htU).mono fun U hU => ?_
  /-
    case succ.intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    n : Nat
    ihn : ∀ {s : Set (Prod α α)}, Membership.mem (uniformity α) s → Filter.Eventua …
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    t : Set (Prod α α)
    htU : Membership.mem (uniformity α) t
    hts : HasSubset.Subset (compRel t t) s
    U : Set (Prod α α)
    hU : And (HasSubset.Subset U t) (HasSubset.Subset (Nat.iterate (fun x => compR …
    ⊢ And (HasSubset.Subset U s) (HasSubset.Subset (Nat.iterate (fun x => compRel  …
  -/
  rw [Function.iterate_succ_apply']
  exact
    ⟨hU.1.trans <| (subset_comp_self <| refl_le_uniformity htU).trans hts,
      (compRel_mono hU.1 hU.2).trans hts⟩


/-- If `s ∈ 𝓤 α`, then for a subset `t` of a sufficiently small set in `𝓤 α`,
we have `t ○ t ⊆ s`. -/
theorem eventually_uniformity_comp_subset {s : Set (α × α)} (hs : s ∈ 𝓤 α) :
    ∀ᶠ t in (𝓤 α).smallSets, t ○ t ⊆ s :=
  eventually_uniformity_iterate_comp_subset hs 1


/-- Relation `fun f g ↦ Tendsto (fun x ↦ (f x, g x)) l (𝓤 α)` is transitive. -/
theorem Filter.Tendsto.uniformity_trans {l : Filter β} {f₁ f₂ f₃ : β → α}
    (h₁₂ : Tendsto (fun x => (f₁ x, f₂ x)) l (𝓤 α))
    (h₂₃ : Tendsto (fun x => (f₂ x, f₃ x)) l (𝓤 α)) : Tendsto (fun x => (f₁ x, f₃ x)) l (𝓤 α) := by
  /-
    α : Type ua
    β : Type ub
    inst✝ : UniformSpace α
    l : Filter β
    f₁ f₂ f₃ : β → α
    h₁₂ : Filter.Tendsto (fun x => { fst := f₁ x, snd := f₂ x }) l (uniformity α)
    h₂₃ : Filter.Tendsto (fun x => { fst := f₂ x, snd := f₃ x }) l (uniformity α)
    ⊢ Filter.Tendsto (fun x => { fst := f₁ x, snd := f₃ x }) l (uniformity α)
  -/
  refine le_trans (le_lift'.2 fun s hs => mem_map.2 ?_) comp_le_uniformity
  /-
    α : Type ua
    β : Type ub
    inst✝ : UniformSpace α
    l : Filter β
    f₁ f₂ f₃ : β → α
    h₁₂ : Filter.Tendsto (fun x => { fst := f₁ x, snd := f₂ x }) l (uniformity α)
    h₂₃ : Filter.Tendsto (fun x => { fst := f₂ x, snd := f₃ x }) l (uniformity α)
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    ⊢ Membership.mem l (Set.preimage (fun x => { fst := f₁ x, snd := f₃ x }) (comp …
  -/
  filter_upwards [mem_map.1 (h₁₂ hs), mem_map.1 (h₂₃ hs)] with x hx₁₂ hx₂₃ using ⟨_, hx₁₂, hx₂₃⟩
  /-
    🎉 no goals
  -/


/-- Relation `fun f g ↦ Tendsto (fun x ↦ (f x, g x)) l (𝓤 α)` is symmetric. -/
theorem Filter.Tendsto.uniformity_symm {l : Filter β} {f : β → α × α} (h : Tendsto f l (𝓤 α)) :
    Tendsto (fun x => ((f x).2, (f x).1)) l (𝓤 α) :=
  tendsto_swap_uniformity.comp h


/-- Relation `fun f g ↦ Tendsto (fun x ↦ (f x, g x)) l (𝓤 α)` is reflexive. -/
theorem tendsto_diag_uniformity (f : β → α) (l : Filter β) :
    Tendsto (fun x => (f x, f x)) l (𝓤 α) := fun _s hs =>
  mem_map.2 <| univ_mem' fun _ => refl_mem_uniformity hs


theorem tendsto_const_uniformity {a : α} {f : Filter β} : Tendsto (fun _ => (a, a)) f (𝓤 α) :=
  tendsto_diag_uniformity (fun _ => a) f


theorem symm_of_uniformity {s : Set (α × α)} (hs : s ∈ 𝓤 α) :
    ∃ t ∈ 𝓤 α, (∀ a b, (a, b) ∈ t → (b, a) ∈ t) ∧ t ⊆ s :=
  have : preimage Prod.swap s ∈ 𝓤 α := symm_le_uniformity hs
  ⟨s ∩ preimage Prod.swap s, inter_mem hs this, fun _ _ ⟨h₁, h₂⟩ => ⟨h₂, h₁⟩, inter_subset_left⟩


theorem comp_symm_of_uniformity {s : Set (α × α)} (hs : s ∈ 𝓤 α) :
    ∃ t ∈ 𝓤 α, (∀ {a b}, (a, b) ∈ t → (b, a) ∈ t) ∧ t ○ t ⊆ s :=
  let ⟨_t, ht₁, ht₂⟩ := comp_mem_uniformity_sets hs
  let ⟨t', ht', ht'₁, ht'₂⟩ := symm_of_uniformity ht₁
  ⟨t', ht', ht'₁ _ _, Subset.trans (monotone_id.compRel monotone_id ht'₂) ht₂⟩


theorem uniformity_le_symm : 𝓤 α ≤ @Prod.swap α α <$> 𝓤 α := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    ⊢ LE.le (uniformity α) (Functor.map Prod.swap (uniformity α))
  -/
  rw [map_swap_eq_comap_swap]; exact tendsto_swap_uniformity.le_comap
                               /-
                                 🎉 no goals
                               -/


theorem uniformity_eq_symm : 𝓤 α = @Prod.swap α α <$> 𝓤 α :=
  le_antisymm uniformity_le_symm symm_le_uniformity


@[simp]
theorem comap_swap_uniformity : comap (@Prod.swap α α) (𝓤 α) = 𝓤 α :=
  (congr_arg _ uniformity_eq_symm).trans <| comap_map Prod.swap_injective


theorem symmetrize_mem_uniformity {V : Set (α × α)} (h : V ∈ 𝓤 α) : symmetrizeRel V ∈ 𝓤 α := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    V : Set (Prod α α)
    h : Membership.mem (uniformity α) V
    ⊢ Membership.mem (uniformity α) (symmetrizeRel V)
  -/
  apply (𝓤 α).inter_sets h
  /-
    α : Type ua
    inst✝ : UniformSpace α
    V : Set (Prod α α)
    h : Membership.mem (uniformity α) V
    ⊢ Membership.mem (uniformity α).sets (Set.preimage Prod.swap V)
  -/
  rw [← image_swap_eq_preimage_swap, uniformity_eq_symm]
  /-
    α : Type ua
    inst✝ : UniformSpace α
    V : Set (Prod α α)
    h : Membership.mem (uniformity α) V
    ⊢ Membership.mem (Functor.map Prod.swap (uniformity α)).sets (Set.image Prod.s …
  -/
  exact image_mem_map h
  /-
    🎉 no goals
  -/


/-- Symmetric entourages form a basis of `𝓤 α` -/
theorem UniformSpace.hasBasis_symmetric :
    (𝓤 α).HasBasis (fun s : Set (α × α) => s ∈ 𝓤 α ∧ SymmetricRel s) id :=
  hasBasis_self.2 fun t t_in =>
    ⟨symmetrizeRel t, symmetrize_mem_uniformity t_in, symmetric_symmetrizeRel t,
      symmetrizeRel_subset_self t⟩


theorem uniformity_lift_le_swap {g : Set (α × α) → Filter β} {f : Filter β} (hg : Monotone g)
    (h : ((𝓤 α).lift fun s => g (preimage Prod.swap s)) ≤ f) : (𝓤 α).lift g ≤ f :=
  calc
    (𝓤 α).lift g ≤ (Filter.map (@Prod.swap α α) <| 𝓤 α).lift g :=
      lift_mono uniformity_le_symm le_rfl
                /-
                  α : Type ua
                  β : Type ub
                  inst✝ : UniformSpace α
                  g : Set (Prod α α) → Filter β
                  f : Filter β
                  hg : Monotone g
                  h : LE.le ((uniformity α).lift fun s => g (Set.preimage Prod.swap s)) f
                  ⊢ LE.le ((Filter.map Prod.swap (uniformity α)).lift g) f
                -/
    _ ≤ _ := by rw [map_lift_eq2 hg, image_swap_eq_preimage_swap]; exact h
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem uniformity_lift_le_comp {f : Set (α × α) → Filter β} (h : Monotone f) :
    ((𝓤 α).lift fun s => f (s ○ s)) ≤ (𝓤 α).lift f :=
  calc
    ((𝓤 α).lift fun s => f (s ○ s)) = ((𝓤 α).lift' fun s : Set (α × α) => s ○ s).lift f := by
      /-
        α : Type ua
        β : Type ub
        inst✝ : UniformSpace α
        f : Set (Prod α α) → Filter β
        h : Monotone f
        ⊢ Eq ((uniformity α).lift fun s => f (compRel s s)) (((uniformity α).lift' fun …
      -/
      rw [lift_lift'_assoc]
        /-
          case hg
          α : Type ua
          β : Type ub
          inst✝ : UniformSpace α
          f : Set (Prod α α) → Filter β
          h : Monotone f
          ⊢ Monotone fun s => compRel s s
        -/
      · exact monotone_id.compRel monotone_id
        /-
          🎉 no goals
        -/
        /-
          case hh
          α : Type ua
          β : Type ub
          inst✝ : UniformSpace α
          f : Set (Prod α α) → Filter β
          h : Monotone f
          ⊢ Monotone f
        -/
      · exact h
        /-
          🎉 no goals
        -/
    _ ≤ (𝓤 α).lift f := lift_mono comp_le_uniformity le_rfl


theorem comp3_mem_uniformity {s : Set (α × α)} (hs : s ∈ 𝓤 α) : ∃ t ∈ 𝓤 α, t ○ (t ○ t) ⊆ s :=
  let ⟨_t', ht', ht's⟩ := comp_mem_uniformity_sets hs
  let ⟨t, ht, htt'⟩ := comp_mem_uniformity_sets ht'
  ⟨t, ht, (compRel_mono ((subset_comp_self (refl_le_uniformity ht)).trans htt') htt').trans ht's⟩


/-- See also `comp3_mem_uniformity`. -/
theorem comp_le_uniformity3 : ((𝓤 α).lift' fun s : Set (α × α) => s ○ (s ○ s)) ≤ 𝓤 α := fun _ h =>
  let ⟨_t, htU, ht⟩ := comp3_mem_uniformity h
  mem_of_superset (mem_lift' htU) ht


/-- See also `comp_open_symm_mem_uniformity_sets`. -/
theorem comp_symm_mem_uniformity_sets {s : Set (α × α)} (hs : s ∈ 𝓤 α) :
    ∃ t ∈ 𝓤 α, SymmetricRel t ∧ t ○ t ⊆ s := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    ⊢ Exists fun t => And (Membership.mem (uniformity α) t) (And (SymmetricRel t)  …
  -/
  obtain ⟨w, w_in, w_sub⟩ : ∃ w ∈ 𝓤 α, w ○ w ⊆ s := comp_mem_uniformity_sets hs
  /-
    case intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    w : Set (Prod α α)
    w_in : Membership.mem (uniformity α) w
    w_sub : HasSubset.Subset (compRel w w) s
    ⊢ Exists fun t => And (Membership.mem (uniformity α) t) (And (SymmetricRel t)  …
  -/
  use symmetrizeRel w, symmetrize_mem_uniformity w_in, symmetric_symmetrizeRel w
  /-
    case right
    α : Type ua
    inst✝ : UniformSpace α
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    w : Set (Prod α α)
    w_in : Membership.mem (uniformity α) w
    w_sub : HasSubset.Subset (compRel w w) s
    ⊢ HasSubset.Subset (compRel (symmetrizeRel w) (symmetrizeRel w)) s
  -/
  have : symmetrizeRel w ⊆ w := symmetrizeRel_subset_self w
  calc symmetrizeRel w ○ symmetrizeRel w
    _ ⊆ w ○ w := by gcongr
    _ ⊆ s     := w_sub


theorem subset_comp_self_of_mem_uniformity {s : Set (α × α)} (h : s ∈ 𝓤 α) : s ⊆ s ○ s :=
  subset_comp_self (refl_le_uniformity h)


theorem comp_comp_symm_mem_uniformity_sets {s : Set (α × α)} (hs : s ∈ 𝓤 α) :
    ∃ t ∈ 𝓤 α, SymmetricRel t ∧ t ○ t ○ t ⊆ s := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    ⊢ Exists fun t => And (Membership.mem (uniformity α) t) (And (SymmetricRel t)  …
  -/
  rcases comp_symm_mem_uniformity_sets hs with ⟨w, w_in, _, w_sub⟩
  /-
    case intro.intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    w : Set (Prod α α)
    w_in : Membership.mem (uniformity α) w
    left✝ : SymmetricRel w
    w_sub : HasSubset.Subset (compRel w w) s
    ⊢ Exists fun t => And (Membership.mem (uniformity α) t) (And (SymmetricRel t)  …
  -/
  rcases comp_symm_mem_uniformity_sets w_in with ⟨t, t_in, t_symm, t_sub⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    w : Set (Prod α α)
    w_in : Membership.mem (uniformity α) w
    left✝ : SymmetricRel w
    w_sub : HasSubset.Subset (compRel w w) s
    t : Set (Prod α α)
    t_in : Membership.mem (uniformity α) t
    t_symm : SymmetricRel t
    t_sub : HasSubset.Subset (compRel t t) w
    ⊢ Exists fun t => And (Membership.mem (uniformity α) t) (And (SymmetricRel t)  …
  -/
  use t, t_in, t_symm
  /-
    case right
    α : Type ua
    inst✝ : UniformSpace α
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    w : Set (Prod α α)
    w_in : Membership.mem (uniformity α) w
    left✝ : SymmetricRel w
    w_sub : HasSubset.Subset (compRel w w) s
    t : Set (Prod α α)
    t_in : Membership.mem (uniformity α) t
    t_symm : SymmetricRel t
    t_sub : HasSubset.Subset (compRel t t) w
    ⊢ HasSubset.Subset (compRel (compRel t t) t) s
  -/
  have : t ⊆ t ○ t := subset_comp_self_of_mem_uniformity t_in
  -- Porting note: Needed the following `have`s to make `mono` work
  /-
    case right
    α : Type ua
    inst✝ : UniformSpace α
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    w : Set (Prod α α)
    w_in : Membership.mem (uniformity α) w
    left✝ : SymmetricRel w
    w_sub : HasSubset.Subset (compRel w w) s
    t : Set (Prod α α)
    t_in : Membership.mem (uniformity α) t
    t_symm : SymmetricRel t
    t_sub : HasSubset.Subset (compRel t t) w
    this : HasSubset.Subset t (compRel t t)
    ⊢ HasSubset.Subset (compRel (compRel t t) t) s
  -/
  have ht := Subset.refl t
  /-
    case right
    α : Type ua
    inst✝ : UniformSpace α
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    w : Set (Prod α α)
    w_in : Membership.mem (uniformity α) w
    left✝ : SymmetricRel w
    w_sub : HasSubset.Subset (compRel w w) s
    t : Set (Prod α α)
    t_in : Membership.mem (uniformity α) t
    t_symm : SymmetricRel t
    t_sub : HasSubset.Subset (compRel t t) w
    this : HasSubset.Subset t (compRel t t)
    ht : HasSubset.Subset t t
    ⊢ HasSubset.Subset (compRel (compRel t t) t) s
  -/
  have hw := Subset.refl w
  calc
    t ○ t ○ t ⊆ w ○ t := by mono
    _ ⊆ w ○ (t ○ t) := by mono
    _ ⊆ w ○ w := by mono
    _ ⊆ s := w_sub


/-- The ball around `(x : β)` with respect to `(V : Set (β × β))`. Intended to be
used for `V ∈ 𝓤 β`, but this is not needed for the definition. Recovers the
notions of metric space ball when `V = {p | dist p.1 p.2 < r }`. -/
def ball (x : β) (V : Set (β × β)) : Set β := Prod.mk x ⁻¹' V


lemma mem_ball_self (x : α) {V : Set (α × α)} : V ∈ 𝓤 α → x ∈ ball x V := refl_mem_uniformity


/-- The triangle inequality for `UniformSpace.ball` -/
theorem mem_ball_comp {V W : Set (β × β)} {x y z} (h : y ∈ ball x V) (h' : z ∈ ball y W) :
    z ∈ ball x (V ○ W) :=
  prod_mk_mem_compRel h h'


theorem ball_subset_of_comp_subset {V W : Set (β × β)} {x y} (h : x ∈ ball y W) (h' : W ○ W ⊆ V) :
    ball x W ⊆ ball y V := fun _z z_in => h' (mem_ball_comp h z_in)


theorem ball_mono {V W : Set (β × β)} (h : V ⊆ W) (x : β) : ball x V ⊆ ball x W :=
  preimage_mono h


theorem ball_inter (x : β) (V W : Set (β × β)) : ball x (V ∩ W) = ball x V ∩ ball x W :=
  preimage_inter


theorem ball_inter_left (x : β) (V W : Set (β × β)) : ball x (V ∩ W) ⊆ ball x V :=
  ball_mono inter_subset_left x


theorem ball_inter_right (x : β) (V W : Set (β × β)) : ball x (V ∩ W) ⊆ ball x W :=
  ball_mono inter_subset_right x


theorem mem_ball_symmetry {V : Set (β × β)} (hV : SymmetricRel V) {x y} :
    x ∈ ball y V ↔ y ∈ ball x V :=
  show (x, y) ∈ Prod.swap ⁻¹' V ↔ (x, y) ∈ V by
    /-
      β : Type ub
      V : Set (Prod β β)
      hV : SymmetricRel V
      x y : β
      ⊢ Iff (Membership.mem (Set.preimage Prod.swap V) { fst := x, snd := y }) (Memb …
    -/
    unfold SymmetricRel at hV
    /-
      β : Type ub
      V : Set (Prod β β)
      hV : Eq (Set.preimage Prod.swap V) V
      x y : β
      ⊢ Iff (Membership.mem (Set.preimage Prod.swap V) { fst := x, snd := y }) (Memb …
    -/
    rw [hV]
    /-
      🎉 no goals
    -/


theorem ball_eq_of_symmetry {V : Set (β × β)} (hV : SymmetricRel V) {x} :
    ball x V = { y | (y, x) ∈ V } := by
  /-
    β : Type ub
    V : Set (Prod β β)
    hV : SymmetricRel V
    x : β
    ⊢ Eq (UniformSpace.ball x V) (setOf fun y => Membership.mem V { fst := y, snd  …
  -/
  ext y
  /-
    case h
    β : Type ub
    V : Set (Prod β β)
    hV : SymmetricRel V
    x y : β
    ⊢ Iff (Membership.mem (UniformSpace.ball x V) y) (Membership.mem (setOf fun y  …
  -/
  rw [mem_ball_symmetry hV]
  /-
    case h
    β : Type ub
    V : Set (Prod β β)
    hV : SymmetricRel V
    x y : β
    ⊢ Iff (Membership.mem (UniformSpace.ball y V) x) (Membership.mem (setOf fun y  …
  -/
  exact Iff.rfl
  /-
    🎉 no goals
  -/


theorem mem_comp_of_mem_ball {V W : Set (β × β)} {x y z : β} (hV : SymmetricRel V)
    (hx : x ∈ ball z V) (hy : y ∈ ball z W) : (x, y) ∈ V ○ W := by
  /-
    β : Type ub
    V W : Set (Prod β β)
    x y z : β
    hV : SymmetricRel V
    hx : Membership.mem (UniformSpace.ball z V) x
    hy : Membership.mem (UniformSpace.ball z W) y
    ⊢ Membership.mem (compRel V W) { fst := x, snd := y }
  -/
  rw [mem_ball_symmetry hV] at hx
  /-
    β : Type ub
    V W : Set (Prod β β)
    x y z : β
    hV : SymmetricRel V
    hx : Membership.mem (UniformSpace.ball x V) z
    hy : Membership.mem (UniformSpace.ball z W) y
    ⊢ Membership.mem (compRel V W) { fst := x, snd := y }
  -/
  exact ⟨z, hx, hy⟩
  /-
    🎉 no goals
  -/


lemma isOpen_ball (x : α) {V : Set (α × α)} (hV : IsOpen V) : IsOpen (ball x V) :=
  hV.preimage <| continuous_const.prod_mk continuous_id


lemma isClosed_ball (x : α) {V : Set (α × α)} (hV : IsClosed V) : IsClosed (ball x V) :=
  hV.preimage <| continuous_const.prod_mk continuous_id


theorem mem_comp_comp {V W M : Set (β × β)} (hW' : SymmetricRel W) {p : β × β} :
    p ∈ V ○ M ○ W ↔ (ball p.1 V ×ˢ ball p.2 W ∩ M).Nonempty := by
  /-
    β : Type ub
    V W M : Set (Prod β β)
    hW' : SymmetricRel W
    p : Prod β β
    ⊢ Iff (Membership.mem (compRel (compRel V M) W) p) (Inter.inter (SProd.sprod ( …
  -/
  cases' p with x y
  /-
    case mk
    β : Type ub
    V W M : Set (Prod β β)
    hW' : SymmetricRel W
    x y : β
    ⊢ Iff (Membership.mem (compRel (compRel V M) W) { fst := x, snd := y }) (Inter …
  -/
  constructor
    /-
      case mk.mp
      β : Type ub
      V W M : Set (Prod β β)
      hW' : SymmetricRel W
      x y : β
      ⊢ Membership.mem (compRel (compRel V M) W) { fst := x, snd := y } → (Inter.int …
    -/
  · rintro ⟨z, ⟨w, hpw, hwz⟩, hzy⟩
    /-
      case mk.mp.intro.intro.intro.intro
      β : Type ub
      V W M : Set (Prod β β)
      hW' : SymmetricRel W
      x y z : β
      hzy : Membership.mem W { fst := z, snd := { fst := x, snd := y }.2 }
      w : β
      hpw : Membership.mem V { fst := { fst := { fst := x, snd := y }.1, snd := z }. …
      hwz : Membership.mem M { fst := w, snd := { fst := { fst := x, snd := y }.1, s …
      ⊢ (Inter.inter (SProd.sprod (UniformSpace.ball { fst := x, snd := y }.1 V) (Un …
    -/
    exact ⟨(w, z), ⟨hpw, by rwa [mem_ball_symmetry hW']⟩, hwz⟩
    /-
      🎉 no goals
    -/
    /-
      case mk.mpr
      β : Type ub
      V W M : Set (Prod β β)
      hW' : SymmetricRel W
      x y : β
      ⊢ (Inter.inter (SProd.sprod (UniformSpace.ball { fst := x, snd := y }.1 V) (Un …
    -/
  · rintro ⟨⟨w, z⟩, ⟨w_in, z_in⟩, hwz⟩
    /-
      case mk.mpr.intro.mk.intro.intro
      β : Type ub
      V W M : Set (Prod β β)
      hW' : SymmetricRel W
      x y w z : β
      hwz : Membership.mem M { fst := w, snd := z }
      w_in : Membership.mem (UniformSpace.ball { fst := x, snd := y }.1 V) { fst :=  …
      z_in : Membership.mem (UniformSpace.ball { fst := x, snd := y }.2 W) { fst :=  …
      ⊢ Membership.mem (compRel (compRel V M) W) { fst := x, snd := y }
    -/
    rw [mem_ball_symmetry hW'] at z_in
    /-
      case mk.mpr.intro.mk.intro.intro
      β : Type ub
      V W M : Set (Prod β β)
      hW' : SymmetricRel W
      x y w z : β
      hwz : Membership.mem M { fst := w, snd := z }
      w_in : Membership.mem (UniformSpace.ball { fst := x, snd := y }.1 V) { fst :=  …
      z_in : Membership.mem (UniformSpace.ball { fst := w, snd := z }.2 W) { fst :=  …
      ⊢ Membership.mem (compRel (compRel V M) W) { fst := x, snd := y }
    -/
    exact ⟨z, ⟨w, w_in, hwz⟩, z_in⟩
    /-
      🎉 no goals
    -/


theorem mem_nhds_uniformity_iff_right {x : α} {s : Set α} :
    s ∈ 𝓝 x ↔ { p : α × α | p.1 = x → p.2 ∈ s } ∈ 𝓤 α := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    x : α
    s : Set α
    ⊢ Iff (Membership.mem (nhds x) s) (Membership.mem (uniformity α) (setOf fun p  …
  -/
  simp only [nhds_eq_comap_uniformity, mem_comap_prod_mk]
  /-
    🎉 no goals
  -/


theorem mem_nhds_uniformity_iff_left {x : α} {s : Set α} :
    s ∈ 𝓝 x ↔ { p : α × α | p.2 = x → p.1 ∈ s } ∈ 𝓤 α := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    x : α
    s : Set α
    ⊢ Iff (Membership.mem (nhds x) s) (Membership.mem (uniformity α) (setOf fun p  …
  -/
  rw [uniformity_eq_symm, mem_nhds_uniformity_iff_right]
  /-
    α : Type ua
    inst✝ : UniformSpace α
    x : α
    s : Set α
    ⊢ Iff (Membership.mem (uniformity α) (setOf fun p => Eq p.1 x → Membership.mem …
  -/
  simp only [map_def, mem_map, preimage_setOf_eq, Prod.snd_swap, Prod.fst_swap]
  /-
    🎉 no goals
  -/


theorem nhdsWithin_eq_comap_uniformity_of_mem {x : α} {T : Set α} (hx : x ∈ T) (S : Set α) :
    𝓝[S] x = (𝓤 α ⊓ 𝓟 (T ×ˢ S)).comap (Prod.mk x) := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    x : α
    T : Set α
    hx : Membership.mem T x
    S : Set α
    ⊢ Eq (nhdsWithin x S) (Filter.comap (Prod.mk x) (Min.min (uniformity α) (Filte …
  -/
  simp [nhdsWithin, nhds_eq_comap_uniformity, hx]
  /-
    🎉 no goals
  -/


theorem nhdsWithin_eq_comap_uniformity {x : α} (S : Set α) :
    𝓝[S] x = (𝓤 α ⊓ 𝓟 (univ ×ˢ S)).comap (Prod.mk x) :=
  nhdsWithin_eq_comap_uniformity_of_mem (mem_univ _) S


/-- See also `isOpen_iff_isOpen_ball_subset`. -/
theorem isOpen_iff_ball_subset {s : Set α} : IsOpen s ↔ ∀ x ∈ s, ∃ V ∈ 𝓤 α, ball x V ⊆ s := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    s : Set α
    ⊢ Iff (IsOpen s) (∀ (x : α), Membership.mem s x → Exists fun V => And (Members …
  -/
  simp_rw [isOpen_iff_mem_nhds, nhds_eq_comap_uniformity, mem_comap, ball]
  /-
    🎉 no goals
  -/


theorem nhds_basis_uniformity' {p : ι → Prop} {s : ι → Set (α × α)} (h : (𝓤 α).HasBasis p s)
    {x : α} : (𝓝 x).HasBasis p fun i => ball x (s i) := by
  /-
    α : Type ua
    ι : Sort u_1
    inst✝ : UniformSpace α
    p : ι → Prop
    s : ι → Set (Prod α α)
    h : (uniformity α).HasBasis p s
    x : α
    ⊢ (nhds x).HasBasis p fun i => UniformSpace.ball x (s i)
  -/
  rw [nhds_eq_comap_uniformity]
  /-
    α : Type ua
    ι : Sort u_1
    inst✝ : UniformSpace α
    p : ι → Prop
    s : ι → Set (Prod α α)
    h : (uniformity α).HasBasis p s
    x : α
    ⊢ (Filter.comap (Prod.mk x) (uniformity α)).HasBasis p fun i => UniformSpace.b …
  -/
  exact h.comap (Prod.mk x)
  /-
    🎉 no goals
  -/


theorem nhds_basis_uniformity {p : ι → Prop} {s : ι → Set (α × α)} (h : (𝓤 α).HasBasis p s)
    {x : α} : (𝓝 x).HasBasis p fun i => { y | (y, x) ∈ s i } := by
  /-
    α : Type ua
    ι : Sort u_1
    inst✝ : UniformSpace α
    p : ι → Prop
    s : ι → Set (Prod α α)
    h : (uniformity α).HasBasis p s
    x : α
    ⊢ (nhds x).HasBasis p fun i => setOf fun y => Membership.mem (s i) { fst := y, …
  -/
  replace h := h.comap Prod.swap
  /-
    α : Type ua
    ι : Sort u_1
    inst✝ : UniformSpace α
    p : ι → Prop
    s : ι → Set (Prod α α)
    x : α
    h : (Filter.comap Prod.swap (uniformity α)).HasBasis p fun i => Set.preimage P …
    ⊢ (nhds x).HasBasis p fun i => setOf fun y => Membership.mem (s i) { fst := y, …
  -/
  rw [comap_swap_uniformity] at h
  /-
    α : Type ua
    ι : Sort u_1
    inst✝ : UniformSpace α
    p : ι → Prop
    s : ι → Set (Prod α α)
    x : α
    h : (uniformity α).HasBasis p fun i => Set.preimage Prod.swap (s i)
    ⊢ (nhds x).HasBasis p fun i => setOf fun y => Membership.mem (s i) { fst := y, …
  -/
  exact nhds_basis_uniformity' h
  /-
    🎉 no goals
  -/


theorem nhds_eq_comap_uniformity' {x : α} : 𝓝 x = (𝓤 α).comap fun y => (y, x) :=
  (nhds_basis_uniformity (𝓤 α).basis_sets).eq_of_same_basis <| (𝓤 α).basis_sets.comap _


theorem UniformSpace.mem_nhds_iff {x : α} {s : Set α} : s ∈ 𝓝 x ↔ ∃ V ∈ 𝓤 α, ball x V ⊆ s := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    x : α
    s : Set α
    ⊢ Iff (Membership.mem (nhds x) s) (Exists fun V => And (Membership.mem (unifor …
  -/
  rw [nhds_eq_comap_uniformity, mem_comap]
  /-
    α : Type ua
    inst✝ : UniformSpace α
    x : α
    s : Set α
    ⊢ Iff (Exists fun t => And (Membership.mem (uniformity α) t) (HasSubset.Subset …
  -/
  simp_rw [ball]
  /-
    🎉 no goals
  -/


theorem UniformSpace.ball_mem_nhds (x : α) ⦃V : Set (α × α)⦄ (V_in : V ∈ 𝓤 α) : ball x V ∈ 𝓝 x := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    x : α
    V : Set (Prod α α)
    V_in : Membership.mem (uniformity α) V
    ⊢ Membership.mem (nhds x) (UniformSpace.ball x V)
  -/
  rw [UniformSpace.mem_nhds_iff]
  /-
    α : Type ua
    inst✝ : UniformSpace α
    x : α
    V : Set (Prod α α)
    V_in : Membership.mem (uniformity α) V
    ⊢ Exists fun V_1 => And (Membership.mem (uniformity α) V_1) (HasSubset.Subset  …
  -/
  exact ⟨V, V_in, Subset.rfl⟩
  /-
    🎉 no goals
  -/


theorem UniformSpace.ball_mem_nhdsWithin {x : α} {S : Set α} ⦃V : Set (α × α)⦄ (x_in : x ∈ S)
    (V_in : V ∈ 𝓤 α ⊓ 𝓟 (S ×ˢ S)) : ball x V ∈ 𝓝[S] x := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    x : α
    S : Set α
    V : Set (Prod α α)
    x_in : Membership.mem S x
    V_in : Membership.mem (Min.min (uniformity α) (Filter.principal (SProd.sprod S …
    ⊢ Membership.mem (nhdsWithin x S) (UniformSpace.ball x V)
  -/
  rw [nhdsWithin_eq_comap_uniformity_of_mem x_in, mem_comap]
  /-
    α : Type ua
    inst✝ : UniformSpace α
    x : α
    S : Set α
    V : Set (Prod α α)
    x_in : Membership.mem S x
    V_in : Membership.mem (Min.min (uniformity α) (Filter.principal (SProd.sprod S …
    ⊢ Exists fun t => And (Membership.mem (Min.min (uniformity α) (Filter.principa …
  -/
  exact ⟨V, V_in, Subset.rfl⟩
  /-
    🎉 no goals
  -/


theorem UniformSpace.mem_nhds_iff_symm {x : α} {s : Set α} :
    s ∈ 𝓝 x ↔ ∃ V ∈ 𝓤 α, SymmetricRel V ∧ ball x V ⊆ s := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    x : α
    s : Set α
    ⊢ Iff (Membership.mem (nhds x) s) (Exists fun V => And (Membership.mem (unifor …
  -/
  rw [UniformSpace.mem_nhds_iff]
  /-
    α : Type ua
    inst✝ : UniformSpace α
    x : α
    s : Set α
    ⊢ Iff (Exists fun V => And (Membership.mem (uniformity α) V) (HasSubset.Subset …
  -/
  constructor
    /-
      case mp
      α : Type ua
      inst✝ : UniformSpace α
      x : α
      s : Set α
      ⊢ (Exists fun V => And (Membership.mem (uniformity α) V) (HasSubset.Subset (Un …
    -/
  · rintro ⟨V, V_in, V_sub⟩
    /-
      case mp.intro.intro
      α : Type ua
      inst✝ : UniformSpace α
      x : α
      s : Set α
      V : Set (Prod α α)
      V_in : Membership.mem (uniformity α) V
      V_sub : HasSubset.Subset (UniformSpace.ball x V) s
      ⊢ Exists fun V => And (Membership.mem (uniformity α) V) (And (SymmetricRel V)  …
    -/
    use symmetrizeRel V, symmetrize_mem_uniformity V_in, symmetric_symmetrizeRel V
    /-
      case right
      α : Type ua
      inst✝ : UniformSpace α
      x : α
      s : Set α
      V : Set (Prod α α)
      V_in : Membership.mem (uniformity α) V
      V_sub : HasSubset.Subset (UniformSpace.ball x V) s
      ⊢ HasSubset.Subset (UniformSpace.ball x (symmetrizeRel V)) s
    -/
    exact Subset.trans (ball_mono (symmetrizeRel_subset_self V) x) V_sub
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type ua
      inst✝ : UniformSpace α
      x : α
      s : Set α
      ⊢ (Exists fun V => And (Membership.mem (uniformity α) V) (And (SymmetricRel V) …
    -/
  · rintro ⟨V, V_in, _, V_sub⟩
    /-
      case mpr.intro.intro.intro
      α : Type ua
      inst✝ : UniformSpace α
      x : α
      s : Set α
      V : Set (Prod α α)
      V_in : Membership.mem (uniformity α) V
      left✝ : SymmetricRel V
      V_sub : HasSubset.Subset (UniformSpace.ball x V) s
      ⊢ Exists fun V => And (Membership.mem (uniformity α) V) (HasSubset.Subset (Uni …
    -/
    exact ⟨V, V_in, V_sub⟩
    /-
      🎉 no goals
    -/


theorem UniformSpace.hasBasis_nhds (x : α) :
    HasBasis (𝓝 x) (fun s : Set (α × α) => s ∈ 𝓤 α ∧ SymmetricRel s) fun s => ball x s :=
               /-
                 α : Type ua
                 inst✝ : UniformSpace α
                 x : α
                 t : Set α
                 ⊢ Iff (Membership.mem (nhds x) t) (Exists fun i => And (And (Membership.mem (u …
               -/
  ⟨fun t => by simp [UniformSpace.mem_nhds_iff_symm, and_assoc]⟩
               /-
                 🎉 no goals
               -/


theorem UniformSpace.mem_closure_iff_symm_ball {s : Set α} {x} :
    x ∈ closure s ↔ ∀ {V}, V ∈ 𝓤 α → SymmetricRel V → (s ∩ ball x V).Nonempty := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    s : Set α
    x : α
    ⊢ Iff (Membership.mem (closure s) x) (∀ {V : Set (Prod α α)}, Membership.mem ( …
  -/
  simp [mem_closure_iff_nhds_basis (hasBasis_nhds x), Set.Nonempty]
  /-
    🎉 no goals
  -/


theorem UniformSpace.mem_closure_iff_ball {s : Set α} {x} :
    x ∈ closure s ↔ ∀ {V}, V ∈ 𝓤 α → (ball x V ∩ s).Nonempty := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    s : Set α
    x : α
    ⊢ Iff (Membership.mem (closure s) x) (∀ {V : Set (Prod α α)}, Membership.mem ( …
  -/
  simp [mem_closure_iff_nhds_basis' (nhds_basis_uniformity' (𝓤 α).basis_sets)]
  /-
    🎉 no goals
  -/


theorem UniformSpace.hasBasis_nhds_prod (x y : α) :
    HasBasis (𝓝 (x, y)) (fun s => s ∈ 𝓤 α ∧ SymmetricRel s) fun s => ball x s ×ˢ ball y s := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    x y : α
    ⊢ (nhds { fst := x, snd := y }).HasBasis (fun s => And (Membership.mem (unifor …
  -/
  rw [nhds_prod_eq]
  /-
    α : Type ua
    inst✝ : UniformSpace α
    x y : α
    ⊢ (SProd.sprod (nhds x) (nhds y)).HasBasis (fun s => And (Membership.mem (unif …
  -/
  apply (hasBasis_nhds x).prod_same_index (hasBasis_nhds y)
  /-
    α : Type ua
    inst✝ : UniformSpace α
    x y : α
    ⊢ ∀ {i j : Set (Prod α α)}, And (Membership.mem (uniformity α) i) (SymmetricRe …
  -/
  rintro U V ⟨U_in, U_symm⟩ ⟨V_in, V_symm⟩
  exact
    ⟨U ∩ V, ⟨(𝓤 α).inter_sets U_in V_in, U_symm.inter V_symm⟩, ball_inter_left x U V,
      ball_inter_right y U V⟩


theorem nhds_eq_uniformity {x : α} : 𝓝 x = (𝓤 α).lift' (ball x) :=
  (nhds_basis_uniformity' (𝓤 α).basis_sets).eq_biInf


theorem nhds_eq_uniformity' {x : α} : 𝓝 x = (𝓤 α).lift' fun s => { y | (y, x) ∈ s } :=
  (nhds_basis_uniformity (𝓤 α).basis_sets).eq_biInf


theorem mem_nhds_left (x : α) {s : Set (α × α)} (h : s ∈ 𝓤 α) : { y : α | (x, y) ∈ s } ∈ 𝓝 x :=
  ball_mem_nhds x h


theorem mem_nhds_right (y : α) {s : Set (α × α)} (h : s ∈ 𝓤 α) : { x : α | (x, y) ∈ s } ∈ 𝓝 y :=
  mem_nhds_left _ (symm_le_uniformity h)


theorem exists_mem_nhds_ball_subset_of_mem_nhds {a : α} {U : Set α} (h : U ∈ 𝓝 a) :
    ∃ V ∈ 𝓝 a, ∃ t ∈ 𝓤 α, ∀ a' ∈ V, UniformSpace.ball a' t ⊆ U :=
  let ⟨t, ht, htU⟩ := comp_mem_uniformity_sets (mem_nhds_uniformity_iff_right.1 h)
  ⟨_, mem_nhds_left a ht, t, ht, fun a₁ h₁ a₂ h₂ => @htU (a, a₂) ⟨a₁, h₁, h₂⟩ rfl⟩


theorem tendsto_right_nhds_uniformity {a : α} : Tendsto (fun a' => (a', a)) (𝓝 a) (𝓤 α) := fun _ =>
  mem_nhds_right a


theorem tendsto_left_nhds_uniformity {a : α} : Tendsto (fun a' => (a, a')) (𝓝 a) (𝓤 α) := fun _ =>
  mem_nhds_left a


theorem lift_nhds_left {x : α} {g : Set α → Filter β} (hg : Monotone g) :
    (𝓝 x).lift g = (𝓤 α).lift fun s : Set (α × α) => g (ball x s) := by
  /-
    α : Type ua
    β : Type ub
    inst✝ : UniformSpace α
    x : α
    g : Set α → Filter β
    hg : Monotone g
    ⊢ Eq ((nhds x).lift g) ((uniformity α).lift fun s => g (UniformSpace.ball x s))
  -/
  rw [nhds_eq_comap_uniformity, comap_lift_eq2 hg]
  /-
    α : Type ua
    β : Type ub
    inst✝ : UniformSpace α
    x : α
    g : Set α → Filter β
    hg : Monotone g
    ⊢ Eq ((uniformity α).lift (Function.comp g (Set.preimage (Prod.mk x)))) ((unif …
  -/
  simp_rw [ball, Function.comp_def]
  /-
    🎉 no goals
  -/


theorem lift_nhds_right {x : α} {g : Set α → Filter β} (hg : Monotone g) :
    (𝓝 x).lift g = (𝓤 α).lift fun s : Set (α × α) => g { y | (y, x) ∈ s } := by
  /-
    α : Type ua
    β : Type ub
    inst✝ : UniformSpace α
    x : α
    g : Set α → Filter β
    hg : Monotone g
    ⊢ Eq ((nhds x).lift g) ((uniformity α).lift fun s => g (setOf fun y => Members …
  -/
  rw [nhds_eq_comap_uniformity', comap_lift_eq2 hg]
  /-
    α : Type ua
    β : Type ub
    inst✝ : UniformSpace α
    x : α
    g : Set α → Filter β
    hg : Monotone g
    ⊢ Eq ((uniformity α).lift (Function.comp g (Set.preimage fun y => { fst := y,  …
  -/
  simp_rw [Function.comp_def, preimage]
  /-
    🎉 no goals
  -/


theorem nhds_nhds_eq_uniformity_uniformity_prod {a b : α} :
    𝓝 a ×ˢ 𝓝 b = (𝓤 α).lift fun s : Set (α × α) =>
      (𝓤 α).lift' fun t => { y : α | (y, a) ∈ s } ×ˢ { y : α | (b, y) ∈ t } := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    a b : α
    ⊢ Eq (SProd.sprod (nhds a) (nhds b)) ((uniformity α).lift fun s => (uniformity …
  -/
  rw [nhds_eq_uniformity', nhds_eq_uniformity, prod_lift'_lift']
  /-
    α : Type ua
    inst✝ : UniformSpace α
    a b : α
    ⊢ Eq ((uniformity α).lift fun s => (uniformity α).lift' fun t => SProd.sprod ( …
  -/
  exacts [rfl, monotone_preimage, monotone_preimage]
  /-
    🎉 no goals
  -/


theorem nhds_eq_uniformity_prod {a b : α} :
    𝓝 (a, b) =
      (𝓤 α).lift' fun s : Set (α × α) => { y : α | (y, a) ∈ s } ×ˢ { y : α | (b, y) ∈ s } := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    a b : α
    ⊢ Eq (nhds { fst := a, snd := b }) ((uniformity α).lift' fun s => SProd.sprod  …
  -/
  rw [nhds_prod_eq, nhds_nhds_eq_uniformity_uniformity_prod, lift_lift'_same_eq_lift']
    /-
      case hg₁
      α : Type ua
      inst✝ : UniformSpace α
      a b : α
      ⊢ ∀ (s : Set (Prod α α)), Monotone fun t => SProd.sprod (setOf fun y => Member …
    -/
  · exact fun s => monotone_const.set_prod monotone_preimage
    /-
      🎉 no goals
    -/
    /-
      case hg₂
      α : Type ua
      inst✝ : UniformSpace α
      a b : α
      ⊢ ∀ (t : Set (Prod α α)), Monotone fun s => SProd.sprod (setOf fun y => Member …
    -/
  · refine fun t => Monotone.set_prod ?_ monotone_const
    /-
      case hg₂
      α : Type ua
      inst✝ : UniformSpace α
      a b : α
      t : Set (Prod α α)
      ⊢ Monotone fun s => setOf fun y => Membership.mem s { fst := y, snd := a }
    -/
    exact monotone_preimage (f := fun y => (y, a))
    /-
      🎉 no goals
    -/


theorem nhdset_of_mem_uniformity {d : Set (α × α)} (s : Set (α × α)) (hd : d ∈ 𝓤 α) :
    ∃ t : Set (α × α), IsOpen t ∧ s ⊆ t ∧
      t ⊆ { p | ∃ x y, (p.1, x) ∈ d ∧ (x, y) ∈ s ∧ (y, p.2) ∈ d } := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    d s : Set (Prod α α)
    hd : Membership.mem (uniformity α) d
    ⊢ Exists fun t => And (IsOpen t) (And (HasSubset.Subset s t) (HasSubset.Subset …
  -/
  let cl_d := { p : α × α | ∃ x y, (p.1, x) ∈ d ∧ (x, y) ∈ s ∧ (y, p.2) ∈ d }
  have : ∀ p ∈ s, ∃ t, t ⊆ cl_d ∧ IsOpen t ∧ p ∈ t := fun ⟨x, y⟩ hp =>
    mem_nhds_iff.mp <|
      show cl_d ∈ 𝓝 (x, y) by
        rw [nhds_eq_uniformity_prod, mem_lift'_sets]
        · exact ⟨d, hd, fun ⟨a, b⟩ ⟨ha, hb⟩ => ⟨x, y, ha, hp, hb⟩⟩
        · exact fun _ _ h _ h' => ⟨h h'.1, h h'.2⟩
  /-
    α : Type ua
    inst✝ : UniformSpace α
    d s : Set (Prod α α)
    hd : Membership.mem (uniformity α) d
    cl_d : Set (Prod α α) := setOf fun p => Exists fun x => Exists fun y => And (M …
    this : ∀ (p : Prod α α), Membership.mem s p → Exists fun t => And (HasSubset.S …
    ⊢ Exists fun t => And (IsOpen t) (And (HasSubset.Subset s t) (HasSubset.Subset …
  -/
  choose t ht using this
  exact ⟨(⋃ p : α × α, ⋃ h : p ∈ s, t p h : Set (α × α)),
    isOpen_iUnion fun p : α × α => isOpen_iUnion fun hp => (ht p hp).right.left,
    fun ⟨a, b⟩ hp => by
      simp only [mem_iUnion, Prod.exists]; exact ⟨a, b, hp, (ht (a, b) hp).right.right⟩,
    iUnion_subset fun p => iUnion_subset fun hp => (ht p hp).left⟩


/-- Entourages are neighborhoods of the diagonal. -/
theorem nhds_le_uniformity (x : α) : 𝓝 (x, x) ≤ 𝓤 α := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    x : α
    ⊢ LE.le (nhds { fst := x, snd := x }) (uniformity α)
  -/
  intro V V_in
  /-
    α : Type ua
    inst✝ : UniformSpace α
    x : α
    V : Set (Prod α α)
    V_in : Membership.mem (uniformity α) V
    ⊢ Membership.mem (nhds { fst := x, snd := x }) V
  -/
  rcases comp_symm_mem_uniformity_sets V_in with ⟨w, w_in, w_symm, w_sub⟩
  have : ball x w ×ˢ ball x w ∈ 𝓝 (x, x) := by
    rw [nhds_prod_eq]
    exact prod_mem_prod (ball_mem_nhds x w_in) (ball_mem_nhds x w_in)
  /-
    case intro.intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    x : α
    V : Set (Prod α α)
    V_in : Membership.mem (uniformity α) V
    w : Set (Prod α α)
    w_in : Membership.mem (uniformity α) w
    w_symm : SymmetricRel w
    w_sub : HasSubset.Subset (compRel w w) V
    this : Membership.mem (nhds { fst := x, snd := x }) (SProd.sprod (UniformSpace …
    ⊢ Membership.mem (nhds { fst := x, snd := x }) V
  -/
  apply mem_of_superset this
  /-
    case intro.intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    x : α
    V : Set (Prod α α)
    V_in : Membership.mem (uniformity α) V
    w : Set (Prod α α)
    w_in : Membership.mem (uniformity α) w
    w_symm : SymmetricRel w
    w_sub : HasSubset.Subset (compRel w w) V
    this : Membership.mem (nhds { fst := x, snd := x }) (SProd.sprod (UniformSpace …
    ⊢ HasSubset.Subset (SProd.sprod (UniformSpace.ball x w) (UniformSpace.ball x w …
  -/
  rintro ⟨u, v⟩ ⟨u_in, v_in⟩
  /-
    case intro.intro.intro.mk.intro
    α : Type ua
    inst✝ : UniformSpace α
    x : α
    V : Set (Prod α α)
    V_in : Membership.mem (uniformity α) V
    w : Set (Prod α α)
    w_in : Membership.mem (uniformity α) w
    w_symm : SymmetricRel w
    w_sub : HasSubset.Subset (compRel w w) V
    this : Membership.mem (nhds { fst := x, snd := x }) (SProd.sprod (UniformSpace …
    u v : α
    u_in : Membership.mem (UniformSpace.ball x w) { fst := u, snd := v }.1
    v_in : Membership.mem (UniformSpace.ball x w) { fst := u, snd := v }.2
    ⊢ Membership.mem V { fst := u, snd := v }
  -/
  exact w_sub (mem_comp_of_mem_ball w_symm u_in v_in)
  /-
    🎉 no goals
  -/


/-- Entourages are neighborhoods of the diagonal. -/
theorem iSup_nhds_le_uniformity : ⨆ x : α, 𝓝 (x, x) ≤ 𝓤 α :=
  iSup_le nhds_le_uniformity


/-- Entourages are neighborhoods of the diagonal. -/
theorem nhdsSet_diagonal_le_uniformity : 𝓝ˢ (diagonal α) ≤ 𝓤 α :=
  (nhdsSet_diagonal α).trans_le iSup_nhds_le_uniformity


theorem closure_eq_uniformity (s : Set <| α × α) :
    closure s = ⋂ V ∈ { V | V ∈ 𝓤 α ∧ SymmetricRel V }, V ○ s ○ V := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    s : Set (Prod α α)
    ⊢ Eq (closure s) (Set.iInter fun V => Set.iInter fun h => compRel (compRel V s …
  -/
  ext ⟨x, y⟩
  simp +contextual only
    [mem_closure_iff_nhds_basis (UniformSpace.hasBasis_nhds_prod x y), mem_iInter, mem_setOf_eq,
      and_imp, mem_comp_comp, exists_prop, ← mem_inter_iff, inter_comm, Set.Nonempty]


theorem uniformity_hasBasis_closed :
    HasBasis (𝓤 α) (fun V : Set (α × α) => V ∈ 𝓤 α ∧ IsClosed V) id := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    ⊢ (uniformity α).HasBasis (fun V => And (Membership.mem (uniformity α) V) (IsC …
  -/
  refine Filter.hasBasis_self.2 fun t h => ?_
  /-
    α : Type ua
    inst✝ : UniformSpace α
    t : Set (Prod α α)
    h : Membership.mem (uniformity α) t
    ⊢ Exists fun r => And (Membership.mem (uniformity α) r) (And (IsClosed r) (Has …
  -/
  rcases comp_comp_symm_mem_uniformity_sets h with ⟨w, w_in, w_symm, r⟩
  /-
    case intro.intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    t : Set (Prod α α)
    h : Membership.mem (uniformity α) t
    w : Set (Prod α α)
    w_in : Membership.mem (uniformity α) w
    w_symm : SymmetricRel w
    r : HasSubset.Subset (compRel (compRel w w) w) t
    ⊢ Exists fun r => And (Membership.mem (uniformity α) r) (And (IsClosed r) (Has …
  -/
  refine ⟨closure w, mem_of_superset w_in subset_closure, isClosed_closure, ?_⟩
  /-
    case intro.intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    t : Set (Prod α α)
    h : Membership.mem (uniformity α) t
    w : Set (Prod α α)
    w_in : Membership.mem (uniformity α) w
    w_symm : SymmetricRel w
    r : HasSubset.Subset (compRel (compRel w w) w) t
    ⊢ HasSubset.Subset (closure w) t
  -/
  refine Subset.trans ?_ r
  /-
    case intro.intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    t : Set (Prod α α)
    h : Membership.mem (uniformity α) t
    w : Set (Prod α α)
    w_in : Membership.mem (uniformity α) w
    w_symm : SymmetricRel w
    r : HasSubset.Subset (compRel (compRel w w) w) t
    ⊢ HasSubset.Subset (closure w) (compRel (compRel w w) w)
  -/
  rw [closure_eq_uniformity]
  /-
    case intro.intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    t : Set (Prod α α)
    h : Membership.mem (uniformity α) t
    w : Set (Prod α α)
    w_in : Membership.mem (uniformity α) w
    w_symm : SymmetricRel w
    r : HasSubset.Subset (compRel (compRel w w) w) t
    ⊢ HasSubset.Subset (Set.iInter fun V => Set.iInter fun h => compRel (compRel V …
  -/
  apply iInter_subset_of_subset
  /-
    case intro.intro.intro.h
    α : Type ua
    inst✝ : UniformSpace α
    t : Set (Prod α α)
    h : Membership.mem (uniformity α) t
    w : Set (Prod α α)
    w_in : Membership.mem (uniformity α) w
    w_symm : SymmetricRel w
    r : HasSubset.Subset (compRel (compRel w w) w) t
    ⊢ HasSubset.Subset (Set.iInter fun h => compRel (compRel ?intro.intro.intro.i  …
  -/
  apply iInter_subset
  /-
    case intro.intro.intro.h.i
    α : Type ua
    inst✝ : UniformSpace α
    t : Set (Prod α α)
    h : Membership.mem (uniformity α) t
    w : Set (Prod α α)
    w_in : Membership.mem (uniformity α) w
    w_symm : SymmetricRel w
    r : HasSubset.Subset (compRel (compRel w w) w) t
    ⊢ Membership.mem (setOf fun V => And (Membership.mem (uniformity α) V) (Symmet …
  -/
  exact ⟨w_in, w_symm⟩
  /-
    🎉 no goals
  -/


theorem uniformity_eq_uniformity_closure : 𝓤 α = (𝓤 α).lift' closure :=
  Eq.symm <| uniformity_hasBasis_closed.lift'_closure_eq_self fun _ => And.right


theorem Filter.HasBasis.uniformity_closure {p : ι → Prop} {U : ι → Set (α × α)}
    (h : (𝓤 α).HasBasis p U) : (𝓤 α).HasBasis p fun i => closure (U i) :=
  (@uniformity_eq_uniformity_closure α _).symm ▸ h.lift'_closure


/-- Closed entourages form a basis of the uniformity filter. -/
theorem uniformity_hasBasis_closure : HasBasis (𝓤 α) (fun V : Set (α × α) => V ∈ 𝓤 α) closure :=
  (𝓤 α).basis_sets.uniformity_closure


theorem closure_eq_inter_uniformity {t : Set (α × α)} : closure t = ⋂ d ∈ 𝓤 α, d ○ (t ○ d) :=
  calc
    closure t = ⋂ (V) (_ : V ∈ 𝓤 α ∧ SymmetricRel V), V ○ t ○ V := closure_eq_uniformity t
    _ = ⋂ V ∈ 𝓤 α, V ○ t ○ V :=
      Eq.symm <|
        UniformSpace.hasBasis_symmetric.biInter_mem fun _ _ hV =>
          compRel_mono (compRel_mono hV Subset.rfl) hV
                                     /-
                                       α : Type ua
                                       inst✝ : UniformSpace α
                                       t : Set (Prod α α)
                                       ⊢ Eq (Set.iInter fun V => Set.iInter fun h => compRel (compRel V t) V) (Set.iI …
                                     -/
    _ = ⋂ V ∈ 𝓤 α, V ○ (t ○ V) := by simp only [compRel_assoc]
                                     /-
                                       🎉 no goals
                                     -/


theorem uniformity_eq_uniformity_interior : 𝓤 α = (𝓤 α).lift' interior :=
  le_antisymm
    (le_iInf₂ fun d hd => by
      /-
        α : Type ua
        inst✝ : UniformSpace α
        d : Set (Prod α α)
        hd : Membership.mem (uniformity α) d
        ⊢ LE.le (uniformity α) (Function.comp Filter.principal interior d)
      -/
      let ⟨s, hs, hs_comp⟩ := comp3_mem_uniformity hd
      /-
        α : Type ua
        inst✝ : UniformSpace α
        d : Set (Prod α α)
        hd : Membership.mem (uniformity α) d
        s : Set (Prod α α)
        hs : Membership.mem (uniformity α) s
        hs_comp : HasSubset.Subset (compRel s (compRel s s)) d
        ⊢ LE.le (uniformity α) (Function.comp Filter.principal interior d)
      -/
      let ⟨t, ht, hst, ht_comp⟩ := nhdset_of_mem_uniformity s hs
      have : s ⊆ interior d :=
        calc
          s ⊆ t := hst
          _ ⊆ interior d :=
            ht.subset_interior_iff.mpr fun x (hx : x ∈ t) =>
              let ⟨x, y, h₁, h₂, h₃⟩ := ht_comp hx
              hs_comp ⟨x, h₁, y, h₂, h₃⟩
      /-
        α : Type ua
        inst✝ : UniformSpace α
        d : Set (Prod α α)
        hd : Membership.mem (uniformity α) d
        s : Set (Prod α α)
        hs : Membership.mem (uniformity α) s
        hs_comp : HasSubset.Subset (compRel s (compRel s s)) d
        t : Set (Prod α α)
        ht : IsOpen t
        hst : HasSubset.Subset s t
        ht_comp : HasSubset.Subset t (setOf fun p => Exists fun x => Exists fun y => A …
        this : HasSubset.Subset s (interior d)
        ⊢ LE.le (uniformity α) (Function.comp Filter.principal interior d)
      -/
      have : interior d ∈ 𝓤 α := by filter_upwards [hs] using this
      /-
        α : Type ua
        inst✝ : UniformSpace α
        d : Set (Prod α α)
        hd : Membership.mem (uniformity α) d
        s : Set (Prod α α)
        hs : Membership.mem (uniformity α) s
        hs_comp : HasSubset.Subset (compRel s (compRel s s)) d
        t : Set (Prod α α)
        ht : IsOpen t
        hst : HasSubset.Subset s t
        ht_comp : HasSubset.Subset t (setOf fun p => Exists fun x => Exists fun y => A …
        this✝ : HasSubset.Subset s (interior d)
        this : Membership.mem (uniformity α) (interior d)
        ⊢ LE.le (uniformity α) (Function.comp Filter.principal interior d)
      -/
      simp [this])
      /-
        🎉 no goals
      -/
    fun _ hs => ((𝓤 α).lift' interior).sets_of_superset (mem_lift' hs) interior_subset


theorem interior_mem_uniformity {s : Set (α × α)} (hs : s ∈ 𝓤 α) : interior s ∈ 𝓤 α := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    ⊢ Membership.mem (uniformity α) (interior s)
  -/
  rw [uniformity_eq_uniformity_interior]; exact mem_lift' hs
                                          /-
                                            🎉 no goals
                                          -/


theorem mem_uniformity_isClosed {s : Set (α × α)} (h : s ∈ 𝓤 α) : ∃ t ∈ 𝓤 α, IsClosed t ∧ t ⊆ s :=
  let ⟨t, ⟨ht_mem, htc⟩, hts⟩ := uniformity_hasBasis_closed.mem_iff.1 h
  ⟨t, ht_mem, htc, hts⟩


theorem isOpen_iff_isOpen_ball_subset {s : Set α} :
    IsOpen s ↔ ∀ x ∈ s, ∃ V ∈ 𝓤 α, IsOpen V ∧ ball x V ⊆ s := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    s : Set α
    ⊢ Iff (IsOpen s) (∀ (x : α), Membership.mem s x → Exists fun V => And (Members …
  -/
  rw [isOpen_iff_ball_subset]
  /-
    α : Type ua
    inst✝ : UniformSpace α
    s : Set α
    ⊢ Iff (∀ (x : α), Membership.mem s x → Exists fun V => And (Membership.mem (un …
  -/
  constructor <;> intro h x hx
    /-
      case mp
      α : Type ua
      inst✝ : UniformSpace α
      s : Set α
      h : ∀ (x : α), Membership.mem s x → Exists fun V => And (Membership.mem (unifo …
      x : α
      hx : Membership.mem s x
      ⊢ Exists fun V => And (Membership.mem (uniformity α) V) (And (IsOpen V) (HasSu …
    -/
  · obtain ⟨V, hV, hV'⟩ := h x hx
    exact
      ⟨interior V, interior_mem_uniformity hV, isOpen_interior,
        (ball_mono interior_subset x).trans hV'⟩
    /-
      case mpr
      α : Type ua
      inst✝ : UniformSpace α
      s : Set α
      h : ∀ (x : α), Membership.mem s x → Exists fun V => And (Membership.mem (unifo …
      x : α
      hx : Membership.mem s x
      ⊢ Exists fun V => And (Membership.mem (uniformity α) V) (HasSubset.Subset (Uni …
    -/
  · obtain ⟨V, hV, -, hV'⟩ := h x hx
    /-
      case mpr.intro.intro.intro
      α : Type ua
      inst✝ : UniformSpace α
      s : Set α
      h : ∀ (x : α), Membership.mem s x → Exists fun V => And (Membership.mem (unifo …
      x : α
      hx : Membership.mem s x
      V : Set (Prod α α)
      hV : Membership.mem (uniformity α) V
      hV' : HasSubset.Subset (UniformSpace.ball x V) s
      ⊢ Exists fun V => And (Membership.mem (uniformity α) V) (HasSubset.Subset (Uni …
    -/
    exact ⟨V, hV, hV'⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-18")] alias
isOpen_iff_open_ball_subset := isOpen_iff_isOpen_ball_subset


/-- The uniform neighborhoods of all points of a dense set cover the whole space. -/
theorem Dense.biUnion_uniformity_ball {s : Set α} {U : Set (α × α)} (hs : Dense s) (hU : U ∈ 𝓤 α) :
    ⋃ x ∈ s, ball x U = univ := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    s : Set α
    U : Set (Prod α α)
    hs : Dense s
    hU : Membership.mem (uniformity α) U
    ⊢ Eq (Set.iUnion fun x => Set.iUnion fun h => UniformSpace.ball x U) Set.univ
  -/
  refine iUnion₂_eq_univ_iff.2 fun y => ?_
  /-
    α : Type ua
    inst✝ : UniformSpace α
    s : Set α
    U : Set (Prod α α)
    hs : Dense s
    hU : Membership.mem (uniformity α) U
    y : α
    ⊢ Exists fun i => Exists fun j => Membership.mem (UniformSpace.ball i U) y
  -/
  rcases hs.inter_nhds_nonempty (mem_nhds_right y hU) with ⟨x, hxs, hxy : (x, y) ∈ U⟩
  /-
    case intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    s : Set α
    U : Set (Prod α α)
    hs : Dense s
    hU : Membership.mem (uniformity α) U
    y x : α
    hxs : Membership.mem s x
    hxy : Membership.mem U { fst := x, snd := y }
    ⊢ Exists fun i => Exists fun j => Membership.mem (UniformSpace.ball i U) y
  -/
  exact ⟨x, hxs, hxy⟩
  /-
    🎉 no goals
  -/


/-- The uniform neighborhoods of all points of a dense indexed collection cover the whole space. -/
lemma DenseRange.iUnion_uniformity_ball {ι : Type*} {xs : ι → α}
    (xs_dense : DenseRange xs) {U : Set (α × α)} (hU : U ∈ uniformity α) :
    ⋃ i, UniformSpace.ball (xs i) U = univ := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    ι : Type u_2
    xs : ι → α
    xs_dense : DenseRange xs
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    ⊢ Eq (Set.iUnion fun i => UniformSpace.ball (xs i) U) Set.univ
  -/
  rw [← biUnion_range (f := xs) (g := fun x ↦ UniformSpace.ball x U)]
  /-
    α : Type ua
    inst✝ : UniformSpace α
    ι : Type u_2
    xs : ι → α
    xs_dense : DenseRange xs
    U : Set (Prod α α)
    hU : Membership.mem (uniformity α) U
    ⊢ Eq (Set.iUnion fun x => Set.iUnion fun h => UniformSpace.ball x U) Set.univ
  -/
  exact Dense.biUnion_uniformity_ball xs_dense hU
  /-
    🎉 no goals
  -/


/-- Open elements of `𝓤 α` form a basis of `𝓤 α`. -/
theorem uniformity_hasBasis_open : HasBasis (𝓤 α) (fun V : Set (α × α) => V ∈ 𝓤 α ∧ IsOpen V) id :=
  hasBasis_self.2 fun s hs =>
    ⟨interior s, interior_mem_uniformity hs, isOpen_interior, interior_subset⟩


theorem Filter.HasBasis.mem_uniformity_iff {p : β → Prop} {s : β → Set (α × α)}
    (h : (𝓤 α).HasBasis p s) {t : Set (α × α)} :
    t ∈ 𝓤 α ↔ ∃ i, p i ∧ ∀ a b, (a, b) ∈ s i → (a, b) ∈ t :=
                        /-
                          α : Type ua
                          β : Type ub
                          inst✝ : UniformSpace α
                          p : β → Prop
                          s : β → Set (Prod α α)
                          h : (uniformity α).HasBasis p s
                          t : Set (Prod α α)
                          ⊢ Iff (Exists fun i => And (p i) (HasSubset.Subset (s i) t)) (Exists fun i =>  …
                        -/
  h.mem_iff.trans <| by simp only [Prod.forall, subset_def]
                        /-
                          🎉 no goals
                        -/


/-- Open elements `s : Set (α × α)` of `𝓤 α` such that `(x, y) ∈ s ↔ (y, x) ∈ s` form a basis
of `𝓤 α`. -/
theorem uniformity_hasBasis_open_symmetric :
    HasBasis (𝓤 α) (fun V : Set (α × α) => V ∈ 𝓤 α ∧ IsOpen V ∧ SymmetricRel V) id := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    ⊢ (uniformity α).HasBasis (fun V => And (Membership.mem (uniformity α) V) (And …
  -/
  simp only [← and_assoc]
  /-
    α : Type ua
    inst✝ : UniformSpace α
    ⊢ (uniformity α).HasBasis (fun V => And (And (Membership.mem (uniformity α) V) …
  -/
  refine uniformity_hasBasis_open.restrict fun s hs => ⟨symmetrizeRel s, ?_⟩
  exact
    ⟨⟨symmetrize_mem_uniformity hs.1, IsOpen.inter hs.2 (hs.2.preimage continuous_swap)⟩,
      symmetric_symmetrizeRel s, symmetrizeRel_subset_self s⟩


theorem comp_open_symm_mem_uniformity_sets {s : Set (α × α)} (hs : s ∈ 𝓤 α) :
    ∃ t ∈ 𝓤 α, IsOpen t ∧ SymmetricRel t ∧ t ○ t ⊆ s := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    ⊢ Exists fun t => And (Membership.mem (uniformity α) t) (And (IsOpen t) (And ( …
  -/
  obtain ⟨t, ht₁, ht₂⟩ := comp_mem_uniformity_sets hs
  /-
    case intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    t : Set (Prod α α)
    ht₁ : Membership.mem (uniformity α) t
    ht₂ : HasSubset.Subset (compRel t t) s
    ⊢ Exists fun t => And (Membership.mem (uniformity α) t) (And (IsOpen t) (And ( …
  -/
  obtain ⟨u, ⟨hu₁, hu₂, hu₃⟩, hu₄ : u ⊆ t⟩ := uniformity_hasBasis_open_symmetric.mem_iff.mp ht₁
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type ua
    inst✝ : UniformSpace α
    s : Set (Prod α α)
    hs : Membership.mem (uniformity α) s
    t : Set (Prod α α)
    ht₁ : Membership.mem (uniformity α) t
    ht₂ : HasSubset.Subset (compRel t t) s
    u : Set (Prod α α)
    hu₄ : HasSubset.Subset u t
    hu₁ : Membership.mem (uniformity α) u
    hu₂ : IsOpen u
    hu₃ : SymmetricRel u
    ⊢ Exists fun t => And (Membership.mem (uniformity α) t) (And (IsOpen t) (And ( …
  -/
  exact ⟨u, hu₁, hu₂, hu₃, (compRel_mono hu₄ hu₄).trans ht₂⟩
  /-
    🎉 no goals
  -/


theorem UniformSpace.has_seq_basis [IsCountablyGenerated <| 𝓤 α] :
    ∃ V : ℕ → Set (α × α), HasAntitoneBasis (𝓤 α) V ∧ ∀ n, SymmetricRel (V n) :=
  let ⟨U, hsym, hbasis⟩ := (@UniformSpace.hasBasis_symmetric α _).exists_antitone_subbasis
  ⟨U, hbasis, fun n => (hsym n).2⟩


theorem Filter.HasBasis.biInter_biUnion_ball {p : ι → Prop} {U : ι → Set (α × α)}
    (h : HasBasis (𝓤 α) p U) (s : Set α) :
    (⋂ (i) (_ : p i), ⋃ x ∈ s, ball x (U i)) = closure s := by
  /-
    α : Type ua
    ι : Sort u_1
    inst✝ : UniformSpace α
    p : ι → Prop
    U : ι → Set (Prod α α)
    h : (uniformity α).HasBasis p U
    s : Set α
    ⊢ Eq (Set.iInter fun i => Set.iInter fun x => Set.iUnion fun x => Set.iUnion f …
  -/
  ext x
  /-
    case h
    α : Type ua
    ι : Sort u_1
    inst✝ : UniformSpace α
    p : ι → Prop
    U : ι → Set (Prod α α)
    h : (uniformity α).HasBasis p U
    s : Set α
    x : α
    ⊢ Iff (Membership.mem (Set.iInter fun i => Set.iInter fun x => Set.iUnion fun  …
  -/
  simp [mem_closure_iff_nhds_basis (nhds_basis_uniformity h), ball]
  /-
    🎉 no goals
  -/


/-- A function `f : α → β` is *uniformly continuous* if `(f x, f y)` tends to the diagonal
as `(x, y)` tends to the diagonal. In other words, if `x` is sufficiently close to `y`, then
`f x` is close to `f y` no matter where `x` and `y` are located in `α`. -/
def UniformContinuous [UniformSpace β] (f : α → β) :=
  Tendsto (fun x : α × α => (f x.1, f x.2)) (𝓤 α) (𝓤 β)


/-- Notation for uniform continuity with respect to non-standard `UniformSpace` instances. -/
scoped[Uniformity] notation "UniformContinuous[" u₁ ", " u₂ "]" => @UniformContinuous _ _ u₁ u₂


/-- A function `f : α → β` is *uniformly continuous* on `s : Set α` if `(f x, f y)` tends to
the diagonal as `(x, y)` tends to the diagonal while remaining in `s ×ˢ s`.
In other words, if `x` is sufficiently close to `y`, then `f x` is close to
`f y` no matter where `x` and `y` are located in `s`. -/
def UniformContinuousOn [UniformSpace β] (f : α → β) (s : Set α) : Prop :=
  Tendsto (fun x : α × α => (f x.1, f x.2)) (𝓤 α ⊓ 𝓟 (s ×ˢ s)) (𝓤 β)


theorem uniformContinuous_def [UniformSpace β] {f : α → β} :
    UniformContinuous f ↔ ∀ r ∈ 𝓤 β, { x : α × α | (f x.1, f x.2) ∈ r } ∈ 𝓤 α :=
  Iff.rfl


theorem uniformContinuous_iff_eventually [UniformSpace β] {f : α → β} :
    UniformContinuous f ↔ ∀ r ∈ 𝓤 β, ∀ᶠ x : α × α in 𝓤 α, (f x.1, f x.2) ∈ r :=
  Iff.rfl


theorem uniformContinuousOn_univ [UniformSpace β] {f : α → β} :
    UniformContinuousOn f univ ↔ UniformContinuous f := by
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    ⊢ Iff (UniformContinuousOn f Set.univ) (UniformContinuous f)
  -/
  rw [UniformContinuousOn, UniformContinuous, univ_prod_univ, principal_univ, inf_top_eq]
  /-
    🎉 no goals
  -/


theorem uniformContinuous_of_const [UniformSpace β] {c : α → β} (h : ∀ a b, c a = c b) :
    UniformContinuous c :=
  have : (fun x : α × α => (c x.fst, c x.snd)) ⁻¹' idRel = univ :=
    eq_univ_iff_forall.2 fun ⟨a, b⟩ => h a b
                                        /-
                                          α : Type ua
                                          β : Type ub
                                          inst✝¹ : UniformSpace α
                                          inst✝ : UniformSpace β
                                          c : α → β
                                          h : ∀ (a b : α), Eq (c a) (c b)
                                          this : Eq (Set.preimage (fun x => { fst := c x.1, snd := c x.2 }) idRel) Set.u …
                                          ⊢ LE.le (uniformity α) (Filter.comap (fun x => { fst := c x.1, snd := c x.2 }) …
                                        -/
  le_trans (map_le_iff_le_comap.2 <| by simp [comap_principal, this, univ_mem]) refl_le_uniformity
                                        /-
                                          🎉 no goals
                                        -/


theorem uniformContinuous_id : UniformContinuous (@id α) := tendsto_id


theorem uniformContinuous_const [UniformSpace β] {b : β} : UniformContinuous fun _ : α => b :=
  uniformContinuous_of_const fun _ _ => rfl


nonrec theorem UniformContinuous.comp [UniformSpace β] [UniformSpace γ] {g : β → γ} {f : α → β}
    (hg : UniformContinuous g) (hf : UniformContinuous f) : UniformContinuous (g ∘ f) :=
  hg.comp hf


/--If a function `T` is uniformly continuous in a uniform space `β`,
then its `n`-th iterate `T^[n]` is also uniformly continuous.-/
theorem UniformContinuous.iterate [UniformSpace β] (T : β → β) (n : ℕ) (h : UniformContinuous T) :
    UniformContinuous T^[n] := by
  induction n with
  | zero => exact uniformContinuous_id
  | succ n hn => exact Function.iterate_succ _ _ ▸ UniformContinuous.comp hn h


theorem Filter.HasBasis.uniformContinuous_iff {ι'} [UniformSpace β] {p : ι → Prop}
    {s : ι → Set (α × α)} (ha : (𝓤 α).HasBasis p s) {q : ι' → Prop} {t : ι' → Set (β × β)}
    (hb : (𝓤 β).HasBasis q t) {f : α → β} :
    UniformContinuous f ↔ ∀ i, q i → ∃ j, p j ∧ ∀ x y, (x, y) ∈ s j → (f x, f y) ∈ t i :=
                                  /-
                                    α : Type ua
                                    β : Type ub
                                    ι : Sort u_1
                                    inst✝¹ : UniformSpace α
                                    ι' : Sort u_2
                                    inst✝ : UniformSpace β
                                    p : ι → Prop
                                    s : ι → Set (Prod α α)
                                    ha : (uniformity α).HasBasis p s
                                    q : ι' → Prop
                                    t : ι' → Set (Prod β β)
                                    hb : (uniformity β).HasBasis q t
                                    f : α → β
                                    ⊢ Iff (∀ (ib : ι'), q ib → Exists fun ia => And (p ia) (∀ (x : Prod α α), Memb …
                                  -/
  (ha.tendsto_iff hb).trans <| by simp only [Prod.forall]
                                  /-
                                    🎉 no goals
                                  -/


theorem Filter.HasBasis.uniformContinuousOn_iff {ι'} [UniformSpace β] {p : ι → Prop}
    {s : ι → Set (α × α)} (ha : (𝓤 α).HasBasis p s) {q : ι' → Prop} {t : ι' → Set (β × β)}
    (hb : (𝓤 β).HasBasis q t) {f : α → β} {S : Set α} :
    UniformContinuousOn f S ↔
      ∀ i, q i → ∃ j, p j ∧ ∀ x, x ∈ S → ∀ y, y ∈ S → (x, y) ∈ s j → (f x, f y) ∈ t i :=
  ((ha.inf_principal (S ×ˢ S)).tendsto_iff hb).trans <| by
    /-
      α : Type ua
      β : Type ub
      ι : Sort u_1
      inst✝¹ : UniformSpace α
      ι' : Sort u_2
      inst✝ : UniformSpace β
      p : ι → Prop
      s : ι → Set (Prod α α)
      ha : (uniformity α).HasBasis p s
      q : ι' → Prop
      t : ι' → Set (Prod β β)
      hb : (uniformity β).HasBasis q t
      f : α → β
      S : Set α
      ⊢ Iff (∀ (ib : ι'), q ib → Exists fun ia => And (p ia) (∀ (x : Prod α α), Memb …
    -/
    simp_rw [Prod.forall, Set.inter_comm (s _), forall_mem_comm, mem_inter_iff, mem_prod, and_imp]
    /-
      🎉 no goals
    -/


instance : PartialOrder (UniformSpace α) :=
  PartialOrder.lift (fun u => 𝓤[u]) fun _ _ => UniformSpace.ext


protected theorem UniformSpace.le_def {u₁ u₂ : UniformSpace α} : u₁ ≤ u₂ ↔ 𝓤[u₁] ≤ 𝓤[u₂] := Iff.rfl


instance : InfSet (UniformSpace α) :=
  ⟨fun s =>
    UniformSpace.ofCore
      { uniformity := ⨅ u ∈ s, 𝓤[u]
        refl := le_iInf fun u => le_iInf fun _ => u.toCore.refl
        symm := le_iInf₂ fun u hu =>
          le_trans (map_mono <| iInf_le_of_le _ <| iInf_le _ hu) u.symm
        comp := le_iInf₂ fun u hu =>
          le_trans (lift'_mono (iInf_le_of_le _ <| iInf_le _ hu) <| le_rfl) u.comp }⟩


protected theorem UniformSpace.sInf_le {tt : Set (UniformSpace α)} {t : UniformSpace α}
    (h : t ∈ tt) : sInf tt ≤ t :=
  show ⨅ u ∈ tt, 𝓤[u] ≤ 𝓤[t] from iInf₂_le t h


protected theorem UniformSpace.le_sInf {tt : Set (UniformSpace α)} {t : UniformSpace α}
    (h : ∀ t' ∈ tt, t ≤ t') : t ≤ sInf tt :=
  show 𝓤[t] ≤ ⨅ u ∈ tt, 𝓤[u] from le_iInf₂ h


instance : Top (UniformSpace α) :=
                                                   /-
                                                     α : Type ua
                                                     β : Type ub
                                                     γ : Type uc
                                                     δ : Type ud
                                                     ι : Sort u_1
                                                     x : α
                                                     ⊢ Eq (nhds x) (Filter.comap (Prod.mk x) Top.top)
                                                   -/
  ⟨@UniformSpace.mk α ⊤ ⊤ le_top le_top fun x ↦ by simp only [nhds_top, comap_top]⟩
                                                   /-
                                                     🎉 no goals
                                                   -/


instance : Bot (UniformSpace α) :=
  ⟨{  toTopologicalSpace := ⊥
      uniformity := 𝓟 idRel
                 /-
                   α : Type ua
                   β : Type ub
                   γ : Type uc
                   δ : Type ud
                   ι : Sort u_1
                   ⊢ Filter.Tendsto Prod.swap (Filter.principal idRel) (Filter.principal idRel)
                 -/
      symm := by simp [Tendsto]
                 /-
                   🎉 no goals
                 -/
      comp := lift'_le (mem_principal_self _) <| principal_mono.2 id_compRel.subset
      nhds_eq_comap_uniformity := fun s => by
        /-
          α : Type ua
          β : Type ub
          γ : Type uc
          δ : Type ud
          ι : Sort u_1
          s : α
          ⊢ Eq (nhds s) (Filter.comap (Prod.mk s) (Filter.principal idRel))
        -/
        let _ : TopologicalSpace α := ⊥; have := discreteTopology_bot α
        /-
          α : Type ua
          β : Type ub
          γ : Type uc
          δ : Type ud
          ι : Sort u_1
          s : α
          x✝ : TopologicalSpace α := Bot.bot
          this : DiscreteTopology α
          ⊢ Eq (nhds s) (Filter.comap (Prod.mk s) (Filter.principal idRel))
        -/
        simp [idRel] }⟩
        /-
          🎉 no goals
        -/


instance : Min (UniformSpace α) :=
  ⟨fun u₁ u₂ =>
    { uniformity := 𝓤[u₁] ⊓ 𝓤[u₂]
      symm := u₁.symm.inf u₂.symm
      comp := (lift'_inf_le _ _ _).trans <| inf_le_inf u₁.comp u₂.comp
      toTopologicalSpace := u₁.toTopologicalSpace ⊓ u₂.toTopologicalSpace
      nhds_eq_comap_uniformity := fun _ ↦ by
        rw [@nhds_inf _ u₁.toTopologicalSpace _, @nhds_eq_comap_uniformity _ u₁,
          @nhds_eq_comap_uniformity _ u₂, comap_inf] }⟩


instance : CompleteLattice (UniformSpace α) :=
  { inferInstanceAs (PartialOrder (UniformSpace α)) with
    sup := fun a b => sInf { x | a ≤ x ∧ b ≤ x }
    le_sup_left := fun _ _ => UniformSpace.le_sInf fun _ ⟨h, _⟩ => h
    le_sup_right := fun _ _ => UniformSpace.le_sInf fun _ ⟨_, h⟩ => h
    sup_le := fun _ _ _ h₁ h₂ => UniformSpace.sInf_le ⟨h₁, h₂⟩
    inf := (· ⊓ ·)
    le_inf := fun a _ _ h₁ h₂ => show a.uniformity ≤ _ from le_inf h₁ h₂
    inf_le_left := fun a _ => show _ ≤ a.uniformity from inf_le_left
    inf_le_right := fun _ b => show _ ≤ b.uniformity from inf_le_right
    top := ⊤
    le_top := fun a => show a.uniformity ≤ ⊤ from le_top
    bot := ⊥
    bot_le := fun u => u.toCore.refl
    sSup := fun tt => sInf { t | ∀ t' ∈ tt, t' ≤ t }
    le_sSup := fun _ _ h => UniformSpace.le_sInf fun _ h' => h' _ h
    sSup_le := fun _ _ h => UniformSpace.sInf_le h
    sInf := sInf
    le_sInf := fun _ _ hs => UniformSpace.le_sInf hs
    sInf_le := fun _ _ ha => UniformSpace.sInf_le ha }


theorem iInf_uniformity {ι : Sort*} {u : ι → UniformSpace α} : 𝓤[iInf u] = ⨅ i, 𝓤[u i] :=
  iInf_range


theorem inf_uniformity {u v : UniformSpace α} : 𝓤[u ⊓ v] = 𝓤[u] ⊓ 𝓤[v] := rfl


lemma bot_uniformity : 𝓤[(⊥ : UniformSpace α)] = 𝓟 idRel := rfl


lemma top_uniformity : 𝓤[(⊤ : UniformSpace α)] = ⊤ := rfl


instance inhabitedUniformSpace : Inhabited (UniformSpace α) :=
  ⟨⊥⟩


instance inhabitedUniformSpaceCore : Inhabited (UniformSpace.Core α) :=
  ⟨@UniformSpace.toCore _ default⟩


instance [Subsingleton α] : Unique (UniformSpace α) where
  uniq u := bot_unique <| le_principal_iff.2 <| by
    /-
      α : Type ua
      β : Type ub
      γ : Type uc
      δ : Type ud
      ι : Sort u_1
      inst✝ : Subsingleton α
      u : UniformSpace α
      ⊢ Membership.mem ((fun u => uniformity α) u) idRel
    -/
    rw [idRel, ← diagonal, diagonal_eq_univ]; exact univ_mem
                                              /-
                                                🎉 no goals
                                              -/


/-- Given `f : α → β` and a uniformity `u` on `β`, the inverse image of `u` under `f`
  is the inverse image in the filter sense of the induced function `α × α → β × β`.
  See note [reducible non-instances]. -/
abbrev UniformSpace.comap (f : α → β) (u : UniformSpace β) : UniformSpace α where
  uniformity := 𝓤[u].comap fun p : α × α => (f p.1, f p.2)
  symm := by
    /-
      α : Type ua
      β : Type ub
      γ : Type uc
      δ : Type ud
      ι : Sort u_1
      f : α → β
      u : UniformSpace β
      ⊢ Filter.Tendsto Prod.swap (Filter.comap (fun p => { fst := f p.1, snd := f p. …
    -/
    simp only [tendsto_comap_iff, Prod.swap, (· ∘ ·)]
    /-
      α : Type ua
      β : Type ub
      γ : Type uc
      δ : Type ud
      ι : Sort u_1
      f : α → β
      u : UniformSpace β
      ⊢ Filter.Tendsto (Function.comp (fun p => { fst := f p.1, snd := f p.2 }) Prod …
    -/
    exact tendsto_swap_uniformity.comp tendsto_comap
    /-
      🎉 no goals
    -/
  comp := le_trans
    (by
      /-
        α : Type ua
        β : Type ub
        γ : Type uc
        δ : Type ud
        ι : Sort u_1
        f : α → β
        u : UniformSpace β
        ⊢ LE.le ((Filter.comap (fun p => { fst := f p.1, snd := f p.2 }) (uniformity β …
      -/
      rw [comap_lift'_eq, comap_lift'_eq2]
        /-
          α : Type ua
          β : Type ub
          γ : Type uc
          δ : Type ud
          ι : Sort u_1
          f : α → β
          u : UniformSpace β
          ⊢ LE.le ((uniformity β).lift' (Function.comp (fun s => compRel s s) (Set.preim …
        -/
      · exact lift'_mono' fun s _ ⟨a₁, a₂⟩ ⟨x, h₁, h₂⟩ => ⟨f x, h₁, h₂⟩
        /-
          🎉 no goals
        -/
        /-
          α : Type ua
          β : Type ub
          γ : Type uc
          δ : Type ud
          ι : Sort u_1
          f : α → β
          u : UniformSpace β
          ⊢ Monotone fun s => compRel s s
        -/
      · exact monotone_id.compRel monotone_id)
        /-
          🎉 no goals
        -/
    (comap_mono u.comp)
  toTopologicalSpace := u.toTopologicalSpace.induced f
  nhds_eq_comap_uniformity x := by
    /-
      α : Type ua
      β : Type ub
      γ : Type uc
      δ : Type ud
      ι : Sort u_1
      f : α → β
      u : UniformSpace β
      x : α
      ⊢ Eq (nhds x) (Filter.comap (Prod.mk x) (Filter.comap (fun p => { fst := f p.1 …
    -/
    simp only [nhds_induced, nhds_eq_comap_uniformity, comap_comap, Function.comp_def]
    /-
      🎉 no goals
    -/


theorem uniformity_comap {_ : UniformSpace β} (f : α → β) :
    𝓤[UniformSpace.comap f ‹_›] = comap (Prod.map f f) (𝓤 β) :=
  rfl


lemma ball_preimage {f : α → β} {U : Set (β × β)} {x : α} :
    UniformSpace.ball x (Prod.map f f ⁻¹' U) = f ⁻¹' UniformSpace.ball (f x) U := by
  /-
    α : Type ua
    β : Type ub
    f : α → β
    U : Set (Prod β β)
    x : α
    ⊢ Eq (UniformSpace.ball x (Set.preimage (Prod.map f f) U)) (Set.preimage f (Un …
  -/
  ext : 1
  /-
    case h
    α : Type ua
    β : Type ub
    f : α → β
    U : Set (Prod β β)
    x x✝ : α
    ⊢ Iff (Membership.mem (UniformSpace.ball x (Set.preimage (Prod.map f f) U)) x✝ …
  -/
  simp only [UniformSpace.ball, mem_preimage, Prod.map_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem uniformSpace_comap_id {α : Type*} : UniformSpace.comap (id : α → α) = id := by
  /-
    α : Type u_2
    ⊢ Eq (UniformSpace.comap id) id
  -/
  ext : 2
  /-
    case h.h
    α : Type u_2
    x✝ : UniformSpace α
    ⊢ Eq (uniformity α) (uniformity α)
  -/
  rw [uniformity_comap, Prod.map_id, comap_id]
  /-
    🎉 no goals
  -/


theorem UniformSpace.comap_comap {α β γ} {uγ : UniformSpace γ} {f : α → β} {g : β → γ} :
    UniformSpace.comap (g ∘ f) uγ = UniformSpace.comap f (UniformSpace.comap g uγ) := by
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    uγ : UniformSpace γ
    f : α → β
    g : β → γ
    ⊢ Eq (UniformSpace.comap (Function.comp g f) uγ) (UniformSpace.comap f (Unifor …
  -/
  ext1
  /-
    case h
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    uγ : UniformSpace γ
    f : α → β
    g : β → γ
    ⊢ Eq (uniformity α) (uniformity α)
  -/
  simp only [uniformity_comap, Filter.comap_comap, Prod.map_comp_map]
  /-
    🎉 no goals
  -/


theorem UniformSpace.comap_inf {α γ} {u₁ u₂ : UniformSpace γ} {f : α → γ} :
    (u₁ ⊓ u₂).comap f = u₁.comap f ⊓ u₂.comap f :=
  UniformSpace.ext Filter.comap_inf


theorem UniformSpace.comap_iInf {ι α γ} {u : ι → UniformSpace γ} {f : α → γ} :
    (⨅ i, u i).comap f = ⨅ i, (u i).comap f := by
  /-
    ι : Sort u_2
    α : Type u_3
    γ : Type u_4
    u : ι → UniformSpace γ
    f : α → γ
    ⊢ Eq (UniformSpace.comap f (iInf fun i => u i)) (iInf fun i => UniformSpace.co …
  -/
  ext : 1
  /-
    case h
    ι : Sort u_2
    α : Type u_3
    γ : Type u_4
    u : ι → UniformSpace γ
    f : α → γ
    ⊢ Eq (uniformity α) (uniformity α)
  -/
  simp [uniformity_comap, iInf_uniformity]
  /-
    🎉 no goals
  -/


theorem UniformSpace.comap_mono {α γ} {f : α → γ} :
    Monotone fun u : UniformSpace γ => u.comap f := fun _ _ hu =>
  Filter.comap_mono hu


theorem uniformContinuous_iff {α β} {uα : UniformSpace α} {uβ : UniformSpace β} {f : α → β} :
    UniformContinuous f ↔ uα ≤ uβ.comap f :=
  Filter.map_le_iff_le_comap


theorem le_iff_uniformContinuous_id {u v : UniformSpace α} :
    u ≤ v ↔ @UniformContinuous _ _ u v id := by
  /-
    α : Type ua
    u v : UniformSpace α
    ⊢ Iff (LE.le u v) (UniformContinuous id)
  -/
  rw [uniformContinuous_iff, uniformSpace_comap_id, id]
  /-
    🎉 no goals
  -/


theorem uniformContinuous_comap {f : α → β} [u : UniformSpace β] :
    @UniformContinuous α β (UniformSpace.comap f u) u f :=
  tendsto_comap


theorem uniformContinuous_comap' {f : γ → β} {g : α → γ} [v : UniformSpace β] [u : UniformSpace α]
    (h : UniformContinuous (f ∘ g)) : @UniformContinuous α γ u (UniformSpace.comap f v) g :=
  tendsto_comap_iff.2 h


theorem to_nhds_mono {u₁ u₂ : UniformSpace α} (h : u₁ ≤ u₂) (a : α) :
    @nhds _ (@UniformSpace.toTopologicalSpace _ u₁) a ≤
      @nhds _ (@UniformSpace.toTopologicalSpace _ u₂) a := by
  /-
    α : Type ua
    u₁ u₂ : UniformSpace α
    h : LE.le u₁ u₂
    a : α
    ⊢ LE.le (nhds a) (nhds a)
  -/
  rw [@nhds_eq_uniformity α u₁ a, @nhds_eq_uniformity α u₂ a]; exact lift'_mono h le_rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem toTopologicalSpace_mono {u₁ u₂ : UniformSpace α} (h : u₁ ≤ u₂) :
    @UniformSpace.toTopologicalSpace _ u₁ ≤ @UniformSpace.toTopologicalSpace _ u₂ :=
  le_of_nhds_le_nhds <| to_nhds_mono h


theorem toTopologicalSpace_comap {f : α → β} {u : UniformSpace β} :
    @UniformSpace.toTopologicalSpace _ (UniformSpace.comap f u) =
      TopologicalSpace.induced f (@UniformSpace.toTopologicalSpace β u) :=
  rfl


lemma uniformSpace_eq_bot {u : UniformSpace α} : u = ⊥ ↔ idRel ∈ 𝓤[u] :=
  le_bot_iff.symm.trans le_principal_iff


protected lemma _root_.Filter.HasBasis.uniformSpace_eq_bot {ι p} {s : ι → Set (α × α)}
    {u : UniformSpace α} (h : 𝓤[u].HasBasis p s) :
    u = ⊥ ↔ ∃ i, p i ∧ Pairwise fun x y : α ↦ (x, y) ∉ s i := by
  /-
    α : Type ua
    ι : Sort u_2
    p : ι → Prop
    s : ι → Set (Prod α α)
    u : UniformSpace α
    h : (uniformity α).HasBasis p s
    ⊢ Iff (Eq u Bot.bot) (Exists fun i => And (p i) (Pairwise fun x y => Not (Memb …
  -/
  simp [uniformSpace_eq_bot, h.mem_iff, subset_def, Pairwise, not_imp_not]
  /-
    🎉 no goals
  -/


theorem toTopologicalSpace_bot : @UniformSpace.toTopologicalSpace α ⊥ = ⊥ := rfl


theorem toTopologicalSpace_top : @UniformSpace.toTopologicalSpace α ⊤ = ⊤ := rfl


theorem toTopologicalSpace_iInf {ι : Sort*} {u : ι → UniformSpace α} :
    (iInf u).toTopologicalSpace = ⨅ i, (u i).toTopologicalSpace :=
  TopologicalSpace.ext_nhds fun a ↦ by simp only [@nhds_eq_comap_uniformity _ (iInf u), nhds_iInf,
    iInf_uniformity, @nhds_eq_comap_uniformity _ (u _), Filter.comap_iInf]


theorem toTopologicalSpace_sInf {s : Set (UniformSpace α)} :
    (sInf s).toTopologicalSpace = ⨅ i ∈ s, @UniformSpace.toTopologicalSpace α i := by
  /-
    α : Type ua
    s : Set (UniformSpace α)
    ⊢ Eq UniformSpace.toTopologicalSpace (iInf fun i => iInf fun h => UniformSpace …
  -/
  rw [sInf_eq_iInf]
  /-
    α : Type ua
    s : Set (UniformSpace α)
    ⊢ Eq UniformSpace.toTopologicalSpace (iInf fun i => iInf fun h => UniformSpace …
  -/
  simp only [← toTopologicalSpace_iInf]
  /-
    🎉 no goals
  -/


theorem toTopologicalSpace_inf {u v : UniformSpace α} :
    (u ⊓ v).toTopologicalSpace = u.toTopologicalSpace ⊓ v.toTopologicalSpace :=
  rfl


theorem UniformContinuous.continuous [UniformSpace α] [UniformSpace β] {f : α → β}
    (hf : UniformContinuous f) : Continuous f :=
  continuous_iff_le_induced.mpr <| UniformSpace.toTopologicalSpace_mono <|
    uniformContinuous_iff.1 hf


/-- Uniform space structure on `ULift α`. -/
instance ULift.uniformSpace [UniformSpace α] : UniformSpace (ULift α) :=
  UniformSpace.comap ULift.down ‹_›


/-- Uniform space structure on `αᵒᵈ`. -/
instance OrderDual.instUniformSpace [UniformSpace α] : UniformSpace (αᵒᵈ) :=
  ‹UniformSpace α›


theorem UniformContinuous.inf_rng {f : α → β} {u₁ : UniformSpace α} {u₂ u₃ : UniformSpace β}
    (h₁ : UniformContinuous[u₁, u₂] f) (h₂ : UniformContinuous[u₁, u₃] f) :
    UniformContinuous[u₁, u₂ ⊓ u₃] f :=
  tendsto_inf.mpr ⟨h₁, h₂⟩

-- Porting note: renamed for dot notation

theorem UniformContinuous.inf_dom_left {f : α → β} {u₁ u₂ : UniformSpace α} {u₃ : UniformSpace β}
    (hf : UniformContinuous[u₁, u₃] f) : UniformContinuous[u₁ ⊓ u₂, u₃] f :=
  tendsto_inf_left hf

-- Porting note: renamed for dot notation

theorem UniformContinuous.inf_dom_right {f : α → β} {u₁ u₂ : UniformSpace α} {u₃ : UniformSpace β}
    (hf : UniformContinuous[u₂, u₃] f) : UniformContinuous[u₁ ⊓ u₂, u₃] f :=
  tendsto_inf_right hf


theorem uniformContinuous_sInf_dom {f : α → β} {u₁ : Set (UniformSpace α)} {u₂ : UniformSpace β}
    {u : UniformSpace α} (h₁ : u ∈ u₁) (hf : UniformContinuous[u, u₂] f) :
    UniformContinuous[sInf u₁, u₂] f := by
  /-
    α : Type ua
    β : Type ub
    f : α → β
    u₁ : Set (UniformSpace α)
    u₂ : UniformSpace β
    u : UniformSpace α
    h₁ : Membership.mem u₁ u
    hf : UniformContinuous f
    ⊢ UniformContinuous f
  -/
  delta UniformContinuous
  /-
    α : Type ua
    β : Type ub
    f : α → β
    u₁ : Set (UniformSpace α)
    u₂ : UniformSpace β
    u : UniformSpace α
    h₁ : Membership.mem u₁ u
    hf : UniformContinuous f
    ⊢ Filter.Tendsto (fun x => { fst := f x.1, snd := f x.2 }) (uniformity α) (uni …
  -/
  rw [sInf_eq_iInf', iInf_uniformity]
  /-
    α : Type ua
    β : Type ub
    f : α → β
    u₁ : Set (UniformSpace α)
    u₂ : UniformSpace β
    u : UniformSpace α
    h₁ : Membership.mem u₁ u
    hf : UniformContinuous f
    ⊢ Filter.Tendsto (fun x => { fst := f x.1, snd := f x.2 }) (iInf fun i => unif …
  -/
  exact tendsto_iInf' ⟨u, h₁⟩ hf
  /-
    🎉 no goals
  -/


theorem uniformContinuous_sInf_rng {f : α → β} {u₁ : UniformSpace α} {u₂ : Set (UniformSpace β)} :
    UniformContinuous[u₁, sInf u₂] f ↔ ∀ u ∈ u₂, UniformContinuous[u₁, u] f := by
  /-
    α : Type ua
    β : Type ub
    f : α → β
    u₁ : UniformSpace α
    u₂ : Set (UniformSpace β)
    ⊢ Iff (UniformContinuous f) (∀ (u : UniformSpace β), Membership.mem u₂ u → Uni …
  -/
  delta UniformContinuous
  /-
    α : Type ua
    β : Type ub
    f : α → β
    u₁ : UniformSpace α
    u₂ : Set (UniformSpace β)
    ⊢ Iff (Filter.Tendsto (fun x => { fst := f x.1, snd := f x.2 }) (uniformity α) …
  -/
  rw [sInf_eq_iInf', iInf_uniformity, tendsto_iInf, SetCoe.forall]
  /-
    🎉 no goals
  -/


theorem uniformContinuous_iInf_dom {f : α → β} {u₁ : ι → UniformSpace α} {u₂ : UniformSpace β}
    {i : ι} (hf : UniformContinuous[u₁ i, u₂] f) : UniformContinuous[iInf u₁, u₂] f := by
  /-
    α : Type ua
    β : Type ub
    ι : Sort u_1
    f : α → β
    u₁ : ι → UniformSpace α
    u₂ : UniformSpace β
    i : ι
    hf : UniformContinuous f
    ⊢ UniformContinuous f
  -/
  delta UniformContinuous
  /-
    α : Type ua
    β : Type ub
    ι : Sort u_1
    f : α → β
    u₁ : ι → UniformSpace α
    u₂ : UniformSpace β
    i : ι
    hf : UniformContinuous f
    ⊢ Filter.Tendsto (fun x => { fst := f x.1, snd := f x.2 }) (uniformity α) (uni …
  -/
  rw [iInf_uniformity]
  /-
    α : Type ua
    β : Type ub
    ι : Sort u_1
    f : α → β
    u₁ : ι → UniformSpace α
    u₂ : UniformSpace β
    i : ι
    hf : UniformContinuous f
    ⊢ Filter.Tendsto (fun x => { fst := f x.1, snd := f x.2 }) (iInf fun i => unif …
  -/
  exact tendsto_iInf' i hf
  /-
    🎉 no goals
  -/


theorem uniformContinuous_iInf_rng {f : α → β} {u₁ : UniformSpace α} {u₂ : ι → UniformSpace β} :
    UniformContinuous[u₁, iInf u₂] f ↔ ∀ i, UniformContinuous[u₁, u₂ i] f := by
  /-
    α : Type ua
    β : Type ub
    ι : Sort u_1
    f : α → β
    u₁ : UniformSpace α
    u₂ : ι → UniformSpace β
    ⊢ Iff (UniformContinuous f) (∀ (i : ι), UniformContinuous f)
  -/
  delta UniformContinuous
  /-
    α : Type ua
    β : Type ub
    ι : Sort u_1
    f : α → β
    u₁ : UniformSpace α
    u₂ : ι → UniformSpace β
    ⊢ Iff (Filter.Tendsto (fun x => { fst := f x.1, snd := f x.2 }) (uniformity α) …
  -/
  rw [iInf_uniformity, tendsto_iInf]
  /-
    🎉 no goals
  -/


/-- A uniform space with the discrete uniformity has the discrete topology. -/
theorem discreteTopology_of_discrete_uniformity [hα : UniformSpace α] (h : uniformity α = 𝓟 idRel) :
    DiscreteTopology α :=
  ⟨(UniformSpace.ext h.symm : ⊥ = hα) ▸ rfl⟩


instance : UniformSpace Empty := ⊥

instance : UniformSpace PUnit := ⊥

instance : UniformSpace Bool := ⊥

instance : UniformSpace ℕ := ⊥

instance : UniformSpace ℤ := ⊥


instance : UniformSpace (Additive α) := ‹UniformSpace α›

instance : UniformSpace (Multiplicative α) := ‹UniformSpace α›


theorem uniformContinuous_ofMul : UniformContinuous (ofMul : α → Additive α) :=
  uniformContinuous_id


theorem uniformContinuous_toMul : UniformContinuous (toMul : Additive α → α) :=
  uniformContinuous_id


theorem uniformContinuous_ofAdd : UniformContinuous (ofAdd : α → Multiplicative α) :=
  uniformContinuous_id


theorem uniformContinuous_toAdd : UniformContinuous (toAdd : Multiplicative α → α) :=
  uniformContinuous_id


theorem uniformity_additive : 𝓤 (Additive α) = (𝓤 α).map (Prod.map ofMul ofMul) := rfl


theorem uniformity_multiplicative : 𝓤 (Multiplicative α) = (𝓤 α).map (Prod.map ofAdd ofAdd) := rfl


instance instUniformSpaceSubtype {p : α → Prop} [t : UniformSpace α] : UniformSpace (Subtype p) :=
  UniformSpace.comap Subtype.val t


theorem uniformity_subtype {p : α → Prop} [UniformSpace α] :
    𝓤 (Subtype p) = comap (fun q : Subtype p × Subtype p => (q.1.1, q.2.1)) (𝓤 α) :=
  rfl


theorem uniformity_setCoe {s : Set α} [UniformSpace α] :
    𝓤 s = comap (Prod.map ((↑) : s → α) ((↑) : s → α)) (𝓤 α) :=
  rfl


theorem map_uniformity_set_coe {s : Set α} [UniformSpace α] :
    map (Prod.map (↑) (↑)) (𝓤 s) = 𝓤 α ⊓ 𝓟 (s ×ˢ s) := by
  /-
    α : Type ua
    s : Set α
    inst✝ : UniformSpace α
    ⊢ Eq (Filter.map (Prod.map Subtype.val Subtype.val) (uniformity ↑s)) (Min.min  …
  -/
  rw [uniformity_setCoe, map_comap, range_prod_map, Subtype.range_val]
  /-
    🎉 no goals
  -/


theorem uniformContinuous_subtype_val {p : α → Prop} [UniformSpace α] :
    UniformContinuous (Subtype.val : { a : α // p a } → α) :=
  uniformContinuous_comap


theorem UniformContinuous.subtype_mk {p : α → Prop} [UniformSpace α] [UniformSpace β] {f : β → α}
    (hf : UniformContinuous f) (h : ∀ x, p (f x)) :
    UniformContinuous (fun x => ⟨f x, h x⟩ : β → Subtype p) :=
  uniformContinuous_comap' hf


theorem uniformContinuousOn_iff_restrict [UniformSpace α] [UniformSpace β] {f : α → β} {s : Set α} :
    UniformContinuousOn f s ↔ UniformContinuous (s.restrict f) := by
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    s : Set α
    ⊢ Iff (UniformContinuousOn f s) (UniformContinuous (s.restrict f))
  -/
  delta UniformContinuousOn UniformContinuous
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    s : Set α
    ⊢ Iff (Filter.Tendsto (fun x => { fst := f x.1, snd := f x.2 }) (Min.min (unif …
  -/
  rw [← map_uniformity_set_coe, tendsto_map'_iff]; rfl
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem tendsto_of_uniformContinuous_subtype [UniformSpace α] [UniformSpace β] {f : α → β}
    {s : Set α} {a : α} (hf : UniformContinuous fun x : s => f x.val) (ha : s ∈ 𝓝 a) :
    Tendsto f (𝓝 a) (𝓝 (f a)) := by
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    s : Set α
    a : α
    hf : UniformContinuous fun x => f ↑x
    ha : Membership.mem (nhds a) s
    ⊢ Filter.Tendsto f (nhds a) (nhds (f a))
  -/
  rw [(@map_nhds_subtype_coe_eq_nhds α _ s a (mem_of_mem_nhds ha) ha).symm]
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    s : Set α
    a : α
    hf : UniformContinuous fun x => f ↑x
    ha : Membership.mem (nhds a) s
    ⊢ Filter.Tendsto f (Filter.map Subtype.val (nhds ⟨a, ⋯⟩)) (nhds (f a))
  -/
  exact tendsto_map' hf.continuous.continuousAt
  /-
    🎉 no goals
  -/


theorem UniformContinuousOn.continuousOn [UniformSpace α] [UniformSpace β] {f : α → β} {s : Set α}
    (h : UniformContinuousOn f s) : ContinuousOn f s := by
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    s : Set α
    h : UniformContinuousOn f s
    ⊢ ContinuousOn f s
  -/
  rw [uniformContinuousOn_iff_restrict] at h
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    s : Set α
    h : UniformContinuous (s.restrict f)
    ⊢ ContinuousOn f s
  -/
  rw [continuousOn_iff_continuous_restrict]
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    s : Set α
    h : UniformContinuous (s.restrict f)
    ⊢ Continuous (s.restrict f)
  -/
  exact h.continuous
  /-
    🎉 no goals
  -/


@[to_additive]
instance [UniformSpace α] : UniformSpace αᵐᵒᵖ :=
  UniformSpace.comap MulOpposite.unop ‹_›


@[to_additive]
theorem uniformity_mulOpposite [UniformSpace α] :
    𝓤 αᵐᵒᵖ = comap (fun q : αᵐᵒᵖ × αᵐᵒᵖ => (q.1.unop, q.2.unop)) (𝓤 α) :=
  rfl


@[to_additive (attr := simp)]
theorem comap_uniformity_mulOpposite [UniformSpace α] :
    comap (fun p : α × α => (MulOpposite.op p.1, MulOpposite.op p.2)) (𝓤 αᵐᵒᵖ) = 𝓤 α := by
  /-
    α : Type ua
    inst✝ : UniformSpace α
    ⊢ Eq (Filter.comap (fun p => { fst := MulOpposite.op p.1, snd := MulOpposite.o …
  -/
  simpa [uniformity_mulOpposite, comap_comap, (· ∘ ·)] using comap_id
  /-
    🎉 no goals
  -/


@[to_additive]
theorem uniformContinuous_unop [UniformSpace α] : UniformContinuous (unop : αᵐᵒᵖ → α) :=
  uniformContinuous_comap


@[to_additive]
theorem uniformContinuous_op [UniformSpace α] : UniformContinuous (op : α → αᵐᵒᵖ) :=
  uniformContinuous_comap' uniformContinuous_id


instance instUniformSpaceProd [u₁ : UniformSpace α] [u₂ : UniformSpace β] : UniformSpace (α × β) :=
  u₁.comap Prod.fst ⊓ u₂.comap Prod.snd

-- check the above produces no diamond for `simp` and typeclass search

theorem uniformity_prod [UniformSpace α] [UniformSpace β] :
    𝓤 (α × β) =
      ((𝓤 α).comap fun p : (α × β) × α × β => (p.1.1, p.2.1)) ⊓
        (𝓤 β).comap fun p : (α × β) × α × β => (p.1.2, p.2.2) :=
  rfl


instance [UniformSpace α] [IsCountablyGenerated (𝓤 α)]
    [UniformSpace β] [IsCountablyGenerated (𝓤 β)] : IsCountablyGenerated (𝓤 (α × β)) := by
  /-
    α : Type ua
    β : Type ub
    γ : Type uc
    δ : Type ud
    ι : Sort u_1
    inst✝³ : UniformSpace α
    inst✝² : (uniformity α).IsCountablyGenerated
    inst✝¹ : UniformSpace β
    inst✝ : (uniformity β).IsCountablyGenerated
    ⊢ (uniformity (Prod α β)).IsCountablyGenerated
  -/
  rw [uniformity_prod]
  /-
    α : Type ua
    β : Type ub
    γ : Type uc
    δ : Type ud
    ι : Sort u_1
    inst✝³ : UniformSpace α
    inst✝² : (uniformity α).IsCountablyGenerated
    inst✝¹ : UniformSpace β
    inst✝ : (uniformity β).IsCountablyGenerated
    ⊢ (Min.min (Filter.comap (fun p => { fst := p.1.1, snd := p.2.1 }) (uniformity …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem uniformity_prod_eq_comap_prod [UniformSpace α] [UniformSpace β] :
    𝓤 (α × β) =
      comap (fun p : (α × β) × α × β => ((p.1.1, p.2.1), (p.1.2, p.2.2))) (𝓤 α ×ˢ 𝓤 β) := by
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    ⊢ Eq (uniformity (Prod α β)) (Filter.comap (fun p => { fst := { fst := p.1.1,  …
  -/
  simp_rw [uniformity_prod, prod_eq_inf, Filter.comap_inf, Filter.comap_comap, Function.comp_def]
  /-
    🎉 no goals
  -/


theorem uniformity_prod_eq_prod [UniformSpace α] [UniformSpace β] :
    𝓤 (α × β) = map (fun p : (α × α) × β × β => ((p.1.1, p.2.1), (p.1.2, p.2.2))) (𝓤 α ×ˢ 𝓤 β) := by
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    ⊢ Eq (uniformity (Prod α β)) (Filter.map (fun p => { fst := { fst := p.1.1, sn …
  -/
  rw [map_swap4_eq_comap, uniformity_prod_eq_comap_prod]
  /-
    🎉 no goals
  -/


theorem mem_uniformity_of_uniformContinuous_invariant [UniformSpace α] [UniformSpace β]
    {s : Set (β × β)} {f : α → α → β} (hf : UniformContinuous fun p : α × α => f p.1 p.2)
    (hs : s ∈ 𝓤 β) : ∃ u ∈ 𝓤 α, ∀ a b c, (a, b) ∈ u → (f a c, f b c) ∈ s := by
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    s : Set (Prod β β)
    f : α → α → β
    hf : UniformContinuous fun p => f p.1 p.2
    hs : Membership.mem (uniformity β) s
    ⊢ Exists fun u => And (Membership.mem (uniformity α) u) (∀ (a b c : α), Member …
  -/
  rw [UniformContinuous, uniformity_prod_eq_prod, tendsto_map'_iff] at hf
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    s : Set (Prod β β)
    f : α → α → β
    hf : Filter.Tendsto (Function.comp (fun x => { fst := f x.1.1 x.1.2, snd := f  …
    hs : Membership.mem (uniformity β) s
    ⊢ Exists fun u => And (Membership.mem (uniformity α) u) (∀ (a b c : α), Member …
  -/
  rcases mem_prod_iff.1 (mem_map.1 <| hf hs) with ⟨u, hu, v, hv, huvt⟩
  /-
    case intro.intro.intro.intro
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    s : Set (Prod β β)
    f : α → α → β
    hf : Filter.Tendsto (Function.comp (fun x => { fst := f x.1.1 x.1.2, snd := f  …
    hs : Membership.mem (uniformity β) s
    u : Set (Prod α α)
    hu : Membership.mem (uniformity α) u
    v : Set (Prod α α)
    hv : Membership.mem (uniformity α) v
    huvt : HasSubset.Subset (SProd.sprod u v) (Set.preimage (Function.comp (fun x  …
    ⊢ Exists fun u => And (Membership.mem (uniformity α) u) (∀ (a b c : α), Member …
  -/
  exact ⟨u, hu, fun a b c hab => @huvt ((_, _), (_, _)) ⟨hab, refl_mem_uniformity hv⟩⟩
  /-
    🎉 no goals
  -/


/-- An entourage of the diagonal in `α` and an entourage in `β` yield an entourage in `α × β`
once we permute coordinates.-/
def entourageProd (u : Set (α × α)) (v : Set (β × β)) : Set ((α × β) × α × β) :=
  {((a₁, b₁),(a₂, b₂)) | (a₁, a₂) ∈ u ∧ (b₁, b₂) ∈ v}


theorem mem_entourageProd {u : Set (α × α)} {v : Set (β × β)} {p : (α × β) × α × β} :
    p ∈ entourageProd u v ↔ (p.1.1, p.2.1) ∈ u ∧ (p.1.2, p.2.2) ∈ v := Iff.rfl


theorem entourageProd_mem_uniformity [t₁ : UniformSpace α] [t₂ : UniformSpace β] {u : Set (α × α)}
    {v : Set (β × β)} (hu : u ∈ 𝓤 α) (hv : v ∈ 𝓤 β) :
    entourageProd u v ∈ 𝓤 (α × β) := by
  /-
    α : Type ua
    β : Type ub
    t₁ : UniformSpace α
    t₂ : UniformSpace β
    u : Set (Prod α α)
    v : Set (Prod β β)
    hu : Membership.mem (uniformity α) u
    hv : Membership.mem (uniformity β) v
    ⊢ Membership.mem (uniformity (Prod α β)) (entourageProd u v)
  -/
  rw [uniformity_prod]; exact inter_mem_inf (preimage_mem_comap hu) (preimage_mem_comap hv)
                        /-
                          🎉 no goals
                        -/


theorem ball_entourageProd (u : Set (α × α)) (v : Set (β × β)) (x : α × β) :
    ball x (entourageProd u v) = ball x.1 u ×ˢ ball x.2 v := by
  /-
    α : Type ua
    β : Type ub
    u : Set (Prod α α)
    v : Set (Prod β β)
    x : Prod α β
    ⊢ Eq (UniformSpace.ball x (entourageProd u v)) (SProd.sprod (UniformSpace.ball …
  -/
  ext p; simp only [ball, entourageProd, Set.mem_setOf_eq, Set.mem_prod, Set.mem_preimage]
         /-
           🎉 no goals
         -/


theorem Filter.HasBasis.uniformity_prod {ιa ιb : Type*} [UniformSpace α] [UniformSpace β]
    {pa : ιa → Prop} {pb : ιb → Prop} {sa : ιa → Set (α × α)} {sb : ιb → Set (β × β)}
    (ha : (𝓤 α).HasBasis pa sa) (hb : (𝓤 β).HasBasis pb sb) :
    (𝓤 (α × β)).HasBasis (fun i : ιa × ιb ↦ pa i.1 ∧ pb i.2)
    (fun i ↦ entourageProd (sa i.1) (sb i.2)) :=
  (ha.comap _).inf (hb.comap _)


theorem entourageProd_subset [UniformSpace α] [UniformSpace β]
    {s : Set ((α × β) × α × β)} (h : s ∈ 𝓤 (α × β)) :
    ∃ u ∈ 𝓤 α, ∃ v ∈ 𝓤 β, entourageProd u v ⊆ s := by
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    s : Set (Prod (Prod α β) (Prod α β))
    h : Membership.mem (uniformity (Prod α β)) s
    ⊢ Exists fun u => And (Membership.mem (uniformity α) u) (Exists fun v => And ( …
  -/
  rcases (((𝓤 α).basis_sets.uniformity_prod (𝓤 β).basis_sets).mem_iff' s).1 h with ⟨w, hw⟩
  /-
    case intro
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    s : Set (Prod (Prod α β) (Prod α β))
    h : Membership.mem (uniformity (Prod α β)) s
    w : Prod (Set (Prod α α)) (Set (Prod β β))
    hw : And (And (Membership.mem (uniformity α) w.1) (Membership.mem (uniformity  …
    ⊢ Exists fun u => And (Membership.mem (uniformity α) u) (Exists fun v => And ( …
  -/
  use w.1, hw.1.1, w.2, hw.1.2, hw.2
  /-
    🎉 no goals
  -/


theorem tendsto_prod_uniformity_fst [UniformSpace α] [UniformSpace β] :
    Tendsto (fun p : (α × β) × α × β => (p.1.1, p.2.1)) (𝓤 (α × β)) (𝓤 α) :=
  le_trans (map_mono inf_le_left) map_comap_le


theorem tendsto_prod_uniformity_snd [UniformSpace α] [UniformSpace β] :
    Tendsto (fun p : (α × β) × α × β => (p.1.2, p.2.2)) (𝓤 (α × β)) (𝓤 β) :=
  le_trans (map_mono inf_le_right) map_comap_le


theorem uniformContinuous_fst [UniformSpace α] [UniformSpace β] :
    UniformContinuous fun p : α × β => p.1 :=
  tendsto_prod_uniformity_fst


theorem uniformContinuous_snd [UniformSpace α] [UniformSpace β] :
    UniformContinuous fun p : α × β => p.2 :=
  tendsto_prod_uniformity_snd


theorem UniformContinuous.prod_mk {f₁ : α → β} {f₂ : α → γ} (h₁ : UniformContinuous f₁)
    (h₂ : UniformContinuous f₂) : UniformContinuous fun a => (f₁ a, f₂ a) := by
  /-
    α : Type ua
    β : Type ub
    γ : Type uc
    inst✝² : UniformSpace α
    inst✝¹ : UniformSpace β
    inst✝ : UniformSpace γ
    f₁ : α → β
    f₂ : α → γ
    h₁ : UniformContinuous f₁
    h₂ : UniformContinuous f₂
    ⊢ UniformContinuous fun a => { fst := f₁ a, snd := f₂ a }
  -/
  rw [UniformContinuous, uniformity_prod]
  /-
    α : Type ua
    β : Type ub
    γ : Type uc
    inst✝² : UniformSpace α
    inst✝¹ : UniformSpace β
    inst✝ : UniformSpace γ
    f₁ : α → β
    f₂ : α → γ
    h₁ : UniformContinuous f₁
    h₂ : UniformContinuous f₂
    ⊢ Filter.Tendsto (fun x => { fst := { fst := f₁ x.1, snd := f₂ x.1 }, snd := { …
  -/
  exact tendsto_inf.2 ⟨tendsto_comap_iff.2 h₁, tendsto_comap_iff.2 h₂⟩
  /-
    🎉 no goals
  -/


theorem UniformContinuous.prod_mk_left {f : α × β → γ} (h : UniformContinuous f) (b) :
    UniformContinuous fun a => f (a, b) :=
  h.comp (uniformContinuous_id.prod_mk uniformContinuous_const)


theorem UniformContinuous.prod_mk_right {f : α × β → γ} (h : UniformContinuous f) (a) :
    UniformContinuous fun b => f (a, b) :=
  h.comp (uniformContinuous_const.prod_mk uniformContinuous_id)


theorem UniformContinuous.prodMap [UniformSpace δ] {f : α → γ} {g : β → δ}
    (hf : UniformContinuous f) (hg : UniformContinuous g) : UniformContinuous (Prod.map f g) :=
  (hf.comp uniformContinuous_fst).prod_mk (hg.comp uniformContinuous_snd)


@[deprecated (since := "2024-10-06")] alias UniformContinuous.prod_map := UniformContinuous.prodMap


theorem toTopologicalSpace_prod {α} {β} [u : UniformSpace α] [v : UniformSpace β] :
    @UniformSpace.toTopologicalSpace (α × β) instUniformSpaceProd =
      @instTopologicalSpaceProd α β u.toTopologicalSpace v.toTopologicalSpace :=
  rfl


/-- A version of `UniformContinuous.inf_dom_left` for binary functions -/
theorem uniformContinuous_inf_dom_left₂ {α β γ} {f : α → β → γ} {ua1 ua2 : UniformSpace α}
    {ub1 ub2 : UniformSpace β} {uc1 : UniformSpace γ}
            /-
              α✝ : Type ua
              β✝ : Type ub
              γ✝ : Type uc
              δ : Type ud
              ι : Sort u_1
              inst✝² : UniformSpace α✝
              inst✝¹ : UniformSpace β✝
              inst✝ : UniformSpace γ✝
              α : Type ?u.130828
              β : Type ?u.130834
              γ : Type ?u.130840
              f : α → β → γ
              ua1 ua2 : UniformSpace α
              ub1 ub2 : UniformSpace β
              uc1 : UniformSpace γ
              ⊢ Sort ?u.130842
            -/
    (h : by haveI := ua1; haveI := ub1; exact UniformContinuous fun p : α × β => f p.1 p.2) : by
                                        /-
                                          🎉 no goals
                                        -/
      /-
        α✝ : Type ua
        β✝ : Type ub
        γ✝ : Type uc
        δ : Type ud
        ι : Sort u_1
        inst✝² : UniformSpace α✝
        inst✝¹ : UniformSpace β✝
        inst✝ : UniformSpace γ✝
        α : Type ?u.130828
        β : Type ?u.130834
        γ : Type ?u.130840
        f : α → β → γ
        ua1 ua2 : UniformSpace α
        ub1 ub2 : UniformSpace β
        uc1 : UniformSpace γ
        h : UniformContinuous fun p => f p.1 p.2
        ⊢ Sort ?u.130845
      -/
      haveI := ua1 ⊓ ua2; haveI := ub1 ⊓ ub2
      /-
        α✝ : Type ua
        β✝ : Type ub
        γ✝ : Type uc
        δ : Type ud
        ι : Sort u_1
        inst✝² : UniformSpace α✝
        inst✝¹ : UniformSpace β✝
        inst✝ : UniformSpace γ✝
        α : Type ?u.130828
        β : Type ?u.130834
        γ : Type ?u.130840
        f : α → β → γ
        ua1 ua2 : UniformSpace α
        ub1 ub2 : UniformSpace β
        uc1 : UniformSpace γ
        h : UniformContinuous fun p => f p.1 p.2
        this✝ : UniformSpace α
        this : UniformSpace β
        ⊢ Sort ?u.130845
      -/
      exact UniformContinuous fun p : α × β => f p.1 p.2 := by
      /-
        🎉 no goals
      -/
  -- proof essentially copied from `continuous_inf_dom_left₂`
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    f : α → β → γ
    ua1 ua2 : UniformSpace α
    ub1 ub2 : UniformSpace β
    uc1 : UniformSpace γ
    h : UniformContinuous fun p => f p.1 p.2
    ⊢ UniformContinuous fun p => f p.1 p.2
  -/
  have ha := @UniformContinuous.inf_dom_left _ _ id ua1 ua2 ua1 (@uniformContinuous_id _ (id _))
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    f : α → β → γ
    ua1 ua2 : UniformSpace α
    ub1 ub2 : UniformSpace β
    uc1 : UniformSpace γ
    h : UniformContinuous fun p => f p.1 p.2
    ha : UniformContinuous id
    ⊢ UniformContinuous fun p => f p.1 p.2
  -/
  have hb := @UniformContinuous.inf_dom_left _ _ id ub1 ub2 ub1 (@uniformContinuous_id _ (id _))
  have h_unif_cont_id :=
    @UniformContinuous.prodMap _ _ _ _ (ua1 ⊓ ua2) (ub1 ⊓ ub2) ua1 ub1 _ _ ha hb
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    f : α → β → γ
    ua1 ua2 : UniformSpace α
    ub1 ub2 : UniformSpace β
    uc1 : UniformSpace γ
    h : UniformContinuous fun p => f p.1 p.2
    ha : UniformContinuous id
    hb : UniformContinuous id
    h_unif_cont_id : UniformContinuous (Prod.map id id)
    ⊢ UniformContinuous fun p => f p.1 p.2
  -/
  exact @UniformContinuous.comp _ _ _ (id _) (id _) _ _ _ h h_unif_cont_id
  /-
    🎉 no goals
  -/


/-- A version of `UniformContinuous.inf_dom_right` for binary functions -/
theorem uniformContinuous_inf_dom_right₂ {α β γ} {f : α → β → γ} {ua1 ua2 : UniformSpace α}
    {ub1 ub2 : UniformSpace β} {uc1 : UniformSpace γ}
            /-
              α✝ : Type ua
              β✝ : Type ub
              γ✝ : Type uc
              δ : Type ud
              ι : Sort u_1
              inst✝² : UniformSpace α✝
              inst✝¹ : UniformSpace β✝
              inst✝ : UniformSpace γ✝
              α : Type ?u.132684
              β : Type ?u.132690
              γ : Type ?u.132696
              f : α → β → γ
              ua1 ua2 : UniformSpace α
              ub1 ub2 : UniformSpace β
              uc1 : UniformSpace γ
              ⊢ Sort ?u.132698
            -/
    (h : by haveI := ua2; haveI := ub2; exact UniformContinuous fun p : α × β => f p.1 p.2) : by
                                        /-
                                          🎉 no goals
                                        -/
      /-
        α✝ : Type ua
        β✝ : Type ub
        γ✝ : Type uc
        δ : Type ud
        ι : Sort u_1
        inst✝² : UniformSpace α✝
        inst✝¹ : UniformSpace β✝
        inst✝ : UniformSpace γ✝
        α : Type ?u.132684
        β : Type ?u.132690
        γ : Type ?u.132696
        f : α → β → γ
        ua1 ua2 : UniformSpace α
        ub1 ub2 : UniformSpace β
        uc1 : UniformSpace γ
        h : UniformContinuous fun p => f p.1 p.2
        ⊢ Sort ?u.132701
      -/
      haveI := ua1 ⊓ ua2; haveI := ub1 ⊓ ub2
      /-
        α✝ : Type ua
        β✝ : Type ub
        γ✝ : Type uc
        δ : Type ud
        ι : Sort u_1
        inst✝² : UniformSpace α✝
        inst✝¹ : UniformSpace β✝
        inst✝ : UniformSpace γ✝
        α : Type ?u.132684
        β : Type ?u.132690
        γ : Type ?u.132696
        f : α → β → γ
        ua1 ua2 : UniformSpace α
        ub1 ub2 : UniformSpace β
        uc1 : UniformSpace γ
        h : UniformContinuous fun p => f p.1 p.2
        this✝ : UniformSpace α
        this : UniformSpace β
        ⊢ Sort ?u.132701
      -/
      exact UniformContinuous fun p : α × β => f p.1 p.2 := by
      /-
        🎉 no goals
      -/
  -- proof essentially copied from `continuous_inf_dom_right₂`
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    f : α → β → γ
    ua1 ua2 : UniformSpace α
    ub1 ub2 : UniformSpace β
    uc1 : UniformSpace γ
    h : UniformContinuous fun p => f p.1 p.2
    ⊢ UniformContinuous fun p => f p.1 p.2
  -/
  have ha := @UniformContinuous.inf_dom_right _ _ id ua1 ua2 ua2 (@uniformContinuous_id _ (id _))
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    f : α → β → γ
    ua1 ua2 : UniformSpace α
    ub1 ub2 : UniformSpace β
    uc1 : UniformSpace γ
    h : UniformContinuous fun p => f p.1 p.2
    ha : UniformContinuous id
    ⊢ UniformContinuous fun p => f p.1 p.2
  -/
  have hb := @UniformContinuous.inf_dom_right _ _ id ub1 ub2 ub2 (@uniformContinuous_id _ (id _))
  have h_unif_cont_id :=
    @UniformContinuous.prodMap _ _ _ _ (ua1 ⊓ ua2) (ub1 ⊓ ub2) ua2 ub2 _ _ ha hb
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    f : α → β → γ
    ua1 ua2 : UniformSpace α
    ub1 ub2 : UniformSpace β
    uc1 : UniformSpace γ
    h : UniformContinuous fun p => f p.1 p.2
    ha : UniformContinuous id
    hb : UniformContinuous id
    h_unif_cont_id : UniformContinuous (Prod.map id id)
    ⊢ UniformContinuous fun p => f p.1 p.2
  -/
  exact @UniformContinuous.comp _ _ _ (id _) (id _) _ _ _ h h_unif_cont_id
  /-
    🎉 no goals
  -/


/-- A version of `uniformContinuous_sInf_dom` for binary functions -/
theorem uniformContinuous_sInf_dom₂ {α β γ} {f : α → β → γ} {uas : Set (UniformSpace α)}
    {ubs : Set (UniformSpace β)} {ua : UniformSpace α} {ub : UniformSpace β} {uc : UniformSpace γ}
    (ha : ua ∈ uas) (hb : ub ∈ ubs) (hf : UniformContinuous fun p : α × β => f p.1 p.2) : by
      /-
        α✝ : Type ua
        β✝ : Type ub
        γ✝ : Type uc
        δ : Type ud
        ι : Sort u_1
        inst✝² : UniformSpace α✝
        inst✝¹ : UniformSpace β✝
        inst✝ : UniformSpace γ✝
        α : Type ?u.134541
        β : Type ?u.134545
        γ : Type ?u.134554
        f : α → β → γ
        uas : Set (UniformSpace α)
        ubs : Set (UniformSpace β)
        ua : UniformSpace α
        ub : UniformSpace β
        uc : UniformSpace γ
        ha : Membership.mem uas ua
        hb : Membership.mem ubs ub
        hf : UniformContinuous fun p => f p.1 p.2
        ⊢ Sort ?u.134721
      -/
      haveI := sInf uas; haveI := sInf ubs
      /-
        α✝ : Type ua
        β✝ : Type ub
        γ✝ : Type uc
        δ : Type ud
        ι : Sort u_1
        inst✝² : UniformSpace α✝
        inst✝¹ : UniformSpace β✝
        inst✝ : UniformSpace γ✝
        α : Type ?u.134541
        β : Type ?u.134545
        γ : Type ?u.134554
        f : α → β → γ
        uas : Set (UniformSpace α)
        ubs : Set (UniformSpace β)
        ua : UniformSpace α
        ub : UniformSpace β
        uc : UniformSpace γ
        ha : Membership.mem uas ua
        hb : Membership.mem ubs ub
        hf : UniformContinuous fun p => f p.1 p.2
        this✝ : UniformSpace α
        this : UniformSpace β
        ⊢ Sort ?u.134721
      -/
      exact @UniformContinuous _ _ _ uc fun p : α × β => f p.1 p.2 := by
      /-
        🎉 no goals
      -/
  -- proof essentially copied from `continuous_sInf_dom`
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    f : α → β → γ
    uas : Set (UniformSpace α)
    ubs : Set (UniformSpace β)
    ua : UniformSpace α
    ub : UniformSpace β
    uc : UniformSpace γ
    ha : Membership.mem uas ua
    hb : Membership.mem ubs ub
    hf : UniformContinuous fun p => f p.1 p.2
    ⊢ UniformContinuous fun p => f p.1 p.2
  -/
  let _ : UniformSpace (α × β) := instUniformSpaceProd
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    f : α → β → γ
    uas : Set (UniformSpace α)
    ubs : Set (UniformSpace β)
    ua : UniformSpace α
    ub : UniformSpace β
    uc : UniformSpace γ
    ha : Membership.mem uas ua
    hb : Membership.mem ubs ub
    hf : UniformContinuous fun p => f p.1 p.2
    x✝ : UniformSpace (Prod α β) := instUniformSpaceProd
    ⊢ UniformContinuous fun p => f p.1 p.2
  -/
  have ha := uniformContinuous_sInf_dom ha uniformContinuous_id
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    f : α → β → γ
    uas : Set (UniformSpace α)
    ubs : Set (UniformSpace β)
    ua : UniformSpace α
    ub : UniformSpace β
    uc : UniformSpace γ
    ha✝ : Membership.mem uas ua
    hb : Membership.mem ubs ub
    hf : UniformContinuous fun p => f p.1 p.2
    x✝ : UniformSpace (Prod α β) := instUniformSpaceProd
    ha : UniformContinuous id
    ⊢ UniformContinuous fun p => f p.1 p.2
  -/
  have hb := uniformContinuous_sInf_dom hb uniformContinuous_id
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    f : α → β → γ
    uas : Set (UniformSpace α)
    ubs : Set (UniformSpace β)
    ua : UniformSpace α
    ub : UniformSpace β
    uc : UniformSpace γ
    ha✝ : Membership.mem uas ua
    hb✝ : Membership.mem ubs ub
    hf : UniformContinuous fun p => f p.1 p.2
    x✝ : UniformSpace (Prod α β) := instUniformSpaceProd
    ha : UniformContinuous id
    hb : UniformContinuous id
    ⊢ UniformContinuous fun p => f p.1 p.2
  -/
  have h_unif_cont_id := @UniformContinuous.prodMap _ _ _ _ (sInf uas) (sInf ubs) ua ub _ _ ha hb
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    f : α → β → γ
    uas : Set (UniformSpace α)
    ubs : Set (UniformSpace β)
    ua : UniformSpace α
    ub : UniformSpace β
    uc : UniformSpace γ
    ha✝ : Membership.mem uas ua
    hb✝ : Membership.mem ubs ub
    hf : UniformContinuous fun p => f p.1 p.2
    x✝ : UniformSpace (Prod α β) := instUniformSpaceProd
    ha : UniformContinuous id
    hb : UniformContinuous id
    h_unif_cont_id : UniformContinuous (Prod.map id id)
    ⊢ UniformContinuous fun p => f p.1 p.2
  -/
  exact @UniformContinuous.comp _ _ _ (id _) (id _) _ _ _ hf h_unif_cont_id
  /-
    🎉 no goals
  -/


local notation f " ∘₂ " g => Function.bicompr f g


/-- Uniform continuity for functions of two variables. -/
def UniformContinuous₂ (f : α → β → γ) :=
  UniformContinuous (uncurry f)


theorem uniformContinuous₂_def (f : α → β → γ) :
    UniformContinuous₂ f ↔ UniformContinuous (uncurry f) :=
  Iff.rfl


theorem UniformContinuous₂.uniformContinuous {f : α → β → γ} (h : UniformContinuous₂ f) :
    UniformContinuous (uncurry f) :=
  h


theorem uniformContinuous₂_curry (f : α × β → γ) :
    UniformContinuous₂ (Function.curry f) ↔ UniformContinuous f := by
  /-
    α : Type ua
    β : Type ub
    γ : Type uc
    inst✝² : UniformSpace α
    inst✝¹ : UniformSpace β
    inst✝ : UniformSpace γ
    f : Prod α β → γ
    ⊢ Iff (UniformContinuous₂ (Function.curry f)) (UniformContinuous f)
  -/
  rw [UniformContinuous₂, uncurry_curry]
  /-
    🎉 no goals
  -/


theorem UniformContinuous₂.comp {f : α → β → γ} {g : γ → δ} (hg : UniformContinuous g)
    (hf : UniformContinuous₂ f) : UniformContinuous₂ (g ∘₂ f) :=
  hg.comp hf


theorem UniformContinuous₂.bicompl {f : α → β → γ} {ga : δ → α} {gb : δ' → β}
    (hf : UniformContinuous₂ f) (hga : UniformContinuous ga) (hgb : UniformContinuous gb) :
    UniformContinuous₂ (bicompl f ga gb) :=
  hf.uniformContinuous.comp (hga.prodMap hgb)


theorem toTopologicalSpace_subtype [u : UniformSpace α] {p : α → Prop} :
    @UniformSpace.toTopologicalSpace (Subtype p) instUniformSpaceSubtype =
      @instTopologicalSpaceSubtype α p u.toTopologicalSpace :=
  rfl


/-- Uniformity on a disjoint union. Entourages of the diagonal in the union are obtained
by taking independently an entourage of the diagonal in the first part, and an entourage of
the diagonal in the second part. -/
instance Sum.instUniformSpace : UniformSpace (α ⊕ β) where
  uniformity := map (fun p : α × α => (inl p.1, inl p.2)) (𝓤 α) ⊔
    map (fun p : β × β => (inr p.1, inr p.2)) (𝓤 β)
  symm := fun _ hs ↦ ⟨symm_le_uniformity hs.1, symm_le_uniformity hs.2⟩
  comp := fun s hs ↦ by
    /-
      α : Type ua
      β : Type ub
      γ : Type uc
      δ : Type ud
      ι : Sort u_1
      inst✝¹ : UniformSpace α
      inst✝ : UniformSpace β
      s : Set (Prod (Sum α β) (Sum α β))
      hs : Membership.mem (Max.max (Filter.map (fun p => { fst := Sum.inl p.1, snd : …
      ⊢ Membership.mem ((Max.max (Filter.map (fun p => { fst := Sum.inl p.1, snd :=  …
    -/
    rcases comp_mem_uniformity_sets hs.1 with ⟨tα, htα, Htα⟩
    /-
      case intro.intro
      α : Type ua
      β : Type ub
      γ : Type uc
      δ : Type ud
      ι : Sort u_1
      inst✝¹ : UniformSpace α
      inst✝ : UniformSpace β
      s : Set (Prod (Sum α β) (Sum α β))
      hs : Membership.mem (Max.max (Filter.map (fun p => { fst := Sum.inl p.1, snd : …
      tα : Set (Prod α α)
      htα : Membership.mem (uniformity α) tα
      Htα : HasSubset.Subset (compRel tα tα) (Set.preimage (fun p => { fst := Sum.in …
      ⊢ Membership.mem ((Max.max (Filter.map (fun p => { fst := Sum.inl p.1, snd :=  …
    -/
    rcases comp_mem_uniformity_sets hs.2 with ⟨tβ, htβ, Htβ⟩
    /-
      case intro.intro.intro.intro
      α : Type ua
      β : Type ub
      γ : Type uc
      δ : Type ud
      ι : Sort u_1
      inst✝¹ : UniformSpace α
      inst✝ : UniformSpace β
      s : Set (Prod (Sum α β) (Sum α β))
      hs : Membership.mem (Max.max (Filter.map (fun p => { fst := Sum.inl p.1, snd : …
      tα : Set (Prod α α)
      htα : Membership.mem (uniformity α) tα
      Htα : HasSubset.Subset (compRel tα tα) (Set.preimage (fun p => { fst := Sum.in …
      tβ : Set (Prod β β)
      htβ : Membership.mem (uniformity β) tβ
      Htβ : HasSubset.Subset (compRel tβ tβ) (Set.preimage (fun p => { fst := Sum.in …
      ⊢ Membership.mem ((Max.max (Filter.map (fun p => { fst := Sum.inl p.1, snd :=  …
    -/
    filter_upwards [mem_lift' (union_mem_sup (image_mem_map htα) (image_mem_map htβ))]
    /-
      case h
      α : Type ua
      β : Type ub
      γ : Type uc
      δ : Type ud
      ι : Sort u_1
      inst✝¹ : UniformSpace α
      inst✝ : UniformSpace β
      s : Set (Prod (Sum α β) (Sum α β))
      hs : Membership.mem (Max.max (Filter.map (fun p => { fst := Sum.inl p.1, snd : …
      tα : Set (Prod α α)
      htα : Membership.mem (uniformity α) tα
      Htα : HasSubset.Subset (compRel tα tα) (Set.preimage (fun p => { fst := Sum.in …
      tβ : Set (Prod β β)
      htβ : Membership.mem (uniformity β) tβ
      Htβ : HasSubset.Subset (compRel tβ tβ) (Set.preimage (fun p => { fst := Sum.in …
      ⊢ ∀ (a : Prod (Sum α β) (Sum α β)), Membership.mem (compRel (Union.union (Set. …
    -/
    rintro ⟨_, _⟩ ⟨z, ⟨⟨a, b⟩, hab, ⟨⟩⟩ | ⟨⟨a, b⟩, hab, ⟨⟩⟩, ⟨⟨_, c⟩, hbc, ⟨⟩⟩ | ⟨⟨_, c⟩, hbc, ⟨⟩⟩⟩
    /-
      case h.mk.intro.intro.inl.intro.mk.intro.refl.inl.intro.mk.intro.refl
      α : Type ua
      β : Type ub
      γ : Type uc
      δ : Type ud
      ι : Sort u_1
      inst✝¹ : UniformSpace α
      inst✝ : UniformSpace β
      s : Set (Prod (Sum α β) (Sum α β))
      hs : Membership.mem (Max.max (Filter.map (fun p => { fst := Sum.inl p.1, snd : …
      tα : Set (Prod α α)
      htα : Membership.mem (uniformity α) tα
      Htα : HasSubset.Subset (compRel tα tα) (Set.preimage (fun p => { fst := Sum.in …
      tβ : Set (Prod β β)
      htβ : Membership.mem (uniformity β) tβ
      Htβ : HasSubset.Subset (compRel tβ tβ) (Set.preimage (fun p => { fst := Sum.in …
      a b : α
      hab : Membership.mem tα { fst := a, snd := b }
      c : α
      hbc : Membership.mem tα { fst := b, snd := c }
      ⊢ Membership.mem s { fst := Sum.inl { fst := a, snd := b }.1, snd := Sum.inl { …
    -/
    exacts [@Htα (_, _) ⟨b, hab, hbc⟩, @Htβ (_, _) ⟨b, hab, hbc⟩]
    /-
      🎉 no goals
    -/
  nhds_eq_comap_uniformity x := by
    /-
      α : Type ua
      β : Type ub
      γ : Type uc
      δ : Type ud
      ι : Sort u_1
      inst✝¹ : UniformSpace α
      inst✝ : UniformSpace β
      x : Sum α β
      ⊢ Eq (nhds x) (Filter.comap (Prod.mk x) (Max.max (Filter.map (fun p => { fst : …
    -/
    ext
    /-
      case h
      α : Type ua
      β : Type ub
      γ : Type uc
      δ : Type ud
      ι : Sort u_1
      inst✝¹ : UniformSpace α
      inst✝ : UniformSpace β
      x : Sum α β
      s✝ : Set (Sum α β)
      ⊢ Iff (Membership.mem (nhds x) s✝) (Membership.mem (Filter.comap (Prod.mk x) ( …
    -/
    cases x <;> simp [mem_comap', -mem_comap, nhds_inl, nhds_inr, nhds_eq_comap_uniformity,
      Prod.ext_iff]


@[reducible, deprecated (since := "2024-02-15")] alias Sum.uniformSpace := Sum.instUniformSpace


/-- The union of an entourage of the diagonal in each set of a disjoint union is again an entourage
of the diagonal. -/
theorem union_mem_uniformity_sum {a : Set (α × α)} (ha : a ∈ 𝓤 α) {b : Set (β × β)} (hb : b ∈ 𝓤 β) :
    Prod.map inl inl '' a ∪ Prod.map inr inr '' b ∈ 𝓤 (α ⊕ β) :=
  union_mem_sup (image_mem_map ha) (image_mem_map hb)


theorem Sum.uniformity : 𝓤 (α ⊕ β) = map (Prod.map inl inl) (𝓤 α) ⊔ map (Prod.map inr inr) (𝓤 β) :=
  rfl


lemma uniformContinuous_inl : UniformContinuous (Sum.inl : α → α ⊕ β) := le_sup_left

lemma uniformContinuous_inr : UniformContinuous (Sum.inr : β → α ⊕ β) := le_sup_right


instance [IsCountablyGenerated (𝓤 α)] [IsCountablyGenerated (𝓤 β)] :
    IsCountablyGenerated (𝓤 (α ⊕ β)) := by
  /-
    α : Type ua
    β : Type ub
    γ : Type uc
    δ : Type ud
    ι : Sort u_1
    inst✝³ : UniformSpace α
    inst✝² : UniformSpace β
    inst✝¹ : (uniformity α).IsCountablyGenerated
    inst✝ : (uniformity β).IsCountablyGenerated
    ⊢ (uniformity (Sum α β)).IsCountablyGenerated
  -/
  rw [Sum.uniformity]
  /-
    α : Type ua
    β : Type ub
    γ : Type uc
    δ : Type ud
    ι : Sort u_1
    inst✝³ : UniformSpace α
    inst✝² : UniformSpace β
    inst✝¹ : (uniformity α).IsCountablyGenerated
    inst✝ : (uniformity β).IsCountablyGenerated
    ⊢ (Max.max (Filter.map (Prod.map Sum.inl Sum.inl) (uniformity α)) (Filter.map  …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem tendsto_nhds_right {f : Filter β} {u : β → α} {a : α} :
    Tendsto u f (𝓝 a) ↔ Tendsto (fun x => (a, u x)) f (𝓤 α) := by
  /-
    α : Type ua
    β : Type ub
    inst✝ : UniformSpace α
    f : Filter β
    u : β → α
    a : α
    ⊢ Iff (Filter.Tendsto u f (nhds a)) (Filter.Tendsto (fun x => { fst := a, snd  …
  -/
  rw [nhds_eq_comap_uniformity, tendsto_comap_iff]; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem tendsto_nhds_left {f : Filter β} {u : β → α} {a : α} :
    Tendsto u f (𝓝 a) ↔ Tendsto (fun x => (u x, a)) f (𝓤 α) := by
  /-
    α : Type ua
    β : Type ub
    inst✝ : UniformSpace α
    f : Filter β
    u : β → α
    a : α
    ⊢ Iff (Filter.Tendsto u f (nhds a)) (Filter.Tendsto (fun x => { fst := u x, sn …
  -/
  rw [nhds_eq_comap_uniformity', tendsto_comap_iff]; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem continuousAt_iff'_right [TopologicalSpace β] {f : β → α} {b : β} :
    ContinuousAt f b ↔ Tendsto (fun x => (f b, f x)) (𝓝 b) (𝓤 α) := by
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : TopologicalSpace β
    f : β → α
    b : β
    ⊢ Iff (ContinuousAt f b) (Filter.Tendsto (fun x => { fst := f b, snd := f x }) …
  -/
  rw [ContinuousAt, tendsto_nhds_right]
  /-
    🎉 no goals
  -/


theorem continuousAt_iff'_left [TopologicalSpace β] {f : β → α} {b : β} :
    ContinuousAt f b ↔ Tendsto (fun x => (f x, f b)) (𝓝 b) (𝓤 α) := by
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : TopologicalSpace β
    f : β → α
    b : β
    ⊢ Iff (ContinuousAt f b) (Filter.Tendsto (fun x => { fst := f x, snd := f b }) …
  -/
  rw [ContinuousAt, tendsto_nhds_left]
  /-
    🎉 no goals
  -/


theorem continuousAt_iff_prod [TopologicalSpace β] {f : β → α} {b : β} :
    ContinuousAt f b ↔ Tendsto (fun x : β × β => (f x.1, f x.2)) (𝓝 (b, b)) (𝓤 α) :=
  ⟨fun H => le_trans (H.prodMap' H) (nhds_le_uniformity _), fun H =>
    continuousAt_iff'_left.2 <| H.comp <| tendsto_id.prod_mk_nhds tendsto_const_nhds⟩


theorem continuousWithinAt_iff'_right [TopologicalSpace β] {f : β → α} {b : β} {s : Set β} :
    ContinuousWithinAt f s b ↔ Tendsto (fun x => (f b, f x)) (𝓝[s] b) (𝓤 α) := by
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : TopologicalSpace β
    f : β → α
    b : β
    s : Set β
    ⊢ Iff (ContinuousWithinAt f s b) (Filter.Tendsto (fun x => { fst := f b, snd : …
  -/
  rw [ContinuousWithinAt, tendsto_nhds_right]
  /-
    🎉 no goals
  -/


theorem continuousWithinAt_iff'_left [TopologicalSpace β] {f : β → α} {b : β} {s : Set β} :
    ContinuousWithinAt f s b ↔ Tendsto (fun x => (f x, f b)) (𝓝[s] b) (𝓤 α) := by
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : TopologicalSpace β
    f : β → α
    b : β
    s : Set β
    ⊢ Iff (ContinuousWithinAt f s b) (Filter.Tendsto (fun x => { fst := f x, snd : …
  -/
  rw [ContinuousWithinAt, tendsto_nhds_left]
  /-
    🎉 no goals
  -/


theorem continuousOn_iff'_right [TopologicalSpace β] {f : β → α} {s : Set β} :
    ContinuousOn f s ↔ ∀ b ∈ s, Tendsto (fun x => (f b, f x)) (𝓝[s] b) (𝓤 α) := by
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : TopologicalSpace β
    f : β → α
    s : Set β
    ⊢ Iff (ContinuousOn f s) (∀ (b : β), Membership.mem s b → Filter.Tendsto (fun  …
  -/
  simp [ContinuousOn, continuousWithinAt_iff'_right]
  /-
    🎉 no goals
  -/


theorem continuousOn_iff'_left [TopologicalSpace β] {f : β → α} {s : Set β} :
    ContinuousOn f s ↔ ∀ b ∈ s, Tendsto (fun x => (f x, f b)) (𝓝[s] b) (𝓤 α) := by
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : TopologicalSpace β
    f : β → α
    s : Set β
    ⊢ Iff (ContinuousOn f s) (∀ (b : β), Membership.mem s b → Filter.Tendsto (fun  …
  -/
  simp [ContinuousOn, continuousWithinAt_iff'_left]
  /-
    🎉 no goals
  -/


theorem continuous_iff'_right [TopologicalSpace β] {f : β → α} :
    Continuous f ↔ ∀ b, Tendsto (fun x => (f b, f x)) (𝓝 b) (𝓤 α) :=
  continuous_iff_continuousAt.trans <| forall_congr' fun _ => tendsto_nhds_right


theorem continuous_iff'_left [TopologicalSpace β] {f : β → α} :
    Continuous f ↔ ∀ b, Tendsto (fun x => (f x, f b)) (𝓝 b) (𝓤 α) :=
  continuous_iff_continuousAt.trans <| forall_congr' fun _ => tendsto_nhds_left


/-- Consider two functions `f` and `g` which coincide on a set `s` and are continuous there.
Then there is an open neighborhood of `s` on which `f` and `g` are uniformly close. -/
lemma exists_is_open_mem_uniformity_of_forall_mem_eq
    [TopologicalSpace β] {r : Set (α × α)} {s : Set β}
    {f g : β → α} (hf : ∀ x ∈ s, ContinuousAt f x) (hg : ∀ x ∈ s, ContinuousAt g x)
    (hfg : s.EqOn f g) (hr : r ∈ 𝓤 α) :
    ∃ t, IsOpen t ∧ s ⊆ t ∧ ∀ x ∈ t, (f x, g x) ∈ r := by
  have A : ∀ x ∈ s, ∃ t, IsOpen t ∧ x ∈ t ∧ ∀ z ∈ t, (f z, g z) ∈ r := by
    intro x hx
    obtain ⟨t, ht, htsymm, htr⟩ := comp_symm_mem_uniformity_sets hr
    have A : {z | (f x, f z) ∈ t} ∈ 𝓝 x := (hf x hx).preimage_mem_nhds (mem_nhds_left (f x) ht)
    have B : {z | (g x, g z) ∈ t} ∈ 𝓝 x := (hg x hx).preimage_mem_nhds (mem_nhds_left (g x) ht)
    rcases _root_.mem_nhds_iff.1 (inter_mem A B) with ⟨u, hu, u_open, xu⟩
    refine ⟨u, u_open, xu, fun y hy ↦ ?_⟩
    have I1 : (f y, f x) ∈ t := (htsymm.mk_mem_comm).2 (hu hy).1
    have I2 : (g x, g y) ∈ t := (hu hy).2
    rw [hfg hx] at I1
    exact htr (prod_mk_mem_compRel I1 I2)
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : TopologicalSpace β
    r : Set (Prod α α)
    s : Set β
    f g : β → α
    hf : ∀ (x : β), Membership.mem s x → ContinuousAt f x
    hg : ∀ (x : β), Membership.mem s x → ContinuousAt g x
    hfg : Set.EqOn f g s
    hr : Membership.mem (uniformity α) r
    A : ∀ (x : β), Membership.mem s x → Exists fun t => And (IsOpen t) (And (Membe …
    ⊢ Exists fun t => And (IsOpen t) (And (HasSubset.Subset s t) (∀ (x : β), Membe …
  -/
  choose! t t_open xt ht using A
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : TopologicalSpace β
    r : Set (Prod α α)
    s : Set β
    f g : β → α
    hf : ∀ (x : β), Membership.mem s x → ContinuousAt f x
    hg : ∀ (x : β), Membership.mem s x → ContinuousAt g x
    hfg : Set.EqOn f g s
    hr : Membership.mem (uniformity α) r
    t : β → Set β
    t_open : ∀ (x : β), Membership.mem s x → IsOpen (t x)
    xt : ∀ (x : β), Membership.mem s x → Membership.mem (t x) x
    ht : ∀ (x : β), Membership.mem s x → ∀ (z : β), Membership.mem (t x) z → Membe …
    ⊢ Exists fun t => And (IsOpen t) (And (HasSubset.Subset s t) (∀ (x : β), Membe …
  -/
  refine ⟨⋃ x ∈ s, t x, isOpen_biUnion t_open, fun x hx ↦ mem_biUnion hx (xt x hx), ?_⟩
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : TopologicalSpace β
    r : Set (Prod α α)
    s : Set β
    f g : β → α
    hf : ∀ (x : β), Membership.mem s x → ContinuousAt f x
    hg : ∀ (x : β), Membership.mem s x → ContinuousAt g x
    hfg : Set.EqOn f g s
    hr : Membership.mem (uniformity α) r
    t : β → Set β
    t_open : ∀ (x : β), Membership.mem s x → IsOpen (t x)
    xt : ∀ (x : β), Membership.mem s x → Membership.mem (t x) x
    ht : ∀ (x : β), Membership.mem s x → ∀ (z : β), Membership.mem (t x) z → Membe …
    ⊢ ∀ (x : β), Membership.mem (Set.iUnion fun x => Set.iUnion fun h => t x) x →  …
  -/
  rintro x hx
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : TopologicalSpace β
    r : Set (Prod α α)
    s : Set β
    f g : β → α
    hf : ∀ (x : β), Membership.mem s x → ContinuousAt f x
    hg : ∀ (x : β), Membership.mem s x → ContinuousAt g x
    hfg : Set.EqOn f g s
    hr : Membership.mem (uniformity α) r
    t : β → Set β
    t_open : ∀ (x : β), Membership.mem s x → IsOpen (t x)
    xt : ∀ (x : β), Membership.mem s x → Membership.mem (t x) x
    ht : ∀ (x : β), Membership.mem s x → ∀ (z : β), Membership.mem (t x) z → Membe …
    x : β
    hx : Membership.mem (Set.iUnion fun x => Set.iUnion fun h => t x) x
    ⊢ Membership.mem r { fst := f x, snd := g x }
  -/
  simp only [mem_iUnion, exists_prop] at hx
  /-
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : TopologicalSpace β
    r : Set (Prod α α)
    s : Set β
    f g : β → α
    hf : ∀ (x : β), Membership.mem s x → ContinuousAt f x
    hg : ∀ (x : β), Membership.mem s x → ContinuousAt g x
    hfg : Set.EqOn f g s
    hr : Membership.mem (uniformity α) r
    t : β → Set β
    t_open : ∀ (x : β), Membership.mem s x → IsOpen (t x)
    xt : ∀ (x : β), Membership.mem s x → Membership.mem (t x) x
    ht : ∀ (x : β), Membership.mem s x → ∀ (z : β), Membership.mem (t x) z → Membe …
    x : β
    hx : Exists fun i => And (Membership.mem s i) (Membership.mem (t i) x)
    ⊢ Membership.mem r { fst := f x, snd := g x }
  -/
  rcases hx with ⟨y, ys, hy⟩
  /-
    case intro.intro
    α : Type ua
    β : Type ub
    inst✝¹ : UniformSpace α
    inst✝ : TopologicalSpace β
    r : Set (Prod α α)
    s : Set β
    f g : β → α
    hf : ∀ (x : β), Membership.mem s x → ContinuousAt f x
    hg : ∀ (x : β), Membership.mem s x → ContinuousAt g x
    hfg : Set.EqOn f g s
    hr : Membership.mem (uniformity α) r
    t : β → Set β
    t_open : ∀ (x : β), Membership.mem s x → IsOpen (t x)
    xt : ∀ (x : β), Membership.mem s x → Membership.mem (t x) x
    ht : ∀ (x : β), Membership.mem s x → ∀ (z : β), Membership.mem (t x) z → Membe …
    x y : β
    ys : Membership.mem s y
    hy : Membership.mem (t y) x
    ⊢ Membership.mem r { fst := f x, snd := g x }
  -/
  exact ht y ys x hy
  /-
    🎉 no goals
  -/


theorem Filter.Tendsto.congr_uniformity {α β} [UniformSpace β] {f g : α → β} {l : Filter α} {b : β}
    (hf : Tendsto f l (𝓝 b)) (hg : Tendsto (fun x => (f x, g x)) l (𝓤 β)) : Tendsto g l (𝓝 b) :=
  Uniform.tendsto_nhds_right.2 <| (Uniform.tendsto_nhds_right.1 hf).uniformity_trans hg


theorem Uniform.tendsto_congr {α β} [UniformSpace β] {f g : α → β} {l : Filter α} {b : β}
    (hfg : Tendsto (fun x => (f x, g x)) l (𝓤 β)) : Tendsto f l (𝓝 b) ↔ Tendsto g l (𝓝 b) :=
  ⟨fun h => h.congr_uniformity hfg, fun h => h.congr_uniformity hfg.uniformity_symm⟩


