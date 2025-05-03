/-- A `Partition` of a simple graph `G` is a structure constituted by
* `parts`: a set of subsets of the vertices `V` of `G`
* `isPartition`: a proof that `parts` is a proper partition of `V`
* `independent`: a proof that each element of `parts` doesn't have a pair of adjacent vertices
-/
structure Partition where
  /-- `parts`: a set of subsets of the vertices `V` of `G`. -/
  parts : Set (Set V)
  /-- `isPartition`: a proof that `parts` is a proper partition of `V`. -/
  isPartition : Setoid.IsPartition parts
  /-- `independent`: a proof that each element of `parts` doesn't have a pair of adjacent vertices.
-/
  independent : ∀ s ∈ parts, IsAntichain G.Adj s


/-- Whether a partition `P` has at most `n` parts. A graph with a partition
satisfying this predicate called `n`-partite. (See `SimpleGraph.Partitionable`.) -/
def Partition.PartsCardLe {G : SimpleGraph V} (P : G.Partition) (n : ℕ) : Prop :=
  ∃ h : P.parts.Finite, h.toFinset.card ≤ n


/-- Whether a graph is `n`-partite, which is whether its vertex set
can be partitioned in at most `n` independent sets. -/
def Partitionable (n : ℕ) : Prop := ∃ P : G.Partition, P.PartsCardLe n


/-- The part in the partition that `v` belongs to -/
def partOfVertex (v : V) : Set V := Classical.choose (P.isPartition.2 v)


theorem partOfVertex_mem (v : V) : P.partOfVertex v ∈ P.parts := by
  /-
    V : Type u
    G : SimpleGraph V
    P : G.Partition
    v : V
    ⊢ Membership.mem P.parts (P.partOfVertex v)
  -/
  obtain ⟨h, -⟩ := (P.isPartition.2 v).choose_spec.1
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    P : G.Partition
    v : V
    h : Membership.mem P.parts (Exists.choose ⋯)
    ⊢ Membership.mem P.parts (P.partOfVertex v)
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem mem_partOfVertex (v : V) : v ∈ P.partOfVertex v := by
  /-
    V : Type u
    G : SimpleGraph V
    P : G.Partition
    v : V
    ⊢ Membership.mem (P.partOfVertex v) v
  -/
  obtain ⟨⟨_, h⟩, _⟩ := (P.isPartition.2 v).choose_spec
  /-
    case intro.intro
    V : Type u
    G : SimpleGraph V
    P : G.Partition
    v : V
    right✝ : ∀ (y : Set V), (fun b => And (Membership.mem P.parts b) (Membership.m …
    left✝ : Membership.mem P.parts (Exists.choose ⋯)
    h : Membership.mem (Exists.choose ⋯) v
    ⊢ Membership.mem (P.partOfVertex v) v
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem partOfVertex_ne_of_adj {v w : V} (h : G.Adj v w) : P.partOfVertex v ≠ P.partOfVertex w := by
  /-
    V : Type u
    G : SimpleGraph V
    P : G.Partition
    v w : V
    h : G.Adj v w
    ⊢ Ne (P.partOfVertex v) (P.partOfVertex w)
  -/
  intro hn
  /-
    V : Type u
    G : SimpleGraph V
    P : G.Partition
    v w : V
    h : G.Adj v w
    hn : Eq (P.partOfVertex v) (P.partOfVertex w)
    ⊢ False
  -/
  have hw := P.mem_partOfVertex w
  /-
    V : Type u
    G : SimpleGraph V
    P : G.Partition
    v w : V
    h : G.Adj v w
    hn : Eq (P.partOfVertex v) (P.partOfVertex w)
    hw : Membership.mem (P.partOfVertex w) w
    ⊢ False
  -/
  rw [← hn] at hw
  /-
    V : Type u
    G : SimpleGraph V
    P : G.Partition
    v w : V
    h : G.Adj v w
    hn : Eq (P.partOfVertex v) (P.partOfVertex w)
    hw : Membership.mem (P.partOfVertex v) w
    ⊢ False
  -/
  exact P.independent _ (P.partOfVertex_mem v) (P.mem_partOfVertex v) hw (G.ne_of_adj h) h
  /-
    🎉 no goals
  -/


/-- Create a coloring using the parts themselves as the colors.
Each vertex is colored by the part it's contained in. -/
def toColoring : G.Coloring P.parts :=
  Coloring.mk (fun v ↦ ⟨P.partOfVertex v, P.partOfVertex_mem v⟩) fun hvw ↦ by
    /-
      V : Type u
      G : SimpleGraph V
      P : G.Partition
      v✝ w✝ : V
      hvw : G.Adj v✝ w✝
      ⊢ Ne ((fun v => ⟨P.partOfVertex v, ⋯⟩) v✝) ((fun v => ⟨P.partOfVertex v, ⋯⟩) w✝)
    -/
    rw [Ne, Subtype.mk_eq_mk]
    /-
      V : Type u
      G : SimpleGraph V
      P : G.Partition
      v✝ w✝ : V
      hvw : G.Adj v✝ w✝
      ⊢ Not (Eq (P.partOfVertex v✝) (P.partOfVertex w✝))
    -/
    exact P.partOfVertex_ne_of_adj hvw
    /-
      🎉 no goals
    -/


/-- Like `SimpleGraph.Partition.toColoring` but uses `Set V` as the coloring type. -/
def toColoring' : G.Coloring (Set V) :=
  Coloring.mk P.partOfVertex fun hvw ↦ P.partOfVertex_ne_of_adj hvw


theorem colorable [Fintype P.parts] : G.Colorable (Fintype.card P.parts) :=
  P.toColoring.colorable


/-- Creates a partition from a coloring. -/
@[simps]
def Coloring.toPartition {α : Type v} (C : G.Coloring α) : G.Partition where
  parts := C.colorClasses
  isPartition := C.colorClasses_isPartition
  independent := by
    /-
      V : Type u
      G : SimpleGraph V
      α : Type v
      C : G.Coloring α
      ⊢ ∀ (s : Set V), Membership.mem C.colorClasses s → IsAntichain G.Adj s
    -/
    rintro s ⟨c, rfl⟩
    /-
      case intro
      V : Type u
      G : SimpleGraph V
      α : Type v
      C : G.Coloring α
      c : V
      ⊢ IsAntichain G.Adj (setOf fun x => (Setoid.ker ⇑C) x c)
    -/
    apply C.color_classes_independent
    /-
      🎉 no goals
    -/


/-- The partition where every vertex is in its own part. -/
@[simps]
instance : Inhabited (Partition G) := ⟨G.selfColoring.toPartition⟩


theorem partitionable_iff_colorable {n : ℕ} : G.Partitionable n ↔ G.Colorable n := by
  /-
    V : Type u
    G : SimpleGraph V
    n : Nat
    ⊢ Iff (G.Partitionable n) (G.Colorable n)
  -/
  constructor
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      n : Nat
      ⊢ G.Partitionable n → G.Colorable n
    -/
  · rintro ⟨P, hf, hc⟩
    /-
      case mp.intro.intro
      V : Type u
      G : SimpleGraph V
      n : Nat
      P : G.Partition
      hf : P.parts.Finite
      hc : LE.le hf.toFinset.card n
      ⊢ G.Colorable n
    -/
    have : Fintype P.parts := hf.fintype
    /-
      case mp.intro.intro
      V : Type u
      G : SimpleGraph V
      n : Nat
      P : G.Partition
      hf : P.parts.Finite
      hc : LE.le hf.toFinset.card n
      this : Fintype ↑P.parts
      ⊢ G.Colorable n
    -/
    rw [Set.Finite.card_toFinset hf] at hc
    /-
      case mp.intro.intro
      V : Type u
      G : SimpleGraph V
      n : Nat
      P : G.Partition
      hf : P.parts.Finite
      this : Fintype ↑P.parts
      hc : LE.le (Fintype.card ↑P.parts) n
      ⊢ G.Colorable n
    -/
    apply P.colorable.mono hc
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u
      G : SimpleGraph V
      n : Nat
      ⊢ G.Colorable n → G.Partitionable n
    -/
  · rintro ⟨C⟩
    /-
      case mpr.intro
      V : Type u
      G : SimpleGraph V
      n : Nat
      C : G.Coloring (Fin n)
      ⊢ G.Partitionable n
    -/
    refine ⟨C.toPartition, C.colorClasses_finite, le_trans ?_ (Fintype.card_fin n).le⟩
    /-
      case mpr.intro
      V : Type u
      G : SimpleGraph V
      n : Nat
      C : G.Coloring (Fin n)
      ⊢ LE.le ⋯.toFinset.card (Fintype.card (Fin n))
    -/
    generalize_proofs h
    /-
      case mpr.intro
      V : Type u
      G : SimpleGraph V
      n : Nat
      C : G.Coloring (Fin n)
      h : C.toPartition.parts.Finite
      ⊢ LE.le h.toFinset.card (Fintype.card (Fin n))
    -/
    change Set.Finite (Coloring.colorClasses C) at h
    /-
      case mpr.intro
      V : Type u
      G : SimpleGraph V
      n : Nat
      C : G.Coloring (Fin n)
      h : C.colorClasses.Finite
      ⊢ LE.le h.toFinset.card (Fintype.card (Fin n))
    -/
    have : Fintype C.colorClasses := C.colorClasses_finite.fintype
    /-
      case mpr.intro
      V : Type u
      G : SimpleGraph V
      n : Nat
      C : G.Coloring (Fin n)
      h : C.colorClasses.Finite
      this : Fintype ↑C.colorClasses
      ⊢ LE.le h.toFinset.card (Fintype.card (Fin n))
    -/
    rw [h.card_toFinset]
    /-
      case mpr.intro
      V : Type u
      G : SimpleGraph V
      n : Nat
      C : G.Coloring (Fin n)
      h : C.colorClasses.Finite
      this : Fintype ↑C.colorClasses
      ⊢ LE.le (Fintype.card ↑C.colorClasses) (Fintype.card (Fin n))
    -/
    exact C.card_colorClasses_le
    /-
      🎉 no goals
    -/


