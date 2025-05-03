/-- A predicate `P` on sets satisfies the **exchange property** if,
  for all `X` and `Y` satisfying `P` and all `a ∈ X \ Y`, there exists `b ∈ Y \ X` so that
  swapping `a` for `b` in `X` maintains `P`. -/
def Matroid.ExchangeProperty {α : Type _} (P : Set α → Prop) : Prop :=
  ∀ X Y, P X → P Y → ∀ a ∈ X \ Y, ∃ b ∈ Y \ X, P (insert b (X \ {a}))


/-- A set `X` has the maximal subset property for a predicate `P` if every subset of `X` satisfying
  `P` is contained in a maximal subset of `X` satisfying `P`. -/
def Matroid.ExistsMaximalSubsetProperty {α : Type _} (P : Set α → Prop) (X : Set α) : Prop :=
  ∀ I, P I → I ⊆ X → ∃ J, I ⊆ J ∧ Maximal (fun K ↦ P K ∧ K ⊆ X) J


/-- A `Matroid α` is a ground set `E` of type `Set α`, and a nonempty collection of its subsets
  satisfying the exchange property and the maximal subset property. Each such set is called a
  `Base` of `M`. An `Indep`endent set is just a set contained in a base, but we include this
  predicate as a structure field for better definitional properties.

  In most cases, using this definition directly is not the best way to construct a matroid,
  since it requires specifying both the bases and independent sets. If the bases are known,
  use `Matroid.ofBase` or a variant. If just the independent sets are known,
  define an `IndepMatroid`, and then use `IndepMatroid.matroid`.
  -/
structure Matroid (α : Type _) where
  /-- `M` has a ground set `E`. -/
  (E : Set α)
  /-- `M` has a predicate `Base` defining its bases. -/
  (Base : Set α → Prop)
  /-- `M` has a predicate `Indep` defining its independent sets. -/
  (Indep : Set α → Prop)
  /-- The `Indep`endent sets are those contained in `Base`s. -/
  (indep_iff' : ∀ ⦃I⦄, Indep I ↔ ∃ B, Base B ∧ I ⊆ B)
  /-- There is at least one `Base`. -/
  (exists_base : ∃ B, Base B)
  /-- For any bases `B`, `B'` and `e ∈ B \ B'`, there is some `f ∈ B' \ B` for which `B-e+f`
    is a base. -/
  (base_exchange : Matroid.ExchangeProperty Base)
  /-- Every independent subset `I` of a set `X` for is contained in a maximal independent
    subset of `X`. -/
  (maximality : ∀ X, X ⊆ E → Matroid.ExistsMaximalSubsetProperty Indep X)
  /-- Every base is contained in the ground set. -/
  (subset_ground : ∀ B, Base B → B ⊆ E)


attribute [local ext] Matroid


/-- Typeclass for a matroid having finite ground set. Just a wrapper for `M.E.Finite`-/
protected class Finite (M : Matroid α) : Prop where
  /-- The ground set is finite -/
  (ground_finite : M.E.Finite)


/-- Typeclass for a matroid having nonempty ground set. Just a wrapper for `M.E.Nonempty`-/
protected class Nonempty (M : Matroid α) : Prop where
  /-- The ground set is nonempty -/
  (ground_nonempty : M.E.Nonempty)


theorem ground_nonempty (M : Matroid α) [M.Nonempty] : M.E.Nonempty :=
  Nonempty.ground_nonempty


theorem ground_nonempty_iff (M : Matroid α) : M.E.Nonempty ↔ M.Nonempty :=
  ⟨fun h ↦ ⟨h⟩, fun ⟨h⟩ ↦ h⟩


theorem ground_finite (M : Matroid α) [M.Finite] : M.E.Finite :=
  Finite.ground_finite


theorem set_finite (M : Matroid α) [M.Finite] (X : Set α) (hX : X ⊆ M.E := by aesop) : X.Finite :=
  M.ground_finite.subset hX


instance finite_of_finite [Finite α] {M : Matroid α} : M.Finite :=
  ⟨Set.toFinite _⟩


/-- A `FiniteRk` matroid is one whose bases are finite -/
class FiniteRk (M : Matroid α) : Prop where
  /-- There is a finite base -/
  exists_finite_base : ∃ B, M.Base B ∧ B.Finite


instance finiteRk_of_finite (M : Matroid α) [M.Finite] : FiniteRk M :=
  ⟨M.exists_base.imp (fun B hB ↦ ⟨hB, M.set_finite B (M.subset_ground _ hB)⟩)⟩


/-- An `InfiniteRk` matroid is one whose bases are infinite. -/
class InfiniteRk (M : Matroid α) : Prop where
  /-- There is an infinite base -/
  exists_infinite_base : ∃ B, M.Base B ∧ B.Infinite


/-- A `RkPos` matroid is one whose bases are nonempty. -/
class RkPos (M : Matroid α) : Prop where
  /-- The empty set isn't a base -/
  empty_not_base : ¬M.Base ∅


theorem rkPos_iff_empty_not_base : M.RkPos ↔ ¬M.Base ∅ :=
  ⟨fun ⟨h⟩ ↦ h, fun h ↦ ⟨h⟩⟩


/-- A family of sets with the exchange property is an antichain. -/
theorem antichain (exch : ExchangeProperty Base) (hB : Base B) (hB' : Base B') (h : B ⊆ B') :
    B = B' :=
  h.antisymm (fun x hx ↦ by_contra
    (fun hxB ↦ let ⟨_, hy, _⟩ := exch B' B hB' hB x ⟨hx, hxB⟩; hy.2 <| h hy.1))


theorem encard_diff_le_aux {B₁ B₂ : Set α}
    (exch : ExchangeProperty Base) (hB₁ : Base B₁) (hB₂ : Base B₂) :
    (B₁ \ B₂).encard ≤ (B₂ \ B₁).encard := by
  obtain (he | hinf | ⟨e, he, hcard⟩) :=
    (B₂ \ B₁).eq_empty_or_encard_eq_top_or_encard_diff_singleton_lt
    /-
      case inl
      α : Type u_1
      Base : Set α → Prop
      B₁ B₂ : Set α
      exch : Matroid.ExchangeProperty Base
      hB₁ : Base B₁
      hB₂ : Base B₂
      he : Eq (SDiff.sdiff B₂ B₁) EmptyCollection.emptyCollection
      ⊢ LE.le (SDiff.sdiff B₁ B₂).encard (SDiff.sdiff B₂ B₁).encard
    -/
  · rw [exch.antichain hB₂ hB₁ (diff_eq_empty.mp he)]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      Base : Set α → Prop
      B₁ B₂ : Set α
      exch : Matroid.ExchangeProperty Base
      hB₁ : Base B₁
      hB₂ : Base B₂
      hinf : Eq (SDiff.sdiff B₂ B₁).encard Top.top
      ⊢ LE.le (SDiff.sdiff B₁ B₂).encard (SDiff.sdiff B₂ B₁).encard
    -/
  · exact le_top.trans_eq hinf.symm
    /-
      🎉 no goals
    -/

  /-
    case inr.inr.intro.intro
    α : Type u_1
    Base : Set α → Prop
    B₁ B₂ : Set α
    exch : Matroid.ExchangeProperty Base
    hB₁ : Base B₁
    hB₂ : Base B₂
    e : α
    he : Membership.mem (SDiff.sdiff B₂ B₁) e
    hcard : LT.lt (SDiff.sdiff (SDiff.sdiff B₂ B₁) (Singleton.singleton e)).encard …
    ⊢ LE.le (SDiff.sdiff B₁ B₂).encard (SDiff.sdiff B₂ B₁).encard
  -/
  obtain ⟨f, hf, hB'⟩ := exch B₂ B₁ hB₂ hB₁ e he

  have : encard (insert f (B₂ \ {e}) \ B₁) < encard (B₂ \ B₁) := by
    rw [insert_diff_of_mem _ hf.1, diff_diff_comm]; exact hcard

  /-
    case inr.inr.intro.intro.intro.intro
    α : Type u_1
    Base : Set α → Prop
    B₁ B₂ : Set α
    exch : Matroid.ExchangeProperty Base
    hB₁ : Base B₁
    hB₂ : Base B₂
    e : α
    he : Membership.mem (SDiff.sdiff B₂ B₁) e
    hcard : LT.lt (SDiff.sdiff (SDiff.sdiff B₂ B₁) (Singleton.singleton e)).encard …
    f : α
    hf : Membership.mem (SDiff.sdiff B₁ B₂) f
    hB' : Base (Insert.insert f (SDiff.sdiff B₂ (Singleton.singleton e)))
    this : LT.lt (SDiff.sdiff (Insert.insert f (SDiff.sdiff B₂ (Singleton.singleto …
    ⊢ LE.le (SDiff.sdiff B₁ B₂).encard (SDiff.sdiff B₂ B₁).encard
  -/
  have hencard := encard_diff_le_aux exch hB₁ hB'
  rw [insert_diff_of_mem _ hf.1, diff_diff_comm, ← union_singleton, ← diff_diff, diff_diff_right,
    inter_singleton_eq_empty.mpr he.2, union_empty] at hencard

  /-
    case inr.inr.intro.intro.intro.intro
    α : Type u_1
    Base : Set α → Prop
    B₁ B₂ : Set α
    exch : Matroid.ExchangeProperty Base
    hB₁ : Base B₁
    hB₂ : Base B₂
    e : α
    he : Membership.mem (SDiff.sdiff B₂ B₁) e
    hcard : LT.lt (SDiff.sdiff (SDiff.sdiff B₂ B₁) (Singleton.singleton e)).encard …
    f : α
    hf : Membership.mem (SDiff.sdiff B₁ B₂) f
    hB' : Base (Insert.insert f (SDiff.sdiff B₂ (Singleton.singleton e)))
    this : LT.lt (SDiff.sdiff (Insert.insert f (SDiff.sdiff B₂ (Singleton.singleto …
    hencard : LE.le (SDiff.sdiff (SDiff.sdiff B₁ B₂) (Singleton.singleton f)).enca …
    ⊢ LE.le (SDiff.sdiff B₁ B₂).encard (SDiff.sdiff B₂ B₁).encard
  -/
  rw [← encard_diff_singleton_add_one he, ← encard_diff_singleton_add_one hf]
  /-
    case inr.inr.intro.intro.intro.intro
    α : Type u_1
    Base : Set α → Prop
    B₁ B₂ : Set α
    exch : Matroid.ExchangeProperty Base
    hB₁ : Base B₁
    hB₂ : Base B₂
    e : α
    he : Membership.mem (SDiff.sdiff B₂ B₁) e
    hcard : LT.lt (SDiff.sdiff (SDiff.sdiff B₂ B₁) (Singleton.singleton e)).encard …
    f : α
    hf : Membership.mem (SDiff.sdiff B₁ B₂) f
    hB' : Base (Insert.insert f (SDiff.sdiff B₂ (Singleton.singleton e)))
    this : LT.lt (SDiff.sdiff (Insert.insert f (SDiff.sdiff B₂ (Singleton.singleto …
    hencard : LE.le (SDiff.sdiff (SDiff.sdiff B₁ B₂) (Singleton.singleton f)).enca …
    ⊢ LE.le (HAdd.hAdd (SDiff.sdiff (SDiff.sdiff B₁ B₂) (Singleton.singleton f)).e …
  -/
  exact add_le_add_right hencard 1
  /-
    🎉 no goals
  -/
termination_by (B₂ \ B₁).encard


/-- For any two sets `B₁`, `B₂` in a family with the exchange property, the differences `B₁ \ B₂`
and `B₂ \ B₁` have the same `ℕ∞`-cardinality. -/
theorem encard_diff_eq (exch : ExchangeProperty Base) (hB₁ : Base B₁) (hB₂ : Base B₂) :
    (B₁ \ B₂).encard = (B₂ \ B₁).encard :=
  (encard_diff_le_aux exch hB₁ hB₂).antisymm (encard_diff_le_aux exch hB₂ hB₁)


/-- Any two sets `B₁`, `B₂` in a family with the exchange property have the same
`ℕ∞`-cardinality. -/
theorem encard_base_eq (exch : ExchangeProperty Base) (hB₁ : Base B₁) (hB₂ : Base B₂) :
    B₁.encard = B₂.encard := by
  rw [← encard_diff_add_encard_inter B₁ B₂, exch.encard_diff_eq hB₁ hB₂, inter_comm,
    encard_diff_add_encard_inter]


/-- The `aesop_mat` tactic attempts to prove a set is contained in the ground set of a matroid.
  It uses a `[Matroid]` ruleset, and is allowed to fail. -/
macro (name := aesop_mat) "aesop_mat" c:Aesop.tactic_clause* : tactic =>
`(tactic|
  aesop $c* (config := { terminal := true })
  (rule_sets := [$(Lean.mkIdent `Matroid):ident]))

/- We add a number of trivial lemmas (deliberately specialized to statements in terms of the
  ground set of a matroid) to the ruleset `Matroid` for `aesop`. -/


@[aesop unsafe 5% (rule_sets := [Matroid])]
private theorem inter_right_subset_ground (hX : X ⊆ M.E) :
    X ∩ Y ⊆ M.E := inter_subset_left.trans hX


@[aesop unsafe 5% (rule_sets := [Matroid])]
private theorem inter_left_subset_ground (hX : X ⊆ M.E) :
    Y ∩ X ⊆ M.E := inter_subset_right.trans hX


@[aesop unsafe 5% (rule_sets := [Matroid])]
private theorem diff_subset_ground (hX : X ⊆ M.E) : X \ Y ⊆ M.E :=
  diff_subset.trans hX


@[aesop unsafe 10% (rule_sets := [Matroid])]
private theorem ground_diff_subset_ground : M.E \ X ⊆ M.E :=
  diff_subset_ground rfl.subset


@[aesop unsafe 10% (rule_sets := [Matroid])]
private theorem singleton_subset_ground (he : e ∈ M.E) : {e} ⊆ M.E :=
  singleton_subset_iff.mpr he


@[aesop unsafe 5% (rule_sets := [Matroid])]
private theorem subset_ground_of_subset (hXY : X ⊆ Y) (hY : Y ⊆ M.E) : X ⊆ M.E :=
  hXY.trans hY


@[aesop unsafe 5% (rule_sets := [Matroid])]
private theorem mem_ground_of_mem_of_subset (hX : X ⊆ M.E) (heX : e ∈ X) : e ∈ M.E :=
  hX heX


@[aesop safe (rule_sets := [Matroid])]
private theorem insert_subset_ground {e : α} {X : Set α} {M : Matroid α}
    (he : e ∈ M.E) (hX : X ⊆ M.E) : insert e X ⊆ M.E :=
    insert_subset he hX


@[aesop safe (rule_sets := [Matroid])]
private theorem ground_subset_ground {M : Matroid α} : M.E ⊆ M.E :=
  rfl.subset


@[aesop unsafe 10% (rule_sets := [Matroid])]
theorem Base.subset_ground (hB : M.Base B) : B ⊆ M.E :=
  M.subset_ground B hB


theorem Base.exchange {e : α} (hB₁ : M.Base B₁) (hB₂ : M.Base B₂) (hx : e ∈ B₁ \ B₂) :
    ∃ y ∈ B₂ \ B₁, M.Base (insert y (B₁ \ {e}))  :=
  M.base_exchange B₁ B₂ hB₁ hB₂ _ hx


theorem Base.exchange_mem {e : α}
    (hB₁ : M.Base B₁) (hB₂ : M.Base B₂) (hxB₁ : e ∈ B₁) (hxB₂ : e ∉ B₂) :
    ∃ y, (y ∈ B₂ ∧ y ∉ B₁) ∧ M.Base (insert y (B₁ \ {e})) := by
  /-
    α : Type u_1
    M : Matroid α
    B₁ B₂ : Set α
    e : α
    hB₁ : M.Base B₁
    hB₂ : M.Base B₂
    hxB₁ : Membership.mem B₁ e
    hxB₂ : Not (Membership.mem B₂ e)
    ⊢ Exists fun y => And (And (Membership.mem B₂ y) (Not (Membership.mem B₁ y)))  …
  -/
  simpa using hB₁.exchange hB₂ ⟨hxB₁, hxB₂⟩
  /-
    🎉 no goals
  -/


theorem Base.eq_of_subset_base (hB₁ : M.Base B₁) (hB₂ : M.Base B₂) (hB₁B₂ : B₁ ⊆ B₂) :
    B₁ = B₂ :=
  M.base_exchange.antichain hB₁ hB₂ hB₁B₂


theorem Base.not_base_of_ssubset {X : Set α} (hB : M.Base B) (hX : X ⊂ B) : ¬ M.Base X :=
  fun h ↦ hX.ne (h.eq_of_subset_base hB hX.subset)


theorem Base.insert_not_base {e : α} (hB : M.Base B) (heB : e ∉ B) : ¬ M.Base (insert e B) :=
  fun h ↦ h.not_base_of_ssubset (ssubset_insert heB) hB


theorem Base.encard_diff_comm (hB₁ : M.Base B₁) (hB₂ : M.Base B₂) :
    (B₁ \ B₂).encard = (B₂ \ B₁).encard :=
  M.base_exchange.encard_diff_eq hB₁ hB₂


theorem Base.ncard_diff_comm (hB₁ : M.Base B₁) (hB₂ : M.Base B₂) :
    (B₁ \ B₂).ncard = (B₂ \ B₁).ncard := by
  /-
    α : Type u_1
    M : Matroid α
    B₁ B₂ : Set α
    hB₁ : M.Base B₁
    hB₂ : M.Base B₂
    ⊢ Eq (SDiff.sdiff B₁ B₂).ncard (SDiff.sdiff B₂ B₁).ncard
  -/
  rw [ncard_def, hB₁.encard_diff_comm hB₂, ← ncard_def]
  /-
    🎉 no goals
  -/


theorem Base.card_eq_card_of_base (hB₁ : M.Base B₁) (hB₂ : M.Base B₂) :
    B₁.encard = B₂.encard := by
  /-
    α : Type u_1
    M : Matroid α
    B₁ B₂ : Set α
    hB₁ : M.Base B₁
    hB₂ : M.Base B₂
    ⊢ Eq B₁.encard B₂.encard
  -/
  rw [M.base_exchange.encard_base_eq hB₁ hB₂]
  /-
    🎉 no goals
  -/


theorem Base.ncard_eq_ncard_of_base (hB₁ : M.Base B₁) (hB₂ : M.Base B₂) : B₁.ncard = B₂.ncard := by
  /-
    α : Type u_1
    M : Matroid α
    B₁ B₂ : Set α
    hB₁ : M.Base B₁
    hB₂ : M.Base B₂
    ⊢ Eq B₁.ncard B₂.ncard
  -/
  rw [ncard_def B₁, hB₁.card_eq_card_of_base hB₂, ← ncard_def]
  /-
    🎉 no goals
  -/


theorem Base.finite_of_finite {B' : Set α}
    (hB : M.Base B) (h : B.Finite) (hB' : M.Base B') : B'.Finite :=
  (finite_iff_finite_of_encard_eq_encard (hB.card_eq_card_of_base hB')).mp h


theorem Base.infinite_of_infinite (hB : M.Base B) (h : B.Infinite) (hB₁ : M.Base B₁) :
    B₁.Infinite :=
  by_contra (fun hB_inf ↦ (hB₁.finite_of_finite (not_infinite.mp hB_inf) hB).not_infinite h)


theorem Base.finite [FiniteRk M] (hB : M.Base B) : B.Finite :=
  let ⟨_,hB₀⟩ := ‹FiniteRk M›.exists_finite_base
  hB₀.1.finite_of_finite hB₀.2 hB


theorem Base.infinite [InfiniteRk M] (hB : M.Base B) : B.Infinite :=
  let ⟨_,hB₀⟩ := ‹InfiniteRk M›.exists_infinite_base
  hB₀.1.infinite_of_infinite hB₀.2 hB


theorem empty_not_base [h : RkPos M] : ¬M.Base ∅ :=
  h.empty_not_base


theorem Base.nonempty [RkPos M] (hB : M.Base B) : B.Nonempty := by
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    inst✝ : M.RkPos
    hB : M.Base B
    ⊢ B.Nonempty
  -/
  rw [nonempty_iff_ne_empty]; rintro rfl; exact M.empty_not_base hB
                                          /-
                                            🎉 no goals
                                          -/


theorem Base.rkPos_of_nonempty (hB : M.Base B) (h : B.Nonempty) : M.RkPos := by
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    hB : M.Base B
    h : B.Nonempty
    ⊢ M.RkPos
  -/
  rw [rkPos_iff_empty_not_base]
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    hB : M.Base B
    h : B.Nonempty
    ⊢ Not (M.Base EmptyCollection.emptyCollection)
  -/
  intro he
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    hB : M.Base B
    h : B.Nonempty
    he : M.Base EmptyCollection.emptyCollection
    ⊢ False
  -/
  obtain rfl := he.eq_of_subset_base hB (empty_subset B)
  /-
    α : Type u_1
    M : Matroid α
    he hB : M.Base EmptyCollection.emptyCollection
    h : EmptyCollection.emptyCollection.Nonempty
    ⊢ False
  -/
  simp at h
  /-
    🎉 no goals
  -/


theorem Base.finiteRk_of_finite (hB : M.Base B) (hfin : B.Finite) : FiniteRk M :=
  ⟨⟨B, hB, hfin⟩⟩


theorem Base.infiniteRk_of_infinite (hB : M.Base B) (h : B.Infinite) : InfiniteRk M :=
  ⟨⟨B, hB, h⟩⟩


theorem not_finiteRk (M : Matroid α) [InfiniteRk M] : ¬ FiniteRk M := by
  /-
    α : Type u_1
    M : Matroid α
    inst✝ : M.InfiniteRk
    ⊢ Not M.FiniteRk
  -/
  intro h; obtain ⟨B,hB⟩ := M.exists_base; exact hB.infinite hB.finite
                                           /-
                                             🎉 no goals
                                           -/


theorem not_infiniteRk (M : Matroid α) [FiniteRk M] : ¬ InfiniteRk M := by
  /-
    α : Type u_1
    M : Matroid α
    inst✝ : M.FiniteRk
    ⊢ Not M.InfiniteRk
  -/
  intro h; obtain ⟨B,hB⟩ := M.exists_base; exact hB.infinite hB.finite
                                           /-
                                             🎉 no goals
                                           -/


theorem finite_or_infiniteRk (M : Matroid α) : FiniteRk M ∨ InfiniteRk M :=
  let ⟨B, hB⟩ := M.exists_base
  B.finite_or_infinite.elim
  (Or.inl ∘ hB.finiteRk_of_finite) (Or.inr ∘ hB.infiniteRk_of_infinite)


theorem Base.diff_finite_comm (hB₁ : M.Base B₁) (hB₂ : M.Base B₂) :
    (B₁ \ B₂).Finite ↔ (B₂ \ B₁).Finite :=
  finite_iff_finite_of_encard_eq_encard (hB₁.encard_diff_comm hB₂)


theorem Base.diff_infinite_comm (hB₁ : M.Base B₁) (hB₂ : M.Base B₂) :
    (B₁ \ B₂).Infinite ↔ (B₂ \ B₁).Infinite :=
  infinite_iff_infinite_of_encard_eq_encard (hB₁.encard_diff_comm hB₂)


theorem ext_base {M₁ M₂ : Matroid α} (hE : M₁.E = M₂.E)
    (h : ∀ ⦃B⦄, B ⊆ M₁.E → (M₁.Base B ↔ M₂.Base B)) : M₁ = M₂ := by
  have h' : ∀ B, M₁.Base B ↔ M₂.Base B :=
    fun B ↦ ⟨fun hB ↦ (h hB.subset_ground).1 hB,
      fun hB ↦ (h <| hB.subset_ground.trans_eq hE.symm).2 hB⟩
  /-
    α : Type u_1
    M₁ M₂ : Matroid α
    hE : Eq M₁.E M₂.E
    h : ∀ ⦃B : Set α⦄, HasSubset.Subset B M₁.E → Iff (M₁.Base B) (M₂.Base B)
    h' : ∀ (B : Set α), Iff (M₁.Base B) (M₂.Base B)
    ⊢ Eq M₁ M₂
  -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
  ext <;> simp [hE, M₁.indep_iff', M₂.indep_iff', h']
          /-
            🎉 no goals
          -/


@[deprecated (since := "2024-12-25")] alias eq_of_base_iff_base_forall := ext_base


theorem ext_iff_base {M₁ M₂ : Matroid α} :
    M₁ = M₂ ↔ M₁.E = M₂.E ∧ ∀ ⦃B⦄, B ⊆ M₁.E → (M₁.Base B ↔ M₂.Base B) :=
              /-
                α : Type u_1
                M₁ M₂ : Matroid α
                h : Eq M₁ M₂
                ⊢ And (Eq M₁.E M₂.E) (∀ ⦃B : Set α⦄, HasSubset.Subset B M₁.E → Iff (M₁.Base B) …
              -/
  ⟨fun h ↦ by simp [h], fun ⟨hE, h⟩ ↦ ext_base hE h⟩
              /-
                🎉 no goals
              -/


theorem base_compl_iff_maximal_disjoint_base (hB : B ⊆ M.E := by aesop_mat) :
    M.Base (M.E \ B) ↔ Maximal (fun I ↦ I ⊆ M.E ∧ ∃ B, M.Base B ∧ Disjoint I B) B := by
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    hB : autoParam (HasSubset.Subset B M.E) _auto✝
    ⊢ Iff (M.Base (SDiff.sdiff M.E B)) (Maximal (fun I => And (HasSubset.Subset I  …
  -/
  simp_rw [maximal_iff, and_iff_right hB, and_imp, forall_exists_index]
  refine ⟨fun h ↦ ⟨⟨_, h, disjoint_sdiff_right⟩,
    fun I hI B' ⟨hB', hIB'⟩ hBI ↦ hBI.antisymm ?_⟩, fun ⟨⟨B', hB', hBB'⟩,h⟩ ↦ ?_⟩
  · rw [hB'.eq_of_subset_base h, ← subset_compl_iff_disjoint_right, diff_eq, compl_inter,
      compl_compl] at hIB'
      /-
        case refine_1
        α : Type u_1
        M : Matroid α
        B : Set α
        hB : autoParam (HasSubset.Subset B M.E) _auto✝
        h : M.Base (SDiff.sdiff M.E B)
        I : Set α
        hI : HasSubset.Subset I M.E
        B' : Set α
        x✝ : And (M.Base B') (Disjoint I B')
        hBI : LE.le B I
        hB' : M.Base B'
        hIB' : HasSubset.Subset I (Union.union (HasCompl.compl M.E) B)
        ⊢ LE.le I B
      -/
    · exact fun e he ↦ (hIB' he).elim (fun h' ↦ (h' (hI he)).elim) id
      /-
        🎉 no goals
      -/
    /-
      case refine_1
      α : Type u_1
      M : Matroid α
      B : Set α
      hB : autoParam (HasSubset.Subset B M.E) _auto✝
      h : M.Base (SDiff.sdiff M.E B)
      I : Set α
      hI : HasSubset.Subset I M.E
      B' : Set α
      x✝ : And (M.Base B') (Disjoint I B')
      hBI : LE.le B I
      hB' : M.Base B'
      hIB' : Disjoint I B'
      ⊢ HasSubset.Subset B' (SDiff.sdiff M.E B)
    -/
    rw [subset_diff, and_iff_right hB'.subset_ground, disjoint_comm]
    /-
      case refine_1
      α : Type u_1
      M : Matroid α
      B : Set α
      hB : autoParam (HasSubset.Subset B M.E) _auto✝
      h : M.Base (SDiff.sdiff M.E B)
      I : Set α
      hI : HasSubset.Subset I M.E
      B' : Set α
      x✝ : And (M.Base B') (Disjoint I B')
      hBI : LE.le B I
      hB' : M.Base B'
      hIB' : Disjoint I B'
      ⊢ Disjoint B B'
    -/
    exact disjoint_of_subset_left hBI hIB'
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    M : Matroid α
    B : Set α
    hB : autoParam (HasSubset.Subset B M.E) _auto✝
    x✝ : And (Exists fun B_1 => And (M.Base B_1) (Disjoint B B_1)) (∀ ⦃y : Set α⦄, …
    B' : Set α
    hB' : M.Base B'
    hBB' : Disjoint B B'
    h : ∀ ⦃y : Set α⦄, HasSubset.Subset y M.E → ∀ (x : Set α), And (M.Base x) (Dis …
    ⊢ M.Base (SDiff.sdiff M.E B)
  -/
  rw [h diff_subset B' ⟨hB', disjoint_sdiff_left⟩]
    /-
      case refine_2
      α : Type u_1
      M : Matroid α
      B : Set α
      hB : autoParam (HasSubset.Subset B M.E) _auto✝
      x✝ : And (Exists fun B_1 => And (M.Base B_1) (Disjoint B B_1)) (∀ ⦃y : Set α⦄, …
      B' : Set α
      hB' : M.Base B'
      hBB' : Disjoint B B'
      h : ∀ ⦃y : Set α⦄, HasSubset.Subset y M.E → ∀ (x : Set α), And (M.Base x) (Dis …
      ⊢ M.Base (SDiff.sdiff M.E (SDiff.sdiff M.E B'))
    -/
  · simpa [hB'.subset_ground]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    M : Matroid α
    B : Set α
    hB : autoParam (HasSubset.Subset B M.E) _auto✝
    x✝ : And (Exists fun B_1 => And (M.Base B_1) (Disjoint B B_1)) (∀ ⦃y : Set α⦄, …
    B' : Set α
    hB' : M.Base B'
    hBB' : Disjoint B B'
    h : ∀ ⦃y : Set α⦄, HasSubset.Subset y M.E → ∀ (x : Set α), And (M.Base x) (Dis …
    ⊢ LE.le B (SDiff.sdiff M.E B')
  -/
  simp [subset_diff, hB, hBB']
  /-
    🎉 no goals
  -/


/-- A subset of `M.E` is `Dep`endent if it is not `Indep`endent . -/
def Dep (M : Matroid α) (D : Set α) : Prop := ¬M.Indep D ∧ D ⊆ M.E


theorem indep_iff : M.Indep I ↔ ∃ B, M.Base B ∧ I ⊆ B :=
  M.indep_iff' (I := I)


theorem setOf_indep_eq (M : Matroid α) : {I | M.Indep I} = lowerClosure ({B | M.Base B}) := by
  /-
    α : Type u_1
    M : Matroid α
    ⊢ Eq (setOf fun I => M.Indep I) ↑(lowerClosure (setOf fun B => M.Base B))
  -/
  simp_rw [indep_iff]
  /-
    α : Type u_1
    M : Matroid α
    ⊢ Eq (setOf fun I => Exists fun B => And (M.Base B) (HasSubset.Subset I B)) ↑( …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Indep.exists_base_superset (hI : M.Indep I) : ∃ B, M.Base B ∧ I ⊆ B :=
  indep_iff.1 hI


theorem dep_iff : M.Dep D ↔ ¬M.Indep D ∧ D ⊆ M.E := Iff.rfl


theorem setOf_dep_eq (M : Matroid α) : {D | M.Dep D} = {I | M.Indep I}ᶜ ∩ Iic M.E := rfl


@[aesop unsafe 30% (rule_sets := [Matroid])]
theorem Indep.subset_ground (hI : M.Indep I) : I ⊆ M.E := by
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    hI : M.Indep I
    ⊢ HasSubset.Subset I M.E
  -/
  obtain ⟨B, hB, hIB⟩ := hI.exists_base_superset
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    I : Set α
    hI : M.Indep I
    B : Set α
    hB : M.Base B
    hIB : HasSubset.Subset I B
    ⊢ HasSubset.Subset I M.E
  -/
  exact hIB.trans hB.subset_ground
  /-
    🎉 no goals
  -/


@[aesop unsafe 20% (rule_sets := [Matroid])]
theorem Dep.subset_ground (hD : M.Dep D) : D ⊆ M.E :=
  hD.2


theorem indep_or_dep (hX : X ⊆ M.E := by aesop_mat) : M.Indep X ∨ M.Dep X := by
  /-
    α : Type u_1
    M : Matroid α
    X : Set α
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ Or (M.Indep X) (M.Dep X)
  -/
  rw [Dep, and_iff_left hX]
  /-
    α : Type u_1
    M : Matroid α
    X : Set α
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ Or (M.Indep X) (Not (M.Indep X))
  -/
  apply em
  /-
    🎉 no goals
  -/


theorem Indep.not_dep (hI : M.Indep I) : ¬ M.Dep I :=
  fun h ↦ h.1 hI


theorem Dep.not_indep (hD : M.Dep D) : ¬ M.Indep D :=
  hD.1


theorem dep_of_not_indep (hD : ¬ M.Indep D) (hDE : D ⊆ M.E := by aesop_mat) : M.Dep D :=
  ⟨hD, hDE⟩


theorem indep_of_not_dep (hI : ¬ M.Dep I) (hIE : I ⊆ M.E := by aesop_mat) : M.Indep I :=
  by_contra (fun h ↦ hI ⟨h, hIE⟩)


@[simp] theorem not_dep_iff (hX : X ⊆ M.E := by aesop_mat) : ¬ M.Dep X ↔ M.Indep X := by
  /-
    α : Type u_1
    M : Matroid α
    X : Set α
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ Iff (Not (M.Dep X)) (M.Indep X)
  -/
  rw [Dep, and_iff_left hX, not_not]
  /-
    🎉 no goals
  -/


@[simp] theorem not_indep_iff (hX : X ⊆ M.E := by aesop_mat) : ¬ M.Indep X ↔ M.Dep X := by
  /-
    α : Type u_1
    M : Matroid α
    X : Set α
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ Iff (Not (M.Indep X)) (M.Dep X)
  -/
  rw [Dep, and_iff_left hX]
  /-
    🎉 no goals
  -/


theorem indep_iff_not_dep : M.Indep I ↔ ¬M.Dep I ∧ I ⊆ M.E := by
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    ⊢ Iff (M.Indep I) (And (Not (M.Dep I)) (HasSubset.Subset I M.E))
  -/
  rw [dep_iff, not_and, not_imp_not]
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    ⊢ Iff (M.Indep I) (And (HasSubset.Subset I M.E → M.Indep I) (HasSubset.Subset  …
  -/
  exact ⟨fun h ↦ ⟨fun _ ↦ h, h.subset_ground⟩, fun h ↦ h.1 h.2⟩
  /-
    🎉 no goals
  -/


theorem Indep.subset (hJ : M.Indep J) (hIJ : I ⊆ J) : M.Indep I := by
  /-
    α : Type u_1
    M : Matroid α
    I J : Set α
    hJ : M.Indep J
    hIJ : HasSubset.Subset I J
    ⊢ M.Indep I
  -/
  obtain ⟨B, hB, hJB⟩ := hJ.exists_base_superset
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    I J : Set α
    hJ : M.Indep J
    hIJ : HasSubset.Subset I J
    B : Set α
    hB : M.Base B
    hJB : HasSubset.Subset J B
    ⊢ M.Indep I
  -/
  exact indep_iff.2 ⟨B, hB, hIJ.trans hJB⟩
  /-
    🎉 no goals
  -/


theorem Dep.superset (hD : M.Dep D) (hDX : D ⊆ X) (hXE : X ⊆ M.E := by aesop_mat) : M.Dep X :=
  /-
    α : Type u_1
    M : Matroid α
    D X : Set α
    hD : M.Dep D
    hDX : HasSubset.Subset D X
    hXE : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ HasSubset.Subset X M.E
  -/
  dep_of_not_indep (fun hI ↦ (hI.subset hDX).not_dep hD)
  /-
    🎉 no goals
  -/


theorem Base.indep (hB : M.Base B) : M.Indep B :=
  indep_iff.2 ⟨B, hB, subset_rfl⟩


@[simp] theorem empty_indep (M : Matroid α) : M.Indep ∅ :=
  Exists.elim M.exists_base (fun _ hB ↦ hB.indep.subset (empty_subset _))


theorem Dep.nonempty (hD : M.Dep D) : D.Nonempty := by
  /-
    α : Type u_1
    M : Matroid α
    D : Set α
    hD : M.Dep D
    ⊢ D.Nonempty
  -/
  rw [nonempty_iff_ne_empty]; rintro rfl; exact hD.not_indep M.empty_indep
                                          /-
                                            🎉 no goals
                                          -/


theorem Indep.finite [FiniteRk M] (hI : M.Indep I) : I.Finite :=
  let ⟨_, hB, hIB⟩ := hI.exists_base_superset
  hB.finite.subset hIB


theorem Indep.rkPos_of_nonempty (hI : M.Indep I) (hne : I.Nonempty) : M.RkPos := by
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    hI : M.Indep I
    hne : I.Nonempty
    ⊢ M.RkPos
  -/
  obtain ⟨B, hB, hIB⟩ := hI.exists_base_superset
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    I : Set α
    hI : M.Indep I
    hne : I.Nonempty
    B : Set α
    hB : M.Base B
    hIB : HasSubset.Subset I B
    ⊢ M.RkPos
  -/
  exact hB.rkPos_of_nonempty (hne.mono hIB)
  /-
    🎉 no goals
  -/


theorem Indep.inter_right (hI : M.Indep I) (X : Set α) : M.Indep (I ∩ X) :=
  hI.subset inter_subset_left


theorem Indep.inter_left (hI : M.Indep I) (X : Set α) : M.Indep (X ∩ I) :=
  hI.subset inter_subset_right


theorem Indep.diff (hI : M.Indep I) (X : Set α) : M.Indep (I \ X) :=
  hI.subset diff_subset


theorem Base.eq_of_subset_indep (hB : M.Base B) (hI : M.Indep I) (hBI : B ⊆ I) : B = I :=
  let ⟨B', hB', hB'I⟩ := hI.exists_base_superset
                   /-
                     α : Type u_1
                     M : Matroid α
                     B I : Set α
                     hB : M.Base B
                     hI : M.Indep I
                     hBI : HasSubset.Subset B I
                     B' : Set α
                     hB' : M.Base B'
                     hB'I : HasSubset.Subset I B'
                     ⊢ HasSubset.Subset I B
                   -/
  hBI.antisymm (by rwa [hB.eq_of_subset_base hB' (hBI.trans hB'I)])
                   /-
                     🎉 no goals
                   -/


theorem base_iff_maximal_indep : M.Base B ↔ Maximal M.Indep B := by
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    ⊢ Iff (M.Base B) (Maximal M.Indep B)
  -/
  rw [maximal_subset_iff]
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    ⊢ Iff (M.Base B) (And (M.Indep B) (∀ ⦃t : Set α⦄, M.Indep t → HasSubset.Subset …
  -/
  refine ⟨fun h ↦ ⟨h.indep, fun _ ↦ h.eq_of_subset_indep⟩, fun ⟨h, h'⟩ ↦ ?_⟩
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    x✝ : And (M.Indep B) (∀ ⦃t : Set α⦄, M.Indep t → HasSubset.Subset B t → Eq B t)
    h : M.Indep B
    h' : ∀ ⦃t : Set α⦄, M.Indep t → HasSubset.Subset B t → Eq B t
    ⊢ M.Base B
  -/
  obtain ⟨B', hB', hBB'⟩ := h.exists_base_superset
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    B : Set α
    x✝ : And (M.Indep B) (∀ ⦃t : Set α⦄, M.Indep t → HasSubset.Subset B t → Eq B t)
    h : M.Indep B
    h' : ∀ ⦃t : Set α⦄, M.Indep t → HasSubset.Subset B t → Eq B t
    B' : Set α
    hB' : M.Base B'
    hBB' : HasSubset.Subset B B'
    ⊢ M.Base B
  -/
  rwa [h' hB'.indep hBB']
  /-
    🎉 no goals
  -/


theorem Indep.base_of_maximal (hI : M.Indep I) (h : ∀ ⦃J⦄, M.Indep J → I ⊆ J → I = J) :
    M.Base I := by
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    hI : M.Indep I
    h : ∀ ⦃J : Set α⦄, M.Indep J → HasSubset.Subset I J → Eq I J
    ⊢ M.Base I
  -/
  rwa [base_iff_maximal_indep, maximal_subset_iff, and_iff_right hI]
  /-
    🎉 no goals
  -/


theorem Base.dep_of_ssubset (hB : M.Base B) (h : B ⊂ X) (hX : X ⊆ M.E := by aesop_mat) : M.Dep X :=
  ⟨fun hX ↦ h.ne (hB.eq_of_subset_indep hX h.subset), hX⟩


theorem Base.dep_of_insert (hB : M.Base B) (heB : e ∉ B) (he : e ∈ M.E := by aesop_mat) :
    M.Dep (insert e B) := hB.dep_of_ssubset (ssubset_insert heB) (insert_subset he hB.subset_ground)


theorem Base.mem_of_insert_indep (hB : M.Base B) (heB : M.Indep (insert e B)) : e ∈ B :=
  by_contra fun he ↦ (hB.dep_of_insert he (heB.subset_ground (mem_insert _ _))).not_indep heB


/-- If the difference of two Bases is a singleton, then they differ by an insertion/removal -/
theorem Base.eq_exchange_of_diff_eq_singleton (hB : M.Base B) (hB' : M.Base B') (h : B \ B' = {e}) :
    ∃ f ∈ B' \ B, B' = (insert f B) \ {e} := by
  /-
    α : Type u_1
    M : Matroid α
    B B' : Set α
    e : α
    hB : M.Base B
    hB' : M.Base B'
    h : Eq (SDiff.sdiff B B') (Singleton.singleton e)
    ⊢ Exists fun f => And (Membership.mem (SDiff.sdiff B' B) f) (Eq B' (SDiff.sdif …
  -/
  obtain ⟨f, hf, hb⟩ := hB.exchange hB' (h.symm.subset (mem_singleton e))
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    B B' : Set α
    e : α
    hB : M.Base B
    hB' : M.Base B'
    h : Eq (SDiff.sdiff B B') (Singleton.singleton e)
    f : α
    hf : Membership.mem (SDiff.sdiff B' B) f
    hb : M.Base (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
    ⊢ Exists fun f => And (Membership.mem (SDiff.sdiff B' B) f) (Eq B' (SDiff.sdif …
  -/
  have hne : f ≠ e := by rintro rfl; exact hf.2 (h.symm.subset (mem_singleton f)).1
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    B B' : Set α
    e : α
    hB : M.Base B
    hB' : M.Base B'
    h : Eq (SDiff.sdiff B B') (Singleton.singleton e)
    f : α
    hf : Membership.mem (SDiff.sdiff B' B) f
    hb : M.Base (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
    hne : Ne f e
    ⊢ Exists fun f => And (Membership.mem (SDiff.sdiff B' B) f) (Eq B' (SDiff.sdif …
  -/
  rw [insert_diff_singleton_comm hne] at hb
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    B B' : Set α
    e : α
    hB : M.Base B
    hB' : M.Base B'
    h : Eq (SDiff.sdiff B B') (Singleton.singleton e)
    f : α
    hf : Membership.mem (SDiff.sdiff B' B) f
    hb : M.Base (SDiff.sdiff (Insert.insert f B) (Singleton.singleton e))
    hne : Ne f e
    ⊢ Exists fun f => And (Membership.mem (SDiff.sdiff B' B) f) (Eq B' (SDiff.sdif …
  -/
  refine ⟨f, hf, (hb.eq_of_subset_base hB' ?_).symm⟩
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    B B' : Set α
    e : α
    hB : M.Base B
    hB' : M.Base B'
    h : Eq (SDiff.sdiff B B') (Singleton.singleton e)
    f : α
    hf : Membership.mem (SDiff.sdiff B' B) f
    hb : M.Base (SDiff.sdiff (Insert.insert f B) (Singleton.singleton e))
    hne : Ne f e
    ⊢ HasSubset.Subset (SDiff.sdiff (Insert.insert f B) (Singleton.singleton e)) B'
  -/
  rw [diff_subset_iff, insert_subset_iff, union_comm, ← diff_subset_iff, h, and_iff_left rfl.subset]
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    B B' : Set α
    e : α
    hB : M.Base B
    hB' : M.Base B'
    h : Eq (SDiff.sdiff B B') (Singleton.singleton e)
    f : α
    hf : Membership.mem (SDiff.sdiff B' B) f
    hb : M.Base (SDiff.sdiff (Insert.insert f B) (Singleton.singleton e))
    hne : Ne f e
    ⊢ Membership.mem (Union.union B' (Singleton.singleton e)) f
  -/
  exact Or.inl hf.1
  /-
    🎉 no goals
  -/


theorem Base.exchange_base_of_indep (hB : M.Base B) (hf : f ∉ B)
    (hI : M.Indep (insert f (B \ {e}))) : M.Base (insert f (B \ {e})) := by
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    e f : α
    hB : M.Base B
    hf : Not (Membership.mem B f)
    hI : M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
    ⊢ M.Base (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
  -/
  obtain ⟨B', hB', hIB'⟩ := hI.exists_base_superset
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    B : Set α
    e f : α
    hB : M.Base B
    hf : Not (Membership.mem B f)
    hI : M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
    B' : Set α
    hB' : M.Base B'
    hIB' : HasSubset.Subset (Insert.insert f (SDiff.sdiff B (Singleton.singleton e …
    ⊢ M.Base (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
  -/
  have hcard := hB'.encard_diff_comm hB
  rw [insert_subset_iff, ← diff_eq_empty, diff_diff_comm, diff_eq_empty, subset_singleton_iff_eq]
    at hIB'
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    B : Set α
    e f : α
    hB : M.Base B
    hf : Not (Membership.mem B f)
    hI : M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
    B' : Set α
    hB' : M.Base B'
    hIB' : And (Membership.mem B' f) (Or (Eq (SDiff.sdiff B B') EmptyCollection.em …
    hcard : Eq (SDiff.sdiff B' B).encard (SDiff.sdiff B B').encard
    ⊢ M.Base (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
  -/
  obtain ⟨hfB, (h | h)⟩ := hIB'
    /-
      case intro.intro.intro.inl
      α : Type u_1
      M : Matroid α
      B : Set α
      e f : α
      hB : M.Base B
      hf : Not (Membership.mem B f)
      hI : M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
      B' : Set α
      hB' : M.Base B'
      hcard : Eq (SDiff.sdiff B' B).encard (SDiff.sdiff B B').encard
      hfB : Membership.mem B' f
      h : Eq (SDiff.sdiff B B') EmptyCollection.emptyCollection
      ⊢ M.Base (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
    -/
  · rw [h, encard_empty, encard_eq_zero, eq_empty_iff_forall_not_mem] at hcard
    /-
      case intro.intro.intro.inl
      α : Type u_1
      M : Matroid α
      B : Set α
      e f : α
      hB : M.Base B
      hf : Not (Membership.mem B f)
      hI : M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
      B' : Set α
      hB' : M.Base B'
      hcard : ∀ (x : α), Not (Membership.mem (SDiff.sdiff B' B) x)
      hfB : Membership.mem B' f
      h : Eq (SDiff.sdiff B B') EmptyCollection.emptyCollection
      ⊢ M.Base (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
    -/
    exact (hcard f ⟨hfB, hf⟩).elim
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.intro.inr
    α : Type u_1
    M : Matroid α
    B : Set α
    e f : α
    hB : M.Base B
    hf : Not (Membership.mem B f)
    hI : M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
    B' : Set α
    hB' : M.Base B'
    hcard : Eq (SDiff.sdiff B' B).encard (SDiff.sdiff B B').encard
    hfB : Membership.mem B' f
    h : Eq (SDiff.sdiff B B') (Singleton.singleton e)
    ⊢ M.Base (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
  -/
  rw [h, encard_singleton, encard_eq_one] at hcard
  /-
    case intro.intro.intro.inr
    α : Type u_1
    M : Matroid α
    B : Set α
    e f : α
    hB : M.Base B
    hf : Not (Membership.mem B f)
    hI : M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
    B' : Set α
    hB' : M.Base B'
    hcard : Exists fun x => Eq (SDiff.sdiff B' B) (Singleton.singleton x)
    hfB : Membership.mem B' f
    h : Eq (SDiff.sdiff B B') (Singleton.singleton e)
    ⊢ M.Base (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
  -/
  obtain ⟨x, hx⟩ := hcard
  /-
    case intro.intro.intro.inr.intro
    α : Type u_1
    M : Matroid α
    B : Set α
    e f : α
    hB : M.Base B
    hf : Not (Membership.mem B f)
    hI : M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
    B' : Set α
    hB' : M.Base B'
    hfB : Membership.mem B' f
    h : Eq (SDiff.sdiff B B') (Singleton.singleton e)
    x : α
    hx : Eq (SDiff.sdiff B' B) (Singleton.singleton x)
    ⊢ M.Base (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
  -/
  obtain (rfl : f = x) := hx.subset ⟨hfB, hf⟩
  simp_rw [← h, ← singleton_union, ← hx, sdiff_sdiff_right_self, inf_eq_inter, inter_comm B,
    diff_union_inter]
  /-
    case intro.intro.intro.inr.intro
    α : Type u_1
    M : Matroid α
    B : Set α
    e f : α
    hB : M.Base B
    hf : Not (Membership.mem B f)
    hI : M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
    B' : Set α
    hB' : M.Base B'
    hfB : Membership.mem B' f
    h : Eq (SDiff.sdiff B B') (Singleton.singleton e)
    hx : Eq (SDiff.sdiff B' B) (Singleton.singleton f)
    ⊢ M.Base B'
  -/
  exact hB'
  /-
    🎉 no goals
  -/


theorem Base.exchange_base_of_indep' (hB : M.Base B) (he : e ∈ B) (hf : f ∉ B)
    (hI : M.Indep (insert f B \ {e})) : M.Base (insert f B \ {e}) := by
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    e f : α
    hB : M.Base B
    he : Membership.mem B e
    hf : Not (Membership.mem B f)
    hI : M.Indep (SDiff.sdiff (Insert.insert f B) (Singleton.singleton e))
    ⊢ M.Base (SDiff.sdiff (Insert.insert f B) (Singleton.singleton e))
  -/
  have hfe : f ≠ e := by rintro rfl; exact hf he
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    e f : α
    hB : M.Base B
    he : Membership.mem B e
    hf : Not (Membership.mem B f)
    hI : M.Indep (SDiff.sdiff (Insert.insert f B) (Singleton.singleton e))
    hfe : Ne f e
    ⊢ M.Base (SDiff.sdiff (Insert.insert f B) (Singleton.singleton e))
  -/
  rw [← insert_diff_singleton_comm hfe] at *
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    e f : α
    hB : M.Base B
    he : Membership.mem B e
    hf : Not (Membership.mem B f)
    hI : M.Indep (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
    hfe : Ne f e
    ⊢ M.Base (Insert.insert f (SDiff.sdiff B (Singleton.singleton e)))
  -/
  exact hB.exchange_base_of_indep hf hI
  /-
    🎉 no goals
  -/


theorem Base.insert_dep (hB : M.Base B) (h : e ∈ M.E \ B) : M.Dep (insert e B) := by
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    e : α
    hB : M.Base B
    h : Membership.mem (SDiff.sdiff M.E B) e
    ⊢ M.Dep (Insert.insert e B)
  -/
  rw [← not_indep_iff (insert_subset h.1 hB.subset_ground)]
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    e : α
    hB : M.Base B
    h : Membership.mem (SDiff.sdiff M.E B) e
    ⊢ Not (M.Indep (Insert.insert e B))
  -/
  exact h.2 ∘ (fun hi ↦ insert_eq_self.mp (hB.eq_of_subset_indep hi (subset_insert e B)).symm)
  /-
    🎉 no goals
  -/


theorem Indep.exists_insert_of_not_base (hI : M.Indep I) (hI' : ¬M.Base I) (hB : M.Base B) :
    ∃ e ∈ B \ I, M.Indep (insert e I) := by
  /-
    α : Type u_1
    M : Matroid α
    B I : Set α
    hI : M.Indep I
    hI' : Not (M.Base I)
    hB : M.Base B
    ⊢ Exists fun e => And (Membership.mem (SDiff.sdiff B I) e) (M.Indep (Insert.in …
  -/
  obtain ⟨B', hB', hIB'⟩ := hI.exists_base_superset
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    B I : Set α
    hI : M.Indep I
    hI' : Not (M.Base I)
    hB : M.Base B
    B' : Set α
    hB' : M.Base B'
    hIB' : HasSubset.Subset I B'
    ⊢ Exists fun e => And (Membership.mem (SDiff.sdiff B I) e) (M.Indep (Insert.in …
  -/
  obtain ⟨x, hxB', hx⟩ := exists_of_ssubset (hIB'.ssubset_of_ne (by (rintro rfl; exact hI' hB')))
  /-
    case intro.intro.intro.intro
    α : Type u_1
    M : Matroid α
    B I : Set α
    hI : M.Indep I
    hI' : Not (M.Base I)
    hB : M.Base B
    B' : Set α
    hB' : M.Base B'
    hIB' : HasSubset.Subset I B'
    x : α
    hxB' : Membership.mem B' x
    hx : Not (Membership.mem I x)
    ⊢ Exists fun e => And (Membership.mem (SDiff.sdiff B I) e) (M.Indep (Insert.in …
  -/
  by_cases hxB : x ∈ B
    /-
      case pos
      α : Type u_1
      M : Matroid α
      B I : Set α
      hI : M.Indep I
      hI' : Not (M.Base I)
      hB : M.Base B
      B' : Set α
      hB' : M.Base B'
      hIB' : HasSubset.Subset I B'
      x : α
      hxB' : Membership.mem B' x
      hx : Not (Membership.mem I x)
      hxB : Membership.mem B x
      ⊢ Exists fun e => And (Membership.mem (SDiff.sdiff B I) e) (M.Indep (Insert.in …
    -/
  · exact ⟨x, ⟨hxB, hx⟩, hB'.indep.subset (insert_subset hxB' hIB')⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    M : Matroid α
    B I : Set α
    hI : M.Indep I
    hI' : Not (M.Base I)
    hB : M.Base B
    B' : Set α
    hB' : M.Base B'
    hIB' : HasSubset.Subset I B'
    x : α
    hxB' : Membership.mem B' x
    hx : Not (Membership.mem I x)
    hxB : Not (Membership.mem B x)
    ⊢ Exists fun e => And (Membership.mem (SDiff.sdiff B I) e) (M.Indep (Insert.in …
  -/
  obtain ⟨e,he, hBase⟩ := hB'.exchange hB ⟨hxB',hxB⟩
  exact ⟨e, ⟨he.1, not_mem_subset hIB' he.2⟩,
    indep_iff.2 ⟨_, hBase, insert_subset_insert (subset_diff_singleton hIB' hx)⟩⟩


/-- This is the same as `Indep.exists_insert_of_not_base`, but phrased so that
  it is defeq to the augmentation axiom for independent sets. -/
theorem Indep.exists_insert_of_not_maximal (M : Matroid α) ⦃I B : Set α⦄ (hI : M.Indep I)
    (hInotmax : ¬ Maximal M.Indep I) (hB : Maximal M.Indep B) :
    ∃ x ∈ B \ I, M.Indep (insert x I) := by
  /-
    α : Type u_1
    M : Matroid α
    I B : Set α
    hI : M.Indep I
    hInotmax : Not (Maximal M.Indep I)
    hB : Maximal M.Indep B
    ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff B I) x) (M.Indep (Insert.in …
  -/
  simp only [maximal_subset_iff, hI, not_and, not_forall, exists_prop, true_imp_iff] at hB hInotmax
  /-
    α : Type u_1
    M : Matroid α
    I B : Set α
    hI : M.Indep I
    hB : And (M.Indep B) (∀ ⦃t : Set α⦄, M.Indep t → HasSubset.Subset B t → Eq B t)
    hInotmax : Exists fun x => And (M.Indep x) (And (HasSubset.Subset I x) (Not (E …
    ⊢ Exists fun x => And (Membership.mem (SDiff.sdiff B I) x) (M.Indep (Insert.in …
  -/
  refine hI.exists_insert_of_not_base (fun hIb ↦ ?_) ?_
    /-
      case refine_1
      α : Type u_1
      M : Matroid α
      I B : Set α
      hI : M.Indep I
      hB : And (M.Indep B) (∀ ⦃t : Set α⦄, M.Indep t → HasSubset.Subset B t → Eq B t)
      hInotmax : Exists fun x => And (M.Indep x) (And (HasSubset.Subset I x) (Not (E …
      hIb : M.Base I
      ⊢ False
    -/
  · obtain ⟨I', hII', hI', hne⟩ := hInotmax
    /-
      case refine_1.intro.intro.intro
      α : Type u_1
      M : Matroid α
      I B : Set α
      hI : M.Indep I
      hB : And (M.Indep B) (∀ ⦃t : Set α⦄, M.Indep t → HasSubset.Subset B t → Eq B t)
      hIb : M.Base I
      I' : Set α
      hII' : M.Indep I'
      hI' : HasSubset.Subset I I'
      hne : Not (Eq I I')
      ⊢ False
    -/
    exact hne <| hIb.eq_of_subset_indep hII' hI'
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    M : Matroid α
    I B : Set α
    hI : M.Indep I
    hB : And (M.Indep B) (∀ ⦃t : Set α⦄, M.Indep t → HasSubset.Subset B t → Eq B t)
    hInotmax : Exists fun x => And (M.Indep x) (And (HasSubset.Subset I x) (Not (E …
    ⊢ M.Base B
  -/
  exact hB.1.base_of_maximal fun J hJ hBJ ↦ hB.2 hJ hBJ
  /-
    🎉 no goals
  -/


theorem Indep.base_of_forall_insert (hB : M.Indep B)
    (hBmax : ∀ e ∈ M.E \ B, ¬ M.Indep (insert e B)) : M.Base B := by
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    hB : M.Indep B
    hBmax : ∀ (e : α), Membership.mem (SDiff.sdiff M.E B) e → Not (M.Indep (Insert …
    ⊢ M.Base B
  -/
  refine by_contra fun hnb ↦ ?_
  /-
    α : Type u_1
    M : Matroid α
    B : Set α
    hB : M.Indep B
    hBmax : ∀ (e : α), Membership.mem (SDiff.sdiff M.E B) e → Not (M.Indep (Insert …
    hnb : Not (M.Base B)
    ⊢ False
  -/
  obtain ⟨B', hB'⟩ := M.exists_base
  /-
    case intro
    α : Type u_1
    M : Matroid α
    B : Set α
    hB : M.Indep B
    hBmax : ∀ (e : α), Membership.mem (SDiff.sdiff M.E B) e → Not (M.Indep (Insert …
    hnb : Not (M.Base B)
    B' : Set α
    hB' : M.Base B'
    ⊢ False
  -/
  obtain ⟨e, he, h⟩ := hB.exists_insert_of_not_base hnb hB'
  /-
    case intro.intro.intro
    α : Type u_1
    M : Matroid α
    B : Set α
    hB : M.Indep B
    hBmax : ∀ (e : α), Membership.mem (SDiff.sdiff M.E B) e → Not (M.Indep (Insert …
    hnb : Not (M.Base B)
    B' : Set α
    hB' : M.Base B'
    e : α
    he : Membership.mem (SDiff.sdiff B' B) e
    h : M.Indep (Insert.insert e B)
    ⊢ False
  -/
  exact hBmax e ⟨hB'.subset_ground he.1, he.2⟩ h
  /-
    🎉 no goals
  -/


theorem ground_indep_iff_base : M.Indep M.E ↔ M.Base M.E :=
  ⟨fun h ↦ h.base_of_maximal (fun _ hJ hEJ ↦ hEJ.antisymm hJ.subset_ground), Base.indep⟩


theorem Base.exists_insert_of_ssubset (hB : M.Base B) (hIB : I ⊂ B) (hB' : M.Base B') :
    ∃ e ∈ B' \ I, M.Indep (insert e I) :=
  (hB.indep.subset hIB.subset).exists_insert_of_not_base
    (fun hI ↦ hIB.ne (hI.eq_of_subset_base hB hIB.subset)) hB'


@[ext] theorem ext_indep {M₁ M₂ : Matroid α} (hE : M₁.E = M₂.E)
    (h : ∀ ⦃I⦄, I ⊆ M₁.E → (M₁.Indep I ↔ M₂.Indep I)) : M₁ = M₂ :=
  have h' : M₁.Indep = M₂.Indep := by
    /-
      α : Type u_1
      M₁ M₂ : Matroid α
      hE : Eq M₁.E M₂.E
      h : ∀ ⦃I : Set α⦄, HasSubset.Subset I M₁.E → Iff (M₁.Indep I) (M₂.Indep I)
      ⊢ Eq M₁.Indep M₂.Indep
    -/
    ext I
    /-
      case h.a
      α : Type u_1
      M₁ M₂ : Matroid α
      hE : Eq M₁.E M₂.E
      h : ∀ ⦃I : Set α⦄, HasSubset.Subset I M₁.E → Iff (M₁.Indep I) (M₂.Indep I)
      I : Set α
      ⊢ Iff (M₁.Indep I) (M₂.Indep I)
    -/
    by_cases hI : I ⊆ M₁.E
      /-
        case pos
        α : Type u_1
        M₁ M₂ : Matroid α
        hE : Eq M₁.E M₂.E
        h : ∀ ⦃I : Set α⦄, HasSubset.Subset I M₁.E → Iff (M₁.Indep I) (M₂.Indep I)
        I : Set α
        hI : HasSubset.Subset I M₁.E
        ⊢ Iff (M₁.Indep I) (M₂.Indep I)
      -/
    · rwa [h]
      /-
        🎉 no goals
      -/
    exact iff_of_false (fun hi ↦ hI hi.subset_ground)
      (fun hi ↦ hI (hi.subset_ground.trans_eq hE.symm))
                            /-
                              α : Type u_1
                              M₁ M₂ : Matroid α
                              hE : Eq M₁.E M₂.E
                              h : ∀ ⦃I : Set α⦄, HasSubset.Subset I M₁.E → Iff (M₁.Indep I) (M₂.Indep I)
                              h' : Eq M₁.Indep M₂.Indep
                              B : Set α
                              x✝ : HasSubset.Subset B M₁.E
                              ⊢ Iff (M₁.Base B) (M₂.Base B)
                            -/
  ext_base hE (fun B _ ↦ by simp_rw [base_iff_maximal_indep, h'])
                            /-
                              🎉 no goals
                            -/


@[deprecated (since := "2024-12-25")] alias eq_of_indep_iff_indep_forall := ext_indep


theorem ext_iff_indep {M₁ M₂ : Matroid α} :
    M₁ = M₂ ↔ (M₁.E = M₂.E) ∧ ∀ ⦃I⦄, I ⊆ M₁.E → (M₁.Indep I ↔ M₂.Indep I) :=
             /-
               α : Type u_1
               M₁ M₂ : Matroid α
               h : Eq M₁ M₂
               ⊢ And (Eq M₁.E M₂.E) (∀ ⦃I : Set α⦄, HasSubset.Subset I M₁.E → Iff (M₁.Indep I …
             -/
⟨fun h ↦ by (subst h; simp), fun h ↦ ext_indep h.1 h.2⟩
                      /-
                        🎉 no goals
                      -/


@[deprecated (since := "2024-12-25")] alias eq_iff_indep_iff_indep_forall := ext_iff_indep


/-- If every base of `M₁` is independent in `M₂` and vice versa, then `M₁ = M₂`. -/
lemma ext_base_indep {M₁ M₂ : Matroid α} (hE : M₁.E = M₂.E) (hM₁ : ∀ ⦃B⦄, M₁.Base B → M₂.Indep B)
    (hM₂ : ∀ ⦃B⦄, M₂.Base B → M₁.Indep B) : M₁ = M₂ := by
  /-
    α : Type u_1
    M₁ M₂ : Matroid α
    hE : Eq M₁.E M₂.E
    hM₁ : ∀ ⦃B : Set α⦄, M₁.Base B → M₂.Indep B
    hM₂ : ∀ ⦃B : Set α⦄, M₂.Base B → M₁.Indep B
    ⊢ Eq M₁ M₂
  -/
  refine ext_indep hE fun I hIE ↦ ⟨fun hI ↦ ?_, fun hI ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      M₁ M₂ : Matroid α
      hE : Eq M₁.E M₂.E
      hM₁ : ∀ ⦃B : Set α⦄, M₁.Base B → M₂.Indep B
      hM₂ : ∀ ⦃B : Set α⦄, M₂.Base B → M₁.Indep B
      I : Set α
      hIE : HasSubset.Subset I M₁.E
      hI : M₁.Indep I
      ⊢ M₂.Indep I
    -/
  · obtain ⟨B, hB, hIB⟩ := hI.exists_base_superset
    /-
      case refine_1.intro.intro
      α : Type u_1
      M₁ M₂ : Matroid α
      hE : Eq M₁.E M₂.E
      hM₁ : ∀ ⦃B : Set α⦄, M₁.Base B → M₂.Indep B
      hM₂ : ∀ ⦃B : Set α⦄, M₂.Base B → M₁.Indep B
      I : Set α
      hIE : HasSubset.Subset I M₁.E
      hI : M₁.Indep I
      B : Set α
      hB : M₁.Base B
      hIB : HasSubset.Subset I B
      ⊢ M₂.Indep I
    -/
    exact (hM₁ hB).subset hIB
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    M₁ M₂ : Matroid α
    hE : Eq M₁.E M₂.E
    hM₁ : ∀ ⦃B : Set α⦄, M₁.Base B → M₂.Indep B
    hM₂ : ∀ ⦃B : Set α⦄, M₂.Base B → M₁.Indep B
    I : Set α
    hIE : HasSubset.Subset I M₁.E
    hI : M₂.Indep I
    ⊢ M₁.Indep I
  -/
  obtain ⟨B, hB, hIB⟩ := hI.exists_base_superset
  /-
    case refine_2.intro.intro
    α : Type u_1
    M₁ M₂ : Matroid α
    hE : Eq M₁.E M₂.E
    hM₁ : ∀ ⦃B : Set α⦄, M₁.Base B → M₂.Indep B
    hM₂ : ∀ ⦃B : Set α⦄, M₂.Base B → M₁.Indep B
    I : Set α
    hIE : HasSubset.Subset I M₁.E
    hI : M₂.Indep I
    B : Set α
    hB : M₂.Base B
    hIB : HasSubset.Subset I B
    ⊢ M₁.Indep I
  -/
  exact (hM₂ hB).subset hIB
  /-
    🎉 no goals
  -/


/-- A `Finitary` matroid is one where a set is independent if and only if it all
  its finite subsets are independent, or equivalently a matroid whose circuits are finite. -/
class Finitary (M : Matroid α) : Prop where
  /-- `I` is independent if all its finite subsets are independent. -/
  indep_of_forall_finite : ∀ I, (∀ J, J ⊆ I → J.Finite → M.Indep J) → M.Indep I


theorem indep_of_forall_finite_subset_indep {M : Matroid α} [Finitary M] (I : Set α)
    (h : ∀ J, J ⊆ I → J.Finite → M.Indep J) : M.Indep I :=
  Finitary.indep_of_forall_finite I h


theorem indep_iff_forall_finite_subset_indep {M : Matroid α} [Finitary M] :
    M.Indep I ↔ ∀ J, J ⊆ I → J.Finite → M.Indep J :=
  ⟨fun h _ hJI _ ↦ h.subset hJI, Finitary.indep_of_forall_finite I⟩


instance finitary_of_finiteRk {M : Matroid α} [FiniteRk M] : Finitary M :=
⟨ by
  /-
    α : Type u_1
    M✝ : Matroid α
    B B' I J D X : Set α
    e f : α
    M : Matroid α
    inst✝ : M.FiniteRk
    ⊢ ∀ (I : Set α), (∀ (J : Set α), HasSubset.Subset J I → J.Finite → M.Indep J)  …
  -/
  refine fun I hI ↦ I.finite_or_infinite.elim (hI _ Subset.rfl) (fun h ↦ False.elim ?_)
  /-
    α : Type u_1
    M✝ : Matroid α
    B B' I✝ J D X : Set α
    e f : α
    M : Matroid α
    inst✝ : M.FiniteRk
    I : Set α
    hI : ∀ (J : Set α), HasSubset.Subset J I → J.Finite → M.Indep J
    h : I.Infinite
    ⊢ False
  -/
  obtain ⟨B, hB⟩ := M.exists_base
  /-
    case intro
    α : Type u_1
    M✝ : Matroid α
    B✝ B' I✝ J D X : Set α
    e f : α
    M : Matroid α
    inst✝ : M.FiniteRk
    I : Set α
    hI : ∀ (J : Set α), HasSubset.Subset J I → J.Finite → M.Indep J
    h : I.Infinite
    B : Set α
    hB : M.Base B
    ⊢ False
  -/
  obtain ⟨I₀, hI₀I, hI₀fin, hI₀card⟩ := h.exists_subset_ncard_eq (B.ncard + 1)
  /-
    case intro.intro.intro.intro
    α : Type u_1
    M✝ : Matroid α
    B✝ B' I✝ J D X : Set α
    e f : α
    M : Matroid α
    inst✝ : M.FiniteRk
    I : Set α
    hI : ∀ (J : Set α), HasSubset.Subset J I → J.Finite → M.Indep J
    h : I.Infinite
    B : Set α
    hB : M.Base B
    I₀ : Set α
    hI₀I : HasSubset.Subset I₀ I
    hI₀fin : I₀.Finite
    hI₀card : Eq I₀.ncard (HAdd.hAdd B.ncard 1)
    ⊢ False
  -/
  obtain ⟨B', hB', hI₀B'⟩ := (hI _ hI₀I hI₀fin).exists_base_superset
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    M✝ : Matroid α
    B✝ B'✝ I✝ J D X : Set α
    e f : α
    M : Matroid α
    inst✝ : M.FiniteRk
    I : Set α
    hI : ∀ (J : Set α), HasSubset.Subset J I → J.Finite → M.Indep J
    h : I.Infinite
    B : Set α
    hB : M.Base B
    I₀ : Set α
    hI₀I : HasSubset.Subset I₀ I
    hI₀fin : I₀.Finite
    hI₀card : Eq I₀.ncard (HAdd.hAdd B.ncard 1)
    B' : Set α
    hB' : M.Base B'
    hI₀B' : HasSubset.Subset I₀ B'
    ⊢ False
  -/
  have hle := ncard_le_ncard hI₀B' hB'.finite
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    M✝ : Matroid α
    B✝ B'✝ I✝ J D X : Set α
    e f : α
    M : Matroid α
    inst✝ : M.FiniteRk
    I : Set α
    hI : ∀ (J : Set α), HasSubset.Subset J I → J.Finite → M.Indep J
    h : I.Infinite
    B : Set α
    hB : M.Base B
    I₀ : Set α
    hI₀I : HasSubset.Subset I₀ I
    hI₀fin : I₀.Finite
    hI₀card : Eq I₀.ncard (HAdd.hAdd B.ncard 1)
    B' : Set α
    hB' : M.Base B'
    hI₀B' : HasSubset.Subset I₀ B'
    hle : LE.le I₀.ncard B'.ncard
    ⊢ False
  -/
  rw [hI₀card, hB'.ncard_eq_ncard_of_base hB, Nat.add_one_le_iff] at hle
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    M✝ : Matroid α
    B✝ B'✝ I✝ J D X : Set α
    e f : α
    M : Matroid α
    inst✝ : M.FiniteRk
    I : Set α
    hI : ∀ (J : Set α), HasSubset.Subset J I → J.Finite → M.Indep J
    h : I.Infinite
    B : Set α
    hB : M.Base B
    I₀ : Set α
    hI₀I : HasSubset.Subset I₀ I
    hI₀fin : I₀.Finite
    hI₀card : Eq I₀.ncard (HAdd.hAdd B.ncard 1)
    B' : Set α
    hB' : M.Base B'
    hI₀B' : HasSubset.Subset I₀ B'
    hle : LT.lt B.ncard B.ncard
    ⊢ False
  -/
  exact hle.ne rfl ⟩
  /-
    🎉 no goals
  -/


/-- Matroids obey the maximality axiom -/
theorem existsMaximalSubsetProperty_indep (M : Matroid α) :
    ∀ X, X ⊆ M.E → ExistsMaximalSubsetProperty M.Indep X :=
  M.maximality


/-- A Basis for a set `X ⊆ M.E` is a maximal independent subset of `X`
  (Often in the literature, the word 'Basis' is used to refer to what we call a 'Base'). -/
def Basis (M : Matroid α) (I X : Set α) : Prop :=
  Maximal (fun A ↦ M.Indep A ∧ A ⊆ X) I ∧ X ⊆ M.E


/-- A `Basis'` is a basis without the requirement that `X ⊆ M.E`. This is convenient for some
  API building, especially when working with rank and closure. -/
def Basis' (M : Matroid α) (I X : Set α) : Prop :=
  Maximal (fun A ↦ M.Indep A ∧ A ⊆ X) I


theorem Basis'.indep (hI : M.Basis' I X) : M.Indep I :=
  hI.1.1


theorem Basis.indep (hI : M.Basis I X) : M.Indep I :=
  hI.1.1.1


theorem Basis.subset (hI : M.Basis I X) : I ⊆ X :=
  hI.1.1.2


theorem Basis.basis' (hI : M.Basis I X) : M.Basis' I X :=
  hI.1


theorem Basis'.basis (hI : M.Basis' I X) (hX : X ⊆ M.E := by aesop_mat) : M.Basis I X :=
  ⟨hI, hX⟩


theorem Basis'.subset (hI : M.Basis' I X) : I ⊆ X :=
  hI.1.2



@[aesop unsafe 15% (rule_sets := [Matroid])]
theorem Basis.subset_ground (hI : M.Basis I X) : X ⊆ M.E :=
  hI.2


theorem Basis.basis_inter_ground (hI : M.Basis I X) : M.Basis I (X ∩ M.E) := by
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    hI : M.Basis I X
    ⊢ M.Basis I (Inter.inter X M.E)
  -/
  convert hI
  /-
    case h.e'_4
    α : Type u_1
    M : Matroid α
    I X : Set α
    hI : M.Basis I X
    ⊢ Eq (Inter.inter X M.E) X
  -/
  rw [inter_eq_self_of_subset_left hI.subset_ground]
  /-
    🎉 no goals
  -/


@[aesop unsafe 15% (rule_sets := [Matroid])]
theorem Basis.left_subset_ground (hI : M.Basis I X) : I ⊆ M.E :=
  hI.indep.subset_ground


theorem Basis.eq_of_subset_indep (hI : M.Basis I X) (hJ : M.Indep J) (hIJ : I ⊆ J) (hJX : J ⊆ X) :
    I = J :=
  hIJ.antisymm (hI.1.2 ⟨hJ, hJX⟩ hIJ)


theorem Basis.Finite (hI : M.Basis I X) [FiniteRk M] : I.Finite := hI.indep.finite


theorem basis_iff' :
    M.Basis I X ↔ (M.Indep I ∧ I ⊆ X ∧ ∀ ⦃J⦄, M.Indep J → I ⊆ J → J ⊆ X → I = J) ∧ X ⊆ M.E := by
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    ⊢ Iff (M.Basis I X) (And (And (M.Indep I) (And (HasSubset.Subset I X) (∀ ⦃J :  …
  -/
  rw [Basis, maximal_subset_iff]
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    ⊢ Iff (And (And (And (M.Indep I) (HasSubset.Subset I X)) (∀ ⦃t : Set α⦄, And ( …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem basis_iff (hX : X ⊆ M.E := by aesop_mat) :
    M.Basis I X ↔ (M.Indep I ∧ I ⊆ X ∧ ∀ J, M.Indep J → I ⊆ J → J ⊆ X → I = J) := by
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ Iff (M.Basis I X) (And (M.Indep I) (And (HasSubset.Subset I X) (∀ (J : Set α …
  -/
  rw [basis_iff', and_iff_left hX]
  /-
    🎉 no goals
  -/


theorem basis'_iff_basis_inter_ground : M.Basis' I X ↔ M.Basis I (X ∩ M.E) := by
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    ⊢ Iff (M.Basis' I X) (M.Basis I (Inter.inter X M.E))
  -/
  rw [Basis', Basis, and_iff_left inter_subset_right, maximal_iff_maximal_of_imp_of_forall]
    /-
      case hPQ
      α : Type u_1
      M : Matroid α
      I X : Set α
      ⊢ ∀ ⦃x : Set α⦄, And (M.Indep x) (HasSubset.Subset x (Inter.inter X M.E)) → An …
    -/
  · exact fun I hI ↦ ⟨hI.1, hI.2.trans inter_subset_left⟩
    /-
      🎉 no goals
    -/
  /-
    case h
    α : Type u_1
    M : Matroid α
    I X : Set α
    ⊢ ∀ ⦃x : Set α⦄, And (M.Indep x) (HasSubset.Subset x X) → Exists fun y => And  …
  -/
  exact fun I hI ↦ ⟨I, rfl.le, hI.1, subset_inter hI.2 hI.1.subset_ground⟩
  /-
    🎉 no goals
  -/


theorem basis'_iff_basis (hX : X ⊆ M.E := by aesop_mat) : M.Basis' I X ↔ M.Basis I X := by
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ Iff (M.Basis' I X) (M.Basis I X)
  -/
  rw [basis'_iff_basis_inter_ground, inter_eq_self_of_subset_left hX]
  /-
    🎉 no goals
  -/


theorem basis_iff_basis'_subset_ground : M.Basis I X ↔ M.Basis' I X ∧ X ⊆ M.E :=
  ⟨fun h ↦ ⟨h.basis', h.subset_ground⟩, fun h ↦ (basis'_iff_basis h.2).mp h.1⟩


theorem Basis'.basis_inter_ground (hIX : M.Basis' I X) : M.Basis I (X ∩ M.E) :=
  basis'_iff_basis_inter_ground.mp hIX


theorem Basis'.eq_of_subset_indep (hI : M.Basis' I X) (hJ : M.Indep J) (hIJ : I ⊆ J)
    (hJX : J ⊆ X) : I = J :=
  hIJ.antisymm (hI.2 ⟨hJ, hJX⟩ hIJ)


theorem Basis'.insert_not_indep (hI : M.Basis' I X) (he : e ∈ X \ I) : ¬ M.Indep (insert e I) :=
  fun hi ↦ he.2 <| insert_eq_self.1 <| Eq.symm <|
    hI.eq_of_subset_indep hi (subset_insert _ _) (insert_subset he.1 hI.subset)


theorem basis_iff_maximal (hX : X ⊆ M.E := by aesop_mat) :
    M.Basis I X ↔ Maximal (fun I ↦ M.Indep I ∧ I ⊆ X) I := by
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ Iff (M.Basis I X) (Maximal (fun I => And (M.Indep I) (HasSubset.Subset I X)) …
  -/
  rw [Basis, and_iff_left hX]
  /-
    🎉 no goals
  -/


theorem Indep.basis_of_maximal_subset (hI : M.Indep I) (hIX : I ⊆ X)
    (hmax : ∀ ⦃J⦄, M.Indep J → I ⊆ J → J ⊆ X → J ⊆ I) (hX : X ⊆ M.E := by aesop_mat) :
    M.Basis I X := by
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    hI : M.Indep I
    hIX : HasSubset.Subset I X
    hmax : ∀ ⦃J : Set α⦄, M.Indep J → HasSubset.Subset I J → HasSubset.Subset J X  …
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ M.Basis I X
  -/
  rw [basis_iff (by aesop_mat : X ⊆ M.E), and_iff_right hI, and_iff_right hIX]
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    hI : M.Indep I
    hIX : HasSubset.Subset I X
    hmax : ∀ ⦃J : Set α⦄, M.Indep J → HasSubset.Subset I J → HasSubset.Subset J X  …
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ ∀ (J : Set α), M.Indep J → HasSubset.Subset I J → HasSubset.Subset J X → Eq  …
  -/
  exact fun J hJ hIJ hJX ↦ hIJ.antisymm (hmax hJ hIJ hJX)
  /-
    🎉 no goals
  -/


theorem Basis.basis_subset (hI : M.Basis I X) (hIY : I ⊆ Y) (hYX : Y ⊆ X) : M.Basis I Y := by
  /-
    α : Type u_1
    M : Matroid α
    I X Y : Set α
    hI : M.Basis I X
    hIY : HasSubset.Subset I Y
    hYX : HasSubset.Subset Y X
    ⊢ M.Basis I Y
  -/
  rw [basis_iff (hYX.trans hI.subset_ground), and_iff_right hI.indep, and_iff_right hIY]
  /-
    α : Type u_1
    M : Matroid α
    I X Y : Set α
    hI : M.Basis I X
    hIY : HasSubset.Subset I Y
    hYX : HasSubset.Subset Y X
    ⊢ ∀ (J : Set α), M.Indep J → HasSubset.Subset I J → HasSubset.Subset J Y → Eq  …
  -/
  exact fun J hJ hIJ hJY ↦ hI.eq_of_subset_indep hJ hIJ (hJY.trans hYX)
  /-
    🎉 no goals
  -/


@[simp] theorem basis_self_iff_indep : M.Basis I I ↔ M.Indep I := by
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    ⊢ Iff (M.Basis I I) (M.Indep I)
  -/
  rw [basis_iff', and_iff_right rfl.subset, and_assoc, and_iff_left_iff_imp]
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    ⊢ M.Indep I → And (∀ ⦃J : Set α⦄, M.Indep J → HasSubset.Subset I J → HasSubset …
  -/
  exact fun hi ↦ ⟨fun _ _ ↦ subset_antisymm, hi.subset_ground⟩
  /-
    🎉 no goals
  -/


theorem Indep.basis_self (h : M.Indep I) : M.Basis I I :=
  basis_self_iff_indep.mpr h


@[simp] theorem basis_empty_iff (M : Matroid α) : M.Basis I ∅ ↔ I = ∅ :=
                                                     /-
                                                       α : Type u_1
                                                       I : Set α
                                                       M : Matroid α
                                                       h : Eq I EmptyCollection.emptyCollection
                                                       ⊢ M.Basis I EmptyCollection.emptyCollection
                                                     -/
  ⟨fun h ↦ subset_empty_iff.mp h.subset, fun h ↦ by (rw [h]; exact M.empty_indep.basis_self)⟩
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem Basis.dep_of_ssubset (hI : M.Basis I X) (hIY : I ⊂ Y) (hYX : Y ⊆ X) : M.Dep Y := by
  /-
    α : Type u_1
    M : Matroid α
    I X Y : Set α
    hI : M.Basis I X
    hIY : HasSSubset.SSubset I Y
    hYX : HasSubset.Subset Y X
    ⊢ M.Dep Y
  -/
  have : X ⊆ M.E := hI.subset_ground
  /-
    α : Type u_1
    M : Matroid α
    I X Y : Set α
    hI : M.Basis I X
    hIY : HasSSubset.SSubset I Y
    hYX : HasSubset.Subset Y X
    this : HasSubset.Subset X M.E
    ⊢ M.Dep Y
  -/
  rw [← not_indep_iff]
  /-
    α : Type u_1
    M : Matroid α
    I X Y : Set α
    hI : M.Basis I X
    hIY : HasSSubset.SSubset I Y
    hYX : HasSubset.Subset Y X
    this : HasSubset.Subset X M.E
    ⊢ Not (M.Indep Y)
  -/
  exact fun hY ↦ hIY.ne (hI.eq_of_subset_indep hY hIY.subset hYX)
  /-
    🎉 no goals
  -/


theorem Basis.insert_dep (hI : M.Basis I X) (he : e ∈ X \ I) : M.Dep (insert e I) :=
  hI.dep_of_ssubset (ssubset_insert he.2) (insert_subset he.1 hI.subset)


theorem Basis.mem_of_insert_indep (hI : M.Basis I X) (he : e ∈ X) (hIe : M.Indep (insert e I)) :
    e ∈ I :=
  by_contra (fun heI ↦ (hI.insert_dep ⟨he, heI⟩).not_indep hIe)


theorem Basis'.mem_of_insert_indep (hI : M.Basis' I X) (he : e ∈ X) (hIe : M.Indep (insert e I)) :
    e ∈ I :=
  hI.basis_inter_ground.mem_of_insert_indep ⟨he, hIe.subset_ground (mem_insert _ _)⟩ hIe


theorem Basis.not_basis_of_ssubset (hI : M.Basis I X) (hJI : J ⊂ I) : ¬ M.Basis J X :=
  fun h ↦ hJI.ne (h.eq_of_subset_indep hI.indep hJI.subset hI.subset)


theorem Indep.subset_basis_of_subset (hI : M.Indep I) (hIX : I ⊆ X) (hX : X ⊆ M.E := by aesop_mat) :
    ∃ J, M.Basis J X ∧ I ⊆ J := by
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    hI : M.Indep I
    hIX : HasSubset.Subset I X
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    ⊢ Exists fun J => And (M.Basis J X) (HasSubset.Subset I J)
  -/
  obtain ⟨J, hJ, hJmax⟩ := M.maximality X hX I hI hIX
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    I X : Set α
    hI : M.Indep I
    hIX : HasSubset.Subset I X
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    J : Set α
    hJ : HasSubset.Subset I J
    hJmax : Maximal (fun K => And (M.Indep K) (HasSubset.Subset K X)) J
    ⊢ Exists fun J => And (M.Basis J X) (HasSubset.Subset I J)
  -/
  exact ⟨J, ⟨hJmax, hX⟩, hJ⟩
  /-
    🎉 no goals
  -/


theorem Indep.subset_basis'_of_subset (hI : M.Indep I) (hIX : I ⊆ X) :
    ∃ J, M.Basis' J X ∧ I ⊆ J := by
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    hI : M.Indep I
    hIX : HasSubset.Subset I X
    ⊢ Exists fun J => And (M.Basis' J X) (HasSubset.Subset I J)
  -/
  simp_rw [basis'_iff_basis_inter_ground]
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    hI : M.Indep I
    hIX : HasSubset.Subset I X
    ⊢ Exists fun J => And (M.Basis J (Inter.inter X M.E)) (HasSubset.Subset I J)
  -/
  exact hI.subset_basis_of_subset (subset_inter hIX hI.subset_ground)
  /-
    🎉 no goals
  -/


theorem exists_basis (M : Matroid α) (X : Set α) (hX : X ⊆ M.E := by aesop_mat) :
    ∃ I, M.Basis I X :=
                    /-
                      α : Type u_1
                      M : Matroid α
                      X : Set α
                      hX : autoParam (HasSubset.Subset X M.E) _auto✝
                      ⊢ HasSubset.Subset X M.E
                    -/
  let ⟨_, hI, _⟩ := M.empty_indep.subset_basis_of_subset (empty_subset X)
                    /-
                      🎉 no goals
                    -/
  ⟨_,hI⟩


theorem exists_basis' (M : Matroid α) (X : Set α) : ∃ I, M.Basis' I X :=
  let ⟨_, hI, _⟩ := M.empty_indep.subset_basis'_of_subset (empty_subset X)
  ⟨_,hI⟩


theorem exists_basis_subset_basis (M : Matroid α) (hXY : X ⊆ Y) (hY : Y ⊆ M.E := by aesop_mat) :
    ∃ I J, M.Basis I X ∧ M.Basis J Y ∧ I ⊆ J := by
  /-
    α : Type u_1
    X Y : Set α
    M : Matroid α
    hXY : HasSubset.Subset X Y
    hY : autoParam (HasSubset.Subset Y M.E) _auto✝
    ⊢ Exists fun I => Exists fun J => And (M.Basis I X) (And (M.Basis J Y) (HasSub …
  -/
  obtain ⟨I, hI⟩ := M.exists_basis X (hXY.trans hY)
  /-
    case intro
    α : Type u_1
    X Y : Set α
    M : Matroid α
    hXY : HasSubset.Subset X Y
    hY : autoParam (HasSubset.Subset Y M.E) _auto✝
    I : Set α
    hI : M.Basis I X
    ⊢ Exists fun I => Exists fun J => And (M.Basis I X) (And (M.Basis J Y) (HasSub …
  -/
  obtain ⟨J, hJ, hIJ⟩ := hI.indep.subset_basis_of_subset (hI.subset.trans hXY)
  /-
    case intro.intro.intro
    α : Type u_1
    X Y : Set α
    M : Matroid α
    hXY : HasSubset.Subset X Y
    hY : autoParam (HasSubset.Subset Y M.E) _auto✝
    I : Set α
    hI : M.Basis I X
    J : Set α
    hJ : M.Basis J Y
    hIJ : HasSubset.Subset I J
    ⊢ Exists fun I => Exists fun J => And (M.Basis I X) (And (M.Basis J Y) (HasSub …
  -/
  exact ⟨_, _, hI, hJ, hIJ⟩
  /-
    🎉 no goals
  -/


theorem Basis.exists_basis_inter_eq_of_superset (hI : M.Basis I X) (hXY : X ⊆ Y)
    (hY : Y ⊆ M.E := by aesop_mat) : ∃ J, M.Basis J Y ∧ J ∩ X = I := by
  /-
    α : Type u_1
    M : Matroid α
    I X Y : Set α
    hI : M.Basis I X
    hXY : HasSubset.Subset X Y
    hY : autoParam (HasSubset.Subset Y M.E) _auto✝
    ⊢ Exists fun J => And (M.Basis J Y) (Eq (Inter.inter J X) I)
  -/
  obtain ⟨J, hJ, hIJ⟩ := hI.indep.subset_basis_of_subset (hI.subset.trans hXY)
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    I X Y : Set α
    hI : M.Basis I X
    hXY : HasSubset.Subset X Y
    hY : autoParam (HasSubset.Subset Y M.E) _auto✝
    J : Set α
    hJ : M.Basis J Y
    hIJ : HasSubset.Subset I J
    ⊢ Exists fun J => And (M.Basis J Y) (Eq (Inter.inter J X) I)
  -/
  refine ⟨J, hJ, subset_antisymm ?_ (subset_inter hIJ hI.subset)⟩
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    I X Y : Set α
    hI : M.Basis I X
    hXY : HasSubset.Subset X Y
    hY : autoParam (HasSubset.Subset Y M.E) _auto✝
    J : Set α
    hJ : M.Basis J Y
    hIJ : HasSubset.Subset I J
    ⊢ HasSubset.Subset (Inter.inter J X) I
  -/
  exact fun e he ↦ hI.mem_of_insert_indep he.2 (hJ.indep.subset (insert_subset he.1 hIJ))
  /-
    🎉 no goals
  -/


theorem exists_basis_union_inter_basis (M : Matroid α) (X Y : Set α) (hX : X ⊆ M.E := by aesop_mat)
    (hY : Y ⊆ M.E := by aesop_mat) : ∃ I, M.Basis I (X ∪ Y) ∧ M.Basis (I ∩ Y) Y :=
                 /-
                   α : Type u_1
                   M : Matroid α
                   X Y : Set α
                   hX : autoParam (HasSubset.Subset X M.E) _auto✝
                   hY : autoParam (HasSubset.Subset Y M.E) _auto✝
                   ⊢ HasSubset.Subset Y M.E
                 -/
  let ⟨J, hJ⟩ := M.exists_basis Y
                 /-
                   🎉 no goals
                 -/
   /-
     α : Type u_1
     M : Matroid α
     X Y : Set α
     hX : autoParam (HasSubset.Subset X M.E) _auto✝
     hY : autoParam (HasSubset.Subset Y M.E) _auto✝
     J : Set α
     hJ : M.Basis J Y
     ⊢ HasSubset.Subset (Union.union X Y) M.E
   -/
  (hJ.exists_basis_inter_eq_of_superset subset_union_right).imp
   /-
     🎉 no goals
   -/
                        /-
                          α : Type u_1
                          M : Matroid α
                          X Y : Set α
                          hX : autoParam (HasSubset.Subset X M.E) _auto✝
                          hY : autoParam (HasSubset.Subset Y M.E) _auto✝
                          J : Set α
                          hJ : M.Basis J Y
                          I : Set α
                          hI : And (M.Basis I (Union.union X Y)) (Eq (Inter.inter I Y) J)
                          ⊢ M.Basis (Inter.inter I Y) Y
                        -/
  (fun I hI ↦ ⟨hI.1, by rwa [hI.2]⟩)
                        /-
                          🎉 no goals
                        -/


theorem Indep.eq_of_basis (hI : M.Indep I) (hJ : M.Basis J I) : J = I :=
  hJ.eq_of_subset_indep hI hJ.subset rfl.subset


theorem Basis.exists_base (hI : M.Basis I X) : ∃ B, M.Base B ∧ I = B ∩ X :=
  let ⟨B,hB, hIB⟩ := hI.indep.exists_base_superset
  ⟨B, hB, subset_antisymm (subset_inter hIB hI.subset)
    (by rw [hI.eq_of_subset_indep (hB.indep.inter_right X) (subset_inter hIB hI.subset)
    inter_subset_right])⟩


@[simp] theorem basis_ground_iff : M.Basis B M.E ↔ M.Base B := by
  rw [Basis, and_iff_left rfl.subset, base_iff_maximal_indep,
    maximal_and_iff_right_of_imp (fun _ h ↦ h.subset_ground),
    and_iff_left_of_imp (fun h ↦ h.1.subset_ground)]


theorem Base.basis_ground (hB : M.Base B) : M.Basis B M.E :=
  basis_ground_iff.mpr hB


theorem Indep.basis_iff_forall_insert_dep (hI : M.Indep I) (hIX : I ⊆ X) :
    M.Basis I X ↔ ∀ e ∈ X \ I, M.Dep (insert e I) := by
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    hI : M.Indep I
    hIX : HasSubset.Subset I X
    ⊢ Iff (M.Basis I X) (∀ (e : α), Membership.mem (SDiff.sdiff X I) e → M.Dep (In …
  -/
  rw [Basis, maximal_iff_forall_insert (fun I J hI hIJ ↦ ⟨hI.1.subset hIJ, hIJ.trans hI.2⟩)]
  simp only [hI, hIX, and_self, insert_subset_iff, and_true, not_and, true_and, mem_diff, and_imp,
    Dep, hI.subset_ground]
  exact ⟨fun h e heX heI ↦ ⟨fun hi ↦ h.1 e heI hi heX, h.2 heX⟩,
    fun h ↦ ⟨fun e heI hi heX ↦ (h e heX heI).1 hi,
      fun e heX ↦ (em (e ∈ I)).elim (fun h ↦ hI.subset_ground h) fun heI ↦ (h _ heX heI).2 ⟩⟩


theorem Indep.basis_of_forall_insert (hI : M.Indep I) (hIX : I ⊆ X)
    (he : ∀ e ∈ X \ I, M.Dep (insert e I)) : M.Basis I X :=
  (hI.basis_iff_forall_insert_dep hIX).mpr he


theorem Indep.basis_insert_iff (hI : M.Indep I) :
    M.Basis I (insert e I) ↔ M.Dep (insert e I) ∨ e ∈ I := by
  simp_rw [hI.basis_iff_forall_insert_dep (subset_insert _ _), dep_iff, insert_subset_iff,
    and_iff_left hI.subset_ground, mem_diff, mem_insert_iff, or_and_right, and_not_self,
    or_false, and_imp, forall_eq]
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    e : α
    hI : M.Indep I
    ⊢ Iff (Not (Membership.mem I e) → And (Not (M.Indep (Insert.insert e I))) (Mem …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem Basis.iUnion_basis_iUnion {ι : Type _} (X I : ι → Set α) (hI : ∀ i, M.Basis (I i) (X i))
    (h_ind : M.Indep (⋃ i, I i)) : M.Basis (⋃ i, I i) (⋃ i, X i) := by
  refine h_ind.basis_of_forall_insert
    (iUnion_subset (fun i ↦ (hI i).subset.trans (subset_iUnion _ _))) ?_
  /-
    α : Type u_1
    M : Matroid α
    ι : Type u_2
    X I : ι → Set α
    hI : ∀ (i : ι), M.Basis (I i) (X i)
    h_ind : M.Indep (Set.iUnion fun i => I i)
    ⊢ ∀ (e : α), Membership.mem (SDiff.sdiff (Set.iUnion fun i => X i) (Set.iUnion …
  -/
  rintro e ⟨⟨_, ⟨⟨i, hi, rfl⟩, (hes : e ∈ X i)⟩⟩, he'⟩
  /-
    case intro.intro.intro.intro.refl
    α : Type u_1
    M : Matroid α
    ι : Type u_2
    X I : ι → Set α
    hI : ∀ (i : ι), M.Basis (I i) (X i)
    h_ind : M.Indep (Set.iUnion fun i => I i)
    e : α
    he' : Not (Membership.mem (Set.iUnion fun i => I i) e)
    i : ι
    hes : Membership.mem (X i) e
    ⊢ M.Dep (Insert.insert e (Set.iUnion fun i => I i))
  -/
  rw [mem_iUnion, not_exists] at he'
  /-
    case intro.intro.intro.intro.refl
    α : Type u_1
    M : Matroid α
    ι : Type u_2
    X I : ι → Set α
    hI : ∀ (i : ι), M.Basis (I i) (X i)
    h_ind : M.Indep (Set.iUnion fun i => I i)
    e : α
    he' : ∀ (x : ι), Not (Membership.mem (I x) e)
    i : ι
    hes : Membership.mem (X i) e
    ⊢ M.Dep (Insert.insert e (Set.iUnion fun i => I i))
  -/
  refine ((hI i).insert_dep ⟨hes, he' _⟩).superset (insert_subset_insert (subset_iUnion _ _)) ?_
  /-
    case intro.intro.intro.intro.refl
    α : Type u_1
    M : Matroid α
    ι : Type u_2
    X I : ι → Set α
    hI : ∀ (i : ι), M.Basis (I i) (X i)
    h_ind : M.Indep (Set.iUnion fun i => I i)
    e : α
    he' : ∀ (x : ι), Not (Membership.mem (I x) e)
    i : ι
    hes : Membership.mem (X i) e
    ⊢ HasSubset.Subset (Insert.insert e (Set.iUnion fun i => I i)) M.E
  -/
  rw [insert_subset_iff, iUnion_subset_iff, and_iff_left (fun i ↦ (hI i).indep.subset_ground)]
  /-
    case intro.intro.intro.intro.refl
    α : Type u_1
    M : Matroid α
    ι : Type u_2
    X I : ι → Set α
    hI : ∀ (i : ι), M.Basis (I i) (X i)
    h_ind : M.Indep (Set.iUnion fun i => I i)
    e : α
    he' : ∀ (x : ι), Not (Membership.mem (I x) e)
    i : ι
    hes : Membership.mem (X i) e
    ⊢ Membership.mem M.E e
  -/
  exact (hI i).subset_ground hes
  /-
    🎉 no goals
  -/


theorem Basis.basis_iUnion {ι : Type _} [Nonempty ι] (X : ι → Set α)
    (hI : ∀ i, M.Basis I (X i)) : M.Basis I (⋃ i, X i) := by
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    ι : Type u_2
    inst✝ : Nonempty ι
    X : ι → Set α
    hI : ∀ (i : ι), M.Basis I (X i)
    ⊢ M.Basis I (Set.iUnion fun i => X i)
  -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  convert Basis.iUnion_basis_iUnion X (fun _ ↦ I) (fun i ↦ hI i) _ <;> rw [iUnion_const]
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    ι : Type u_2
    inst✝ : Nonempty ι
    X : ι → Set α
    hI : ∀ (i : ι), M.Basis I (X i)
    ⊢ M.Indep I
  -/
  exact (hI (Classical.arbitrary ι)).indep
  /-
    🎉 no goals
  -/


theorem Basis.basis_sUnion {Xs : Set (Set α)} (hne : Xs.Nonempty) (h : ∀ X ∈ Xs, M.Basis I X) :
    M.Basis I (⋃₀ Xs) := by
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    Xs : Set (Set α)
    hne : Xs.Nonempty
    h : ∀ (X : Set α), Membership.mem Xs X → M.Basis I X
    ⊢ M.Basis I Xs.sUnion
  -/
  rw [sUnion_eq_iUnion]
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    Xs : Set (Set α)
    hne : Xs.Nonempty
    h : ∀ (X : Set α), Membership.mem Xs X → M.Basis I X
    ⊢ M.Basis I (Set.iUnion fun i => ↑i)
  -/
  have := Iff.mpr nonempty_coe_sort hne
  /-
    α : Type u_1
    M : Matroid α
    I : Set α
    Xs : Set (Set α)
    hne : Xs.Nonempty
    h : ∀ (X : Set α), Membership.mem Xs X → M.Basis I X
    this : Nonempty ↑Xs
    ⊢ M.Basis I (Set.iUnion fun i => ↑i)
  -/
  exact Basis.basis_iUnion _ fun X ↦ (h X X.prop)
  /-
    🎉 no goals
  -/


theorem Indep.basis_setOf_insert_basis (hI : M.Indep I) :
    M.Basis I {x | M.Basis I (insert x I)} := by
  refine hI.basis_of_forall_insert (fun e he ↦ (?_ : M.Basis _ _))
    (fun e he ↦ ⟨fun hu ↦ he.2 ?_, he.1.subset_ground⟩)
    /-
      case refine_1
      α : Type u_1
      M : Matroid α
      I : Set α
      hI : M.Indep I
      e : α
      he : Membership.mem I e
      ⊢ M.Basis I (Insert.insert e I)
    -/
  · rw [insert_eq_of_mem he]; exact hI.basis_self
                              /-
                                🎉 no goals
                              -/
  /-
    case refine_2
    α : Type u_1
    M : Matroid α
    I : Set α
    hI : M.Indep I
    e : α
    he : Membership.mem (SDiff.sdiff (setOf fun x => M.Basis I (Insert.insert x I) …
    hu : M.Indep (Insert.insert e I)
    ⊢ Membership.mem I e
  -/
  simpa using (hu.eq_of_basis he.1).symm
  /-
    🎉 no goals
  -/


theorem Basis.union_basis_union (hIX : M.Basis I X) (hJY : M.Basis J Y) (h : M.Indep (I ∪ J)) :
    M.Basis (I ∪ J) (X ∪ Y) := by
  /-
    α : Type u_1
    M : Matroid α
    I J X Y : Set α
    hIX : M.Basis I X
    hJY : M.Basis J Y
    h : M.Indep (Union.union I J)
    ⊢ M.Basis (Union.union I J) (Union.union X Y)
  -/
  rw [union_eq_iUnion, union_eq_iUnion]
  /-
    α : Type u_1
    M : Matroid α
    I J X Y : Set α
    hIX : M.Basis I X
    hJY : M.Basis J Y
    h : M.Indep (Union.union I J)
    ⊢ M.Basis (Set.iUnion fun b => cond b I J) (Set.iUnion fun b => cond b X Y)
  -/
  refine Basis.iUnion_basis_iUnion _ _ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      M : Matroid α
      I J X Y : Set α
      hIX : M.Basis I X
      hJY : M.Basis J Y
      h : M.Indep (Union.union I J)
      ⊢ ∀ (i : Bool), M.Basis (cond i I J) (cond i X Y)
    -/
  · simp only [Bool.forall_bool, cond_false, cond_true]; exact ⟨hJY, hIX⟩
                                                         /-
                                                           🎉 no goals
                                                         -/
  /-
    case refine_2
    α : Type u_1
    M : Matroid α
    I J X Y : Set α
    hIX : M.Basis I X
    hJY : M.Basis J Y
    h : M.Indep (Union.union I J)
    ⊢ M.Indep (Set.iUnion fun i => cond i I J)
  -/
  rwa [← union_eq_iUnion]
  /-
    🎉 no goals
  -/


theorem Basis.basis_union (hIX : M.Basis I X) (hIY : M.Basis I Y) : M.Basis I (X ∪ Y) := by
    /-
      α : Type u_1
      M : Matroid α
      I X Y : Set α
      hIX : M.Basis I X
      hIY : M.Basis I Y
      ⊢ M.Basis I (Union.union X Y)
    -/
                                            /-
                                              🎉 no goals
                                            -/
    convert hIX.union_basis_union hIY _ <;> rw [union_self]; exact hIX.indep
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem Basis.basis_union_of_subset (hI : M.Basis I X) (hJ : M.Indep J) (hIJ : I ⊆ J) :
    M.Basis J (J ∪ X) := by
  /-
    α : Type u_1
    M : Matroid α
    I J X : Set α
    hI : M.Basis I X
    hJ : M.Indep J
    hIJ : HasSubset.Subset I J
    ⊢ M.Basis J (Union.union J X)
  -/
  convert hJ.basis_self.union_basis_union hI _ <;>
  /-
    case h.e'_3
    α : Type u_1
    M : Matroid α
    I J X : Set α
    hI : M.Basis I X
    hJ : M.Indep J
    hIJ : HasSubset.Subset I J
    ⊢ Eq J (Union.union J I)
  -/
  /-
    🎉 no goals
  -/
  rw [union_eq_self_of_subset_right hIJ]
  /-
    α : Type u_1
    M : Matroid α
    I J X : Set α
    hI : M.Basis I X
    hJ : M.Indep J
    hIJ : HasSubset.Subset I J
    ⊢ M.Indep J
  -/
  assumption
  /-
    🎉 no goals
  -/


theorem Basis.insert_basis_insert (hI : M.Basis I X) (h : M.Indep (insert e I)) :
    M.Basis (insert e I) (insert e X) := by
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    e : α
    hI : M.Basis I X
    h : M.Indep (Insert.insert e I)
    ⊢ M.Basis (Insert.insert e I) (Insert.insert e X)
  -/
  simp_rw [← union_singleton] at *
  /-
    α : Type u_1
    M : Matroid α
    I X : Set α
    e : α
    hI : M.Basis I X
    h : M.Indep (Union.union I (Singleton.singleton e))
    ⊢ M.Basis (Union.union I (Singleton.singleton e)) (Union.union X (Singleton.si …
  -/
  exact hI.union_basis_union (h.subset subset_union_right).basis_self h
  /-
    🎉 no goals
  -/


theorem Base.base_of_basis_superset (hB : M.Base B) (hBX : B ⊆ X) (hIX : M.Basis I X) :
    M.Base I := by
  /-
    α : Type u_1
    M : Matroid α
    B I X : Set α
    hB : M.Base B
    hBX : HasSubset.Subset B X
    hIX : M.Basis I X
    ⊢ M.Base I
  -/
  by_contra h
  /-
    α : Type u_1
    M : Matroid α
    B I X : Set α
    hB : M.Base B
    hBX : HasSubset.Subset B X
    hIX : M.Basis I X
    h : Not (M.Base I)
    ⊢ False
  -/
  obtain ⟨e,heBI,he⟩ := hIX.indep.exists_insert_of_not_base h hB
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    B I X : Set α
    hB : M.Base B
    hBX : HasSubset.Subset B X
    hIX : M.Basis I X
    h : Not (M.Base I)
    e : α
    heBI : Membership.mem (SDiff.sdiff B I) e
    he : M.Indep (Insert.insert e I)
    ⊢ False
  -/
  exact heBI.2 (hIX.mem_of_insert_indep (hBX heBI.1) he)
  /-
    🎉 no goals
  -/


theorem Indep.exists_base_subset_union_base (hI : M.Indep I) (hB : M.Base B) :
    ∃ B', M.Base B' ∧ I ⊆ B' ∧ B' ⊆ I ∪ B := by
  /-
    α : Type u_1
    M : Matroid α
    B I : Set α
    hI : M.Indep I
    hB : M.Base B
    ⊢ Exists fun B' => And (M.Base B') (And (HasSubset.Subset I B') (HasSubset.Sub …
  -/
  obtain ⟨B', hB', hIB'⟩ := hI.subset_basis_of_subset <| subset_union_left (t := B)
  /-
    case intro.intro
    α : Type u_1
    M : Matroid α
    B I : Set α
    hI : M.Indep I
    hB : M.Base B
    B' : Set α
    hB' : M.Basis B' (Union.union I B)
    hIB' : HasSubset.Subset I B'
    ⊢ Exists fun B' => And (M.Base B') (And (HasSubset.Subset I B') (HasSubset.Sub …
  -/
  exact ⟨B', hB.base_of_basis_superset subset_union_right hB', hIB', hB'.subset⟩
  /-
    🎉 no goals
  -/


theorem Basis.inter_eq_of_subset_indep (hIX : M.Basis I X) (hIJ : I ⊆ J) (hJ : M.Indep J) :
    J ∩ X = I :=
(subset_inter hIJ hIX.subset).antisymm'
  (fun _ he ↦ hIX.mem_of_insert_indep he.2 (hJ.subset (insert_subset he.1 hIJ)))


theorem Basis'.inter_eq_of_subset_indep (hI : M.Basis' I X) (hIJ : I ⊆ J) (hJ : M.Indep J) :
    J ∩ X = I := by
  rw [← hI.basis_inter_ground.inter_eq_of_subset_indep hIJ hJ, inter_comm X, ← inter_assoc,
    inter_eq_self_of_subset_left hJ.subset_ground]


theorem Base.basis_of_subset (hX : X ⊆ M.E := by aesop_mat) (hB : M.Base B) (hBX : B ⊆ X) :
    M.Basis B X := by
  /-
    α : Type u_1
    M : Matroid α
    B X : Set α
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    hB : M.Base B
    hBX : HasSubset.Subset B X
    ⊢ M.Basis B X
  -/
  rw [basis_iff, and_iff_right hB.indep, and_iff_right hBX]
  /-
    α : Type u_1
    M : Matroid α
    B X : Set α
    hX : autoParam (HasSubset.Subset X M.E) _auto✝
    hB : M.Base B
    hBX : HasSubset.Subset B X
    ⊢ ∀ (J : Set α), M.Indep J → HasSubset.Subset B J → HasSubset.Subset J X → Eq  …
  -/
  exact fun J hJ hBJ _ ↦ hB.eq_of_subset_indep hJ hBJ
  /-
    🎉 no goals
  -/


theorem exists_basis_disjoint_basis_of_subset (M : Matroid α) {X Y : Set α} (hXY : X ⊆ Y)
    (hY : Y ⊆ M.E := by aesop_mat) : ∃ I J, M.Basis I X ∧ M.Basis (I ∪ J) Y ∧ Disjoint X J := by
  /-
    α : Type u_1
    M : Matroid α
    X Y : Set α
    hXY : HasSubset.Subset X Y
    hY : autoParam (HasSubset.Subset Y M.E) _auto✝
    ⊢ Exists fun I => Exists fun J => And (M.Basis I X) (And (M.Basis (Union.union …
  -/
  obtain ⟨I, I', hI, hI', hII'⟩ := M.exists_basis_subset_basis hXY
  /-
    case intro.intro.intro.intro
    α : Type u_1
    M : Matroid α
    X Y : Set α
    hXY : HasSubset.Subset X Y
    hY : autoParam (HasSubset.Subset Y M.E) _auto✝
    I I' : Set α
    hI : M.Basis I X
    hI' : M.Basis I' Y
    hII' : HasSubset.Subset I I'
    ⊢ Exists fun I => Exists fun J => And (M.Basis I X) (And (M.Basis (Union.union …
  -/
  refine ⟨I, I' \ I, hI, by rwa [union_diff_self, union_eq_self_of_subset_left hII'], ?_⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    M : Matroid α
    X Y : Set α
    hXY : HasSubset.Subset X Y
    hY : autoParam (HasSubset.Subset Y M.E) _auto✝
    I I' : Set α
    hI : M.Basis I X
    hI' : M.Basis I' Y
    hII' : HasSubset.Subset I I'
    ⊢ Disjoint X (SDiff.sdiff I' I)
  -/
  rw [disjoint_iff_forall_ne]
  /-
    case intro.intro.intro.intro
    α : Type u_1
    M : Matroid α
    X Y : Set α
    hXY : HasSubset.Subset X Y
    hY : autoParam (HasSubset.Subset Y M.E) _auto✝
    I I' : Set α
    hI : M.Basis I X
    hI' : M.Basis I' Y
    hII' : HasSubset.Subset I I'
    ⊢ ∀ ⦃a : α⦄, Membership.mem X a → ∀ ⦃b : α⦄, Membership.mem (SDiff.sdiff I' I) …
  -/
  rintro e heX _ ⟨heI', heI⟩ rfl
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    M : Matroid α
    X Y : Set α
    hXY : HasSubset.Subset X Y
    hY : autoParam (HasSubset.Subset Y M.E) _auto✝
    I I' : Set α
    hI : M.Basis I X
    hI' : M.Basis I' Y
    hII' : HasSubset.Subset I I'
    e : α
    heX : Membership.mem X e
    heI' : Membership.mem I' e
    heI : Not (Membership.mem I e)
    ⊢ False
  -/
  exact heI <| hI.mem_of_insert_indep heX (hI'.indep.subset (insert_subset heI' hII'))
  /-
    🎉 no goals
  -/


/-- For finite `E`, finitely many matroids have ground set contained in `E`. -/
theorem finite_setOf_matroid {E : Set α} (hE : E.Finite) : {M : Matroid α | M.E ⊆ E}.Finite := by
  /-
    α : Type u_1
    E : Set α
    hE : E.Finite
    ⊢ (setOf fun M => HasSubset.Subset M.E E).Finite
  -/
  set f : Matroid α → Set α × (Set (Set α)) := fun M ↦ ⟨M.E, {B | M.Base B}⟩
  have hf : f.Injective := by
    refine fun M M' hMM' ↦ ?_
    rw [Prod.mk.injEq, and_comm, Set.ext_iff, and_comm] at hMM'
    exact ext_base hMM'.1 (fun B _ ↦ hMM'.2 B)
  /-
    α : Type u_1
    E : Set α
    hE : E.Finite
    f : Matroid α → Prod (Set α) (Set (Set α)) := fun M => { fst := M.E, snd := se …
    hf : Function.Injective f
    ⊢ (setOf fun M => HasSubset.Subset M.E E).Finite
  -/
  rw [← Set.finite_image_iff hf.injOn]
  /-
    α : Type u_1
    E : Set α
    hE : E.Finite
    f : Matroid α → Prod (Set α) (Set (Set α)) := fun M => { fst := M.E, snd := se …
    hf : Function.Injective f
    ⊢ (Set.image f (setOf fun M => HasSubset.Subset M.E E)).Finite
  -/
  refine (hE.finite_subsets.prod hE.finite_subsets.finite_subsets).subset ?_
  /-
    α : Type u_1
    E : Set α
    hE : E.Finite
    f : Matroid α → Prod (Set α) (Set (Set α)) := fun M => { fst := M.E, snd := se …
    hf : Function.Injective f
    ⊢ HasSubset.Subset (Set.image f (setOf fun M => HasSubset.Subset M.E E)) (SPro …
  -/
  rintro _ ⟨M, hE : M.E ⊆ E, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    E : Set α
    hE✝ : E.Finite
    f : Matroid α → Prod (Set α) (Set (Set α)) := fun M => { fst := M.E, snd := se …
    hf : Function.Injective f
    M : Matroid α
    hE : HasSubset.Subset M.E E
    ⊢ Membership.mem (SProd.sprod (setOf fun b => HasSubset.Subset b E) (setOf fun …
  -/
  simp only [Set.mem_prod, Set.mem_setOf_eq, Set.setOf_subset_setOf]
  /-
    case intro.intro
    α : Type u_1
    E : Set α
    hE✝ : E.Finite
    f : Matroid α → Prod (Set α) (Set (Set α)) := fun M => { fst := M.E, snd := se …
    hf : Function.Injective f
    M : Matroid α
    hE : HasSubset.Subset M.E E
    ⊢ And (HasSubset.Subset M.E E) (∀ (a : Set α), M.Base a → HasSubset.Subset a E)
  -/
  exact ⟨hE, fun B hB ↦ hB.subset_ground.trans hE⟩
  /-
    🎉 no goals
  -/


/-- For finite `E`, finitely many matroids have ground set `E`. -/
theorem finite_setOf_matroid' {E : Set α} (hE : E.Finite) : {M : Matroid α | M.E = E}.Finite :=
                                               /-
                                                 α : Type u_1
                                                 E : Set α
                                                 hE : E.Finite
                                                 M : Matroid α
                                                 ⊢ Membership.mem (setOf fun M => Eq M.E E) M → Membership.mem (setOf fun M =>  …
                                               -/
  (finite_setOf_matroid hE).subset (fun M ↦ by rintro rfl; exact rfl.subset)
                                                           /-
                                                             🎉 no goals
                                                           -/


