/-- A filter `F` on a type `α` is a collection of sets of `α` which contains the whole `α`,
is upwards-closed, and is stable under intersection. We do not forbid this collection to be
all sets of `α`. -/
structure Filter (α : Type*) where
  /-- The set of sets that belong to the filter. -/
  sets : Set (Set α)
  /-- The set `Set.univ` belongs to any filter. -/
  univ_sets : Set.univ ∈ sets
  /-- If a set belongs to a filter, then its superset belongs to the filter as well. -/
  sets_of_superset {x y} : x ∈ sets → x ⊆ y → y ∈ sets
  /-- If two sets belong to a filter, then their intersection belongs to the filter as well. -/
  inter_sets {x y} : x ∈ sets → y ∈ sets → x ∩ y ∈ sets


theorem filter_eq : ∀ {f g : Filter α}, f.sets = g.sets → f = g
  | ⟨_, _, _, _⟩, ⟨_, _, _, _⟩, rfl => rfl


/-- If `F` is a filter on `α`, and `U` a subset of `α` then we can write `U ∈ F` as on paper. -/
instance instMembership : Membership (Set α) (Filter α) := ⟨fun F U => U ∈ F.sets⟩


@[ext]
protected theorem ext (h : ∀ s, s ∈ f ↔ s ∈ g) : f = g := filter_eq <| Set.ext h


@[simp]
protected theorem mem_mk {t : Set (Set α)} {h₁ h₂ h₃} : s ∈ mk t h₁ h₂ h₃ ↔ s ∈ t :=
  Iff.rfl


@[simp]
protected theorem mem_sets : s ∈ f.sets ↔ s ∈ f :=
  Iff.rfl


@[simp]
theorem univ_mem : univ ∈ f :=
  f.univ_sets


theorem mem_of_superset {x y : Set α} (hx : x ∈ f) (hxy : x ⊆ y) : y ∈ f :=
  f.sets_of_superset hx hxy


theorem univ_mem' (h : ∀ a, a ∈ s) : s ∈ f :=
  mem_of_superset univ_mem fun x _ => h x


theorem inter_mem (hs : s ∈ f) (ht : t ∈ f) : s ∩ t ∈ f :=
  f.inter_sets hs ht


theorem mp_mem (hs : s ∈ f) (h : { x | x ∈ s → x ∈ t } ∈ f) : t ∈ f :=
  mem_of_superset (inter_mem hs h) fun _ ⟨h₁, h₂⟩ => h₂ h₁


/-- Override `sets` field of a filter to provide better definitional equality. -/
protected def copy (f : Filter α) (S : Set (Set α)) (hmem : ∀ s, s ∈ S ↔ s ∈ f) : Filter α where
  sets := S
  univ_sets := (hmem _).2 univ_mem
  sets_of_superset h hsub := (hmem _).2 <| mem_of_superset ((hmem _).1 h) hsub
  inter_sets h₁ h₂ := (hmem _).2 <| inter_mem ((hmem _).1 h₁) ((hmem _).1 h₂)


@[simp] theorem mem_copy {S hmem} : s ∈ f.copy S hmem ↔ s ∈ S := Iff.rfl


/-- Construct a filter from a property that is stable under finite unions.
A set `s` belongs to `Filter.comk p _ _ _` iff its complement satisfies the predicate `p`.
This constructor is useful to define filters like `Filter.cofinite`. -/
def comk (p : Set α → Prop) (he : p ∅) (hmono : ∀ t, p t → ∀ s ⊆ t, p s)
    (hunion : ∀ s, p s → ∀ t, p t → p (s ∪ t)) : Filter α where
  sets := {t | p tᶜ}
                  /-
                    α : Type u_1
                    β : Type u_2
                    f g : Filter α
                    s t : Set α
                    p : Set α → Prop
                    he : p EmptyCollection.emptyCollection
                    hmono : ∀ (t : Set α), p t → ∀ (s : Set α), HasSubset.Subset s t → p s
                    hunion : ∀ (s : Set α), p s → ∀ (t : Set α), p t → p (Union.union s t)
                    ⊢ Membership.mem (setOf fun t => p (HasCompl.compl t)) Set.univ
                  -/
  univ_sets := by simpa
                  /-
                    🎉 no goals
                  -/
  sets_of_superset := fun ht₁ ht => hmono _ ht₁ _ (compl_subset_compl.2 ht)
                                  /-
                                    α : Type u_1
                                    β : Type u_2
                                    f g : Filter α
                                    s t : Set α
                                    p : Set α → Prop
                                    he : p EmptyCollection.emptyCollection
                                    hmono : ∀ (t : Set α), p t → ∀ (s : Set α), HasSubset.Subset s t → p s
                                    hunion : ∀ (s : Set α), p s → ∀ (t : Set α), p t → p (Union.union s t)
                                    x✝ y✝ : Set α
                                    ht₁ : Membership.mem (setOf fun t => p (HasCompl.compl t)) x✝
                                    ht₂ : Membership.mem (setOf fun t => p (HasCompl.compl t)) y✝
                                    ⊢ Membership.mem (setOf fun t => p (HasCompl.compl t)) (Inter.inter x✝ y✝)
                                  -/
  inter_sets := fun ht₁ ht₂ => by simp [compl_inter, hunion _ ht₁ _ ht₂]
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
lemma mem_comk {p : Set α → Prop} {he hmono hunion s} :
    s ∈ comk p he hmono hunion ↔ p sᶜ :=
  .rfl


/-- The principal filter of `s` is the collection of all supersets of `s`. -/
def principal (s : Set α) : Filter α where
  sets := { t | s ⊆ t }
  univ_sets := subset_univ s
  sets_of_superset hx := Subset.trans hx
  inter_sets := subset_inter


@[inherit_doc]
scoped notation "𝓟" => Filter.principal


@[simp] theorem mem_principal : s ∈ 𝓟 t ↔ t ⊆ s := Iff.rfl


/-- `pure x` is the set of sets that contain `x`. It is equal to `𝓟 {x}` but
with this definition we have `s ∈ pure a` defeq `a ∈ s`. -/
instance : Pure Filter where
                                                 /-
                                                   α : Type u_1
                                                   β : Type u_2
                                                   f g : Filter α
                                                   s t : Set α
                                                   α✝ : Type ?u.5990
                                                   x : α✝
                                                   x✝ : Set α✝
                                                   ⊢ Iff (Membership.mem (setOf fun s => Membership.mem s x) x✝) (Membership.mem  …
                                                 -/
  pure x := .copy (𝓟 {x}) {s | x ∈ s} fun _ ↦ by simp
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem mem_pure {a : α} {s : Set α} : s ∈ (pure a : Filter α) ↔ a ∈ s :=
  Iff.rfl


/-- The *kernel* of a filter is the intersection of all its sets. -/
def ker (f : Filter α) : Set α := ⋂₀ f.sets


/-- The join of a filter of filters is defined by the relation `s ∈ join f ↔ {t | s ∈ t} ∈ f`. -/
def join (f : Filter (Filter α)) : Filter α where
  sets := { s | { t : Filter α | s ∈ t } ∈ f }
                  /-
                    α : Type u_1
                    β : Type u_2
                    f✝ g : Filter α
                    s t : Set α
                    f : Filter (Filter α)
                    ⊢ Membership.mem (setOf fun s => Membership.mem f (setOf fun t => Membership.m …
                  -/
  univ_sets := by simp only [mem_setOf_eq, univ_mem, setOf_true]
                  /-
                    🎉 no goals
                  -/
  sets_of_superset hx xy := mem_of_superset hx fun f h => mem_of_superset h xy
  inter_sets hx hy := mem_of_superset (inter_mem hx hy) fun f ⟨h₁, h₂⟩ => inter_mem h₁ h₂


@[simp]
theorem mem_join {s : Set α} {f : Filter (Filter α)} : s ∈ join f ↔ { t | s ∈ t } ∈ f :=
  Iff.rfl


instance : PartialOrder (Filter α) where
  le f g := ∀ ⦃U : Set α⦄, U ∈ g → U ∈ f
  le_antisymm a b h₁ h₂ := filter_eq <| Subset.antisymm h₂ h₁
  le_refl a := Subset.rfl
  le_trans a b c h₁ h₂ := Subset.trans h₂ h₁


theorem le_def : f ≤ g ↔ ∀ x ∈ g, x ∈ f :=
  Iff.rfl


instance instSupSet : SupSet (Filter α) where
  sSup S := join (𝓟 S)


@[simp] theorem mem_sSup {S : Set (Filter α)} : s ∈ sSup S ↔ ∀ f ∈ S, s ∈ f := .rfl


/-- Infimum of a set of filters.
This definition is marked as irreducible
so that Lean doesn't try to unfold it when unifying expressions. -/
@[irreducible]
protected def sInf (s : Set (Filter α)) : Filter α := sSup (lowerBounds s)


instance instInfSet : InfSet (Filter α) where
  sInf := Filter.sInf


protected theorem sSup_lowerBounds (s : Set (Filter α)) : sSup (lowerBounds s) = sInf s := by
  /-
    α : Type u_1
    s : Set (Filter α)
    ⊢ Eq (SupSet.sSup (lowerBounds s)) (InfSet.sInf s)
  -/
  simp [sInf, Filter.sInf]
  /-
    🎉 no goals
  -/


instance : Top (Filter α) where
                                                              /-
                                                                α : Type u_1
                                                                β : Type u_2
                                                                f g : Filter α
                                                                s t : Set α
                                                                ⊢ ∀ (s : Set α), Iff (Membership.mem (setOf fun s => ∀ (x : α), Membership.mem …
                                                              -/
  top := .copy (sSup (Set.range pure)) {s | ∀ x, x ∈ s} <| by simp
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem mem_top_iff_forall {s : Set α} : s ∈ (⊤ : Filter α) ↔ ∀ x, x ∈ s :=
  Iff.rfl


@[simp]
theorem mem_top {s : Set α} : s ∈ (⊤ : Filter α) ↔ s = univ := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (Membership.mem Top.top s) (Eq s Set.univ)
  -/
  rw [mem_top_iff_forall, eq_univ_iff_forall]
  /-
    🎉 no goals
  -/


instance : Bot (Filter α) where
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     f g : Filter α
                                     s t : Set α
                                     ⊢ ∀ (s : Set α), Iff (Membership.mem Set.univ s) (Membership.mem (SupSet.sSup  …
                                   -/
  bot := .copy (sSup ∅) univ <| by simp
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem mem_bot {s : Set α} : s ∈ (⊥ : Filter α) :=
  trivial


/-- The infimum of filters is the filter generated by intersections
  of elements of the two filters. -/
instance instInf : Min (Filter α) :=
  ⟨fun f g : Filter α =>
    { sets := { s | ∃ a ∈ f, ∃ b ∈ g, s = a ∩ b }
                                                 /-
                                                   α : Type u_1
                                                   β : Type u_2
                                                   f✝ g✝ : Filter α
                                                   s t : Set α
                                                   f g : Filter α
                                                   ⊢ Eq Set.univ (Inter.inter Set.univ Set.univ)
                                                 -/
      univ_sets := ⟨_, univ_mem, _, univ_mem, by simp⟩
                                                 /-
                                                   🎉 no goals
                                                 -/
      sets_of_superset := by
        /-
          α : Type u_1
          β : Type u_2
          f✝ g✝ : Filter α
          s t : Set α
          f g : Filter α
          ⊢ ∀ {x y : Set α}, Membership.mem (setOf fun s => Exists fun a => And (Members …
        -/
        rintro x y ⟨a, ha, b, hb, rfl⟩ xy
        refine ⟨a ∪ y, mem_of_superset ha subset_union_left, b ∪ y,
          mem_of_superset hb subset_union_left, ?_⟩
        /-
          case intro.intro.intro.intro
          α : Type u_1
          β : Type u_2
          f✝ g✝ : Filter α
          s t : Set α
          f g : Filter α
          y a : Set α
          ha : Membership.mem f a
          b : Set α
          hb : Membership.mem g b
          xy : HasSubset.Subset (Inter.inter a b) y
          ⊢ Eq y (Inter.inter (Union.union a y) (Union.union b y))
        -/
        rw [← inter_union_distrib_right, union_eq_self_of_subset_left xy]
        /-
          🎉 no goals
        -/
      inter_sets := by
        /-
          α : Type u_1
          β : Type u_2
          f✝ g✝ : Filter α
          s t : Set α
          f g : Filter α
          ⊢ ∀ {x y : Set α}, Membership.mem (setOf fun s => Exists fun a => And (Members …
        -/
        rintro x y ⟨a, ha, b, hb, rfl⟩ ⟨c, hc, d, hd, rfl⟩
        /-
          case intro.intro.intro.intro.intro.intro.intro.intro
          α : Type u_1
          β : Type u_2
          f✝ g✝ : Filter α
          s t : Set α
          f g : Filter α
          a : Set α
          ha : Membership.mem f a
          b : Set α
          hb : Membership.mem g b
          c : Set α
          hc : Membership.mem f c
          d : Set α
          hd : Membership.mem g d
          ⊢ Membership.mem (setOf fun s => Exists fun a => And (Membership.mem f a) (Exi …
        -/
        refine ⟨a ∩ c, inter_mem ha hc, b ∩ d, inter_mem hb hd, ?_⟩
        /-
          case intro.intro.intro.intro.intro.intro.intro.intro
          α : Type u_1
          β : Type u_2
          f✝ g✝ : Filter α
          s t : Set α
          f g : Filter α
          a : Set α
          ha : Membership.mem f a
          b : Set α
          hb : Membership.mem g b
          c : Set α
          hc : Membership.mem f c
          d : Set α
          hd : Membership.mem g d
          ⊢ Eq (Inter.inter (Inter.inter a b) (Inter.inter c d)) (Inter.inter (Inter.int …
        -/
        ac_rfl }⟩
        /-
          🎉 no goals
        -/


/-- The supremum of two filters is the filter that contains sets that belong to both filters. -/
instance instSup : Max (Filter α) where
                                                           /-
                                                             α : Type u_1
                                                             β : Type u_2
                                                             f✝ g✝ : Filter α
                                                             s t : Set α
                                                             f g : Filter α
                                                             ⊢ ∀ (s : Set α), Iff (Membership.mem (setOf fun s => And (Membership.mem f s)  …
                                                           -/
  max f g := .copy (sSup {f, g}) {s | s ∈ f ∧ s ∈ g} <| by simp
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- A filter is `NeBot` if it is not equal to `⊥`, or equivalently the empty set does not belong to
the filter. Bourbaki include this assumption in the definition of a filter but we prefer to have a
`CompleteLattice` structure on `Filter _`, so we use a typeclass argument in lemmas instead. -/
class NeBot (f : Filter α) : Prop where
  /-- The filter is nontrivial: `f ≠ ⊥` or equivalently, `∅ ∉ f`. -/
  ne' : f ≠ ⊥


theorem neBot_iff {f : Filter α} : NeBot f ↔ f ≠ ⊥ :=
  ⟨fun h => h.1, fun h => ⟨h⟩⟩


/-- `f.Eventually p` or `∀ᶠ x in f, p x` mean that `{x | p x} ∈ f`. E.g., `∀ᶠ x in atTop, p x`
means that `p` holds true for sufficiently large `x`. -/
protected def Eventually (p : α → Prop) (f : Filter α) : Prop :=
  { x | p x } ∈ f


@[inherit_doc Filter.Eventually]
notation3 "∀ᶠ "(...)" in "f", "r:(scoped p => Filter.Eventually p f) => r


/-- `f.Frequently p` or `∃ᶠ x in f, p x` mean that `{x | ¬p x} ∉ f`. E.g., `∃ᶠ x in atTop, p x`
means that there exist arbitrarily large `x` for which `p` holds true. -/
protected def Frequently (p : α → Prop) (f : Filter α) : Prop :=
  ¬∀ᶠ x in f, ¬p x


@[inherit_doc Filter.Frequently]
notation3 "∃ᶠ "(...)" in "f", "r:(scoped p => Filter.Frequently p f) => r


/-- Two functions `f` and `g` are *eventually equal* along a filter `l` if the set of `x` such that
`f x = g x` belongs to `l`. -/
def EventuallyEq (l : Filter α) (f g : α → β) : Prop :=
  ∀ᶠ x in l, f x = g x


@[inherit_doc]
notation:50 f " =ᶠ[" l:50 "] " g:50 => EventuallyEq l f g


/-- A function `f` is eventually less than or equal to a function `g` at a filter `l`. -/
def EventuallyLE [LE β] (l : Filter α) (f g : α → β) : Prop :=
  ∀ᶠ x in l, f x ≤ g x


@[inherit_doc]
notation:50 f " ≤ᶠ[" l:50 "] " g:50 => EventuallyLE l f g


/-- The forward map of a filter -/
def map (m : α → β) (f : Filter α) : Filter β where
  sets := preimage m ⁻¹' f.sets
  univ_sets := univ_mem
  sets_of_superset hs st := mem_of_superset hs fun _x hx ↦ st hx
  inter_sets hs ht := inter_mem hs ht


/-- `Filter.Tendsto` is the generic "limit of a function" predicate.
  `Tendsto f l₁ l₂` asserts that for every `l₂` neighborhood `a`,
  the `f`-preimage of `a` is an `l₁` neighborhood. -/
def Tendsto (f : α → β) (l₁ : Filter α) (l₂ : Filter β) :=
  l₁.map f ≤ l₂


/-- The inverse map of a filter. A set `s` belongs to `Filter.comap m f` if either of the following
equivalent conditions hold.

1. There exists a set `t ∈ f` such that `m ⁻¹' t ⊆ s`. This is used as a definition.
2. The set `kernImage m s = {y | ∀ x, m x = y → x ∈ s}` belongs to `f`, see `Filter.mem_comap'`.
3. The set `(m '' sᶜ)ᶜ` belongs to `f`, see `Filter.mem_comap_iff_compl` and
`Filter.compl_mem_comap`. -/
def comap (m : α → β) (f : Filter β) : Filter α where
  sets := { s | ∃ t ∈ f, m ⁻¹' t ⊆ s }
  univ_sets := ⟨univ, univ_mem, subset_univ _⟩
  sets_of_superset := fun ⟨a', ha', ma'a⟩ ab => ⟨a', ha', ma'a.trans ab⟩
  inter_sets := fun ⟨a', ha₁, ha₂⟩ ⟨b', hb₁, hb₂⟩ =>
    ⟨a' ∩ b', inter_mem ha₁ hb₁, inter_subset_inter ha₂ hb₂⟩


/-- Coproduct of filters. -/
protected def coprod (f : Filter α) (g : Filter β) : Filter (α × β) :=
  f.comap Prod.fst ⊔ g.comap Prod.snd


/-- Product of filters. This is the filter generated by cartesian products
of elements of the component filters. -/
instance instSProd : SProd (Filter α) (Filter β) (Filter (α × β)) where
  sprod f g := f.comap Prod.fst ⊓ g.comap Prod.snd


/-- Product of filters. This is the filter generated by cartesian products
of elements of the component filters.
Deprecated. Use `f ×ˢ g` instead. -/
@[deprecated " Use `f ×ˢ g` instead." (since := "2024-11-29")]
protected def prod (f : Filter α) (g : Filter β) : Filter (α × β) := f ×ˢ g


theorem prod_eq_inf (f : Filter α) (g : Filter β) : f ×ˢ g = f.comap Prod.fst ⊓ g.comap Prod.snd :=
  rfl


/-- The product of an indexed family of filters. -/
def pi {ι : Type*} {α : ι → Type*} (f : ∀ i, Filter (α i)) : Filter (∀ i, α i) :=
  ⨅ i, comap (Function.eval i) (f i)


/-- The monadic bind operation on filter is defined the usual way in terms of `map` and `join`.

Unfortunately, this `bind` does not result in the expected applicative. See `Filter.seq` for the
applicative instance. -/
def bind (f : Filter α) (m : α → Filter β) : Filter β :=
  join (map m f)


/-- The applicative sequentiation operation. This is not induced by the bind operation. -/
def seq (f : Filter (α → β)) (g : Filter α) : Filter β where
  sets := { s | ∃ u ∈ f, ∃ t ∈ g, ∀ m ∈ u, ∀ x ∈ t, (m : α → β) x ∈ s }
  univ_sets := ⟨univ, univ_mem, univ, univ_mem, fun _ _ _ _ => trivial⟩
  sets_of_superset := fun ⟨t₀, t₁, h₀, h₁, h⟩ hst =>
    ⟨t₀, t₁, h₀, h₁, fun _ hx _ hy => hst <| h _ hx _ hy⟩
  inter_sets := fun ⟨t₀, ht₀, t₁, ht₁, ht⟩ ⟨u₀, hu₀, u₁, hu₁, hu⟩ =>
    ⟨t₀ ∩ u₀, inter_mem ht₀ hu₀, t₁ ∩ u₁, inter_mem ht₁ hu₁, fun _ ⟨hx₀, hx₁⟩ _ ⟨hy₀, hy₁⟩ =>
      ⟨ht _ hx₀ _ hy₀, hu _ hx₁ _ hy₁⟩⟩


/-- This filter is characterized by `Filter.eventually_curry_iff`:
`(∀ᶠ (x : α × β) in f.curry g, p x) ↔ ∀ᶠ (x : α) in f, ∀ᶠ (y : β) in g, p (x, y)`. Useful
in adding quantifiers to the middle of `Tendsto`s. See
`hasFDerivAt_of_tendstoUniformlyOnFilter`. -/
def curry (f : Filter α) (g : Filter β) : Filter (α × β) :=
  bind f fun a ↦ map (a, ·) g


instance : Bind Filter :=
  ⟨@Filter.bind⟩


instance : Functor Filter where map := @Filter.map


/-- A variant on `bind` using a function `g` taking a set instead of a member of `α`.
This is essentially a push-forward along a function mapping each set to a filter. -/
protected def lift (f : Filter α) (g : Set α → Filter β) :=
  ⨅ s ∈ f, g s


/-- Specialize `lift` to functions `Set α → Set β`. This can be viewed as a generalization of `map`.
This is essentially a push-forward along a function mapping each set to a set. -/
protected def lift' (f : Filter α) (h : Set α → Set β) :=
  f.lift (𝓟 ∘ h)


/-- `f.IsBounded r`: the filter `f` is eventually bounded w.r.t. the relation `r`,
i.e. eventually, it is bounded by some uniform bound.
`r` will be usually instantiated with `(· ≤ ·)` or `(· ≥ ·)`. -/
def IsBounded (r : α → α → Prop) (f : Filter α) :=
  ∃ b, ∀ᶠ x in f, r x b


/-- `f.IsBoundedUnder (≺) u`: the image of the filter `f` under `u` is eventually bounded w.r.t.
the relation `≺`, i.e. eventually, it is bounded by some uniform bound. -/
def IsBoundedUnder (r : α → α → Prop) (f : Filter β) (u : β → α) :=
  (map u f).IsBounded r


/-- `IsCobounded (≺) f` states that the filter `f` does not tend to infinity w.r.t. `≺`. This is
also called frequently bounded. Will be usually instantiated with `≤` or `≥`.

There is a subtlety in this definition: we want `f.IsCobounded` to hold for any `f` in the case of
complete lattices. This will be relevant to deduce theorems on complete lattices from their
versions on conditionally complete lattices with additional assumptions. We have to be careful in
the edge case of the trivial filter containing the empty set: the other natural definition
  `¬ ∀ a, ∀ᶠ n in f, a ≤ n`
would not work as well in this case.
-/
def IsCobounded (r : α → α → Prop) (f : Filter α) :=
  ∃ b, ∀ a, (∀ᶠ x in f, r x a) → r b a


/-- `IsCoboundedUnder (≺) f u` states that the image of the filter `f` under the map `u` does not
tend to infinity w.r.t. `≺`. This is also called frequently bounded. Will be usually instantiated
with `≤` or `≥`. -/
def IsCoboundedUnder (r : α → α → Prop) (f : Filter β) (u : β → α) :=
  (map u f).IsCobounded r


/--
`filter_upwards [h₁, ⋯, hₙ]` replaces a goal of the form `s ∈ f` and terms
`h₁ : t₁ ∈ f, ⋯, hₙ : tₙ ∈ f` with `∀ x, x ∈ t₁ → ⋯ → x ∈ tₙ → x ∈ s`.
The list is an optional parameter, `[]` being its default value.

`filter_upwards [h₁, ⋯, hₙ] with a₁ a₂ ⋯ aₖ` is a short form for
`{ filter_upwards [h₁, ⋯, hₙ], intros a₁ a₂ ⋯ aₖ }`.

`filter_upwards [h₁, ⋯, hₙ] using e` is a short form for
`{ filter_upwards [h1, ⋯, hn], exact e }`.

Combining both shortcuts is done by writing `filter_upwards [h₁, ⋯, hₙ] with a₁ a₂ ⋯ aₖ using e`.
Note that in this case, the `aᵢ` terms can be used in `e`.
-/
syntax (name := filterUpwards) "filter_upwards" (" [" term,* "]")?
  (" with" (ppSpace colGt term:max)*)? (" using " term)? : tactic


elab_rules : tactic
| `(tactic| filter_upwards $[[$[$args],*]]? $[with $wth*]? $[using $usingArg]?) => do
  let config : ApplyConfig := {newGoals := ApplyNewGoals.nonDependentOnly}
  for e in args.getD #[] |>.reverse do
    let goal ← getMainGoal
    replaceMainGoal <| ← goal.withContext <| runTermElab do
      let m ← mkFreshExprMVar none
      let lem ← Term.elabTermEnsuringType
        (← ``(Filter.mp_mem $e $(← Term.exprToSyntax m))) (← goal.getType)
      goal.assign lem
      return [m.mvarId!]
  liftMetaTactic fun goal => do
    goal.apply (← mkConstWithFreshMVarLevels ``Filter.univ_mem') config
  evalTactic <|← `(tactic| dsimp (config := {zeta := false}) only [Set.mem_setOf_eq])
  if let some l := wth then
    evalTactic <|← `(tactic| intro $[$l]*)
  if let some e := usingArg then
    evalTactic <|← `(tactic| exact $e)


