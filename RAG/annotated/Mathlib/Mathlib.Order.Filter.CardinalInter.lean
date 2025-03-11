/-- A filter `l` has the cardinal `c` intersection property if for any collection
of less than `c` sets `s ∈ l`, their intersection belongs to `l` as well. -/
class CardinalInterFilter (l : Filter α) (c : Cardinal.{u}) : Prop where
  /-- For a collection of sets `s ∈ l` with cardinality below c,
  their intersection belongs to `l` as well. -/
  cardinal_sInter_mem : ∀ S : Set (Set α), (#S < c) → (∀ s ∈ S, s ∈ l) → ⋂₀ S ∈ l


theorem cardinal_sInter_mem {S : Set (Set α)} [CardinalInterFilter l c] (hSc : #S < c) :
    ⋂₀ S ∈ l ↔ ∀ s ∈ S, s ∈ l := ⟨fun hS _s hs => mem_of_superset hS (sInter_subset_of_mem hs),
  CardinalInterFilter.cardinal_sInter_mem _ hSc⟩


/-- Every filter is a CardinalInterFilter with c = ℵ₀ -/
theorem _root_.Filter.cardinalInterFilter_aleph0 (l : Filter α) : CardinalInterFilter l ℵ₀ where
  cardinal_sInter_mem := by
    simp_all only [aleph_zero, lt_aleph0_iff_subtype_finite, setOf_mem_eq, sInter_mem,
      implies_true, forall_const]


/-- Every CardinalInterFilter with c > ℵ₀ is a CountableInterFilter -/
theorem CardinalInterFilter.toCountableInterFilter (l : Filter α) [CardinalInterFilter l c]
    (hc : ℵ₀ < c) : CountableInterFilter l where
  countable_sInter_mem S hS a :=
    CardinalInterFilter.cardinal_sInter_mem S (lt_of_le_of_lt (Set.Countable.le_aleph0 hS) hc) a


/-- Every CountableInterFilter is a CardinalInterFilter with c = ℵ₁ -/
instance CountableInterFilter.toCardinalInterFilter (l : Filter α) [CountableInterFilter l] :
    CardinalInterFilter l ℵ₁ where
  cardinal_sInter_mem S hS a :=
    CountableInterFilter.countable_sInter_mem S ((countable_iff_lt_aleph_one S).mpr hS) a


theorem cardinalInterFilter_aleph_one_iff :
    CardinalInterFilter l ℵ₁ ↔ CountableInterFilter l :=
  ⟨fun _ ↦ ⟨fun S h a ↦
    CardinalInterFilter.cardinal_sInter_mem S ((countable_iff_lt_aleph_one S).1 h) a⟩,
   fun _ ↦ CountableInterFilter.toCardinalInterFilter l⟩


/-- Every CardinalInterFilter for some c also is a CardinalInterFilter for some a ≤ c -/
theorem CardinalInterFilter.of_cardinalInterFilter_of_le (l : Filter α) [CardinalInterFilter l c]
    {a : Cardinal.{u}} (hac : a ≤ c) :
    CardinalInterFilter l a where
  cardinal_sInter_mem S hS a :=
    CardinalInterFilter.cardinal_sInter_mem S (lt_of_lt_of_le hS hac) a


theorem CardinalInterFilter.of_cardinalInterFilter_of_lt (l : Filter α) [CardinalInterFilter l c]
    {a : Cardinal.{u}} (hac : a < c) : CardinalInterFilter l a :=
  CardinalInterFilter.of_cardinalInterFilter_of_le l (hac.le)


theorem cardinal_iInter_mem {s : ι → Set α} (hic : #ι < c) :
    (⋂ i, s i) ∈ l ↔ ∀ i, s i ∈ l := by
  /-
    ι α : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝ : CardinalInterFilter l c
    s : ι → Set α
    hic : LT.lt (Cardinal.mk ι) c
    ⊢ Iff (Membership.mem l (Set.iInter fun i => s i)) (∀ (i : ι), Membership.mem  …
  -/
  rw [← sInter_range _]
  /-
    ι α : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝ : CardinalInterFilter l c
    s : ι → Set α
    hic : LT.lt (Cardinal.mk ι) c
    ⊢ Iff (Membership.mem l (Set.range s).sInter) (∀ (i : ι), Membership.mem l (s  …
  -/
  apply (cardinal_sInter_mem (lt_of_le_of_lt Cardinal.mk_range_le hic)).trans
  /-
    ι α : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝ : CardinalInterFilter l c
    s : ι → Set α
    hic : LT.lt (Cardinal.mk ι) c
    ⊢ Iff (∀ (s_1 : Set α), Membership.mem (Set.range s) s_1 → Membership.mem l s_ …
  -/
  exact forall_mem_range
  /-
    🎉 no goals
  -/


theorem cardinal_bInter_mem {S : Set ι} (hS : #S < c)
    {s : ∀ i ∈ S, Set α} :
    (⋂ i, ⋂ hi : i ∈ S, s i ‹_›) ∈ l ↔ ∀ i, ∀ hi : i ∈ S, s i ‹_› ∈ l := by
  /-
    ι α : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝ : CardinalInterFilter l c
    S : Set ι
    hS : LT.lt (Cardinal.mk ↑S) c
    s : (i : ι) → Membership.mem S i → Set α
    ⊢ Iff (Membership.mem l (Set.iInter fun i => Set.iInter fun hi => s i hi)) (∀  …
  -/
  rw [biInter_eq_iInter]
  /-
    ι α : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝ : CardinalInterFilter l c
    S : Set ι
    hS : LT.lt (Cardinal.mk ↑S) c
    s : (i : ι) → Membership.mem S i → Set α
    ⊢ Iff (Membership.mem l (Set.iInter fun x => s ↑x ⋯)) (∀ (i : ι) (hi : Members …
  -/
  exact (cardinal_iInter_mem hS).trans Subtype.forall
  /-
    🎉 no goals
  -/


theorem eventually_cardinal_forall {p : α → ι → Prop} (hic : #ι < c) :
    (∀ᶠ x in l, ∀ i, p x i) ↔ ∀ i, ∀ᶠ x in l, p x i := by
  /-
    ι α : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝ : CardinalInterFilter l c
    p : α → ι → Prop
    hic : LT.lt (Cardinal.mk ι) c
    ⊢ Iff (Filter.Eventually (fun x => ∀ (i : ι), p x i) l) (∀ (i : ι), Filter.Eve …
  -/
  simp only [Filter.Eventually, setOf_forall]
  /-
    ι α : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝ : CardinalInterFilter l c
    p : α → ι → Prop
    hic : LT.lt (Cardinal.mk ι) c
    ⊢ Iff (Membership.mem l (Set.iInter fun i => setOf fun x => p x i)) (∀ (i : ι) …
  -/
  exact cardinal_iInter_mem hic
  /-
    🎉 no goals
  -/


theorem eventually_cardinal_ball {S : Set ι} (hS : #S < c)
    {p : α → ∀ i ∈ S, Prop} :
    (∀ᶠ x in l, ∀ i hi, p x i hi) ↔ ∀ i hi, ∀ᶠ x in l, p x i hi := by
  /-
    ι α : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝ : CardinalInterFilter l c
    S : Set ι
    hS : LT.lt (Cardinal.mk ↑S) c
    p : α → (i : ι) → Membership.mem S i → Prop
    ⊢ Iff (Filter.Eventually (fun x => ∀ (i : ι) (hi : Membership.mem S i), p x i  …
  -/
  simp only [Filter.Eventually, setOf_forall]
  /-
    ι α : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝ : CardinalInterFilter l c
    S : Set ι
    hS : LT.lt (Cardinal.mk ↑S) c
    p : α → (i : ι) → Membership.mem S i → Prop
    ⊢ Iff (Membership.mem l (Set.iInter fun i => Set.iInter fun i_1 => setOf fun x …
  -/
  exact cardinal_bInter_mem hS
  /-
    🎉 no goals
  -/


theorem EventuallyLE.cardinal_iUnion {s t : ι → Set α} (hic : #ι < c)
    (h : ∀ i, s i ≤ᶠ[l] t i) : ⋃ i, s i ≤ᶠ[l] ⋃ i, t i :=
  ((eventually_cardinal_forall hic).2 h).mono fun _ hst hs => mem_iUnion.2 <|
    (mem_iUnion.1 hs).imp hst


theorem EventuallyEq.cardinal_iUnion {s t : ι → Set α} (hic : #ι < c)
    (h : ∀ i, s i =ᶠ[l] t i) : ⋃ i, s i =ᶠ[l] ⋃ i, t i :=
  (EventuallyLE.cardinal_iUnion hic fun i => (h i).le).antisymm
    (EventuallyLE.cardinal_iUnion hic fun i => (h i).symm.le)


theorem EventuallyLE.cardinal_bUnion {S : Set ι} (hS : #S < c)
    {s t : ∀ i ∈ S, Set α} (h : ∀ i hi, s i hi ≤ᶠ[l] t i hi) :
    ⋃ i ∈ S, s i ‹_› ≤ᶠ[l] ⋃ i ∈ S, t i ‹_› := by
  /-
    ι α : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝ : CardinalInterFilter l c
    S : Set ι
    hS : LT.lt (Cardinal.mk ↑S) c
    s t : (i : ι) → Membership.mem S i → Set α
    h : ∀ (i : ι) (hi : Membership.mem S i), l.EventuallyLE (s i hi) (t i hi)
    ⊢ l.EventuallyLE (Set.iUnion fun i => Set.iUnion fun h => s i h) (Set.iUnion f …
  -/
  simp only [biUnion_eq_iUnion]
  /-
    ι α : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝ : CardinalInterFilter l c
    S : Set ι
    hS : LT.lt (Cardinal.mk ↑S) c
    s t : (i : ι) → Membership.mem S i → Set α
    h : ∀ (i : ι) (hi : Membership.mem S i), l.EventuallyLE (s i hi) (t i hi)
    ⊢ l.EventuallyLE (Set.iUnion fun x => s ↑x ⋯) (Set.iUnion fun x => t ↑x ⋯)
  -/
  exact EventuallyLE.cardinal_iUnion hS fun i => h i i.2
  /-
    🎉 no goals
  -/


theorem EventuallyEq.cardinal_bUnion {S : Set ι} (hS : #S < c)
    {s t : ∀ i ∈ S, Set α} (h : ∀ i hi, s i hi =ᶠ[l] t i hi) :
    ⋃ i ∈ S, s i ‹_› =ᶠ[l] ⋃ i ∈ S, t i ‹_› :=
  (EventuallyLE.cardinal_bUnion hS fun i hi => (h i hi).le).antisymm
    (EventuallyLE.cardinal_bUnion hS fun i hi => (h i hi).symm.le)


theorem EventuallyLE.cardinal_iInter {s t : ι → Set α} (hic : #ι < c)
    (h : ∀ i, s i ≤ᶠ[l] t i) : ⋂ i, s i ≤ᶠ[l] ⋂ i, t i :=
  ((eventually_cardinal_forall hic).2 h).mono fun _ hst hs =>
    mem_iInter.2 fun i => hst _ (mem_iInter.1 hs i)


theorem EventuallyEq.cardinal_iInter {s t : ι → Set α} (hic : #ι < c)
    (h : ∀ i, s i =ᶠ[l] t i) : ⋂ i, s i =ᶠ[l] ⋂ i, t i :=
  (EventuallyLE.cardinal_iInter hic fun i => (h i).le).antisymm
    (EventuallyLE.cardinal_iInter hic fun i => (h i).symm.le)


theorem EventuallyLE.cardinal_bInter {S : Set ι} (hS : #S < c)
    {s t : ∀ i ∈ S, Set α} (h : ∀ i hi, s i hi ≤ᶠ[l] t i hi) :
    ⋂ i ∈ S, s i ‹_› ≤ᶠ[l] ⋂ i ∈ S, t i ‹_› := by
  /-
    ι α : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝ : CardinalInterFilter l c
    S : Set ι
    hS : LT.lt (Cardinal.mk ↑S) c
    s t : (i : ι) → Membership.mem S i → Set α
    h : ∀ (i : ι) (hi : Membership.mem S i), l.EventuallyLE (s i hi) (t i hi)
    ⊢ l.EventuallyLE (Set.iInter fun i => Set.iInter fun h => s i h) (Set.iInter f …
  -/
  simp only [biInter_eq_iInter]
  /-
    ι α : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝ : CardinalInterFilter l c
    S : Set ι
    hS : LT.lt (Cardinal.mk ↑S) c
    s t : (i : ι) → Membership.mem S i → Set α
    h : ∀ (i : ι) (hi : Membership.mem S i), l.EventuallyLE (s i hi) (t i hi)
    ⊢ l.EventuallyLE (Set.iInter fun x => s ↑x ⋯) (Set.iInter fun x => t ↑x ⋯)
  -/
  exact EventuallyLE.cardinal_iInter hS fun i => h i i.2
  /-
    🎉 no goals
  -/


theorem EventuallyEq.cardinal_bInter {S : Set ι} (hS : #S < c)
    {s t : ∀ i ∈ S, Set α} (h : ∀ i hi, s i hi =ᶠ[l] t i hi) :
    ⋂ i ∈ S, s i ‹_› =ᶠ[l] ⋂ i ∈ S, t i ‹_› :=
  (EventuallyLE.cardinal_bInter hS fun i hi => (h i hi).le).antisymm
    (EventuallyLE.cardinal_bInter hS fun i hi => (h i hi).symm.le)


/-- Construct a filter with cardinal `c` intersection property. This constructor deduces
`Filter.univ_sets` and `Filter.inter_sets` from the cardinal `c` intersection property. -/
def ofCardinalInter (l : Set (Set α)) (hc : 2 < c)
    (hl : ∀ S : Set (Set α), (#S < c) → S ⊆ l → ⋂₀ S ∈ l)
    (h_mono : ∀ s t, s ∈ l → s ⊆ t → t ∈ l) : Filter α where
  sets := l
  univ_sets :=
    sInter_empty ▸ hl ∅ (mk_eq_zero (∅ : Set (Set α)) ▸ lt_trans zero_lt_two hc) (empty_subset _)
  sets_of_superset := h_mono _ _
  inter_sets {s t} hs ht := sInter_pair s t ▸ by
    /-
      ι α β : Type u
      c : Cardinal.{u}
      l✝ : Filter α
      inst✝ : CardinalInterFilter l✝ c
      l : Set (Set α)
      hc : LT.lt 2 c
      hl : ∀ (S : Set (Set α)), LT.lt (Cardinal.mk ↑S) c → HasSubset.Subset S l → Me …
      h_mono : ∀ (s t : Set α), Membership.mem l s → HasSubset.Subset s t → Membersh …
      s t : Set α
      hs : Membership.mem l s
      ht : Membership.mem l t
      ⊢ Membership.mem l (Insert.insert s (Singleton.singleton t)).sInter
    -/
    apply hl _ (?_) (insert_subset_iff.2 ⟨hs, singleton_subset_iff.2 ht⟩)
    have : #({s, t} : Set (Set α)) ≤ 2 := by
      calc
      _ ≤ #({t} : Set (Set α)) + 1 := Cardinal.mk_insert_le
      _ = 2 := by norm_num
    /-
      ι α β : Type u
      c : Cardinal.{u}
      l✝ : Filter α
      inst✝ : CardinalInterFilter l✝ c
      l : Set (Set α)
      hc : LT.lt 2 c
      hl : ∀ (S : Set (Set α)), LT.lt (Cardinal.mk ↑S) c → HasSubset.Subset S l → Me …
      h_mono : ∀ (s t : Set α), Membership.mem l s → HasSubset.Subset s t → Membersh …
      s t : Set α
      hs : Membership.mem l s
      ht : Membership.mem l t
      this : LE.le (Cardinal.mk ↑(Insert.insert s (Singleton.singleton t))) 2
      ⊢ LT.lt (Cardinal.mk ↑(Insert.insert s (Singleton.singleton t))) c
    -/
    exact lt_of_le_of_lt this hc
    /-
      🎉 no goals
    -/


instance cardinalInter_ofCardinalInter (l : Set (Set α)) (hc : 2 < c)
    (hl : ∀ S : Set (Set α), (#S < c) → S ⊆ l → ⋂₀ S ∈ l)
    (h_mono : ∀ s t, s ∈ l → s ⊆ t → t ∈ l) :
    CardinalInterFilter (Filter.ofCardinalInter l hc hl h_mono) c :=
  ⟨hl⟩


@[simp]
theorem mem_ofCardinalInter {l : Set (Set α)} (hc : 2 < c)
    (hl : ∀ S : Set (Set α), (#S < c) → S ⊆ l → ⋂₀ S ∈ l) (h_mono : ∀ s t, s ∈ l → s ⊆ t → t ∈ l)
    {s : Set α} : s ∈ Filter.ofCardinalInter l hc hl h_mono ↔ s ∈ l :=
  Iff.rfl


/-- Construct a filter with cardinal `c` intersection property.
Similarly to `Filter.comk`, a set belongs to this filter if its complement satisfies the property.
Similarly to `Filter.ofCardinalInter`,
this constructor deduces some properties from the cardinal `c` intersection property
which becomes the cardinal `c` union property because we take complements of all sets. -/
def ofCardinalUnion (l : Set (Set α)) (hc : 2 < c)
    (hUnion : ∀ S : Set (Set α), (#S < c) → (∀ s ∈ S, s ∈ l) → ⋃₀ S ∈ l)
    (hmono : ∀ t ∈ l, ∀ s ⊆ t, s ∈ l) : Filter α := by
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l✝ : Filter α
    inst✝ : CardinalInterFilter l✝ c
    l : Set (Set α)
    hc : LT.lt 2 c
    hUnion : ∀ (S : Set (Set α)), LT.lt (Cardinal.mk ↑S) c → (∀ (s : Set α), Membe …
    hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
    ⊢ Filter α
  -/
  refine .ofCardinalInter {s | sᶜ ∈ l} hc (fun S hSc hSp ↦ ?_) fun s t ht hsub ↦ ?_
    /-
      case refine_1
      ι α β : Type u
      c : Cardinal.{u}
      l✝ : Filter α
      inst✝ : CardinalInterFilter l✝ c
      l : Set (Set α)
      hc : LT.lt 2 c
      hUnion : ∀ (S : Set (Set α)), LT.lt (Cardinal.mk ↑S) c → (∀ (s : Set α), Membe …
      hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
      S : Set (Set α)
      hSc : LT.lt (Cardinal.mk ↑S) c
      hSp : HasSubset.Subset S (setOf fun s => Membership.mem l (HasCompl.compl s))
      ⊢ Membership.mem (setOf fun s => Membership.mem l (HasCompl.compl s)) S.sInter
    -/
  · rw [mem_setOf_eq, compl_sInter]
    /-
      case refine_1
      ι α β : Type u
      c : Cardinal.{u}
      l✝ : Filter α
      inst✝ : CardinalInterFilter l✝ c
      l : Set (Set α)
      hc : LT.lt 2 c
      hUnion : ∀ (S : Set (Set α)), LT.lt (Cardinal.mk ↑S) c → (∀ (s : Set α), Membe …
      hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
      S : Set (Set α)
      hSc : LT.lt (Cardinal.mk ↑S) c
      hSp : HasSubset.Subset S (setOf fun s => Membership.mem l (HasCompl.compl s))
      ⊢ Membership.mem l (Set.image HasCompl.compl S).sUnion
    -/
    apply hUnion (compl '' S) (lt_of_le_of_lt mk_image_le hSc)
    /-
      case refine_1
      ι α β : Type u
      c : Cardinal.{u}
      l✝ : Filter α
      inst✝ : CardinalInterFilter l✝ c
      l : Set (Set α)
      hc : LT.lt 2 c
      hUnion : ∀ (S : Set (Set α)), LT.lt (Cardinal.mk ↑S) c → (∀ (s : Set α), Membe …
      hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
      S : Set (Set α)
      hSc : LT.lt (Cardinal.mk ↑S) c
      hSp : HasSubset.Subset S (setOf fun s => Membership.mem l (HasCompl.compl s))
      ⊢ ∀ (s : Set α), Membership.mem (Set.image HasCompl.compl S) s → Membership.me …
    -/
    intro s hs
    /-
      case refine_1
      ι α β : Type u
      c : Cardinal.{u}
      l✝ : Filter α
      inst✝ : CardinalInterFilter l✝ c
      l : Set (Set α)
      hc : LT.lt 2 c
      hUnion : ∀ (S : Set (Set α)), LT.lt (Cardinal.mk ↑S) c → (∀ (s : Set α), Membe …
      hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
      S : Set (Set α)
      hSc : LT.lt (Cardinal.mk ↑S) c
      hSp : HasSubset.Subset S (setOf fun s => Membership.mem l (HasCompl.compl s))
      s : Set α
      hs : Membership.mem (Set.image HasCompl.compl S) s
      ⊢ Membership.mem l s
    -/
    rw [mem_image] at hs
    /-
      case refine_1
      ι α β : Type u
      c : Cardinal.{u}
      l✝ : Filter α
      inst✝ : CardinalInterFilter l✝ c
      l : Set (Set α)
      hc : LT.lt 2 c
      hUnion : ∀ (S : Set (Set α)), LT.lt (Cardinal.mk ↑S) c → (∀ (s : Set α), Membe …
      hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
      S : Set (Set α)
      hSc : LT.lt (Cardinal.mk ↑S) c
      hSp : HasSubset.Subset S (setOf fun s => Membership.mem l (HasCompl.compl s))
      s : Set α
      hs : Exists fun x => And (Membership.mem S x) (Eq (HasCompl.compl x) s)
      ⊢ Membership.mem l s
    -/
    rcases hs with ⟨t, ht, rfl⟩
    /-
      case refine_1.intro.intro
      ι α β : Type u
      c : Cardinal.{u}
      l✝ : Filter α
      inst✝ : CardinalInterFilter l✝ c
      l : Set (Set α)
      hc : LT.lt 2 c
      hUnion : ∀ (S : Set (Set α)), LT.lt (Cardinal.mk ↑S) c → (∀ (s : Set α), Membe …
      hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
      S : Set (Set α)
      hSc : LT.lt (Cardinal.mk ↑S) c
      hSp : HasSubset.Subset S (setOf fun s => Membership.mem l (HasCompl.compl s))
      t : Set α
      ht : Membership.mem S t
      ⊢ Membership.mem l (HasCompl.compl t)
    -/
    apply hSp ht
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι α β : Type u
      c : Cardinal.{u}
      l✝ : Filter α
      inst✝ : CardinalInterFilter l✝ c
      l : Set (Set α)
      hc : LT.lt 2 c
      hUnion : ∀ (S : Set (Set α)), LT.lt (Cardinal.mk ↑S) c → (∀ (s : Set α), Membe …
      hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
      s t : Set α
      ht : Membership.mem (setOf fun s => Membership.mem l (HasCompl.compl s)) s
      hsub : HasSubset.Subset s t
      ⊢ Membership.mem (setOf fun s => Membership.mem l (HasCompl.compl s)) t
    -/
  · rw [mem_setOf_eq]
    /-
      case refine_2
      ι α β : Type u
      c : Cardinal.{u}
      l✝ : Filter α
      inst✝ : CardinalInterFilter l✝ c
      l : Set (Set α)
      hc : LT.lt 2 c
      hUnion : ∀ (S : Set (Set α)), LT.lt (Cardinal.mk ↑S) c → (∀ (s : Set α), Membe …
      hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
      s t : Set α
      ht : Membership.mem (setOf fun s => Membership.mem l (HasCompl.compl s)) s
      hsub : HasSubset.Subset s t
      ⊢ Membership.mem l (HasCompl.compl t)
    -/
    rw [← compl_subset_compl] at hsub
    /-
      case refine_2
      ι α β : Type u
      c : Cardinal.{u}
      l✝ : Filter α
      inst✝ : CardinalInterFilter l✝ c
      l : Set (Set α)
      hc : LT.lt 2 c
      hUnion : ∀ (S : Set (Set α)), LT.lt (Cardinal.mk ↑S) c → (∀ (s : Set α), Membe …
      hmono : ∀ (t : Set α), Membership.mem l t → ∀ (s : Set α), HasSubset.Subset s  …
      s t : Set α
      ht : Membership.mem (setOf fun s => Membership.mem l (HasCompl.compl s)) s
      hsub : HasSubset.Subset (HasCompl.compl t) (HasCompl.compl s)
      ⊢ Membership.mem l (HasCompl.compl t)
    -/
    exact hmono sᶜ ht tᶜ hsub
    /-
      🎉 no goals
    -/


instance cardinalInter_ofCardinalUnion (l : Set (Set α)) (hc : 2 < c) (h₁ h₂) :
    CardinalInterFilter (Filter.ofCardinalUnion l hc h₁ h₂) c :=
  cardinalInter_ofCardinalInter ..


@[simp]
theorem mem_ofCardinalUnion {l : Set (Set α)} (hc : 2 < c) {hunion hmono s} :
    s ∈ ofCardinalUnion l hc hunion hmono ↔ l sᶜ :=
  Iff.rfl


instance cardinalInterFilter_principal (s : Set α) : CardinalInterFilter (𝓟 s) c :=
  ⟨fun _ _ hS => subset_sInter hS⟩


instance cardinalInterFilter_bot : CardinalInterFilter (⊥ : Filter α) c := by
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝ : CardinalInterFilter l c
    ⊢ CardinalInterFilter Bot.bot c
  -/
  rw [← principal_empty]
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝ : CardinalInterFilter l c
    ⊢ CardinalInterFilter (Filter.principal EmptyCollection.emptyCollection) c
  -/
  apply cardinalInterFilter_principal
  /-
    🎉 no goals
  -/


instance cardinalInterFilter_top : CardinalInterFilter (⊤ : Filter α) c := by
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝ : CardinalInterFilter l c
    ⊢ CardinalInterFilter Top.top c
  -/
  rw [← principal_univ]
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝ : CardinalInterFilter l c
    ⊢ CardinalInterFilter (Filter.principal Set.univ) c
  -/
  apply cardinalInterFilter_principal
  /-
    🎉 no goals
  -/


instance (l : Filter β) [CardinalInterFilter l c] (f : α → β) :
    CardinalInterFilter (comap f l) c := by
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l✝ : Filter α
    inst✝¹ : CardinalInterFilter l✝ c
    l : Filter β
    inst✝ : CardinalInterFilter l c
    f : α → β
    ⊢ CardinalInterFilter (Filter.comap f l) c
  -/
  refine ⟨fun S hSc hS => ?_⟩
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l✝ : Filter α
    inst✝¹ : CardinalInterFilter l✝ c
    l : Filter β
    inst✝ : CardinalInterFilter l c
    f : α → β
    S : Set (Set α)
    hSc : LT.lt (Cardinal.mk ↑S) c
    hS : ∀ (s : Set α), Membership.mem S s → Membership.mem (Filter.comap f l) s
    ⊢ Membership.mem (Filter.comap f l) S.sInter
  -/
  choose! t htl ht using hS
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l✝ : Filter α
    inst✝¹ : CardinalInterFilter l✝ c
    l : Filter β
    inst✝ : CardinalInterFilter l c
    f : α → β
    S : Set (Set α)
    hSc : LT.lt (Cardinal.mk ↑S) c
    t : Set α → Set β
    htl : ∀ (s : Set α), Membership.mem S s → Membership.mem l (t s)
    ht : ∀ (s : Set α), Membership.mem S s → HasSubset.Subset (Set.preimage f (t s …
    ⊢ Membership.mem (Filter.comap f l) S.sInter
  -/
  refine ⟨_, (cardinal_bInter_mem hSc).2 htl, ?_⟩
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l✝ : Filter α
    inst✝¹ : CardinalInterFilter l✝ c
    l : Filter β
    inst✝ : CardinalInterFilter l c
    f : α → β
    S : Set (Set α)
    hSc : LT.lt (Cardinal.mk ↑S) c
    t : Set α → Set β
    htl : ∀ (s : Set α), Membership.mem S s → Membership.mem l (t s)
    ht : ∀ (s : Set α), Membership.mem S s → HasSubset.Subset (Set.preimage f (t s …
    ⊢ HasSubset.Subset (Set.preimage f (Set.iInter fun i => Set.iInter fun hi => t …
  -/
  simpa [preimage_iInter] using iInter₂_mono ht
  /-
    🎉 no goals
  -/


instance (l : Filter α) [CardinalInterFilter l c] (f : α → β) :
    CardinalInterFilter (map f l) c := by
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l✝ : Filter α
    inst✝¹ : CardinalInterFilter l✝ c
    l : Filter α
    inst✝ : CardinalInterFilter l c
    f : α → β
    ⊢ CardinalInterFilter (Filter.map f l) c
  -/
  refine ⟨fun S hSc hS => ?_⟩
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l✝ : Filter α
    inst✝¹ : CardinalInterFilter l✝ c
    l : Filter α
    inst✝ : CardinalInterFilter l c
    f : α → β
    S : Set (Set β)
    hSc : LT.lt (Cardinal.mk ↑S) c
    hS : ∀ (s : Set β), Membership.mem S s → Membership.mem (Filter.map f l) s
    ⊢ Membership.mem (Filter.map f l) S.sInter
  -/
  simp only [mem_map, sInter_eq_biInter, preimage_iInter₂] at hS ⊢
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l✝ : Filter α
    inst✝¹ : CardinalInterFilter l✝ c
    l : Filter α
    inst✝ : CardinalInterFilter l c
    f : α → β
    S : Set (Set β)
    hSc : LT.lt (Cardinal.mk ↑S) c
    hS : ∀ (s : Set β), Membership.mem S s → Membership.mem l (Set.preimage f s)
    ⊢ Membership.mem l (Set.iInter fun i => Set.iInter fun j => Set.preimage f i)
  -/
  exact (cardinal_bInter_mem hSc).2 hS
  /-
    🎉 no goals
  -/


/-- Infimum of two `CardinalInterFilter`s is a `CardinalInterFilter`. This is useful, e.g.,
to automatically get an instance for `residual α ⊓ 𝓟 s`. -/
instance cardinalInterFilter_inf_eq (l₁ l₂ : Filter α) [CardinalInterFilter l₁ c]
    [CardinalInterFilter l₂ c] : CardinalInterFilter (l₁ ⊓ l₂) c := by
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝² : CardinalInterFilter l c
    l₁ l₂ : Filter α
    inst✝¹ : CardinalInterFilter l₁ c
    inst✝ : CardinalInterFilter l₂ c
    ⊢ CardinalInterFilter (Min.min l₁ l₂) c
  -/
  refine ⟨fun S hSc hS => ?_⟩
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝² : CardinalInterFilter l c
    l₁ l₂ : Filter α
    inst✝¹ : CardinalInterFilter l₁ c
    inst✝ : CardinalInterFilter l₂ c
    S : Set (Set α)
    hSc : LT.lt (Cardinal.mk ↑S) c
    hS : ∀ (s : Set α), Membership.mem S s → Membership.mem (Min.min l₁ l₂) s
    ⊢ Membership.mem (Min.min l₁ l₂) S.sInter
  -/
  choose s hs t ht hst using hS
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝² : CardinalInterFilter l c
    l₁ l₂ : Filter α
    inst✝¹ : CardinalInterFilter l₁ c
    inst✝ : CardinalInterFilter l₂ c
    S : Set (Set α)
    hSc : LT.lt (Cardinal.mk ↑S) c
    s : (s : Set α) → Membership.mem S s → Set α
    hs : ∀ (s_1 : Set α) (a : Membership.mem S s_1), Membership.mem l₁ (s s_1 a)
    t : (s : Set α) → Membership.mem S s → Set α
    ht : ∀ (s : Set α) (a : Membership.mem S s), Membership.mem l₂ (t s a)
    hst : ∀ (s_1 : Set α) (a : Membership.mem S s_1), Eq s_1 (Inter.inter (s s_1 a …
    ⊢ Membership.mem (Min.min l₁ l₂) S.sInter
  -/
  replace hs : (⋂ i ∈ S, s i ‹_›) ∈ l₁ := (cardinal_bInter_mem hSc).2 hs
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝² : CardinalInterFilter l c
    l₁ l₂ : Filter α
    inst✝¹ : CardinalInterFilter l₁ c
    inst✝ : CardinalInterFilter l₂ c
    S : Set (Set α)
    hSc : LT.lt (Cardinal.mk ↑S) c
    s t : (s : Set α) → Membership.mem S s → Set α
    ht : ∀ (s : Set α) (a : Membership.mem S s), Membership.mem l₂ (t s a)
    hst : ∀ (s_1 : Set α) (a : Membership.mem S s_1), Eq s_1 (Inter.inter (s s_1 a …
    hs : Membership.mem l₁ (Set.iInter fun i => Set.iInter fun h => s i h)
    ⊢ Membership.mem (Min.min l₁ l₂) S.sInter
  -/
  replace ht : (⋂ i ∈ S, t i ‹_›) ∈ l₂ := (cardinal_bInter_mem hSc).2 ht
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝² : CardinalInterFilter l c
    l₁ l₂ : Filter α
    inst✝¹ : CardinalInterFilter l₁ c
    inst✝ : CardinalInterFilter l₂ c
    S : Set (Set α)
    hSc : LT.lt (Cardinal.mk ↑S) c
    s t : (s : Set α) → Membership.mem S s → Set α
    hst : ∀ (s_1 : Set α) (a : Membership.mem S s_1), Eq s_1 (Inter.inter (s s_1 a …
    hs : Membership.mem l₁ (Set.iInter fun i => Set.iInter fun h => s i h)
    ht : Membership.mem l₂ (Set.iInter fun i => Set.iInter fun h => t i h)
    ⊢ Membership.mem (Min.min l₁ l₂) S.sInter
  -/
  refine mem_of_superset (inter_mem_inf hs ht) (subset_sInter fun i hi => ?_)
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝² : CardinalInterFilter l c
    l₁ l₂ : Filter α
    inst✝¹ : CardinalInterFilter l₁ c
    inst✝ : CardinalInterFilter l₂ c
    S : Set (Set α)
    hSc : LT.lt (Cardinal.mk ↑S) c
    s t : (s : Set α) → Membership.mem S s → Set α
    hst : ∀ (s_1 : Set α) (a : Membership.mem S s_1), Eq s_1 (Inter.inter (s s_1 a …
    hs : Membership.mem l₁ (Set.iInter fun i => Set.iInter fun h => s i h)
    ht : Membership.mem l₂ (Set.iInter fun i => Set.iInter fun h => t i h)
    i : Set α
    hi : Membership.mem S i
    ⊢ HasSubset.Subset (Inter.inter (Set.iInter fun i => Set.iInter fun h => s i h …
  -/
  rw [hst i hi]
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝² : CardinalInterFilter l c
    l₁ l₂ : Filter α
    inst✝¹ : CardinalInterFilter l₁ c
    inst✝ : CardinalInterFilter l₂ c
    S : Set (Set α)
    hSc : LT.lt (Cardinal.mk ↑S) c
    s t : (s : Set α) → Membership.mem S s → Set α
    hst : ∀ (s_1 : Set α) (a : Membership.mem S s_1), Eq s_1 (Inter.inter (s s_1 a …
    hs : Membership.mem l₁ (Set.iInter fun i => Set.iInter fun h => s i h)
    ht : Membership.mem l₂ (Set.iInter fun i => Set.iInter fun h => t i h)
    i : Set α
    hi : Membership.mem S i
    ⊢ HasSubset.Subset (Inter.inter (Set.iInter fun i => Set.iInter fun h => s i h …
  -/
                               /-
                                 🎉 no goals
                               -/
  apply inter_subset_inter <;> exact iInter_subset_of_subset i (iInter_subset _ _)
                               /-
                                 🎉 no goals
                               -/


instance cardinalInterFilter_inf (l₁ l₂ : Filter α) {c₁ c₂ : Cardinal.{u}}
    [CardinalInterFilter l₁ c₁] [CardinalInterFilter l₂ c₂] : CardinalInterFilter (l₁ ⊓ l₂)
    (c₁ ⊓ c₂) := by
  have : CardinalInterFilter l₁ (c₁ ⊓ c₂) :=
    CardinalInterFilter.of_cardinalInterFilter_of_le l₁ inf_le_left
  have : CardinalInterFilter l₂ (c₁ ⊓ c₂) :=
    CardinalInterFilter.of_cardinalInterFilter_of_le l₂ inf_le_right
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝² : CardinalInterFilter l c
    l₁ l₂ : Filter α
    c₁ c₂ : Cardinal.{u}
    inst✝¹ : CardinalInterFilter l₁ c₁
    inst✝ : CardinalInterFilter l₂ c₂
    this✝ : CardinalInterFilter l₁ (Min.min c₁ c₂)
    this : CardinalInterFilter l₂ (Min.min c₁ c₂)
    ⊢ CardinalInterFilter (Min.min l₁ l₂) (Min.min c₁ c₂)
  -/
  exact cardinalInterFilter_inf_eq _ _
  /-
    🎉 no goals
  -/


/-- Supremum of two `CardinalInterFilter`s is a `CardinalInterFilter`. -/
instance cardinalInterFilter_sup_eq (l₁ l₂ : Filter α) [CardinalInterFilter l₁ c]
    [CardinalInterFilter l₂ c] : CardinalInterFilter (l₁ ⊔ l₂) c := by
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝² : CardinalInterFilter l c
    l₁ l₂ : Filter α
    inst✝¹ : CardinalInterFilter l₁ c
    inst✝ : CardinalInterFilter l₂ c
    ⊢ CardinalInterFilter (Max.max l₁ l₂) c
  -/
  refine ⟨fun S hSc hS => ⟨?_, ?_⟩⟩ <;> refine (cardinal_sInter_mem hSc).2 fun s hs => ?_
  /-
    case refine_1
    ι α β : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝² : CardinalInterFilter l c
    l₁ l₂ : Filter α
    inst✝¹ : CardinalInterFilter l₁ c
    inst✝ : CardinalInterFilter l₂ c
    S : Set (Set α)
    hSc : LT.lt (Cardinal.mk ↑S) c
    hS : ∀ (s : Set α), Membership.mem S s → Membership.mem (Max.max l₁ l₂) s
    s : Set α
    hs : Membership.mem S s
    ⊢ Membership.mem l₁ s
  -/
  exacts [(hS s hs).1, (hS s hs).2]
  /-
    🎉 no goals
  -/


instance cardinalInterFilter_sup (l₁ l₂ : Filter α) {c₁ c₂ : Cardinal.{u}}
    [CardinalInterFilter l₁ c₁] [CardinalInterFilter l₂ c₂] :
    CardinalInterFilter (l₁ ⊔ l₂) (c₁ ⊓ c₂) := by
  have : CardinalInterFilter l₁ (c₁ ⊓ c₂) :=
    CardinalInterFilter.of_cardinalInterFilter_of_le l₁ inf_le_left
  have : CardinalInterFilter l₂ (c₁ ⊓ c₂) :=
    CardinalInterFilter.of_cardinalInterFilter_of_le l₂ inf_le_right
  /-
    ι α β : Type u
    c : Cardinal.{u}
    l : Filter α
    inst✝² : CardinalInterFilter l c
    l₁ l₂ : Filter α
    c₁ c₂ : Cardinal.{u}
    inst✝¹ : CardinalInterFilter l₁ c₁
    inst✝ : CardinalInterFilter l₂ c₂
    this✝ : CardinalInterFilter l₁ (Min.min c₁ c₂)
    this : CardinalInterFilter l₂ (Min.min c₁ c₂)
    ⊢ CardinalInterFilter (Max.max l₁ l₂) (Min.min c₁ c₂)
  -/
  exact cardinalInterFilter_sup_eq _ _
  /-
    🎉 no goals
  -/


/-- `Filter.CardinalGenerateSets c g` is the (sets of the)
greatest `cardinalInterFilter c` containing `g`. -/
inductive CardinalGenerateSets : Set α → Prop
  | basic {s : Set α} : s ∈ g → CardinalGenerateSets s
  | univ : CardinalGenerateSets univ
  | superset {s t : Set α} : CardinalGenerateSets s → s ⊆ t → CardinalGenerateSets t
  | sInter {S : Set (Set α)} :
    (#S < c) → (∀ s ∈ S, CardinalGenerateSets s) → CardinalGenerateSets (⋂₀ S)


/-- `Filter.cardinalGenerate c g` is the greatest `cardinalInterFilter c` containing `g`. -/
def cardinalGenerate (hc : 2 < c) : Filter α :=
  ofCardinalInter (CardinalGenerateSets g) hc (fun _ => CardinalGenerateSets.sInter) fun _ _ =>
    CardinalGenerateSets.superset


lemma cardinalInter_ofCardinalGenerate (hc : 2 < c) :
    CardinalInterFilter (cardinalGenerate g hc) c := by
  /-
    α : Type u
    c : Cardinal.{u}
    g : Set (Set α)
    hc : LT.lt 2 c
    ⊢ CardinalInterFilter (Filter.cardinalGenerate g hc) c
  -/
  delta cardinalGenerate
  /-
    α : Type u
    c : Cardinal.{u}
    g : Set (Set α)
    hc : LT.lt 2 c
    ⊢ CardinalInterFilter (Filter.ofCardinalInter (Filter.CardinalGenerateSets g)  …
  -/
  apply cardinalInter_ofCardinalInter _ _ _
  /-
    🎉 no goals
  -/


/-- A set is in the `cardinalInterFilter` generated by `g` if and only if
it contains an intersection of `c` elements of `g`. -/
theorem mem_cardinaleGenerate_iff {s : Set α} {hreg : c.IsRegular} :
    s ∈ cardinalGenerate g (IsRegular.nat_lt hreg 2) ↔
    ∃ S : Set (Set α), S ⊆ g ∧ (#S < c) ∧ ⋂₀ S ⊆ s := by
  /-
    α : Type u
    c : Cardinal.{u}
    g : Set (Set α)
    s : Set α
    hreg : c.IsRegular
    ⊢ Iff (Membership.mem (Filter.cardinalGenerate g ⋯) s) (Exists fun S => And (H …
  -/
  constructor <;> intro h
  · induction h with
    | @basic s hs =>
      refine ⟨{s}, singleton_subset_iff.mpr hs, ?_⟩
      simpa [subset_refl] using IsRegular.nat_lt hreg 1
    | univ =>
      exact ⟨∅, ⟨empty_subset g, mk_eq_zero (∅ : Set <| Set α) ▸ IsRegular.nat_lt hreg 0, by simp⟩⟩
    | superset _ _ ih => exact Exists.imp (by tauto) ih
    | @sInter S Sct _ ih =>
      choose T Tg Tct hT using ih
      refine ⟨⋃ (s) (H : s ∈ S), T s H, by simpa,
        (Cardinal.card_biUnion_lt_iff_forall_of_isRegular hreg Sct).2 Tct, ?_⟩
      apply subset_sInter
      apply fun s H => subset_trans (sInter_subset_sInter (subset_iUnion₂ s H)) (hT s H)
  /-
    case mpr
    α : Type u
    c : Cardinal.{u}
    g : Set (Set α)
    s : Set α
    hreg : c.IsRegular
    h : Exists fun S => And (HasSubset.Subset S g) (And (LT.lt (Cardinal.mk ↑S) c) …
    ⊢ Membership.mem (Filter.cardinalGenerate g ⋯) s
  -/
  rcases h with ⟨S, Sg, Sct, hS⟩
  have : CardinalInterFilter (cardinalGenerate g (IsRegular.nat_lt hreg 2)) c :=
    cardinalInter_ofCardinalGenerate _ _
  exact mem_of_superset ((cardinal_sInter_mem Sct).mpr
    (fun s H => CardinalGenerateSets.basic (Sg H))) hS


theorem le_cardinalGenerate_iff_of_cardinalInterFilter {f : Filter α} [CardinalInterFilter f c]
    (hc : 2 < c) : f ≤ cardinalGenerate g hc ↔ g ⊆ f.sets := by
  /-
    α : Type u
    c : Cardinal.{u}
    g : Set (Set α)
    f : Filter α
    inst✝ : CardinalInterFilter f c
    hc : LT.lt 2 c
    ⊢ Iff (LE.le f (Filter.cardinalGenerate g hc)) (HasSubset.Subset g f.sets)
  -/
  constructor <;> intro h
    /-
      case mp
      α : Type u
      c : Cardinal.{u}
      g : Set (Set α)
      f : Filter α
      inst✝ : CardinalInterFilter f c
      hc : LT.lt 2 c
      h : LE.le f (Filter.cardinalGenerate g hc)
      ⊢ HasSubset.Subset g f.sets
    -/
  · exact subset_trans (fun s => CardinalGenerateSets.basic) h
    /-
      🎉 no goals
    -/
  /-
    case mpr
    α : Type u
    c : Cardinal.{u}
    g : Set (Set α)
    f : Filter α
    inst✝ : CardinalInterFilter f c
    hc : LT.lt 2 c
    h : HasSubset.Subset g f.sets
    ⊢ LE.le f (Filter.cardinalGenerate g hc)
  -/
  intro s hs
  induction hs with
  | basic hs => exact h hs
  | univ => exact univ_mem
  | superset _ st ih => exact mem_of_superset ih st
  | sInter Sct _ ih => exact (cardinal_sInter_mem Sct).mpr ih


/-- `cardinalGenerate g hc` is the greatest `cardinalInterFilter c` containing `g`. -/
theorem cardinalGenerate_isGreatest (hc : 2 < c) :
    IsGreatest { f : Filter α | CardinalInterFilter f c ∧ g ⊆ f.sets } (cardinalGenerate g hc) := by
  /-
    α : Type u
    c : Cardinal.{u}
    g : Set (Set α)
    hc : LT.lt 2 c
    ⊢ IsGreatest (setOf fun f => And (CardinalInterFilter f c) (HasSubset.Subset g …
  -/
  refine ⟨⟨cardinalInter_ofCardinalGenerate _ _, fun s => CardinalGenerateSets.basic⟩, ?_⟩
  /-
    α : Type u
    c : Cardinal.{u}
    g : Set (Set α)
    hc : LT.lt 2 c
    ⊢ Membership.mem (upperBounds (setOf fun f => And (CardinalInterFilter f c) (H …
  -/
  rintro f ⟨fct, hf⟩
  /-
    case intro
    α : Type u
    c : Cardinal.{u}
    g : Set (Set α)
    hc : LT.lt 2 c
    f : Filter α
    fct : CardinalInterFilter f c
    hf : HasSubset.Subset g f.sets
    ⊢ LE.le f (Filter.cardinalGenerate g hc)
  -/
  rwa [le_cardinalGenerate_iff_of_cardinalInterFilter]
  /-
    🎉 no goals
  -/


