/-- The cofinite filter is the filter of subsets whose complements are finite. -/
def cofinite : Filter α :=
  comk Set.Finite finite_empty (fun _t ht _s hsub ↦ ht.subset hsub) fun _ h _ ↦ h.union


@[simp]
theorem mem_cofinite {s : Set α} : s ∈ @cofinite α ↔ sᶜ.Finite :=
  Iff.rfl


@[simp]
theorem eventually_cofinite {p : α → Prop} : (∀ᶠ x in cofinite, p x) ↔ { x | ¬p x }.Finite :=
  Iff.rfl


theorem hasBasis_cofinite : HasBasis cofinite (fun s : Set α => s.Finite) compl :=
  ⟨fun s =>
    ⟨fun h => ⟨sᶜ, h, (compl_compl s).subset⟩, fun ⟨_t, htf, hts⟩ =>
      htf.subset <| compl_subset_comm.2 hts⟩⟩


instance cofinite_neBot [Infinite α] : NeBot (@cofinite α) :=
  hasBasis_cofinite.neBot_iff.2 fun hs => hs.infinite_compl.nonempty


@[simp]
theorem cofinite_eq_bot_iff : @cofinite α = ⊥ ↔ Finite α := by
  /-
    α : Type u_2
    ⊢ Iff (Eq Filter.cofinite Bot.bot) (Finite α)
  -/
  simp [← empty_mem_iff_bot, finite_univ_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem cofinite_eq_bot [Finite α] : @cofinite α = ⊥ := cofinite_eq_bot_iff.2 ‹_›


theorem frequently_cofinite_iff_infinite {p : α → Prop} :
    (∃ᶠ x in cofinite, p x) ↔ Set.Infinite { x | p x } := by
  /-
    α : Type u_2
    p : α → Prop
    ⊢ Iff (Filter.Frequently (fun x => p x) Filter.cofinite) (setOf fun x => p x). …
  -/
  simp only [Filter.Frequently, eventually_cofinite, not_not, Set.Infinite]
  /-
    🎉 no goals
  -/


lemma frequently_cofinite_mem_iff_infinite {s : Set α} : (∃ᶠ x in cofinite, x ∈ s) ↔ s.Infinite :=
  frequently_cofinite_iff_infinite


alias ⟨_, _root_.Set.Infinite.frequently_cofinite⟩ := frequently_cofinite_mem_iff_infinite


@[simp]
lemma cofinite_inf_principal_neBot_iff {s : Set α} : (cofinite ⊓ 𝓟 s).NeBot ↔ s.Infinite :=
  frequently_mem_iff_neBot.symm.trans frequently_cofinite_mem_iff_infinite


alias ⟨_, _root_.Set.Infinite.cofinite_inf_principal_neBot⟩ := cofinite_inf_principal_neBot_iff


theorem _root_.Set.Finite.compl_mem_cofinite {s : Set α} (hs : s.Finite) : sᶜ ∈ @cofinite α :=
  mem_cofinite.2 <| (compl_compl s).symm ▸ hs


theorem _root_.Set.Finite.eventually_cofinite_nmem {s : Set α} (hs : s.Finite) :
    ∀ᶠ x in cofinite, x ∉ s :=
  hs.compl_mem_cofinite


theorem _root_.Finset.eventually_cofinite_nmem (s : Finset α) : ∀ᶠ x in cofinite, x ∉ s :=
  s.finite_toSet.eventually_cofinite_nmem


theorem _root_.Set.infinite_iff_frequently_cofinite {s : Set α} :
    Set.Infinite s ↔ ∃ᶠ x in cofinite, x ∈ s :=
  frequently_cofinite_iff_infinite.symm


theorem eventually_cofinite_ne (x : α) : ∀ᶠ a in cofinite, a ≠ x :=
  (Set.finite_singleton x).eventually_cofinite_nmem


theorem le_cofinite_iff_compl_singleton_mem : l ≤ cofinite ↔ ∀ x, {x}ᶜ ∈ l := by
  /-
    α : Type u_2
    l : Filter α
    ⊢ Iff (LE.le l Filter.cofinite) (∀ (x : α), Membership.mem l (HasCompl.compl ( …
  -/
  refine ⟨fun h x => h (finite_singleton x).compl_mem_cofinite, fun h s (hs : sᶜ.Finite) => ?_⟩
  /-
    α : Type u_2
    l : Filter α
    h : ∀ (x : α), Membership.mem l (HasCompl.compl (Singleton.singleton x))
    s : Set α
    hs : (HasCompl.compl s).Finite
    ⊢ Membership.mem l s
  -/
  rw [← compl_compl s, ← biUnion_of_singleton sᶜ, compl_iUnion₂, Filter.biInter_mem hs]
  /-
    α : Type u_2
    l : Filter α
    h : ∀ (x : α), Membership.mem l (HasCompl.compl (Singleton.singleton x))
    s : Set α
    hs : (HasCompl.compl s).Finite
    ⊢ ∀ (i : α), Membership.mem (HasCompl.compl s) i → Membership.mem l (HasCompl. …
  -/
  exact fun x _ => h x
  /-
    🎉 no goals
  -/


theorem le_cofinite_iff_eventually_ne : l ≤ cofinite ↔ ∀ x, ∀ᶠ y in l, y ≠ x :=
  le_cofinite_iff_compl_singleton_mem


/-- If `α` is a preorder with no maximal element, then `atTop ≤ cofinite`. -/
theorem atTop_le_cofinite [Preorder α] [NoMaxOrder α] : (atTop : Filter α) ≤ cofinite :=
  le_cofinite_iff_eventually_ne.mpr eventually_ne_atTop


theorem comap_cofinite_le (f : α → β) : comap f cofinite ≤ cofinite :=
  le_cofinite_iff_eventually_ne.mpr fun x =>
    mem_comap.2 ⟨{f x}ᶜ, (finite_singleton _).compl_mem_cofinite, fun _ => ne_of_apply_ne f⟩


/-- The coproduct of the cofinite filters on two types is the cofinite filter on their product. -/
theorem coprod_cofinite : (cofinite : Filter α).coprod (cofinite : Filter β) = cofinite :=
  Filter.coext fun s => by
    /-
      α : Type u_2
      β : Type u_3
      s : Set (Prod α β)
      ⊢ Iff (Membership.mem (Filter.cofinite.coprod Filter.cofinite) (HasCompl.compl …
    -/
    simp only [compl_mem_coprod, mem_cofinite, compl_compl, finite_image_fst_and_snd_iff]
    /-
      🎉 no goals
    -/


theorem coprodᵢ_cofinite {α : ι → Type*} [Finite ι] :
    (Filter.coprodᵢ fun i => (cofinite : Filter (α i))) = cofinite :=
  Filter.coext fun s => by
    /-
      ι : Type u_1
      α : ι → Type u_4
      inst✝ : Finite ι
      s : Set ((i : ι) → α i)
      ⊢ Iff (Membership.mem (Filter.coprodᵢ fun i => Filter.cofinite) (HasCompl.comp …
    -/
    simp only [compl_mem_coprodᵢ, mem_cofinite, compl_compl, forall_finite_image_eval_iff]
    /-
      🎉 no goals
    -/


theorem disjoint_cofinite_left : Disjoint cofinite l ↔ ∃ s ∈ l, Set.Finite s := by
  /-
    α : Type u_2
    l : Filter α
    ⊢ Iff (Disjoint Filter.cofinite l) (Exists fun s => And (Membership.mem l s) s …
  -/
  simp [l.basis_sets.disjoint_iff_right]
  /-
    🎉 no goals
  -/


theorem disjoint_cofinite_right : Disjoint l cofinite ↔ ∃ s ∈ l, Set.Finite s :=
  disjoint_comm.trans disjoint_cofinite_left


/-- If `l ≥ Filter.cofinite` is a countably generated filter, then `l.ker` is cocountable. -/
theorem countable_compl_ker [l.IsCountablyGenerated] (h : cofinite ≤ l) : Set.Countable l.kerᶜ := by
  /-
    α : Type u_2
    l : Filter α
    inst✝ : l.IsCountablyGenerated
    h : LE.le Filter.cofinite l
    ⊢ (HasCompl.compl l.ker).Countable
  -/
  rcases exists_antitone_basis l with ⟨s, hs⟩
  /-
    case intro
    α : Type u_2
    l : Filter α
    inst✝ : l.IsCountablyGenerated
    h : LE.le Filter.cofinite l
    s : Nat → Set α
    hs : l.HasAntitoneBasis s
    ⊢ (HasCompl.compl l.ker).Countable
  -/
  simp only [hs.ker, iInter_true, compl_iInter]
  /-
    case intro
    α : Type u_2
    l : Filter α
    inst✝ : l.IsCountablyGenerated
    h : LE.le Filter.cofinite l
    s : Nat → Set α
    hs : l.HasAntitoneBasis s
    ⊢ (Set.iUnion fun i => HasCompl.compl (s i)).Countable
  -/
  exact countable_iUnion fun n ↦ Set.Finite.countable <| h <| hs.mem _
  /-
    🎉 no goals
  -/


/-- If `f` tends to a countably generated filter `l` along `Filter.cofinite`,
then for all but countably many elements, `f x ∈ l.ker`. -/
theorem Tendsto.countable_compl_preimage_ker {f : α → β}
    {l : Filter β} [l.IsCountablyGenerated] (h : Tendsto f cofinite l) :
                                       /-
                                         α : Type u_2
                                         β : Type u_3
                                         f : α → β
                                         l : Filter β
                                         inst✝ : l.IsCountablyGenerated
                                         h : Filter.Tendsto f Filter.cofinite l
                                         ⊢ (HasCompl.compl (Set.preimage f l.ker)).Countable
                                       -/
    Set.Countable (f ⁻¹' l.ker)ᶜ := by rw [← ker_comap]; exact countable_compl_ker h.le_comap
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- Given a collection of filters `l i : Filter (α i)` and sets `s i ∈ l i`,
if all but finitely many of `s i` are the whole space,
then their indexed product `Set.pi Set.univ s` belongs to the filter `Filter.pi l`. -/
theorem univ_pi_mem_pi {α : ι → Type*} {s : ∀ i, Set (α i)} {l : ∀ i, Filter (α i)}
    (h : ∀ i, s i ∈ l i) (hfin : ∀ᶠ i in cofinite, s i = univ) : univ.pi s ∈ pi l := by
  /-
    ι : Type u_1
    α : ι → Type u_4
    s : (i : ι) → Set (α i)
    l : (i : ι) → Filter (α i)
    h : ∀ (i : ι), Membership.mem (l i) (s i)
    hfin : Filter.Eventually (fun i => Eq (s i) Set.univ) Filter.cofinite
    ⊢ Membership.mem (Filter.pi l) (Set.univ.pi s)
  -/
  filter_upwards [pi_mem_pi hfin fun i _ ↦ h i] with a ha i _
  if hi : s i = univ then
    simp [hi]
  else
    exact ha i hi


/-- Given a family of maps `f i : α i → β i` and a family of filters `l i : Filter (α i)`,
if all but finitely many of `f i` are surjective,
then the indexed product of `f i`s maps the indexed product of the filters `l i`
to the indexed products of their pushforwards under individual `f i`s.

See also `map_piMap_pi_finite` for the case of a finite index type.
-/
theorem map_piMap_pi {α β : ι → Type*} {f : ∀ i, α i → β i}
    (hf : ∀ᶠ i in cofinite, Surjective (f i)) (l : ∀ i, Filter (α i)) :
    map (Pi.map f) (pi l) = pi fun i ↦ map (f i) (l i) := by
  /-
    ι : Type u_1
    α : ι → Type u_4
    β : ι → Type u_5
    f : (i : ι) → α i → β i
    hf : Filter.Eventually (fun i => Function.Surjective (f i)) Filter.cofinite
    l : (i : ι) → Filter (α i)
    ⊢ Eq (Filter.map (Pi.map f) (Filter.pi l)) (Filter.pi fun i => Filter.map (f i …
  -/
  refine le_antisymm (tendsto_piMap_pi fun _ ↦ tendsto_map) ?_
  /-
    ι : Type u_1
    α : ι → Type u_4
    β : ι → Type u_5
    f : (i : ι) → α i → β i
    hf : Filter.Eventually (fun i => Function.Surjective (f i)) Filter.cofinite
    l : (i : ι) → Filter (α i)
    ⊢ LE.le (Filter.pi fun i => Filter.map (f i) (l i)) (Filter.map (Pi.map f) (Fi …
  -/
  refine ((hasBasis_pi fun i ↦ (l i).basis_sets).map _).ge_iff.2 ?_
  /-
    ι : Type u_1
    α : ι → Type u_4
    β : ι → Type u_5
    f : (i : ι) → α i → β i
    hf : Filter.Eventually (fun i => Function.Surjective (f i)) Filter.cofinite
    l : (i : ι) → Filter (α i)
    ⊢ ∀ (i' : Prod (Set ι) ((i : ι) → Set (α i))), And i'.1.Finite (∀ (i : ι), Mem …
  -/
  rintro ⟨I, s⟩ ⟨hI : I.Finite, hs : ∀ i ∈ I, s i ∈ l i⟩
  classical
  rw [← univ_pi_piecewise_univ, piMap_image_univ_pi]
  refine univ_pi_mem_pi (fun i ↦ ?_) ?_
  · by_cases hi : i ∈ I
    · simpa [hi] using image_mem_map (hs i hi)
    · simp [hi]
  · filter_upwards [hf, hI.compl_mem_cofinite] with i hsurj (hiI : i ∉ I)
    simp [hiI, hsurj.range_eq]


/-- Given finite families of maps `f i : α i → β i` and of filters `l i : Filter (α i)`,
the indexed product of `f i`s maps the indexed product of the filters `l i`
to the indexed products of their pushforwards under individual `f i`s.

See also `map_piMap_pi` for a more general case.
-/
theorem map_piMap_pi_finite {α β : ι → Type*} [Finite ι]
    (f : ∀ i, α i → β i) (l : ∀ i, Filter (α i)) :
    map (Pi.map f) (pi l) = pi fun i ↦ map (f i) (l i) :=
                   /-
                     ι : Type u_1
                     α : ι → Type u_4
                     β : ι → Type u_5
                     inst✝ : Finite ι
                     f : (i : ι) → α i → β i
                     l : (i : ι) → Filter (α i)
                     ⊢ Filter.Eventually (fun i => Function.Surjective (f i)) Filter.cofinite
                   -/
  map_piMap_pi (by simp) l
                   /-
                     🎉 no goals
                   -/


lemma Set.Finite.cofinite_inf_principal_compl {s : Set α} (hs : s.Finite) :
    cofinite ⊓ 𝓟 sᶜ = cofinite := by
  /-
    α : Type u_2
    s : Set α
    hs : s.Finite
    ⊢ Eq (Min.min Filter.cofinite (Filter.principal (HasCompl.compl s))) Filter.co …
  -/
  simpa using hs.compl_mem_cofinite
  /-
    🎉 no goals
  -/


lemma Set.Finite.cofinite_inf_principal_diff {s t : Set α} (ht : t.Finite) :
    cofinite ⊓ 𝓟 (s \ t) = cofinite ⊓ 𝓟 s := by
  /-
    α : Type u_2
    s t : Set α
    ht : t.Finite
    ⊢ Eq (Min.min Filter.cofinite (Filter.principal (SDiff.sdiff s t))) (Min.min F …
  -/
  rw [diff_eq, ← inf_principal, ← inf_assoc, inf_right_comm, ht.cofinite_inf_principal_compl]
  /-
    🎉 no goals
  -/


/-- For natural numbers the filters `Filter.cofinite` and `Filter.atTop` coincide. -/
theorem Nat.cofinite_eq_atTop : @cofinite ℕ = atTop := by
  /-
    ⊢ Eq Filter.cofinite Filter.atTop
  -/
  refine le_antisymm ?_ atTop_le_cofinite
  /-
    ⊢ LE.le Filter.cofinite Filter.atTop
  -/
  refine atTop_basis.ge_iff.2 fun N _ => ?_
  /-
    N : Nat
    x✝ : True
    ⊢ Membership.mem Filter.cofinite (Set.Ici N)
  -/
  simpa only [mem_cofinite, compl_Ici] using finite_lt_nat N
  /-
    🎉 no goals
  -/


theorem Nat.frequently_atTop_iff_infinite {p : ℕ → Prop} :
    (∃ᶠ n in atTop, p n) ↔ Set.Infinite { n | p n } := by
  /-
    p : Nat → Prop
    ⊢ Iff (Filter.Frequently (fun n => p n) Filter.atTop) (setOf fun n => p n).Inf …
  -/
  rw [← Nat.cofinite_eq_atTop, frequently_cofinite_iff_infinite]
  /-
    🎉 no goals
  -/


lemma Nat.eventually_pos : ∀ᶠ (k : ℕ) in Filter.atTop, 0 < k :=
  Filter.eventually_of_mem (Filter.mem_atTop_sets.mpr ⟨1, fun _ hx ↦ hx⟩) (fun _ hx ↦ hx)


theorem Filter.Tendsto.exists_within_forall_le {α β : Type*} [LinearOrder β] {s : Set α}
    (hs : s.Nonempty) {f : α → β} (hf : Filter.Tendsto f Filter.cofinite Filter.atTop) :
    ∃ a₀ ∈ s, ∀ a ∈ s, f a₀ ≤ f a := by
  /-
    α : Type u_4
    β : Type u_5
    inst✝ : LinearOrder β
    s : Set α
    hs : s.Nonempty
    f : α → β
    hf : Filter.Tendsto f Filter.cofinite Filter.atTop
    ⊢ Exists fun a₀ => And (Membership.mem s a₀) (∀ (a : α), Membership.mem s a →  …
  -/
  rcases em (∃ y ∈ s, ∃ x, f y < x) with (⟨y, hys, x, hx⟩ | not_all_top)
  · -- the set of points `{y | f y < x}` is nonempty and finite, so we take `min` over this set
    /-
      case inl.intro.intro.intro
      α : Type u_4
      β : Type u_5
      inst✝ : LinearOrder β
      s : Set α
      hs : s.Nonempty
      f : α → β
      hf : Filter.Tendsto f Filter.cofinite Filter.atTop
      y : α
      hys : Membership.mem s y
      x : β
      hx : LT.lt (f y) x
      ⊢ Exists fun a₀ => And (Membership.mem s a₀) (∀ (a : α), Membership.mem s a →  …
    -/
    have : { y | ¬x ≤ f y }.Finite := Filter.eventually_cofinite.mp (tendsto_atTop.1 hf x)
    /-
      case inl.intro.intro.intro
      α : Type u_4
      β : Type u_5
      inst✝ : LinearOrder β
      s : Set α
      hs : s.Nonempty
      f : α → β
      hf : Filter.Tendsto f Filter.cofinite Filter.atTop
      y : α
      hys : Membership.mem s y
      x : β
      hx : LT.lt (f y) x
      this : (setOf fun y => Not (LE.le x (f y))).Finite
      ⊢ Exists fun a₀ => And (Membership.mem s a₀) (∀ (a : α), Membership.mem s a →  …
    -/
    simp only [not_le] at this
    obtain ⟨a₀, ⟨ha₀ : f a₀ < x, ha₀s⟩, others_bigger⟩ :=
      exists_min_image _ f (this.inter_of_left s) ⟨y, hx, hys⟩
    /-
      case inl.intro.intro.intro.intro.intro.intro
      α : Type u_4
      β : Type u_5
      inst✝ : LinearOrder β
      s : Set α
      hs : s.Nonempty
      f : α → β
      hf : Filter.Tendsto f Filter.cofinite Filter.atTop
      y : α
      hys : Membership.mem s y
      x : β
      hx : LT.lt (f y) x
      this : (setOf fun y => LT.lt (f y) x).Finite
      a₀ : α
      others_bigger : ∀ (b : α), Membership.mem (Inter.inter (setOf fun y => LT.lt ( …
      ha₀ : LT.lt (f a₀) x
      ha₀s : Membership.mem s a₀
      ⊢ Exists fun a₀ => And (Membership.mem s a₀) (∀ (a : α), Membership.mem s a →  …
    -/
    refine ⟨a₀, ha₀s, fun a has => (lt_or_le (f a) x).elim ?_ (le_trans ha₀.le)⟩
    /-
      case inl.intro.intro.intro.intro.intro.intro
      α : Type u_4
      β : Type u_5
      inst✝ : LinearOrder β
      s : Set α
      hs : s.Nonempty
      f : α → β
      hf : Filter.Tendsto f Filter.cofinite Filter.atTop
      y : α
      hys : Membership.mem s y
      x : β
      hx : LT.lt (f y) x
      this : (setOf fun y => LT.lt (f y) x).Finite
      a₀ : α
      others_bigger : ∀ (b : α), Membership.mem (Inter.inter (setOf fun y => LT.lt ( …
      ha₀ : LT.lt (f a₀) x
      ha₀s : Membership.mem s a₀
      a : α
      has : Membership.mem s a
      ⊢ LT.lt (f a) x → LE.le (f a₀) (f a)
    -/
    exact fun h => others_bigger a ⟨h, has⟩
    /-
      🎉 no goals
    -/
  · -- in this case, f is constant because all values are at top
    /-
      case inr
      α : Type u_4
      β : Type u_5
      inst✝ : LinearOrder β
      s : Set α
      hs : s.Nonempty
      f : α → β
      hf : Filter.Tendsto f Filter.cofinite Filter.atTop
      not_all_top : Not (Exists fun y => And (Membership.mem s y) (Exists fun x => L …
      ⊢ Exists fun a₀ => And (Membership.mem s a₀) (∀ (a : α), Membership.mem s a →  …
    -/
    push_neg at not_all_top
    /-
      case inr
      α : Type u_4
      β : Type u_5
      inst✝ : LinearOrder β
      s : Set α
      hs : s.Nonempty
      f : α → β
      hf : Filter.Tendsto f Filter.cofinite Filter.atTop
      not_all_top : ∀ (y : α), Membership.mem s y → ∀ (x : β), LE.le x (f y)
      ⊢ Exists fun a₀ => And (Membership.mem s a₀) (∀ (a : α), Membership.mem s a →  …
    -/
    obtain ⟨a₀, ha₀s⟩ := hs
    /-
      case inr.intro
      α : Type u_4
      β : Type u_5
      inst✝ : LinearOrder β
      s : Set α
      f : α → β
      hf : Filter.Tendsto f Filter.cofinite Filter.atTop
      not_all_top : ∀ (y : α), Membership.mem s y → ∀ (x : β), LE.le x (f y)
      a₀ : α
      ha₀s : Membership.mem s a₀
      ⊢ Exists fun a₀ => And (Membership.mem s a₀) (∀ (a : α), Membership.mem s a →  …
    -/
    exact ⟨a₀, ha₀s, fun a ha => not_all_top a ha (f a₀)⟩
    /-
      🎉 no goals
    -/


theorem Filter.Tendsto.exists_forall_le [Nonempty α] [LinearOrder β] {f : α → β}
    (hf : Tendsto f cofinite atTop) : ∃ a₀, ∀ a, f a₀ ≤ f a :=
  let ⟨a₀, _, ha₀⟩ := hf.exists_within_forall_le univ_nonempty
  ⟨a₀, fun a => ha₀ a (mem_univ _)⟩


theorem Filter.Tendsto.exists_within_forall_ge [LinearOrder β] {s : Set α} (hs : s.Nonempty)
    {f : α → β} (hf : Filter.Tendsto f Filter.cofinite Filter.atBot) :
    ∃ a₀ ∈ s, ∀ a ∈ s, f a ≤ f a₀ :=
  @Filter.Tendsto.exists_within_forall_le _ βᵒᵈ _ _ hs _ hf


theorem Filter.Tendsto.exists_forall_ge [Nonempty α] [LinearOrder β] {f : α → β}
    (hf : Tendsto f cofinite atBot) : ∃ a₀, ∀ a, f a ≤ f a₀ :=
  @Filter.Tendsto.exists_forall_le _ βᵒᵈ _ _ _ hf


theorem Function.Surjective.le_map_cofinite {f : α → β} (hf : Surjective f) :
    cofinite ≤ map f cofinite := fun _ h => .of_preimage h hf


/-- For an injective function `f`, inverse images of finite sets are finite. See also
`Filter.comap_cofinite_le` and `Function.Injective.comap_cofinite_eq`. -/
theorem Function.Injective.tendsto_cofinite {f : α → β} (hf : Injective f) :
    Tendsto f cofinite cofinite := fun _ h => h.preimage hf.injOn


/-- The pullback of the `Filter.cofinite` under an injective function is equal to `Filter.cofinite`.
See also `Filter.comap_cofinite_le` and `Function.Injective.tendsto_cofinite`. -/
theorem Function.Injective.comap_cofinite_eq {f : α → β} (hf : Injective f) :
    comap f cofinite = cofinite :=
  (comap_cofinite_le f).antisymm hf.tendsto_cofinite.le_comap


/-- An injective sequence `f : ℕ → ℕ` tends to infinity at infinity. -/
theorem Function.Injective.nat_tendsto_atTop {f : ℕ → ℕ} (hf : Injective f) :
    Tendsto f atTop atTop :=
  Nat.cofinite_eq_atTop ▸ hf.tendsto_cofinite

