/-- If a set `t` is eventually covered by a countable family of sets, all with cardinality at
most `a`, then the cardinality of `t` is also bounded by `a`.
Superseded by `mk_le_of_countable_eventually_mem` which does not assume
that the indexing set lives in the same universe. -/
lemma mk_subtype_le_of_countable_eventually_mem_aux {α ι : Type u} {a : Cardinal}
    [Countable ι] {f : ι → Set α} {l : Filter ι} [NeBot l]
    {t : Set α} (ht : ∀ x ∈ t, ∀ᶠ i in l, x ∈ f i)
    (h'f : ∀ i, #(f i) ≤ a) : #t ≤ a := by
  /-
    α ι : Type u
    a : Cardinal.{u}
    inst✝¹ : Countable ι
    f : ι → Set α
    l : Filter ι
    inst✝ : l.NeBot
    t : Set α
    ht : ∀ (x : α), Membership.mem t x → Filter.Eventually (fun i => Membership.me …
    h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) a
    ⊢ LE.le (Cardinal.mk ↑t) a
  -/
  rcases lt_or_le a ℵ₀ with ha|ha
  /- case `a` finite. In this case, it suffices to show that any finite subset `s` of `t` has
  cardinality at most `a`. For this, we pick `i` such that `f i` contains all the points in `s`,
  and apply the assumption that the cardinality of `f i` is at most `a`.   -/
    /-
      case inl
      α ι : Type u
      a : Cardinal.{u}
      inst✝¹ : Countable ι
      f : ι → Set α
      l : Filter ι
      inst✝ : l.NeBot
      t : Set α
      ht : ∀ (x : α), Membership.mem t x → Filter.Eventually (fun i => Membership.me …
      h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) a
      ha : LT.lt a Cardinal.aleph0
      ⊢ LE.le (Cardinal.mk ↑t) a
    -/
  · obtain ⟨n, rfl⟩ : ∃ (n : ℕ), a = n := lt_aleph0.1 ha
    /-
      case inl.intro
      α ι : Type u
      inst✝¹ : Countable ι
      f : ι → Set α
      l : Filter ι
      inst✝ : l.NeBot
      t : Set α
      ht : ∀ (x : α), Membership.mem t x → Filter.Eventually (fun i => Membership.me …
      n : Nat
      h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) ↑n
      ha : LT.lt (↑n) Cardinal.aleph0
      ⊢ LE.le (Cardinal.mk ↑t) ↑n
    -/
    apply mk_le_iff_forall_finset_subset_card_le.2 (fun s hs ↦ ?_)
    /-
      α ι : Type u
      inst✝¹ : Countable ι
      f : ι → Set α
      l : Filter ι
      inst✝ : l.NeBot
      t : Set α
      ht : ∀ (x : α), Membership.mem t x → Filter.Eventually (fun i => Membership.me …
      n : Nat
      h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) ↑n
      ha : LT.lt (↑n) Cardinal.aleph0
      s : Finset α
      hs : HasSubset.Subset (↑s) t
      ⊢ LE.le s.card n
    -/
    have A : ∀ x ∈ s, ∀ᶠ i in l, x ∈ f i := fun x hx ↦ ht x (hs hx)
    /-
      α ι : Type u
      inst✝¹ : Countable ι
      f : ι → Set α
      l : Filter ι
      inst✝ : l.NeBot
      t : Set α
      ht : ∀ (x : α), Membership.mem t x → Filter.Eventually (fun i => Membership.me …
      n : Nat
      h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) ↑n
      ha : LT.lt (↑n) Cardinal.aleph0
      s : Finset α
      hs : HasSubset.Subset (↑s) t
      A : ∀ (x : α), Membership.mem s x → Filter.Eventually (fun i => Membership.mem …
      ⊢ LE.le s.card n
    -/
    have B : ∀ᶠ i in l, ∀ x ∈ s, x ∈ f i := (s.eventually_all).2 A
    /-
      α ι : Type u
      inst✝¹ : Countable ι
      f : ι → Set α
      l : Filter ι
      inst✝ : l.NeBot
      t : Set α
      ht : ∀ (x : α), Membership.mem t x → Filter.Eventually (fun i => Membership.me …
      n : Nat
      h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) ↑n
      ha : LT.lt (↑n) Cardinal.aleph0
      s : Finset α
      hs : HasSubset.Subset (↑s) t
      A : ∀ (x : α), Membership.mem s x → Filter.Eventually (fun i => Membership.mem …
      B : Filter.Eventually (fun i => ∀ (x : α), Membership.mem s x → Membership.mem …
      ⊢ LE.le s.card n
    -/
    rcases B.exists with ⟨i, hi⟩
    /-
      case intro
      α ι : Type u
      inst✝¹ : Countable ι
      f : ι → Set α
      l : Filter ι
      inst✝ : l.NeBot
      t : Set α
      ht : ∀ (x : α), Membership.mem t x → Filter.Eventually (fun i => Membership.me …
      n : Nat
      h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) ↑n
      ha : LT.lt (↑n) Cardinal.aleph0
      s : Finset α
      hs : HasSubset.Subset (↑s) t
      A : ∀ (x : α), Membership.mem s x → Filter.Eventually (fun i => Membership.mem …
      B : Filter.Eventually (fun i => ∀ (x : α), Membership.mem s x → Membership.mem …
      i : ι
      hi : ∀ (x : α), Membership.mem s x → Membership.mem (f i) x
      ⊢ LE.le s.card n
    -/
    have : ∀ i, Fintype (f i) := fun i ↦ (lt_aleph0_iff_fintype.1 ((h'f i).trans_lt ha)).some
    /-
      case intro
      α ι : Type u
      inst✝¹ : Countable ι
      f : ι → Set α
      l : Filter ι
      inst✝ : l.NeBot
      t : Set α
      ht : ∀ (x : α), Membership.mem t x → Filter.Eventually (fun i => Membership.me …
      n : Nat
      h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) ↑n
      ha : LT.lt (↑n) Cardinal.aleph0
      s : Finset α
      hs : HasSubset.Subset (↑s) t
      A : ∀ (x : α), Membership.mem s x → Filter.Eventually (fun i => Membership.mem …
      B : Filter.Eventually (fun i => ∀ (x : α), Membership.mem s x → Membership.mem …
      i : ι
      hi : ∀ (x : α), Membership.mem s x → Membership.mem (f i) x
      this : (i : ι) → Fintype ↑(f i)
      ⊢ LE.le s.card n
    -/
    let u : Finset α := (f i).toFinset
    have I1 : s.card ≤ u.card := by
      have : s ⊆ u := fun x hx ↦ by simpa only [u, Set.mem_toFinset] using hi x hx
      exact Finset.card_le_card this
    have I2 : (u.card : Cardinal) ≤ n := by
      convert h'f i; simp only [u, Set.toFinset_card, mk_fintype]
    /-
      case intro
      α ι : Type u
      inst✝¹ : Countable ι
      f : ι → Set α
      l : Filter ι
      inst✝ : l.NeBot
      t : Set α
      ht : ∀ (x : α), Membership.mem t x → Filter.Eventually (fun i => Membership.me …
      n : Nat
      h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) ↑n
      ha : LT.lt (↑n) Cardinal.aleph0
      s : Finset α
      hs : HasSubset.Subset (↑s) t
      A : ∀ (x : α), Membership.mem s x → Filter.Eventually (fun i => Membership.mem …
      B : Filter.Eventually (fun i => ∀ (x : α), Membership.mem s x → Membership.mem …
      i : ι
      hi : ∀ (x : α), Membership.mem s x → Membership.mem (f i) x
      this : (i : ι) → Fintype ↑(f i)
      u : Finset α := (f i).toFinset
      I1 : LE.le s.card u.card
      I2 : LE.le ↑u.card ↑n
      ⊢ LE.le s.card n
    -/
    exact I1.trans (Nat.cast_le.1 I2)
    /-
      🎉 no goals
    -/
  -- case `a` infinite:
  · have : t ⊆ ⋃ i, f i := by
      intro x hx
      obtain ⟨i, hi⟩ : ∃ i, x ∈ f i := (ht x hx).exists
      exact mem_iUnion_of_mem i hi
    calc #t ≤ #(⋃ i, f i) := mk_le_mk_of_subset this
      _     ≤ sum (fun i ↦ #(f i)) := mk_iUnion_le_sum_mk
      _     ≤ sum (fun _ ↦ a) := sum_le_sum _ _ h'f
      _     = #ι * a := by simp
      _     ≤ ℵ₀ * a := mul_le_mul_right' mk_le_aleph0 a
      _     = a := aleph0_mul_eq ha


/-- If a set `t` is eventually covered by a countable family of sets, all with cardinality at
most `a`, then the cardinality of `t` is also bounded by `a`. -/
lemma mk_subtype_le_of_countable_eventually_mem {α : Type u} {ι : Type v} {a : Cardinal}
    [Countable ι] {f : ι → Set α} {l : Filter ι} [NeBot l]
    {t : Set α} (ht : ∀ x ∈ t, ∀ᶠ i in l, x ∈ f i)
    (h'f : ∀ i, #(f i) ≤ a) : #t ≤ a := by
  /-
    α : Type u
    ι : Type v
    a : Cardinal.{u}
    inst✝¹ : Countable ι
    f : ι → Set α
    l : Filter ι
    inst✝ : l.NeBot
    t : Set α
    ht : ∀ (x : α), Membership.mem t x → Filter.Eventually (fun i => Membership.me …
    h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) a
    ⊢ LE.le (Cardinal.mk ↑t) a
  -/
  let g : ULift.{u, v} ι → Set (ULift.{v, u} α) := (ULift.down ⁻¹' ·) ∘ f ∘ ULift.down
  /-
    α : Type u
    ι : Type v
    a : Cardinal.{u}
    inst✝¹ : Countable ι
    f : ι → Set α
    l : Filter ι
    inst✝ : l.NeBot
    t : Set α
    ht : ∀ (x : α), Membership.mem t x → Filter.Eventually (fun i => Membership.me …
    h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) a
    g : ULift.{u, v} ι → Set (ULift.{v, u} α) := Function.comp (fun x => Set.preim …
    ⊢ LE.le (Cardinal.mk ↑t) a
  -/
  suffices #(ULift.down.{v} ⁻¹' t) ≤ Cardinal.lift.{v, u} a by simpa
  /-
    α : Type u
    ι : Type v
    a : Cardinal.{u}
    inst✝¹ : Countable ι
    f : ι → Set α
    l : Filter ι
    inst✝ : l.NeBot
    t : Set α
    ht : ∀ (x : α), Membership.mem t x → Filter.Eventually (fun i => Membership.me …
    h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) a
    g : ULift.{u, v} ι → Set (ULift.{v, u} α) := Function.comp (fun x => Set.preim …
    ⊢ LE.le (Cardinal.mk ↑(Set.preimage ULift.down t)) (Cardinal.lift.{v, u} a)
  -/
  let l' : Filter (ULift.{u} ι) := Filter.map ULift.up l
  /-
    α : Type u
    ι : Type v
    a : Cardinal.{u}
    inst✝¹ : Countable ι
    f : ι → Set α
    l : Filter ι
    inst✝ : l.NeBot
    t : Set α
    ht : ∀ (x : α), Membership.mem t x → Filter.Eventually (fun i => Membership.me …
    h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) a
    g : ULift.{u, v} ι → Set (ULift.{v, u} α) := Function.comp (fun x => Set.preim …
    l' : Filter (ULift.{u, v} ι) := Filter.map ULift.up l
    ⊢ LE.le (Cardinal.mk ↑(Set.preimage ULift.down t)) (Cardinal.lift.{v, u} a)
  -/
  have : NeBot l' := map_neBot
  /-
    α : Type u
    ι : Type v
    a : Cardinal.{u}
    inst✝¹ : Countable ι
    f : ι → Set α
    l : Filter ι
    inst✝ : l.NeBot
    t : Set α
    ht : ∀ (x : α), Membership.mem t x → Filter.Eventually (fun i => Membership.me …
    h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) a
    g : ULift.{u, v} ι → Set (ULift.{v, u} α) := Function.comp (fun x => Set.preim …
    l' : Filter (ULift.{u, v} ι) := Filter.map ULift.up l
    this : l'.NeBot
    ⊢ LE.le (Cardinal.mk ↑(Set.preimage ULift.down t)) (Cardinal.lift.{v, u} a)
  -/
  apply mk_subtype_le_of_countable_eventually_mem_aux (ι := ULift.{u} ι) (l := l') (f := g)
    /-
      case ht
      α : Type u
      ι : Type v
      a : Cardinal.{u}
      inst✝¹ : Countable ι
      f : ι → Set α
      l : Filter ι
      inst✝ : l.NeBot
      t : Set α
      ht : ∀ (x : α), Membership.mem t x → Filter.Eventually (fun i => Membership.me …
      h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) a
      g : ULift.{u, v} ι → Set (ULift.{v, u} α) := Function.comp (fun x => Set.preim …
      l' : Filter (ULift.{u, v} ι) := Filter.map ULift.up l
      this : l'.NeBot
      ⊢ ∀ (x : ULift.{v, u} α), Membership.mem (Set.preimage ULift.down t) x → Filte …
    -/
  · intro x hx
    /-
      case ht
      α : Type u
      ι : Type v
      a : Cardinal.{u}
      inst✝¹ : Countable ι
      f : ι → Set α
      l : Filter ι
      inst✝ : l.NeBot
      t : Set α
      ht : ∀ (x : α), Membership.mem t x → Filter.Eventually (fun i => Membership.me …
      h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) a
      g : ULift.{u, v} ι → Set (ULift.{v, u} α) := Function.comp (fun x => Set.preim …
      l' : Filter (ULift.{u, v} ι) := Filter.map ULift.up l
      this : l'.NeBot
      x : ULift.{v, u} α
      hx : Membership.mem (Set.preimage ULift.down t) x
      ⊢ Filter.Eventually (fun i => Membership.mem (g i) x) l'
    -/
    simpa only [Function.comp_apply, mem_preimage, eventually_map] using ht _ hx
    /-
      🎉 no goals
    -/
    /-
      case h'f
      α : Type u
      ι : Type v
      a : Cardinal.{u}
      inst✝¹ : Countable ι
      f : ι → Set α
      l : Filter ι
      inst✝ : l.NeBot
      t : Set α
      ht : ∀ (x : α), Membership.mem t x → Filter.Eventually (fun i => Membership.me …
      h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) a
      g : ULift.{u, v} ι → Set (ULift.{v, u} α) := Function.comp (fun x => Set.preim …
      l' : Filter (ULift.{u, v} ι) := Filter.map ULift.up l
      this : l'.NeBot
      ⊢ ∀ (i : ULift.{u, v} ι), LE.le (Cardinal.mk ↑(g i)) (Cardinal.lift.{v, u} a)
    -/
  · intro i
    /-
      case h'f
      α : Type u
      ι : Type v
      a : Cardinal.{u}
      inst✝¹ : Countable ι
      f : ι → Set α
      l : Filter ι
      inst✝ : l.NeBot
      t : Set α
      ht : ∀ (x : α), Membership.mem t x → Filter.Eventually (fun i => Membership.me …
      h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) a
      g : ULift.{u, v} ι → Set (ULift.{v, u} α) := Function.comp (fun x => Set.preim …
      l' : Filter (ULift.{u, v} ι) := Filter.map ULift.up l
      this : l'.NeBot
      i : ULift.{u, v} ι
      ⊢ LE.le (Cardinal.mk ↑(g i)) (Cardinal.lift.{v, u} a)
    -/
    simpa [g] using h'f i.down
    /-
      🎉 no goals
    -/


/-- If a space is eventually covered by a countable family of sets, all with cardinality at
most `a`, then the cardinality of the space is also bounded by `a`. -/
lemma mk_le_of_countable_eventually_mem {α : Type u} {ι : Type v} {a : Cardinal}
    [Countable ι] {f : ι → Set α} {l : Filter ι} [NeBot l] (ht : ∀ x, ∀ᶠ i in l, x ∈ f i)
    (h'f : ∀ i, #(f i) ≤ a) : #α ≤ a := by
  /-
    α : Type u
    ι : Type v
    a : Cardinal.{u}
    inst✝¹ : Countable ι
    f : ι → Set α
    l : Filter ι
    inst✝ : l.NeBot
    ht : ∀ (x : α), Filter.Eventually (fun i => Membership.mem (f i) x) l
    h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) a
    ⊢ LE.le (Cardinal.mk α) a
  -/
  rw [← mk_univ]
  /-
    α : Type u
    ι : Type v
    a : Cardinal.{u}
    inst✝¹ : Countable ι
    f : ι → Set α
    l : Filter ι
    inst✝ : l.NeBot
    ht : ∀ (x : α), Filter.Eventually (fun i => Membership.mem (f i) x) l
    h'f : ∀ (i : ι), LE.le (Cardinal.mk ↑(f i)) a
    ⊢ LE.le (Cardinal.mk ↑Set.univ) a
  -/
  exact mk_subtype_le_of_countable_eventually_mem (l := l) (fun x _ ↦ ht x) h'f
  /-
    🎉 no goals
  -/


/-- If a space is eventually covered by a countable family of sets, all with cardinality `a`,
then the cardinality of the space is also `a`. -/
lemma mk_of_countable_eventually_mem {α : Type u} {ι : Type v} {a : Cardinal}
    [Countable ι] {f : ι → Set α} {l : Filter ι} [NeBot l] (ht : ∀ x, ∀ᶠ i in l, x ∈ f i)
    (h'f : ∀ i, #(f i) = a) : #α = a := by
  /-
    α : Type u
    ι : Type v
    a : Cardinal.{u}
    inst✝¹ : Countable ι
    f : ι → Set α
    l : Filter ι
    inst✝ : l.NeBot
    ht : ∀ (x : α), Filter.Eventually (fun i => Membership.mem (f i) x) l
    h'f : ∀ (i : ι), Eq (Cardinal.mk ↑(f i)) a
    ⊢ Eq (Cardinal.mk α) a
  -/
  apply le_antisymm
    /-
      case a
      α : Type u
      ι : Type v
      a : Cardinal.{u}
      inst✝¹ : Countable ι
      f : ι → Set α
      l : Filter ι
      inst✝ : l.NeBot
      ht : ∀ (x : α), Filter.Eventually (fun i => Membership.mem (f i) x) l
      h'f : ∀ (i : ι), Eq (Cardinal.mk ↑(f i)) a
      ⊢ LE.le (Cardinal.mk α) a
    -/
  · apply mk_le_of_countable_eventually_mem ht (fun i ↦ (h'f i).le)
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u
      ι : Type v
      a : Cardinal.{u}
      inst✝¹ : Countable ι
      f : ι → Set α
      l : Filter ι
      inst✝ : l.NeBot
      ht : ∀ (x : α), Filter.Eventually (fun i => Membership.mem (f i) x) l
      h'f : ∀ (i : ι), Eq (Cardinal.mk ↑(f i)) a
      ⊢ LE.le a (Cardinal.mk α)
    -/
  · obtain ⟨i⟩ : Nonempty ι := nonempty_of_neBot l
    /-
      case a.intro
      α : Type u
      ι : Type v
      a : Cardinal.{u}
      inst✝¹ : Countable ι
      f : ι → Set α
      l : Filter ι
      inst✝ : l.NeBot
      ht : ∀ (x : α), Filter.Eventually (fun i => Membership.mem (f i) x) l
      h'f : ∀ (i : ι), Eq (Cardinal.mk ↑(f i)) a
      i : ι
      ⊢ LE.le a (Cardinal.mk α)
    -/
    rw [← (h'f i)]
    /-
      case a.intro
      α : Type u
      ι : Type v
      a : Cardinal.{u}
      inst✝¹ : Countable ι
      f : ι → Set α
      l : Filter ι
      inst✝ : l.NeBot
      ht : ∀ (x : α), Filter.Eventually (fun i => Membership.mem (f i) x) l
      h'f : ∀ (i : ι), Eq (Cardinal.mk ↑(f i)) a
      i : ι
      ⊢ LE.le (Cardinal.mk ↑(f i)) (Cardinal.mk α)
    -/
    exact mk_set_le (f i)
    /-
      🎉 no goals
    -/


