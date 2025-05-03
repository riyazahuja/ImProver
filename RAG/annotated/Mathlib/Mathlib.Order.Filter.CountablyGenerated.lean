/-- `IsCountablyGenerated f` means `f = generate s` for some countable `s`. -/
class IsCountablyGenerated (f : Filter α) : Prop where
  /-- There exists a countable set that generates the filter. -/
  out : ∃ s : Set (Set α), s.Countable ∧ f = generate s


/-- `IsCountableBasis p s` means the image of `s` bounded by `p` is a countable filter basis. -/
structure IsCountableBasis (p : ι → Prop) (s : ι → Set α) extends IsBasis p s : Prop where
  /-- The set of `i` that satisfy the predicate `p` is countable. -/
  countable : (setOf p).Countable


/-- We say that a filter `l` has a countable basis `s : ι → Set α` bounded by `p : ι → Prop`,
if `t ∈ l` if and only if `t` includes `s i` for some `i` such that `p i`, and the set
defined by `p` is countable. -/
structure HasCountableBasis (l : Filter α) (p : ι → Prop) (s : ι → Set α)
    extends HasBasis l p s : Prop where
  /-- The set of `i` that satisfy the predicate `p` is countable. -/
  countable : (setOf p).Countable


/-- A countable filter basis `B` on a type `α` is a nonempty countable collection of sets of `α`
such that the intersection of two elements of this collection contains some element
of the collection. -/
structure CountableFilterBasis (α : Type*) extends FilterBasis α where
  /-- The set of sets of the filter basis is countable. -/
  countable : sets.Countable

-- For illustration purposes, the countable filter basis defining `(atTop : Filter ℕ)`

instance Nat.inhabitedCountableFilterBasis : Inhabited (CountableFilterBasis ℕ) :=
  ⟨⟨default, countable_range fun n => Ici n⟩⟩


theorem HasCountableBasis.isCountablyGenerated {f : Filter α} {p : ι → Prop} {s : ι → Set α}
    (h : f.HasCountableBasis p s) : f.IsCountablyGenerated :=
  ⟨⟨{ t | ∃ i, p i ∧ s i = t }, h.countable.image s, h.toHasBasis.eq_generate⟩⟩


theorem HasBasis.isCountablyGenerated [Countable ι] {f : Filter α} {p : ι → Prop} {s : ι → Set α}
    (h : f.HasBasis p s) : f.IsCountablyGenerated :=
  HasCountableBasis.isCountablyGenerated ⟨h, to_countable _⟩


theorem antitone_seq_of_seq (s : ℕ → Set α) :
    ∃ t : ℕ → Set α, Antitone t ∧ ⨅ i, 𝓟 (s i) = ⨅ i, 𝓟 (t i) := by
  /-
    α : Type u_1
    s : Nat → Set α
    ⊢ Exists fun t => And (Antitone t) (Eq (iInf fun i => Filter.principal (s i))  …
  -/
  use fun n => ⋂ m ≤ n, s m; constructor
    /-
      case h.left
      α : Type u_1
      s : Nat → Set α
      ⊢ Antitone fun n => Set.iInter fun m => Set.iInter fun h => s m
    -/
  · exact fun i j hij => biInter_mono (Iic_subset_Iic.2 hij) fun n _ => Subset.rfl
    /-
      🎉 no goals
    -/
  /-
    case h.right
    α : Type u_1
    s : Nat → Set α
    ⊢ Eq (iInf fun i => Filter.principal (s i)) (iInf fun i => Filter.principal (S …
  -/
  apply le_antisymm <;> rw [le_iInf_iff] <;> intro i
    /-
      case h.right.a
      α : Type u_1
      s : Nat → Set α
      i : Nat
      ⊢ LE.le (iInf fun i => Filter.principal (s i)) (Filter.principal (Set.iInter f …
    -/
  · rw [le_principal_iff]
    /-
      case h.right.a
      α : Type u_1
      s : Nat → Set α
      i : Nat
      ⊢ Membership.mem (iInf fun i => Filter.principal (s i)) (Set.iInter fun m => S …
    -/
    refine (biInter_mem (finite_le_nat _)).2 fun j _ => ?_
    /-
      case h.right.a
      α : Type u_1
      s : Nat → Set α
      i j : Nat
      x✝ : Membership.mem (setOf fun i_1 => LE.le i_1 i) j
      ⊢ Membership.mem (iInf fun i => Filter.principal (s i)) (s j)
    -/
    exact mem_iInf_of_mem j (mem_principal_self _)
    /-
      🎉 no goals
    -/
    /-
      case h.right.a
      α : Type u_1
      s : Nat → Set α
      i : Nat
      ⊢ LE.le (iInf fun i => Filter.principal (Set.iInter fun m => Set.iInter fun h  …
    -/
  · refine iInf_le_of_le i (principal_mono.2 <| iInter₂_subset i ?_)
    /-
      case h.right.a
      α : Type u_1
      s : Nat → Set α
      i : Nat
      ⊢ LE.le i i
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem countable_biInf_eq_iInf_seq [CompleteLattice α] {B : Set ι} (Bcbl : B.Countable)
    (Bne : B.Nonempty) (f : ι → α) : ∃ x : ℕ → ι, ⨅ t ∈ B, f t = ⨅ i, f (x i) :=
  let ⟨g, hg⟩ := Bcbl.exists_eq_range Bne
  ⟨g, hg.symm ▸ iInf_range⟩


theorem countable_biInf_eq_iInf_seq' [CompleteLattice α] {B : Set ι} (Bcbl : B.Countable)
    (f : ι → α) {i₀ : ι} (h : f i₀ = ⊤) : ∃ x : ℕ → ι, ⨅ t ∈ B, f t = ⨅ i, f (x i) := by
  /-
    α : Type u_1
    ι : Type u_4
    inst✝ : CompleteLattice α
    B : Set ι
    Bcbl : B.Countable
    f : ι → α
    i₀ : ι
    h : Eq (f i₀) Top.top
    ⊢ Exists fun x => Eq (iInf fun t => iInf fun h => f t) (iInf fun i => f (x i))
  -/
  rcases B.eq_empty_or_nonempty with hB | Bnonempty
    /-
      case inl
      α : Type u_1
      ι : Type u_4
      inst✝ : CompleteLattice α
      B : Set ι
      Bcbl : B.Countable
      f : ι → α
      i₀ : ι
      h : Eq (f i₀) Top.top
      hB : Eq B EmptyCollection.emptyCollection
      ⊢ Exists fun x => Eq (iInf fun t => iInf fun h => f t) (iInf fun i => f (x i))
    -/
  · rw [hB, iInf_emptyset]
    /-
      case inl
      α : Type u_1
      ι : Type u_4
      inst✝ : CompleteLattice α
      B : Set ι
      Bcbl : B.Countable
      f : ι → α
      i₀ : ι
      h : Eq (f i₀) Top.top
      hB : Eq B EmptyCollection.emptyCollection
      ⊢ Exists fun x => Eq Top.top (iInf fun i => f (x i))
    -/
    use fun _ => i₀
    /-
      case h
      α : Type u_1
      ι : Type u_4
      inst✝ : CompleteLattice α
      B : Set ι
      Bcbl : B.Countable
      f : ι → α
      i₀ : ι
      h : Eq (f i₀) Top.top
      hB : Eq B EmptyCollection.emptyCollection
      ⊢ Eq Top.top (iInf fun i => f ((fun x => i₀) i))
    -/
    simp [h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      ι : Type u_4
      inst✝ : CompleteLattice α
      B : Set ι
      Bcbl : B.Countable
      f : ι → α
      i₀ : ι
      h : Eq (f i₀) Top.top
      Bnonempty : B.Nonempty
      ⊢ Exists fun x => Eq (iInf fun t => iInf fun h => f t) (iInf fun i => f (x i))
    -/
  · exact countable_biInf_eq_iInf_seq Bcbl Bnonempty f
    /-
      🎉 no goals
    -/


theorem countable_biInf_principal_eq_seq_iInf {B : Set (Set α)} (Bcbl : B.Countable) :
    ∃ x : ℕ → Set α, ⨅ t ∈ B, 𝓟 t = ⨅ i, 𝓟 (x i) :=
  countable_biInf_eq_iInf_seq' Bcbl 𝓟 principal_univ


protected theorem HasAntitoneBasis.mem_iff [Preorder ι] {l : Filter α} {s : ι → Set α}
    (hs : l.HasAntitoneBasis s) {t : Set α} : t ∈ l ↔ ∃ i, s i ⊆ t :=
                                    /-
                                      α : Type u_1
                                      ι : Type u_4
                                      inst✝ : Preorder ι
                                      l : Filter α
                                      s : ι → Set α
                                      hs : l.HasAntitoneBasis s
                                      t : Set α
                                      ⊢ Iff (Exists fun i => And True (HasSubset.Subset (s i) t)) (Exists fun i => H …
                                    -/
  hs.toHasBasis.mem_iff.trans <| by simp only [exists_prop, true_and]
                                    /-
                                      🎉 no goals
                                    -/


protected theorem HasAntitoneBasis.mem [Preorder ι] {l : Filter α} {s : ι → Set α}
    (hs : l.HasAntitoneBasis s) (i : ι) : s i ∈ l :=
  hs.toHasBasis.mem_of_mem trivial


theorem HasAntitoneBasis.hasBasis_ge [Preorder ι] [IsDirected ι (· ≤ ·)] {l : Filter α}
    {s : ι → Set α} (hs : l.HasAntitoneBasis s) (i : ι) : l.HasBasis (fun j => i ≤ j) s :=
  hs.1.to_hasBasis (fun j _ => (exists_ge_ge i j).imp fun _k hk => ⟨hk.1, hs.2 hk.2⟩) fun j _ =>
    ⟨j, trivial, Subset.rfl⟩


/-- If `f` is countably generated and `f.HasBasis p s`, then `f` admits a decreasing basis
enumerated by natural numbers such that all sets have the form `s i`. More precisely, there is a
sequence `i n` such that `p (i n)` for all `n` and `s (i n)` is a decreasing sequence of sets which
forms a basis of `f`-/
theorem HasBasis.exists_antitone_subbasis {f : Filter α} [h : f.IsCountablyGenerated]
    {p : ι' → Prop} {s : ι' → Set α} (hs : f.HasBasis p s) :
    ∃ x : ℕ → ι', (∀ i, p (x i)) ∧ f.HasAntitoneBasis fun i => s (x i) := by
  obtain ⟨x', hx'⟩ : ∃ x : ℕ → Set α, f = ⨅ i, 𝓟 (x i) := by
    rcases h with ⟨s, hsc, rfl⟩
    rw [generate_eq_biInf]
    exact countable_biInf_principal_eq_seq_iInf hsc
  /-
    case intro
    α : Type u_1
    ι' : Sort u_5
    f : Filter α
    h : f.IsCountablyGenerated
    p : ι' → Prop
    s : ι' → Set α
    hs : f.HasBasis p s
    x' : Nat → Set α
    hx' : Eq f (iInf fun i => Filter.principal (x' i))
    ⊢ Exists fun x => And (∀ (i : Nat), p (x i)) (f.HasAntitoneBasis fun i => s (x …
  -/
  have : ∀ i, x' i ∈ f := fun i => hx'.symm ▸ (iInf_le (fun i => 𝓟 (x' i)) i) (mem_principal_self _)
  let x : ℕ → { i : ι' // p i } := fun n =>
    Nat.recOn n (hs.index _ <| this 0) fun n xn =>
      hs.index _ <| inter_mem (this <| n + 1) (hs.mem_of_mem xn.2)
  have x_anti : Antitone fun i => s (x i).1 :=
    antitone_nat_of_succ_le fun i => (hs.set_index_subset _).trans inter_subset_right
  have x_subset : ∀ i, s (x i).1 ⊆ x' i := by
    rintro (_ | i)
    exacts [hs.set_index_subset _, (hs.set_index_subset _).trans inter_subset_left]
  /-
    case intro
    α : Type u_1
    ι' : Sort u_5
    f : Filter α
    h : f.IsCountablyGenerated
    p : ι' → Prop
    s : ι' → Set α
    hs : f.HasBasis p s
    x' : Nat → Set α
    hx' : Eq f (iInf fun i => Filter.principal (x' i))
    this : ∀ (i : Nat), Membership.mem f (x' i)
    x : Nat → Subtype fun i => p i := fun n => Nat.recOn n (hs.index (x' 0) ⋯) fun …
    x_anti : Antitone fun i => s ↑(x i)
    x_subset : ∀ (i : Nat), HasSubset.Subset (s ↑(x i)) (x' i)
    ⊢ Exists fun x => And (∀ (i : Nat), p (x i)) (f.HasAntitoneBasis fun i => s (x …
  -/
  refine ⟨fun i => (x i).1, fun i => (x i).2, ?_⟩
  /-
    case intro
    α : Type u_1
    ι' : Sort u_5
    f : Filter α
    h : f.IsCountablyGenerated
    p : ι' → Prop
    s : ι' → Set α
    hs : f.HasBasis p s
    x' : Nat → Set α
    hx' : Eq f (iInf fun i => Filter.principal (x' i))
    this : ∀ (i : Nat), Membership.mem f (x' i)
    x : Nat → Subtype fun i => p i := fun n => Nat.recOn n (hs.index (x' 0) ⋯) fun …
    x_anti : Antitone fun i => s ↑(x i)
    x_subset : ∀ (i : Nat), HasSubset.Subset (s ↑(x i)) (x' i)
    ⊢ f.HasAntitoneBasis fun i => s ((fun i => ↑(x i)) i)
  -/
  have : (⨅ i, 𝓟 (s (x i).1)).HasAntitoneBasis fun i => s (x i).1 := .iInf_principal x_anti
  /-
    case intro
    α : Type u_1
    ι' : Sort u_5
    f : Filter α
    h : f.IsCountablyGenerated
    p : ι' → Prop
    s : ι' → Set α
    hs : f.HasBasis p s
    x' : Nat → Set α
    hx' : Eq f (iInf fun i => Filter.principal (x' i))
    this✝ : ∀ (i : Nat), Membership.mem f (x' i)
    x : Nat → Subtype fun i => p i := fun n => Nat.recOn n (hs.index (x' 0) ⋯) fun …
    x_anti : Antitone fun i => s ↑(x i)
    x_subset : ∀ (i : Nat), HasSubset.Subset (s ↑(x i)) (x' i)
    this : (iInf fun i => Filter.principal (s ↑(x i))).HasAntitoneBasis fun i => s …
    ⊢ f.HasAntitoneBasis fun i => s ((fun i => ↑(x i)) i)
  -/
  convert this
  exact
    le_antisymm (le_iInf fun i => le_principal_iff.2 <| by cases i <;> apply hs.set_index_mem)
      (hx'.symm ▸
        le_iInf fun i => le_principal_iff.2 <| this.1.mem_iff.2 ⟨i, trivial, x_subset i⟩)


/-- A countably generated filter admits a basis formed by an antitone sequence of sets. -/
theorem exists_antitone_basis (f : Filter α) [f.IsCountablyGenerated] :
    ∃ x : ℕ → Set α, f.HasAntitoneBasis x :=
  let ⟨x, _, hx⟩ := f.basis_sets.exists_antitone_subbasis
  ⟨x, hx⟩


theorem exists_antitone_seq (f : Filter α) [f.IsCountablyGenerated] :
    ∃ x : ℕ → Set α, Antitone x ∧ ∀ {s}, s ∈ f ↔ ∃ i, x i ⊆ s :=
  let ⟨x, hx⟩ := f.exists_antitone_basis
                      /-
                        α : Type u_1
                        f : Filter α
                        inst✝ : f.IsCountablyGenerated
                        x : Nat → Set α
                        hx : f.HasAntitoneBasis x
                        ⊢ ∀ {s : Set α}, Iff (Membership.mem f s) (Exists fun i => HasSubset.Subset (x …
                      -/
  ⟨x, hx.antitone, by simp [hx.1.mem_iff]⟩
                      /-
                        🎉 no goals
                      -/


instance Inf.isCountablyGenerated (f g : Filter α) [IsCountablyGenerated f]
    [IsCountablyGenerated g] : IsCountablyGenerated (f ⊓ g) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    ι : Type u_4
    ι' : Sort u_5
    f g : Filter α
    inst✝¹ : f.IsCountablyGenerated
    inst✝ : g.IsCountablyGenerated
    ⊢ (Min.min f g).IsCountablyGenerated
  -/
  rcases f.exists_antitone_basis with ⟨s, hs⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    ι : Type u_4
    ι' : Sort u_5
    f g : Filter α
    inst✝¹ : f.IsCountablyGenerated
    inst✝ : g.IsCountablyGenerated
    s : Nat → Set α
    hs : f.HasAntitoneBasis s
    ⊢ (Min.min f g).IsCountablyGenerated
  -/
  rcases g.exists_antitone_basis with ⟨t, ht⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    ι : Type u_4
    ι' : Sort u_5
    f g : Filter α
    inst✝¹ : f.IsCountablyGenerated
    inst✝ : g.IsCountablyGenerated
    s : Nat → Set α
    hs : f.HasAntitoneBasis s
    t : Nat → Set α
    ht : g.HasAntitoneBasis t
    ⊢ (Min.min f g).IsCountablyGenerated
  -/
  exact HasCountableBasis.isCountablyGenerated ⟨hs.1.inf ht.1, Set.to_countable _⟩
  /-
    🎉 no goals
  -/


instance map.isCountablyGenerated (l : Filter α) [l.IsCountablyGenerated] (f : α → β) :
    (map f l).IsCountablyGenerated :=
  let ⟨_x, hxl⟩ := l.exists_antitone_basis
  (hxl.map _).isCountablyGenerated


instance comap.isCountablyGenerated (l : Filter β) [l.IsCountablyGenerated] (f : α → β) :
    (comap f l).IsCountablyGenerated :=
  let ⟨_x, hxl⟩ := l.exists_antitone_basis
  (hxl.comap _).isCountablyGenerated


instance Sup.isCountablyGenerated (f g : Filter α) [IsCountablyGenerated f]
    [IsCountablyGenerated g] : IsCountablyGenerated (f ⊔ g) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    ι : Type u_4
    ι' : Sort u_5
    f g : Filter α
    inst✝¹ : f.IsCountablyGenerated
    inst✝ : g.IsCountablyGenerated
    ⊢ (Max.max f g).IsCountablyGenerated
  -/
  rcases f.exists_antitone_basis with ⟨s, hs⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    ι : Type u_4
    ι' : Sort u_5
    f g : Filter α
    inst✝¹ : f.IsCountablyGenerated
    inst✝ : g.IsCountablyGenerated
    s : Nat → Set α
    hs : f.HasAntitoneBasis s
    ⊢ (Max.max f g).IsCountablyGenerated
  -/
  rcases g.exists_antitone_basis with ⟨t, ht⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    ι : Type u_4
    ι' : Sort u_5
    f g : Filter α
    inst✝¹ : f.IsCountablyGenerated
    inst✝ : g.IsCountablyGenerated
    s : Nat → Set α
    hs : f.HasAntitoneBasis s
    t : Nat → Set α
    ht : g.HasAntitoneBasis t
    ⊢ (Max.max f g).IsCountablyGenerated
  -/
  exact HasCountableBasis.isCountablyGenerated ⟨hs.1.sup ht.1, Set.to_countable _⟩
  /-
    🎉 no goals
  -/


instance prod.isCountablyGenerated (la : Filter α) (lb : Filter β) [IsCountablyGenerated la]
    [IsCountablyGenerated lb] : IsCountablyGenerated (la ×ˢ lb) :=
  Filter.Inf.isCountablyGenerated _ _


instance coprod.isCountablyGenerated (la : Filter α) (lb : Filter β) [IsCountablyGenerated la]
    [IsCountablyGenerated lb] : IsCountablyGenerated (la.coprod lb) :=
  Filter.Sup.isCountablyGenerated _ _


theorem isCountablyGenerated_seq [Countable ι'] (x : ι' → Set α) :
    IsCountablyGenerated (⨅ i, 𝓟 (x i)) := by
  /-
    α : Type u_1
    ι' : Sort u_5
    inst✝ : Countable ι'
    x : ι' → Set α
    ⊢ (iInf fun i => Filter.principal (x i)).IsCountablyGenerated
  -/
  use range x, countable_range x
  /-
    case right
    α : Type u_1
    ι' : Sort u_5
    inst✝ : Countable ι'
    x : ι' → Set α
    ⊢ Eq (iInf fun i => Filter.principal (x i)) (Filter.generate (Set.range x))
  -/
  rw [generate_eq_biInf, iInf_range]
  /-
    🎉 no goals
  -/


theorem isCountablyGenerated_of_seq {f : Filter α} (h : ∃ x : ℕ → Set α, f = ⨅ i, 𝓟 (x i)) :
    f.IsCountablyGenerated := by
  /-
    α : Type u_1
    f : Filter α
    h : Exists fun x => Eq f (iInf fun i => Filter.principal (x i))
    ⊢ f.IsCountablyGenerated
  -/
  rcases h with ⟨x, rfl⟩
  /-
    case intro
    α : Type u_1
    x : Nat → Set α
    ⊢ (iInf fun i => Filter.principal (x i)).IsCountablyGenerated
  -/
  apply isCountablyGenerated_seq
  /-
    🎉 no goals
  -/


theorem isCountablyGenerated_biInf_principal {B : Set (Set α)} (h : B.Countable) :
    IsCountablyGenerated (⨅ s ∈ B, 𝓟 s) :=
  isCountablyGenerated_of_seq (countable_biInf_principal_eq_seq_iInf h)


theorem isCountablyGenerated_iff_exists_antitone_basis {f : Filter α} :
    IsCountablyGenerated f ↔ ∃ x : ℕ → Set α, f.HasAntitoneBasis x := by
  /-
    α : Type u_1
    f : Filter α
    ⊢ Iff f.IsCountablyGenerated (Exists fun x => f.HasAntitoneBasis x)
  -/
  constructor
    /-
      case mp
      α : Type u_1
      f : Filter α
      ⊢ f.IsCountablyGenerated → Exists fun x => f.HasAntitoneBasis x
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      f : Filter α
      h : f.IsCountablyGenerated
      ⊢ Exists fun x => f.HasAntitoneBasis x
    -/
    exact f.exists_antitone_basis
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      f : Filter α
      ⊢ (Exists fun x => f.HasAntitoneBasis x) → f.IsCountablyGenerated
    -/
  · rintro ⟨x, h⟩
    /-
      case mpr.intro
      α : Type u_1
      f : Filter α
      x : Nat → Set α
      h : f.HasAntitoneBasis x
      ⊢ f.IsCountablyGenerated
    -/
    rw [h.1.eq_iInf]
    /-
      case mpr.intro
      α : Type u_1
      f : Filter α
      x : Nat → Set α
      h : f.HasAntitoneBasis x
      ⊢ (iInf fun i => Filter.principal (x i)).IsCountablyGenerated
    -/
    exact isCountablyGenerated_seq x
    /-
      🎉 no goals
    -/


@[instance]
theorem isCountablyGenerated_principal (s : Set α) : IsCountablyGenerated (𝓟 s) :=
  isCountablyGenerated_of_seq ⟨fun _ => s, iInf_const.symm⟩


@[instance]
theorem isCountablyGenerated_pure (a : α) : IsCountablyGenerated (pure a) := by
  /-
    α : Type u_1
    a : α
    ⊢ (Pure.pure a).IsCountablyGenerated
  -/
  rw [← principal_singleton]
  /-
    α : Type u_1
    a : α
    ⊢ (Filter.principal (Singleton.singleton a)).IsCountablyGenerated
  -/
  exact isCountablyGenerated_principal _
  /-
    🎉 no goals
  -/


@[instance]
theorem isCountablyGenerated_bot : IsCountablyGenerated (⊥ : Filter α) :=
  @principal_empty α ▸ isCountablyGenerated_principal _


@[instance]
theorem isCountablyGenerated_top : IsCountablyGenerated (⊤ : Filter α) :=
  @principal_univ α ▸ isCountablyGenerated_principal _

-- Porting note: without explicit `Sort u` and `Type v`, Lean 4 uses `ι : Prop`

instance iInf.isCountablyGenerated {ι : Sort u} {α : Type v} [Countable ι] (f : ι → Filter α)
    [∀ i, IsCountablyGenerated (f i)] : IsCountablyGenerated (⨅ i, f i) := by
  /-
    α✝ : Type u_1
    β : Type u_2
    γ : Type u_3
    ι✝ : Type u_4
    ι' : Sort u_5
    ι : Sort u
    α : Type v
    inst✝¹ : Countable ι
    f : ι → Filter α
    inst✝ : ∀ (i : ι), (f i).IsCountablyGenerated
    ⊢ (iInf fun i => f i).IsCountablyGenerated
  -/
  choose s hs using fun i => exists_antitone_basis (f i)
  /-
    α✝ : Type u_1
    β : Type u_2
    γ : Type u_3
    ι✝ : Type u_4
    ι' : Sort u_5
    ι : Sort u
    α : Type v
    inst✝¹ : Countable ι
    f : ι → Filter α
    inst✝ : ∀ (i : ι), (f i).IsCountablyGenerated
    s : ι → Nat → Set α
    hs : ∀ (i : ι), (f i).HasAntitoneBasis (s i)
    ⊢ (iInf fun i => f i).IsCountablyGenerated
  -/
  rw [← PLift.down_surjective.iInf_comp]
  /-
    α✝ : Type u_1
    β : Type u_2
    γ : Type u_3
    ι✝ : Type u_4
    ι' : Sort u_5
    ι : Sort u
    α : Type v
    inst✝¹ : Countable ι
    f : ι → Filter α
    inst✝ : ∀ (i : ι), (f i).IsCountablyGenerated
    s : ι → Nat → Set α
    hs : ∀ (i : ι), (f i).HasAntitoneBasis (s i)
    ⊢ (iInf fun x => f x.down).IsCountablyGenerated
  -/
  refine HasCountableBasis.isCountablyGenerated ⟨hasBasis_iInf fun n => (hs _).1, ?_⟩
  /-
    α✝ : Type u_1
    β : Type u_2
    γ : Type u_3
    ι✝ : Type u_4
    ι' : Sort u_5
    ι : Sort u
    α : Type v
    inst✝¹ : Countable ι
    f : ι → Filter α
    inst✝ : ∀ (i : ι), (f i).IsCountablyGenerated
    s : ι → Nat → Set α
    hs : ∀ (i : ι), (f i).HasAntitoneBasis (s i)
    ⊢ (setOf fun If => And If.fst.Finite (↑If.fst → True)).Countable
  -/
  refine (countable_range <| Sigma.map ((↑) : Finset (PLift ι) → Set (PLift ι)) fun _ => id).mono ?_
  /-
    α✝ : Type u_1
    β : Type u_2
    γ : Type u_3
    ι✝ : Type u_4
    ι' : Sort u_5
    ι : Sort u
    α : Type v
    inst✝¹ : Countable ι
    f : ι → Filter α
    inst✝ : ∀ (i : ι), (f i).IsCountablyGenerated
    s : ι → Nat → Set α
    hs : ∀ (i : ι), (f i).HasAntitoneBasis (s i)
    ⊢ HasSubset.Subset (setOf fun If => And If.fst.Finite (↑If.fst → True)) (Set.r …
  -/
  rintro ⟨I, f⟩ ⟨hI, -⟩
  /-
    case mk.intro
    α✝ : Type u_1
    β : Type u_2
    γ : Type u_3
    ι✝ : Type u_4
    ι' : Sort u_5
    ι : Sort u
    α : Type v
    inst✝¹ : Countable ι
    f✝ : ι → Filter α
    inst✝ : ∀ (i : ι), (f✝ i).IsCountablyGenerated
    s : ι → Nat → Set α
    hs : ∀ (i : ι), (f✝ i).HasAntitoneBasis (s i)
    I : Set (PLift ι)
    f : ↑I → Nat
    hI : ⟨I, f⟩.fst.Finite
    ⊢ Membership.mem (Set.range (Sigma.map Finset.toSet fun x => id)) ⟨I, f⟩
  -/
  lift I to Finset (PLift ι) using hI
  /-
    case mk.intro.intro
    α✝ : Type u_1
    β : Type u_2
    γ : Type u_3
    ι✝ : Type u_4
    ι' : Sort u_5
    ι : Sort u
    α : Type v
    inst✝¹ : Countable ι
    f✝ : ι → Filter α
    inst✝ : ∀ (i : ι), (f✝ i).IsCountablyGenerated
    s : ι → Nat → Set α
    hs : ∀ (i : ι), (f✝ i).HasAntitoneBasis (s i)
    I : Finset (PLift ι)
    f : ↑↑I → Nat
    ⊢ Membership.mem (Set.range (Sigma.map Finset.toSet fun x => id)) ⟨↑I, f⟩
  -/
  exact ⟨⟨I, f⟩, rfl⟩
  /-
    🎉 no goals
  -/


instance pi.isCountablyGenerated {ι : Type*} {α : ι → Type*} [Countable ι]
    (f : ∀ i, Filter (α i)) [∀ i, IsCountablyGenerated (f i)] : IsCountablyGenerated (pi f) :=
  iInf.isCountablyGenerated _


