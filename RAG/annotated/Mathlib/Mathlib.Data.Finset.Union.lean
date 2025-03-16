/-- `disjiUnion s f h` is the set such that `a ∈ disjiUnion s f` iff `a ∈ f i` for some `i ∈ s`.
It is the same as `s.biUnion f`, but it does not require decidable equality on the type. The
hypothesis ensures that the sets are disjoint. -/
def disjiUnion (s : Finset α) (t : α → Finset β) (hf : (s : Set α).PairwiseDisjoint t) : Finset β :=
  ⟨s.val.bind (Finset.val ∘ t), Multiset.nodup_bind.2
    ⟨fun a _ ↦ (t a).nodup, s.nodup.pairwise fun _ ha _ hb hab ↦ disjoint_val.2 <| hf ha hb hab⟩⟩


@[simp]
lemma disjiUnion_val (s : Finset α) (t : α → Finset β) (h) :
    (s.disjiUnion t h).1 = s.1.bind fun a ↦ (t a).1 := rfl


                                                                       /-
                                                                         α : Type u_1
                                                                         β : Type u_2
                                                                         γ : Type u_3
                                                                         s s₁ s₂ : Finset α
                                                                         t✝ t₁ t₂ t : α → Finset β
                                                                         ⊢ (↑EmptyCollection.emptyCollection).PairwiseDisjoint t
                                                                       -/
@[simp] lemma disjiUnion_empty (t : α → Finset β) : disjiUnion ∅ t (by simp) = ∅ := rfl
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp] lemma mem_disjiUnion {b : β} {h} : b ∈ s.disjiUnion t h ↔ ∃ a ∈ s, b ∈ t a := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : α → Finset β
    b : β
    h : (↑s).PairwiseDisjoint t
    ⊢ Iff (Membership.mem (s.disjiUnion t h) b) (Exists fun a => And (Membership.m …
  -/
  simp only [mem_def, disjiUnion_val, Multiset.mem_bind, exists_prop]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
lemma coe_disjiUnion {h} : (s.disjiUnion t h : Set β) = ⋃ x ∈ (s : Set α), t x := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : α → Finset β
    h : (↑s).PairwiseDisjoint t
    ⊢ Eq (↑(s.disjiUnion t h)) (Set.iUnion fun x => Set.iUnion fun h => ↑(t x))
  -/
  simp [Set.ext_iff, mem_disjiUnion, Set.mem_iUnion, mem_coe, imp_true_iff]
  /-
    🎉 no goals
  -/


@[simp] lemma disjiUnion_cons (a : α) (s : Finset α) (ha : a ∉ s) (f : α → Finset β) (H) :
    disjiUnion (cons a s ha) f H =
    (f a).disjUnion ((s.disjiUnion f) fun _ hb _ hc ↦ H (mem_cons_of_mem hb) (mem_cons_of_mem hc))
      (disjoint_left.2 fun _ hb h ↦
        let ⟨_, hc, h⟩ := mem_disjiUnion.mp h
        disjoint_left.mp
          (H (mem_cons_self a s) (mem_cons_of_mem hc) (ne_of_mem_of_not_mem hc ha).symm) hb h) :=
  eq_of_veq <| Multiset.cons_bind _ _ _


@[simp] lemma singleton_disjiUnion (a : α) {h} : Finset.disjiUnion {a} t h = t a :=
  eq_of_veq <| Multiset.singleton_bind _ _


lemma disjiUnion_disjiUnion (s : Finset α) (f : α → Finset β) (g : β → Finset γ) (h1 h2) :
    (s.disjiUnion f h1).disjiUnion g h2 =
      s.attach.disjiUnion
        (fun a ↦ ((f a).disjiUnion g) fun _ hb _ hc ↦
            h2 (mem_disjiUnion.mpr ⟨_, a.prop, hb⟩) (mem_disjiUnion.mpr ⟨_, a.prop, hc⟩))
        fun a _ b _ hab ↦
        disjoint_left.mpr fun x hxa hxb ↦ by
          /-
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            s✝ s₁ s₂ : Finset α
            t t₁ t₂ : α → Finset β
            s : Finset α
            f : α → Finset β
            g : β → Finset γ
            h1 : (↑s).PairwiseDisjoint f
            h2 : (↑(s.disjiUnion f h1)).PairwiseDisjoint g
            a : Subtype fun x => Membership.mem s x
            x✝¹ : Membership.mem (↑s.attach) a
            b : Subtype fun x => Membership.mem s x
            x✝ : Membership.mem (↑s.attach) b
            hab : Ne a b
            x : γ
            hxa : Membership.mem ((fun a => (f ↑a).disjiUnion g ⋯) a) x
            hxb : Membership.mem ((fun a => (f ↑a).disjiUnion g ⋯) b) x
            ⊢ False
          -/
          obtain ⟨xa, hfa, hga⟩ := mem_disjiUnion.mp hxa
          /-
            case intro.intro
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            s✝ s₁ s₂ : Finset α
            t t₁ t₂ : α → Finset β
            s : Finset α
            f : α → Finset β
            g : β → Finset γ
            h1 : (↑s).PairwiseDisjoint f
            h2 : (↑(s.disjiUnion f h1)).PairwiseDisjoint g
            a : Subtype fun x => Membership.mem s x
            x✝¹ : Membership.mem (↑s.attach) a
            b : Subtype fun x => Membership.mem s x
            x✝ : Membership.mem (↑s.attach) b
            hab : Ne a b
            x : γ
            hxa : Membership.mem ((fun a => (f ↑a).disjiUnion g ⋯) a) x
            hxb : Membership.mem ((fun a => (f ↑a).disjiUnion g ⋯) b) x
            xa : β
            hfa : Membership.mem (f ↑a) xa
            hga : Membership.mem (g xa) x
            ⊢ False
          -/
          obtain ⟨xb, hfb, hgb⟩ := mem_disjiUnion.mp hxb
          refine disjoint_left.mp
            (h2 (mem_disjiUnion.mpr ⟨_, a.prop, hfa⟩) (mem_disjiUnion.mpr ⟨_, b.prop, hfb⟩) ?_) hga
            hgb
          /-
            case intro.intro.intro.intro
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            s✝ s₁ s₂ : Finset α
            t t₁ t₂ : α → Finset β
            s : Finset α
            f : α → Finset β
            g : β → Finset γ
            h1 : (↑s).PairwiseDisjoint f
            h2 : (↑(s.disjiUnion f h1)).PairwiseDisjoint g
            a : Subtype fun x => Membership.mem s x
            x✝¹ : Membership.mem (↑s.attach) a
            b : Subtype fun x => Membership.mem s x
            x✝ : Membership.mem (↑s.attach) b
            hab : Ne a b
            x : γ
            hxa : Membership.mem ((fun a => (f ↑a).disjiUnion g ⋯) a) x
            hxb : Membership.mem ((fun a => (f ↑a).disjiUnion g ⋯) b) x
            xa : β
            hfa : Membership.mem (f ↑a) xa
            hga : Membership.mem (g xa) x
            xb : β
            hfb : Membership.mem (f ↑b) xb
            hgb : Membership.mem (g xb) x
            ⊢ Ne xa xb
          -/
          rintro rfl
          /-
            case intro.intro.intro.intro
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            s✝ s₁ s₂ : Finset α
            t t₁ t₂ : α → Finset β
            s : Finset α
            f : α → Finset β
            g : β → Finset γ
            h1 : (↑s).PairwiseDisjoint f
            h2 : (↑(s.disjiUnion f h1)).PairwiseDisjoint g
            a : Subtype fun x => Membership.mem s x
            x✝¹ : Membership.mem (↑s.attach) a
            b : Subtype fun x => Membership.mem s x
            x✝ : Membership.mem (↑s.attach) b
            hab : Ne a b
            x : γ
            hxa : Membership.mem ((fun a => (f ↑a).disjiUnion g ⋯) a) x
            hxb : Membership.mem ((fun a => (f ↑a).disjiUnion g ⋯) b) x
            xa : β
            hfa : Membership.mem (f ↑a) xa
            hga : Membership.mem (g xa) x
            hfb : Membership.mem (f ↑b) xa
            hgb : Membership.mem (g xa) x
            ⊢ False
          -/
          exact disjoint_left.mp (h1 a.prop b.prop <| Subtype.coe_injective.ne hab) hfa hfb :=
          /-
            🎉 no goals
          -/
  eq_of_veq <| Multiset.bind_assoc.trans (Multiset.attach_bind_coe _ _).symm


private lemma pairwiseDisjoint_fibers : Set.PairwiseDisjoint ↑t fun a ↦ s.filter (f · = a) :=
  fun x' hx y' hy hne ↦ by
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : DecidableEq β
      s : Finset α
      t : Finset β
      f : α → β
      x' : β
      hx : Membership.mem (↑t) x'
      y' : β
      hy : Membership.mem (↑t) y'
      hne : Ne x' y'
      ⊢ Function.onFun Disjoint (fun a => Finset.filter (fun x => Eq (f x) a) s) x' y'
    -/
    simp_rw [disjoint_left, mem_filter]; rintro i ⟨_, rfl⟩ ⟨_, rfl⟩; exact hne rfl
                                                                     /-
                                                                       🎉 no goals
                                                                     -/

-- `simpNF` claims that the statement can't simplify itself, but it can (as of 2024-02-14)

@[simp, nolint simpNF] lemma disjiUnion_filter_eq (s : Finset α) (t : Finset β) (f : α → β) :
    t.disjiUnion (fun a ↦ s.filter (f · = a)) pairwiseDisjoint_fibers =
      s.filter fun c ↦ f c ∈ t :=
                  /-
                    α : Type u_1
                    β : Type u_2
                    inst✝ : DecidableEq β
                    s : Finset α
                    t : Finset β
                    f : α → β
                    b : α
                    ⊢ Iff (Membership.mem (t.disjiUnion (fun a => Finset.filter (fun x => Eq (f x) …
                  -/
  ext fun b => by simpa using and_comm
                  /-
                    🎉 no goals
                  -/


lemma disjiUnion_filter_eq_of_maps_to (h : ∀ x ∈ s, f x ∈ t) :
    t.disjiUnion (fun a ↦ s.filter (f · = a)) pairwiseDisjoint_fibers = s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Finset α
    t : Finset β
    f : α → β
    h : ∀ (x : α), Membership.mem s x → Membership.mem t (f x)
    ⊢ Eq (t.disjiUnion (fun a => Finset.filter (fun x => Eq (f x) a) s) ⋯) s
  -/
  simpa [filter_eq_self]
  /-
    🎉 no goals
  -/


/-- `Finset.biUnion s t` is the union of `t a` over `a ∈ s`.

(This was formerly `bind` due to the monad structure on types with `DecidableEq`.) -/
protected def biUnion (s : Finset α) (t : α → Finset β) : Finset β :=
  (s.1.bind fun a ↦ (t a).1).toFinset


@[simp] lemma biUnion_val (s : Finset α) (t : α → Finset β) :
    (s.biUnion t).1 = (s.1.bind fun a ↦ (t a).1).dedup := rfl


@[simp] lemma biUnion_empty : Finset.biUnion ∅ t = ∅ := rfl


@[simp] lemma mem_biUnion {b : β} : b ∈ s.biUnion t ↔ ∃ a ∈ s, b ∈ t a := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : α → Finset β
    inst✝ : DecidableEq β
    b : β
    ⊢ Iff (Membership.mem (s.biUnion t) b) (Exists fun a => And (Membership.mem s  …
  -/
  simp only [mem_def, biUnion_val, Multiset.mem_dedup, Multiset.mem_bind, exists_prop]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
lemma coe_biUnion : (s.biUnion t : Set β) = ⋃ x ∈ (s : Set α), t x := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : α → Finset β
    inst✝ : DecidableEq β
    ⊢ Eq (↑(s.biUnion t)) (Set.iUnion fun x => Set.iUnion fun h => ↑(t x))
  -/
  simp [Set.ext_iff, mem_biUnion, Set.mem_iUnion, mem_coe, imp_true_iff]
  /-
    🎉 no goals
  -/


@[simp]
lemma biUnion_insert [DecidableEq α] {a : α} : (insert a s).biUnion t = t a ∪ s.biUnion t :=
  ext fun x ↦ by
    simp only [mem_biUnion, exists_prop, mem_union, mem_insert, or_and_right, exists_or,
      exists_eq_left]


lemma biUnion_congr (hs : s₁ = s₂) (ht : ∀ a ∈ s₁, t₁ a = t₂ a) : s₁.biUnion t₁ = s₂.biUnion t₂ :=
  ext fun x ↦ by
    -- Porting note: this entire proof was `simp [or_and_right, exists_or]`
    /-
      α : Type u_1
      β : Type u_2
      s₁ s₂ : Finset α
      t₁ t₂ : α → Finset β
      inst✝ : DecidableEq β
      hs : Eq s₁ s₂
      ht : ∀ (a : α), Membership.mem s₁ a → Eq (t₁ a) (t₂ a)
      x : β
      ⊢ Iff (Membership.mem (s₁.biUnion t₁) x) (Membership.mem (s₂.biUnion t₂) x)
    -/
    simp_rw [mem_biUnion]
    /-
      α : Type u_1
      β : Type u_2
      s₁ s₂ : Finset α
      t₁ t₂ : α → Finset β
      inst✝ : DecidableEq β
      hs : Eq s₁ s₂
      ht : ∀ (a : α), Membership.mem s₁ a → Eq (t₁ a) (t₂ a)
      x : β
      ⊢ Iff (Exists fun a => And (Membership.mem s₁ a) (Membership.mem (t₁ a) x)) (E …
    -/
    apply exists_congr
    /-
      case h
      α : Type u_1
      β : Type u_2
      s₁ s₂ : Finset α
      t₁ t₂ : α → Finset β
      inst✝ : DecidableEq β
      hs : Eq s₁ s₂
      ht : ∀ (a : α), Membership.mem s₁ a → Eq (t₁ a) (t₂ a)
      x : β
      ⊢ ∀ (a : α), Iff (And (Membership.mem s₁ a) (Membership.mem (t₁ a) x)) (And (M …
    -/
    simp +contextual only [hs, and_congr_right_iff, ht, implies_true]
    /-
      🎉 no goals
    -/


@[simp]
lemma disjiUnion_eq_biUnion (s : Finset α) (f : α → Finset β) (hf) :
    s.disjiUnion f hf = s.biUnion f := eq_of_veq (s.disjiUnion f hf).nodup.dedup.symm


lemma biUnion_subset {s' : Finset β} : s.biUnion t ⊆ s' ↔ ∀ x ∈ s, t x ⊆ s' := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : α → Finset β
    inst✝ : DecidableEq β
    s' : Finset β
    ⊢ Iff (HasSubset.Subset (s.biUnion t) s') (∀ (x : α), Membership.mem s x → Has …
  -/
  simp only [subset_iff, mem_biUnion]
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : α → Finset β
    inst✝ : DecidableEq β
    s' : Finset β
    ⊢ Iff (∀ ⦃x : β⦄, (Exists fun a => And (Membership.mem s a) (Membership.mem (t …
  -/
  exact ⟨fun H a ha b hb ↦ H ⟨a, ha, hb⟩, fun H b ⟨a, ha, hb⟩ ↦ H a ha hb⟩
  /-
    🎉 no goals
  -/


@[simp]
lemma singleton_biUnion {a : α} : Finset.biUnion {a} t = t a := by
  /-
    α : Type u_1
    β : Type u_2
    t : α → Finset β
    inst✝ : DecidableEq β
    a : α
    ⊢ Eq ((Singleton.singleton a).biUnion t) (t a)
  -/
  classical rw [← insert_emptyc_eq, biUnion_insert, biUnion_empty, union_empty]
  /-
    🎉 no goals
  -/


lemma biUnion_inter (s : Finset α) (f : α → Finset β) (t : Finset β) :
    s.biUnion f ∩ t = s.biUnion fun x ↦ f x ∩ t := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Finset α
    f : α → Finset β
    t : Finset β
    ⊢ Eq (Inter.inter (s.biUnion f) t) (s.biUnion fun x => Inter.inter (f x) t)
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Finset α
    f : α → Finset β
    t : Finset β
    x : β
    ⊢ Iff (Membership.mem (Inter.inter (s.biUnion f) t) x) (Membership.mem (s.biUn …
  -/
  simp only [mem_biUnion, mem_inter]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Finset α
    f : α → Finset β
    t : Finset β
    x : β
    ⊢ Iff (And (Exists fun a => And (Membership.mem s a) (Membership.mem (f a) x)) …
  -/
  tauto
  /-
    🎉 no goals
  -/


lemma inter_biUnion (t : Finset β) (s : Finset α) (f : α → Finset β) :
    t ∩ s.biUnion f = s.biUnion fun x ↦ t ∩ f x := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    t : Finset β
    s : Finset α
    f : α → Finset β
    ⊢ Eq (Inter.inter t (s.biUnion f)) (s.biUnion fun x => Inter.inter t (f x))
  -/
  rw [inter_comm, biUnion_inter]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    t : Finset β
    s : Finset α
    f : α → Finset β
    ⊢ Eq (s.biUnion fun x => Inter.inter (f x) t) (s.biUnion fun x => Inter.inter  …
  -/
  simp [inter_comm]
  /-
    🎉 no goals
  -/


lemma biUnion_biUnion [DecidableEq γ] (s : Finset α) (f : α → Finset β) (g : β → Finset γ) :
    (s.biUnion f).biUnion g = s.biUnion fun a ↦ (f a).biUnion g := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq γ
    s : Finset α
    f : α → Finset β
    g : β → Finset γ
    ⊢ Eq ((s.biUnion f).biUnion g) (s.biUnion fun a => (f a).biUnion g)
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq γ
    s : Finset α
    f : α → Finset β
    g : β → Finset γ
    a✝ : γ
    ⊢ Iff (Membership.mem ((s.biUnion f).biUnion g) a✝) (Membership.mem (s.biUnion …
  -/
  simp only [Finset.mem_biUnion, exists_prop]
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq γ
    s : Finset α
    f : α → Finset β
    g : β → Finset γ
    a✝ : γ
    ⊢ Iff (Exists fun a => And (Exists fun a_1 => And (Membership.mem s a_1) (Memb …
  -/
  simp_rw [← exists_and_right, ← exists_and_left, and_assoc]
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq γ
    s : Finset α
    f : α → Finset β
    g : β → Finset γ
    a✝ : γ
    ⊢ Iff (Exists fun a => Exists fun x => And (Membership.mem s x) (And (Membersh …
  -/
  rw [exists_comm]
  /-
    🎉 no goals
  -/


lemma bind_toFinset [DecidableEq α] (s : Multiset α) (t : α → Multiset β) :
    (s.bind t).toFinset = s.toFinset.biUnion fun a ↦ (t a).toFinset :=
                 /-
                   α : Type u_1
                   β : Type u_2
                   inst✝¹ : DecidableEq β
                   inst✝ : DecidableEq α
                   s : Multiset α
                   t : α → Multiset β
                   x : β
                   ⊢ Iff (Membership.mem (s.bind t).toFinset x) (Membership.mem (s.toFinset.biUni …
                 -/
  ext fun x ↦ by simp only [Multiset.mem_toFinset, mem_biUnion, Multiset.mem_bind, exists_prop]
                 /-
                   🎉 no goals
                 -/


lemma biUnion_mono (h : ∀ a ∈ s, t₁ a ⊆ t₂ a) : s.biUnion t₁ ⊆ s.biUnion t₂ := by
  have : ∀ b a, a ∈ s → b ∈ t₁ a → ∃ a : α, a ∈ s ∧ b ∈ t₂ a := fun b a ha hb ↦
    ⟨a, ha, Finset.mem_of_subset (h a ha) hb⟩
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t₁ t₂ : α → Finset β
    inst✝ : DecidableEq β
    h : ∀ (a : α), Membership.mem s a → HasSubset.Subset (t₁ a) (t₂ a)
    this : ∀ (b : β) (a : α), Membership.mem s a → Membership.mem (t₁ a) b → Exist …
    ⊢ HasSubset.Subset (s.biUnion t₁) (s.biUnion t₂)
  -/
  simpa only [subset_iff, mem_biUnion, exists_imp, and_imp, exists_prop]
  /-
    🎉 no goals
  -/


lemma biUnion_subset_biUnion_of_subset_left (t : α → Finset β) (h : s₁ ⊆ s₂) :
    s₁.biUnion t ⊆ s₂.biUnion t := fun x ↦ by
  /-
    α : Type u_1
    β : Type u_2
    s₁ s₂ : Finset α
    inst✝ : DecidableEq β
    t : α → Finset β
    h : HasSubset.Subset s₁ s₂
    x : β
    ⊢ Membership.mem (s₁.biUnion t) x → Membership.mem (s₂.biUnion t) x
  -/
  simp only [and_imp, mem_biUnion, exists_prop]; exact Exists.imp fun a ha ↦ ⟨h ha.1, ha.2⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


lemma subset_biUnion_of_mem (u : α → Finset β) {x : α} (xs : x ∈ s) : u x ⊆ s.biUnion u :=
  singleton_biUnion.superset.trans <|
    biUnion_subset_biUnion_of_subset_left u <| singleton_subset_iff.2 xs


@[simp]
lemma biUnion_subset_iff_forall_subset {α β : Type*} [DecidableEq β] {s : Finset α}
    {t : Finset β} {f : α → Finset β} : s.biUnion f ⊆ t ↔ ∀ x ∈ s, f x ⊆ t :=
  ⟨fun h _ hx ↦ (subset_biUnion_of_mem f hx).trans h, fun h _ hx ↦
    let ⟨_, ha₁, ha₂⟩ := mem_biUnion.mp hx
    h _ ha₁ ha₂⟩


@[simp]
lemma biUnion_singleton_eq_self [DecidableEq α] : s.biUnion (singleton : α → Finset α) = s :=
                 /-
                   α : Type u_1
                   s : Finset α
                   inst✝ : DecidableEq α
                   x : α
                   ⊢ Iff (Membership.mem (s.biUnion Singleton.singleton) x) (Membership.mem s x)
                 -/
  ext fun x ↦ by simp only [mem_biUnion, mem_singleton, exists_prop, exists_eq_right']
                 /-
                   🎉 no goals
                 -/


lemma filter_biUnion (s : Finset α) (f : α → Finset β) (p : β → Prop) [DecidablePred p] :
    (s.biUnion f).filter p = s.biUnion fun a ↦ (f a).filter p := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    s : Finset α
    f : α → Finset β
    p : β → Prop
    inst✝ : DecidablePred p
    ⊢ Eq (Finset.filter p (s.biUnion f)) (s.biUnion fun a => Finset.filter p (f a))
  -/
  ext b
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    s : Finset α
    f : α → Finset β
    p : β → Prop
    inst✝ : DecidablePred p
    b : β
    ⊢ Iff (Membership.mem (Finset.filter p (s.biUnion f)) b) (Membership.mem (s.bi …
  -/
  simp only [mem_biUnion, exists_prop, mem_filter]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    s : Finset α
    f : α → Finset β
    p : β → Prop
    inst✝ : DecidablePred p
    b : β
    ⊢ Iff (And (Exists fun a => And (Membership.mem s a) (Membership.mem (f a) b)) …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq β
      s : Finset α
      f : α → Finset β
      p : β → Prop
      inst✝ : DecidablePred p
      b : β
      ⊢ And (Exists fun a => And (Membership.mem s a) (Membership.mem (f a) b)) (p b …
    -/
  · rintro ⟨⟨a, ha, hba⟩, hb⟩
    /-
      case h.mp.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq β
      s : Finset α
      f : α → Finset β
      p : β → Prop
      inst✝ : DecidablePred p
      b : β
      hb : p b
      a : α
      ha : Membership.mem s a
      hba : Membership.mem (f a) b
      ⊢ Exists fun a => And (Membership.mem s a) (And (Membership.mem (f a) b) (p b))
    -/
    exact ⟨a, ha, hba, hb⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq β
      s : Finset α
      f : α → Finset β
      p : β → Prop
      inst✝ : DecidablePred p
      b : β
      ⊢ (Exists fun a => And (Membership.mem s a) (And (Membership.mem (f a) b) (p b …
    -/
  · rintro ⟨a, ha, hba, hb⟩
    /-
      case h.mpr.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq β
      s : Finset α
      f : α → Finset β
      p : β → Prop
      inst✝ : DecidablePred p
      b : β
      a : α
      ha : Membership.mem s a
      hba : Membership.mem (f a) b
      hb : p b
      ⊢ And (Exists fun a => And (Membership.mem s a) (Membership.mem (f a) b)) (p b)
    -/
    exact ⟨⟨a, ha, hba⟩, hb⟩
    /-
      🎉 no goals
    -/


lemma biUnion_filter_eq_of_maps_to [DecidableEq α] {s : Finset α} {t : Finset β} {f : α → β}
    (h : ∀ x ∈ s, f x ∈ t) : (t.biUnion fun a ↦ s.filter fun c ↦ f c = a) = s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq α
    s : Finset α
    t : Finset β
    f : α → β
    h : ∀ (x : α), Membership.mem s x → Membership.mem t (f x)
    ⊢ Eq (t.biUnion fun a => Finset.filter (fun c => Eq (f c) a) s) s
  -/
  simpa only [disjiUnion_eq_biUnion] using disjiUnion_filter_eq_of_maps_to h
  /-
    🎉 no goals
  -/


lemma erase_biUnion (f : α → Finset β) (s : Finset α) (b : β) :
    (s.biUnion f).erase b = s.biUnion fun x ↦ (f x).erase b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f : α → Finset β
    s : Finset α
    b : β
    ⊢ Eq ((s.biUnion f).erase b) (s.biUnion fun x => (f x).erase b)
  -/
  ext a
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f : α → Finset β
    s : Finset α
    b a : β
    ⊢ Iff (Membership.mem ((s.biUnion f).erase b) a) (Membership.mem (s.biUnion fu …
  -/
  simp only [mem_biUnion, not_exists, not_and, mem_erase, ne_eq]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f : α → Finset β
    s : Finset α
    b a : β
    ⊢ Iff (And (Not (Eq a b)) (Exists fun a_1 => And (Membership.mem s a_1) (Membe …
  -/
  tauto
  /-
    🎉 no goals
  -/


@[simp]
lemma biUnion_nonempty : (s.biUnion t).Nonempty ↔ ∃ x ∈ s, (t x).Nonempty := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : α → Finset β
    inst✝ : DecidableEq β
    ⊢ Iff (s.biUnion t).Nonempty (Exists fun x => And (Membership.mem s x) (t x).N …
  -/
  simp only [Finset.Nonempty, mem_biUnion]
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : α → Finset β
    inst✝ : DecidableEq β
    ⊢ Iff (Exists fun x => Exists fun a => And (Membership.mem s a) (Membership.me …
  -/
  rw [exists_swap]
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : α → Finset β
    inst✝ : DecidableEq β
    ⊢ Iff (Exists fun y => Exists fun x => And (Membership.mem s y) (Membership.me …
  -/
  simp [exists_and_left]
  /-
    🎉 no goals
  -/


lemma Nonempty.biUnion (hs : s.Nonempty) (ht : ∀ x ∈ s, (t x).Nonempty) :
    (s.biUnion t).Nonempty := biUnion_nonempty.2 <| hs.imp fun x hx ↦ ⟨hx, ht x hx⟩


lemma disjoint_biUnion_left (s : Finset α) (f : α → Finset β) (t : Finset β) :
    Disjoint (s.biUnion f) t ↔ ∀ i ∈ s, Disjoint (f i) t := by
  classical
  refine s.induction ?_ ?_
  · simp only [forall_mem_empty_iff, biUnion_empty, disjoint_empty_left]
  · intro i s his ih
    simp only [disjoint_union_left, biUnion_insert, his, forall_mem_insert, ih]


lemma disjoint_biUnion_right (s : Finset β) (t : Finset α) (f : α → Finset β) :
    Disjoint s (t.biUnion f) ↔ ∀ i ∈ t, Disjoint s (f i) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Finset β
    t : Finset α
    f : α → Finset β
    ⊢ Iff (Disjoint s (t.biUnion f)) (∀ (i : α), Membership.mem t i → Disjoint s ( …
  -/
  simpa only [_root_.disjoint_comm] using disjoint_biUnion_left t f s
  /-
    🎉 no goals
  -/


