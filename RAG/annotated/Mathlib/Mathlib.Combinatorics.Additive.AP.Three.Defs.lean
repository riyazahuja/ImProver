/-- A set is **3GP-free** if it does not contain any non-trivial geometric progression of length
three. -/
@[to_additive "A set is **3AP-free** if it does not contain any non-trivial arithmetic progression
of length three.

This is also sometimes called a **non averaging set** or **Salem-Spencer set**."]
def ThreeGPFree : Prop := ∀ ⦃a⦄, a ∈ s → ∀ ⦃b⦄, b ∈ s → ∀ ⦃c⦄, c ∈ s → a * c = b * b → a = b


/-- Whether a given finset is 3GP-free is decidable. -/
@[to_additive "Whether a given finset is 3AP-free is decidable."]
instance ThreeGPFree.instDecidable [DecidableEq α] {s : Finset α} :
    Decidable (ThreeGPFree (s : Set α)) :=
  decidable_of_iff (∀ a ∈ s, ∀ b ∈ s, ∀ c ∈ s, a * c = b * b → a = b) Iff.rfl


@[to_additive]
theorem ThreeGPFree.mono (h : t ⊆ s) (hs : ThreeGPFree s) : ThreeGPFree t :=
  fun _ ha _ hb _ hc ↦ hs (h ha) (h hb) (h hc)


@[to_additive (attr := simp)]
theorem threeGPFree_empty : ThreeGPFree (∅ : Set α) := fun _ _ _ ha => ha.elim


@[to_additive]
theorem Set.Subsingleton.threeGPFree (hs : s.Subsingleton) : ThreeGPFree s :=
  fun _ ha _ hb _ _ _ ↦ hs ha hb


@[to_additive (attr := simp)]
theorem threeGPFree_singleton (a : α) : ThreeGPFree ({a} : Set α) :=
  subsingleton_singleton.threeGPFree


@[to_additive ThreeAPFree.prod]
theorem ThreeGPFree.prod {t : Set β} (hs : ThreeGPFree s) (ht : ThreeGPFree t) :
    ThreeGPFree (s ×ˢ t) := fun _ ha _ hb _ hc h ↦
  Prod.ext (hs ha.1 hb.1 hc.1 (Prod.ext_iff.1 h).1) (ht ha.2 hb.2 hc.2 (Prod.ext_iff.1 h).2)


@[to_additive]
theorem threeGPFree_pi {ι : Type*} {α : ι → Type*} [∀ i, Monoid (α i)] {s : ∀ i, Set (α i)}
    (hs : ∀ i, ThreeGPFree (s i)) : ThreeGPFree ((univ : Set ι).pi s) :=
  fun _ ha _ hb _ hc h ↦
  funext fun i => hs i (ha i trivial) (hb i trivial) (hc i trivial) <| congr_fun h i


/-- Geometric progressions of length three are reflected under `2`-Freiman homomorphisms. -/
@[to_additive
"Arithmetic progressions of length three are reflected under `2`-Freiman homomorphisms."]
lemma ThreeGPFree.of_image (hf : IsMulFreimanHom 2 s t f) (hf' : s.InjOn f) (hAs : A ⊆ s)
    (hA : ThreeGPFree (f '' A)) : ThreeGPFree A :=
  fun _ ha _ hb _ hc habc ↦ hf' (hAs ha) (hAs hb) <| hA (mem_image_of_mem _ ha)
    (mem_image_of_mem _ hb) (mem_image_of_mem _ hc) <|
    hf.mul_eq_mul (hAs ha) (hAs hc) (hAs hb) (hAs hb) habc


/-- Geometric progressions of length three are unchanged under `2`-Freiman isomorphisms. -/
@[to_additive
"Arithmetic progressions of length three are unchanged under `2`-Freiman isomorphisms."]
lemma threeGPFree_image (hf : IsMulFreimanIso 2 s t f) (hAs : A ⊆ s) :
    ThreeGPFree (f '' A) ↔ ThreeGPFree A := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : CommMonoid β
    s A : Set α
    t : Set β
    f : α → β
    hf : IsMulFreimanIso 2 s t f
    hAs : HasSubset.Subset A s
    ⊢ Iff (ThreeGPFree (Set.image f A)) (ThreeGPFree A)
  -/
  rw [ThreeGPFree, ThreeGPFree]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : CommMonoid β
    s A : Set α
    t : Set β
    f : α → β
    hf : IsMulFreimanIso 2 s t f
    hAs : HasSubset.Subset A s
    ⊢ Iff (∀ ⦃a : β⦄, Membership.mem (Set.image f A) a → ∀ ⦃b : β⦄, Membership.mem …
  -/
  have := (hf.bijOn.injOn.mono hAs).bijOn_image (f := f)
  simp +contextual only
    [((hf.bijOn.injOn.mono hAs).bijOn_image (f := f)).forall,
    hf.mul_eq_mul (hAs _) (hAs _) (hAs _) (hAs _), this.injOn.eq_iff]


@[to_additive] alias ⟨_, ThreeGPFree.image⟩ := threeGPFree_image


/-- Geometric progressions of length three are reflected under `2`-Freiman homomorphisms. -/
@[to_additive
"Arithmetic progressions of length three are reflected under `2`-Freiman homomorphisms."]
lemma IsMulFreimanHom.threeGPFree (hf : IsMulFreimanHom 2 s t f) (hf' : s.InjOn f)
    (ht : ThreeGPFree t) : ThreeGPFree s :=
  (ht.mono hf.mapsTo.image_subset).of_image hf hf' subset_rfl


/-- Geometric progressions of length three are unchanged under `2`-Freiman isomorphisms. -/
@[to_additive
"Arithmetic progressions of length three are unchanged under `2`-Freiman isomorphisms."]
lemma IsMulFreimanIso.threeGPFree_congr (hf : IsMulFreimanIso 2 s t f) :
    ThreeGPFree s ↔ ThreeGPFree t := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : CommMonoid β
    s : Set α
    t : Set β
    f : α → β
    hf : IsMulFreimanIso 2 s t f
    ⊢ Iff (ThreeGPFree s) (ThreeGPFree t)
  -/
  rw [← threeGPFree_image hf subset_rfl, hf.bijOn.image_eq]
  /-
    🎉 no goals
  -/


/-- Geometric progressions of length three are preserved under semigroup homomorphisms. -/
@[to_additive
"Arithmetic progressions of length three are preserved under semigroup homomorphisms."]
theorem ThreeGPFree.image' [FunLike F α β] [MulHomClass F α β] (f : F) (hf : (s * s).InjOn f)
    (h : ThreeGPFree s) : ThreeGPFree (f '' s) := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : CommMonoid α
    inst✝² : CommMonoid β
    s : Set α
    inst✝¹ : FunLike F α β
    inst✝ : MulHomClass F α β
    f : F
    hf : Set.InjOn (⇑f) (HMul.hMul s s)
    h : ThreeGPFree s
    ⊢ ThreeGPFree (Set.image (⇑f) s)
  -/
  rintro _ ⟨a, ha, rfl⟩ _ ⟨b, hb, rfl⟩ _ ⟨c, hc, rfl⟩ habc
  /-
    case intro.intro.intro.intro.intro.intro
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : CommMonoid α
    inst✝² : CommMonoid β
    s : Set α
    inst✝¹ : FunLike F α β
    inst✝ : MulHomClass F α β
    f : F
    hf : Set.InjOn (⇑f) (HMul.hMul s s)
    h : ThreeGPFree s
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    c : α
    hc : Membership.mem s c
    habc : Eq (HMul.hMul (f a) (f c)) (HMul.hMul (f b) (f b))
    ⊢ Eq (f a) (f b)
  -/
  rw [h ha hb hc (hf (mul_mem_mul ha hc) (mul_mem_mul hb hb) <| by rwa [map_mul, map_mul])]
  /-
    🎉 no goals
  -/


@[to_additive] lemma ThreeGPFree.eq_right (hs : ThreeGPFree s) :
    ∀ ⦃a⦄, a ∈ s → ∀ ⦃b⦄, b ∈ s → ∀ ⦃c⦄, c ∈ s → a * c = b * b → b = c := by
  /-
    α : Type u_2
    inst✝ : CancelCommMonoid α
    s : Set α
    hs : ThreeGPFree s
    ⊢ ∀ ⦃a : α⦄, Membership.mem s a → ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c : α⦄, M …
  -/
  rintro a ha b hb c hc habc
  /-
    α : Type u_2
    inst✝ : CancelCommMonoid α
    s : Set α
    hs : ThreeGPFree s
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    c : α
    hc : Membership.mem s c
    habc : Eq (HMul.hMul a c) (HMul.hMul b b)
    ⊢ Eq b c
  -/
  obtain rfl := hs ha hb hc habc
  /-
    α : Type u_2
    inst✝ : CancelCommMonoid α
    s : Set α
    hs : ThreeGPFree s
    a : α
    ha : Membership.mem s a
    c : α
    hc : Membership.mem s c
    hb : Membership.mem s a
    habc : Eq (HMul.hMul a c) (HMul.hMul a a)
    ⊢ Eq a c
  -/
  simpa using habc.symm
  /-
    🎉 no goals
  -/


@[to_additive] lemma threeGPFree_insert :
    ThreeGPFree (insert a s) ↔ ThreeGPFree s ∧
      (∀ ⦃b⦄, b ∈ s → ∀ ⦃c⦄, c ∈ s → a * c = b * b → a = b) ∧
        ∀ ⦃b⦄, b ∈ s → ∀ ⦃c⦄, c ∈ s → b * c = a * a → b = a := by
  refine ⟨fun hs ↦ ⟨hs.mono (subset_insert _ _),
    fun b hb c hc ↦ hs (Or.inl rfl) (Or.inr hb) (Or.inr hc),
    fun b hb c hc ↦ hs (Or.inr hb) (Or.inl rfl) (Or.inr hc)⟩, ?_⟩
  /-
    α : Type u_2
    inst✝ : CancelCommMonoid α
    s : Set α
    a : α
    ⊢ And (ThreeGPFree s) (And (∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c : α⦄, Members …
  -/
  rintro ⟨hs, ha, ha'⟩ b hb c hc d hd h
  /-
    case intro.intro
    α : Type u_2
    inst✝ : CancelCommMonoid α
    s : Set α
    a : α
    hs : ThreeGPFree s
    ha : ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c : α⦄, Membership.mem s c → Eq (HMul. …
    ha' : ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c : α⦄, Membership.mem s c → Eq (HMul …
    b : α
    hb : Membership.mem (Insert.insert a s) b
    c : α
    hc : Membership.mem (Insert.insert a s) c
    d : α
    hd : Membership.mem (Insert.insert a s) d
    h : Eq (HMul.hMul b d) (HMul.hMul c c)
    ⊢ Eq b c
  -/
  rw [mem_insert_iff] at hb hc hd
  /-
    case intro.intro
    α : Type u_2
    inst✝ : CancelCommMonoid α
    s : Set α
    a : α
    hs : ThreeGPFree s
    ha : ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c : α⦄, Membership.mem s c → Eq (HMul. …
    ha' : ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c : α⦄, Membership.mem s c → Eq (HMul …
    b : α
    hb : Or (Eq b a) (Membership.mem s b)
    c : α
    hc : Or (Eq c a) (Membership.mem s c)
    d : α
    hd : Or (Eq d a) (Membership.mem s d)
    h : Eq (HMul.hMul b d) (HMul.hMul c c)
    ⊢ Eq b c
  -/
  obtain rfl | hb := hb <;> obtain rfl | hc := hc
    /-
      case intro.intro.inl.inl
      α : Type u_2
      inst✝ : CancelCommMonoid α
      s : Set α
      hs : ThreeGPFree s
      c d : α
      h : Eq (HMul.hMul c d) (HMul.hMul c c)
      ha : ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c_1 : α⦄, Membership.mem s c_1 → Eq (H …
      ha' : ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c_1 : α⦄, Membership.mem s c_1 → Eq ( …
      hd : Or (Eq d c) (Membership.mem s d)
      ⊢ Eq c c
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.inl.inr
    α : Type u_2
    inst✝ : CancelCommMonoid α
    s : Set α
    hs : ThreeGPFree s
    b c d : α
    h : Eq (HMul.hMul b d) (HMul.hMul c c)
    ha : ∀ ⦃b_1 : α⦄, Membership.mem s b_1 → ∀ ⦃c : α⦄, Membership.mem s c → Eq (H …
    ha' : ∀ ⦃b_1 : α⦄, Membership.mem s b_1 → ∀ ⦃c : α⦄, Membership.mem s c → Eq ( …
    hd : Or (Eq d b) (Membership.mem s d)
    hc : Membership.mem s c
    ⊢ Eq b c
  -/
  all_goals obtain rfl | hd := hd
    /-
      case intro.intro.inl.inr.inl
      α : Type u_2
      inst✝ : CancelCommMonoid α
      s : Set α
      hs : ThreeGPFree s
      c d : α
      hc : Membership.mem s c
      h : Eq (HMul.hMul d d) (HMul.hMul c c)
      ha : ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c : α⦄, Membership.mem s c → Eq (HMul. …
      ha' : ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c : α⦄, Membership.mem s c → Eq (HMul …
      ⊢ Eq d c
    -/
  · exact (ha' hc hc h.symm).symm
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inl.inr.inr
      α : Type u_2
      inst✝ : CancelCommMonoid α
      s : Set α
      hs : ThreeGPFree s
      b c d : α
      h : Eq (HMul.hMul b d) (HMul.hMul c c)
      ha : ∀ ⦃b_1 : α⦄, Membership.mem s b_1 → ∀ ⦃c : α⦄, Membership.mem s c → Eq (H …
      ha' : ∀ ⦃b_1 : α⦄, Membership.mem s b_1 → ∀ ⦃c : α⦄, Membership.mem s c → Eq ( …
      hc : Membership.mem s c
      hd : Membership.mem s d
      ⊢ Eq b c
    -/
  · exact ha hc hd h
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr.inl.inl
      α : Type u_2
      inst✝ : CancelCommMonoid α
      s : Set α
      hs : ThreeGPFree s
      b d : α
      hb : Membership.mem s b
      h : Eq (HMul.hMul b d) (HMul.hMul d d)
      ha : ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c : α⦄, Membership.mem s c → Eq (HMul. …
      ha' : ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c : α⦄, Membership.mem s c → Eq (HMul …
      ⊢ Eq b d
    -/
  · exact mul_right_cancel h
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr.inl.inr
      α : Type u_2
      inst✝ : CancelCommMonoid α
      s : Set α
      hs : ThreeGPFree s
      b c d : α
      h : Eq (HMul.hMul b d) (HMul.hMul c c)
      hb : Membership.mem s b
      ha : ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c_1 : α⦄, Membership.mem s c_1 → Eq (H …
      ha' : ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c_1 : α⦄, Membership.mem s c_1 → Eq ( …
      hd : Membership.mem s d
      ⊢ Eq b c
    -/
  · exact ha' hb hd h
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr.inr.inl
      α : Type u_2
      inst✝ : CancelCommMonoid α
      s : Set α
      hs : ThreeGPFree s
      b c d : α
      h : Eq (HMul.hMul b d) (HMul.hMul c c)
      hb : Membership.mem s b
      hc : Membership.mem s c
      ha : ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c : α⦄, Membership.mem s c → Eq (HMul. …
      ha' : ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c : α⦄, Membership.mem s c → Eq (HMul …
      ⊢ Eq b c
    -/
  · obtain rfl := ha hc hb ((mul_comm _ _).trans h)
    /-
      case intro.intro.inr.inr.inl
      α : Type u_2
      inst✝ : CancelCommMonoid α
      s : Set α
      hs : ThreeGPFree s
      b d : α
      hb : Membership.mem s b
      ha : ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c : α⦄, Membership.mem s c → Eq (HMul. …
      ha' : ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c : α⦄, Membership.mem s c → Eq (HMul …
      h : Eq (HMul.hMul b d) (HMul.hMul d d)
      hc : Membership.mem s d
      ⊢ Eq b d
    -/
    exact ha' hb hc h
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr.inr.inr
      α : Type u_2
      inst✝ : CancelCommMonoid α
      s : Set α
      a : α
      hs : ThreeGPFree s
      ha : ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c : α⦄, Membership.mem s c → Eq (HMul. …
      ha' : ∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c : α⦄, Membership.mem s c → Eq (HMul …
      b c d : α
      h : Eq (HMul.hMul b d) (HMul.hMul c c)
      hb : Membership.mem s b
      hc : Membership.mem s c
      hd : Membership.mem s d
      ⊢ Eq b c
    -/
  · exact hs hb hc hd h
    /-
      🎉 no goals
    -/


@[to_additive]
theorem ThreeGPFree.smul_set (hs : ThreeGPFree s) : ThreeGPFree (a • s) := by
  /-
    α : Type u_2
    inst✝ : CancelCommMonoid α
    s : Set α
    a : α
    hs : ThreeGPFree s
    ⊢ ThreeGPFree (HSMul.hSMul a s)
  -/
  rintro _ ⟨b, hb, rfl⟩ _ ⟨c, hc, rfl⟩ _ ⟨d, hd, rfl⟩ h
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_2
    inst✝ : CancelCommMonoid α
    s : Set α
    a : α
    hs : ThreeGPFree s
    b : α
    hb : Membership.mem s b
    c : α
    hc : Membership.mem s c
    d : α
    hd : Membership.mem s d
    h : Eq (HMul.hMul ((fun x => HSMul.hSMul a x) b) ((fun x => HSMul.hSMul a x) d …
    ⊢ Eq ((fun x => HSMul.hSMul a x) b) ((fun x => HSMul.hSMul a x) c)
  -/
  exact congr_arg (a • ·) <| hs hb hc hd <| by simpa [mul_mul_mul_comm _ _ a] using h
  /-
    🎉 no goals
  -/


@[to_additive] lemma threeGPFree_smul_set : ThreeGPFree (a • s) ↔ ThreeGPFree s where
  mp hs b hb c hc d hd h := mul_left_cancel
      (hs (mem_image_of_mem _ hb) (mem_image_of_mem _ hc) (mem_image_of_mem _ hd) <| by
        /-
          α : Type u_2
          inst✝ : CancelCommMonoid α
          s : Set α
          a : α
          hs : ThreeGPFree (HSMul.hSMul a s)
          b : α
          hb : Membership.mem s b
          c : α
          hc : Membership.mem s c
          d : α
          hd : Membership.mem s d
          h : Eq (HMul.hMul b d) (HMul.hMul c c)
          ⊢ Eq (HMul.hMul (HSMul.hSMul a b) (HSMul.hSMul a d)) (HMul.hMul (HMul.hMul a c …
        -/
        rw [mul_mul_mul_comm, smul_eq_mul, smul_eq_mul, mul_mul_mul_comm, h])
        /-
          🎉 no goals
        -/
  mpr := ThreeGPFree.smul_set


@[to_additive]
theorem threeGPFree_insert_of_lt (hs : ∀ i ∈ s, i < a) :
    ThreeGPFree (insert a s) ↔
      ThreeGPFree s ∧ ∀ ⦃b⦄, b ∈ s → ∀ ⦃c⦄, c ∈ s → a * c = b * b → a = b := by
  /-
    α : Type u_2
    inst✝ : OrderedCancelCommMonoid α
    s : Set α
    a : α
    hs : ∀ (i : α), Membership.mem s i → LT.lt i a
    ⊢ Iff (ThreeGPFree (Insert.insert a s)) (And (ThreeGPFree s) (∀ ⦃b : α⦄, Membe …
  -/
  refine threeGPFree_insert.trans ?_
  /-
    α : Type u_2
    inst✝ : OrderedCancelCommMonoid α
    s : Set α
    a : α
    hs : ∀ (i : α), Membership.mem s i → LT.lt i a
    ⊢ Iff (And (ThreeGPFree s) (And (∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c : α⦄, Me …
  -/
  rw [← and_assoc]
  /-
    α : Type u_2
    inst✝ : OrderedCancelCommMonoid α
    s : Set α
    a : α
    hs : ∀ (i : α), Membership.mem s i → LT.lt i a
    ⊢ Iff (And (And (ThreeGPFree s) (∀ ⦃b : α⦄, Membership.mem s b → ∀ ⦃c : α⦄, Me …
  -/
  exact and_iff_left fun b hb c hc h => ((mul_lt_mul_of_lt_of_lt (hs _ hb) (hs _ hc)).ne h).elim
  /-
    🎉 no goals
  -/


lemma ThreeGPFree.smul_set₀ (hs : ThreeGPFree s) (ha : a ≠ 0) : ThreeGPFree (a • s) := by
  /-
    α : Type u_2
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NoZeroDivisors α
    s : Set α
    a : α
    hs : ThreeGPFree s
    ha : Ne a 0
    ⊢ ThreeGPFree (HSMul.hSMul a s)
  -/
  rintro _ ⟨b, hb, rfl⟩ _ ⟨c, hc, rfl⟩ _ ⟨d, hd, rfl⟩ h
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_2
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : NoZeroDivisors α
    s : Set α
    a : α
    hs : ThreeGPFree s
    ha : Ne a 0
    b : α
    hb : Membership.mem s b
    c : α
    hc : Membership.mem s c
    d : α
    hd : Membership.mem s d
    h : Eq (HMul.hMul ((fun x => HSMul.hSMul a x) b) ((fun x => HSMul.hSMul a x) d …
    ⊢ Eq ((fun x => HSMul.hSMul a x) b) ((fun x => HSMul.hSMul a x) c)
  -/
  exact congr_arg (a • ·) <| hs hb hc hd <| by simpa [mul_mul_mul_comm _ _ a, ha] using h
  /-
    🎉 no goals
  -/


theorem threeGPFree_smul_set₀ (ha : a ≠ 0) : ThreeGPFree (a • s) ↔ ThreeGPFree s :=
  ⟨fun hs b hb c hc d hd h ↦
    mul_left_cancel₀ ha
      (hs (Set.mem_image_of_mem _ hb) (Set.mem_image_of_mem _ hc) (Set.mem_image_of_mem _ hd) <| by
        /-
          α : Type u_2
          inst✝¹ : CancelCommMonoidWithZero α
          inst✝ : NoZeroDivisors α
          s : Set α
          a : α
          ha : Ne a 0
          hs : ThreeGPFree (HSMul.hSMul a s)
          b : α
          hb : Membership.mem s b
          c : α
          hc : Membership.mem s c
          d : α
          hd : Membership.mem s d
          h : Eq (HMul.hMul b d) (HMul.hMul c c)
          ⊢ Eq (HMul.hMul (HSMul.hSMul a b) (HSMul.hSMul a d)) (HMul.hMul (HMul.hMul a c …
        -/
        rw [smul_eq_mul, smul_eq_mul, mul_mul_mul_comm, h, mul_mul_mul_comm]),
        /-
          🎉 no goals
        -/
    fun hs => hs.smul_set₀ ha⟩


theorem threeAPFree_iff_eq_right {s : Set ℕ} :
    ThreeAPFree s ↔ ∀ ⦃a⦄, a ∈ s → ∀ ⦃b⦄, b ∈ s → ∀ ⦃c⦄, c ∈ s → a + c = b + b → a = c := by
  /-
    s : Set Nat
    ⊢ Iff (ThreeAPFree s) (∀ ⦃a : Nat⦄, Membership.mem s a → ∀ ⦃b : Nat⦄, Membersh …
  -/
  refine forall₄_congr fun a _ha b hb => forall₃_congr fun c hc habc => ⟨?_, ?_⟩
    /-
      case refine_1
      s : Set Nat
      a : Nat
      _ha : Membership.mem s a
      b : Nat
      hb : Membership.mem s b
      c : Nat
      hc : Membership.mem s c
      habc : Eq (HAdd.hAdd a c) (HAdd.hAdd b b)
      ⊢ Eq a b → Eq a c
    -/
  · rintro rfl
    /-
      case refine_1
      s : Set Nat
      a : Nat
      _ha : Membership.mem s a
      c : Nat
      hc : Membership.mem s c
      hb : Membership.mem s a
      habc : Eq (HAdd.hAdd a c) (HAdd.hAdd a a)
      ⊢ Eq a c
    -/
    exact (add_left_cancel habc).symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      s : Set Nat
      a : Nat
      _ha : Membership.mem s a
      b : Nat
      hb : Membership.mem s b
      c : Nat
      hc : Membership.mem s c
      habc : Eq (HAdd.hAdd a c) (HAdd.hAdd b b)
      ⊢ Eq a c → Eq a b
    -/
  · rintro rfl
    /-
      case refine_2
      s : Set Nat
      a : Nat
      _ha : Membership.mem s a
      b : Nat
      hb : Membership.mem s b
      hc : Membership.mem s a
      habc : Eq (HAdd.hAdd a a) (HAdd.hAdd b b)
      ⊢ Eq a b
    -/
    simp_rw [← two_mul] at habc
    /-
      case refine_2
      s : Set Nat
      a : Nat
      _ha : Membership.mem s a
      b : Nat
      hb : Membership.mem s b
      hc : Membership.mem s a
      habc : Eq (HMul.hMul 2 a) (HMul.hMul 2 b)
      ⊢ Eq a b
    -/
    exact mul_left_cancel₀ two_ne_zero habc
    /-
      🎉 no goals
    -/


/-- The multiplicative Roth number of a finset is the cardinality of its biggest 3GP-free subset. -/
@[to_additive "The additive Roth number of a finset is the cardinality of its biggest 3AP-free
subset.

The usual Roth number corresponds to `addRothNumber (Finset.range n)`, see `rothNumberNat`."]
def mulRothNumber : Finset α →o ℕ :=
  ⟨fun s ↦ Nat.findGreatest (fun m ↦ ∃ t ⊆ s, #t = m ∧ ThreeGPFree (t : Set α)) #s, by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝³ : DecidableEq α
      inst✝² : Monoid α
      inst✝¹ : DecidableEq β
      inst✝ : Monoid β
      s t : Finset α
      ⊢ Monotone fun s => Nat.findGreatest (fun m => Exists fun t => And (HasSubset. …
    -/
    rintro t u htu
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝³ : DecidableEq α
      inst✝² : Monoid α
      inst✝¹ : DecidableEq β
      inst✝ : Monoid β
      s t✝ t u : Finset α
      htu : LE.le t u
      ⊢ LE.le ((fun s => Nat.findGreatest (fun m => Exists fun t => And (HasSubset.S …
    -/
    refine Nat.findGreatest_mono (fun m => ?_) (card_le_card htu)
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝³ : DecidableEq α
      inst✝² : Monoid α
      inst✝¹ : DecidableEq β
      inst✝ : Monoid β
      s t✝ t u : Finset α
      htu : LE.le t u
      m : Nat
      ⊢ (Exists fun t_1 => And (HasSubset.Subset t_1 t) (And (Eq t_1.card m) (ThreeG …
    -/
    rintro ⟨v, hvt, hv⟩
    /-
      case intro.intro
      F : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝³ : DecidableEq α
      inst✝² : Monoid α
      inst✝¹ : DecidableEq β
      inst✝ : Monoid β
      s t✝ t u : Finset α
      htu : LE.le t u
      m : Nat
      v : Finset α
      hvt : HasSubset.Subset v t
      hv : And (Eq v.card m) (ThreeGPFree ↑v)
      ⊢ Exists fun t => And (HasSubset.Subset t u) (And (Eq t.card m) (ThreeGPFree ↑ …
    -/
    exact ⟨v, hvt.trans htu, hv⟩⟩
    /-
      🎉 no goals
    -/


@[to_additive]
theorem mulRothNumber_le : mulRothNumber s ≤ #s := Nat.findGreatest_le #s


@[to_additive]
theorem mulRothNumber_spec :
    ∃ t ⊆ s, #t = mulRothNumber s ∧ ThreeGPFree (t : Set α) :=
  Nat.findGreatest_spec (P := fun m ↦ ∃ t ⊆ s, #t = m ∧ ThreeGPFree (t : Set α))
                                                       /-
                                                         α : Type u_2
                                                         inst✝¹ : DecidableEq α
                                                         inst✝ : Monoid α
                                                         s : Finset α
                                                         ⊢ ThreeGPFree ↑EmptyCollection.emptyCollection
                                                       -/
    (Nat.zero_le _) ⟨∅, empty_subset _, card_empty, by norm_cast; exact threeGPFree_empty⟩
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[to_additive]
theorem ThreeGPFree.le_mulRothNumber (hs : ThreeGPFree (s : Set α)) (h : s ⊆ t) :
    #s ≤ mulRothNumber t :=
  Nat.le_findGreatest (card_le_card h) ⟨s, h, rfl, hs⟩


@[to_additive]
theorem ThreeGPFree.mulRothNumber_eq (hs : ThreeGPFree (s : Set α)) :
    mulRothNumber s = #s :=
  (mulRothNumber_le _).antisymm <| hs.le_mulRothNumber <| Subset.refl _


@[to_additive (attr := simp)]
theorem mulRothNumber_empty : mulRothNumber (∅ : Finset α) = 0 :=
  Nat.eq_zero_of_le_zero <| (mulRothNumber_le _).trans card_empty.le


@[to_additive (attr := simp)]
theorem mulRothNumber_singleton (a : α) : mulRothNumber ({a} : Finset α) = 1 := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Monoid α
    a : α
    ⊢ Eq (mulRothNumber (Singleton.singleton a)) 1
  -/
  refine ThreeGPFree.mulRothNumber_eq ?_
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Monoid α
    a : α
    ⊢ ThreeGPFree ↑(Singleton.singleton a)
  -/
  rw [coe_singleton]
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Monoid α
    a : α
    ⊢ ThreeGPFree (Singleton.singleton a)
  -/
  exact threeGPFree_singleton a
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulRothNumber_union_le (s t : Finset α) :
    mulRothNumber (s ∪ t) ≤ mulRothNumber s + mulRothNumber t :=
  let ⟨u, hus, hcard, hu⟩ := mulRothNumber_spec (s ∪ t)
  calc
    mulRothNumber (s ∪ t) = #u := hcard.symm
                               /-
                                 α : Type u_2
                                 inst✝¹ : DecidableEq α
                                 inst✝ : Monoid α
                                 s t u : Finset α
                                 hus : HasSubset.Subset u (Union.union s t)
                                 hcard : Eq u.card (mulRothNumber (Union.union s t))
                                 hu : ThreeGPFree ↑u
                                 ⊢ Eq u.card (Union.union (Inter.inter u s) (Inter.inter u t)).card
                               -/
    _ = #(u ∩ s ∪ u ∩ t) := by rw [← inter_union_distrib_left, inter_eq_left.2 hus]
                               /-
                                 🎉 no goals
                               -/
    _ ≤ #(u ∩ s) + #(u ∩ t) := card_union_le _ _
    _ ≤ mulRothNumber s + mulRothNumber t := _root_.add_le_add
      ((hu.mono inter_subset_left).le_mulRothNumber inter_subset_right)
      ((hu.mono inter_subset_left).le_mulRothNumber inter_subset_right)


@[to_additive]
theorem le_mulRothNumber_product (s : Finset α) (t : Finset β) :
    mulRothNumber s * mulRothNumber t ≤ mulRothNumber (s ×ˢ t) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : Monoid α
    inst✝¹ : DecidableEq β
    inst✝ : Monoid β
    s : Finset α
    t : Finset β
    ⊢ LE.le (HMul.hMul (mulRothNumber s) (mulRothNumber t)) (mulRothNumber (SProd. …
  -/
  obtain ⟨u, hus, hucard, hu⟩ := mulRothNumber_spec s
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : Monoid α
    inst✝¹ : DecidableEq β
    inst✝ : Monoid β
    s : Finset α
    t : Finset β
    u : Finset α
    hus : HasSubset.Subset u s
    hucard : Eq u.card (mulRothNumber s)
    hu : ThreeGPFree ↑u
    ⊢ LE.le (HMul.hMul (mulRothNumber s) (mulRothNumber t)) (mulRothNumber (SProd. …
  -/
  obtain ⟨v, hvt, hvcard, hv⟩ := mulRothNumber_spec t
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : Monoid α
    inst✝¹ : DecidableEq β
    inst✝ : Monoid β
    s : Finset α
    t : Finset β
    u : Finset α
    hus : HasSubset.Subset u s
    hucard : Eq u.card (mulRothNumber s)
    hu : ThreeGPFree ↑u
    v : Finset β
    hvt : HasSubset.Subset v t
    hvcard : Eq v.card (mulRothNumber t)
    hv : ThreeGPFree ↑v
    ⊢ LE.le (HMul.hMul (mulRothNumber s) (mulRothNumber t)) (mulRothNumber (SProd. …
  -/
  rw [← hucard, ← hvcard, ← card_product]
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : Monoid α
    inst✝¹ : DecidableEq β
    inst✝ : Monoid β
    s : Finset α
    t : Finset β
    u : Finset α
    hus : HasSubset.Subset u s
    hucard : Eq u.card (mulRothNumber s)
    hu : ThreeGPFree ↑u
    v : Finset β
    hvt : HasSubset.Subset v t
    hvcard : Eq v.card (mulRothNumber t)
    hv : ThreeGPFree ↑v
    ⊢ LE.le (SProd.sprod u v).card (mulRothNumber (SProd.sprod s t))
  -/
  refine ThreeGPFree.le_mulRothNumber ?_ (product_subset_product hus hvt)
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : Monoid α
    inst✝¹ : DecidableEq β
    inst✝ : Monoid β
    s : Finset α
    t : Finset β
    u : Finset α
    hus : HasSubset.Subset u s
    hucard : Eq u.card (mulRothNumber s)
    hu : ThreeGPFree ↑u
    v : Finset β
    hvt : HasSubset.Subset v t
    hvcard : Eq v.card (mulRothNumber t)
    hv : ThreeGPFree ↑v
    ⊢ ThreeGPFree ↑(SProd.sprod u v)
  -/
  rw [coe_product]
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : Monoid α
    inst✝¹ : DecidableEq β
    inst✝ : Monoid β
    s : Finset α
    t : Finset β
    u : Finset α
    hus : HasSubset.Subset u s
    hucard : Eq u.card (mulRothNumber s)
    hu : ThreeGPFree ↑u
    v : Finset β
    hvt : HasSubset.Subset v t
    hvcard : Eq v.card (mulRothNumber t)
    hv : ThreeGPFree ↑v
    ⊢ ThreeGPFree (SProd.sprod ↑u ↑v)
  -/
  exact hu.prod hv
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulRothNumber_lt_of_forall_not_threeGPFree
    (h : ∀ t ∈ powersetCard n s, ¬ThreeGPFree ((t : Finset α) : Set α)) :
    mulRothNumber s < n := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Monoid α
    s : Finset α
    n : Nat
    h : ∀ (t : Finset α), Membership.mem (Finset.powersetCard n s) t → Not (ThreeG …
    ⊢ LT.lt (mulRothNumber s) n
  -/
  obtain ⟨t, hts, hcard, ht⟩ := mulRothNumber_spec s
  /-
    case intro.intro.intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Monoid α
    s : Finset α
    n : Nat
    h : ∀ (t : Finset α), Membership.mem (Finset.powersetCard n s) t → Not (ThreeG …
    t : Finset α
    hts : HasSubset.Subset t s
    hcard : Eq t.card (mulRothNumber s)
    ht : ThreeGPFree ↑t
    ⊢ LT.lt (mulRothNumber s) n
  -/
  rw [← hcard, ← not_le]
  /-
    case intro.intro.intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Monoid α
    s : Finset α
    n : Nat
    h : ∀ (t : Finset α), Membership.mem (Finset.powersetCard n s) t → Not (ThreeG …
    t : Finset α
    hts : HasSubset.Subset t s
    hcard : Eq t.card (mulRothNumber s)
    ht : ThreeGPFree ↑t
    ⊢ Not (LE.le n t.card)
  -/
  intro hn
  /-
    case intro.intro.intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Monoid α
    s : Finset α
    n : Nat
    h : ∀ (t : Finset α), Membership.mem (Finset.powersetCard n s) t → Not (ThreeG …
    t : Finset α
    hts : HasSubset.Subset t s
    hcard : Eq t.card (mulRothNumber s)
    ht : ThreeGPFree ↑t
    hn : LE.le n t.card
    ⊢ False
  -/
  obtain ⟨u, hut, rfl⟩ := exists_subset_card_eq hn
  /-
    case intro.intro.intro.intro.intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Monoid α
    s t : Finset α
    hts : HasSubset.Subset t s
    hcard : Eq t.card (mulRothNumber s)
    ht : ThreeGPFree ↑t
    u : Finset α
    hut : HasSubset.Subset u t
    h : ∀ (t : Finset α), Membership.mem (Finset.powersetCard u.card s) t → Not (T …
    hn : LE.le u.card t.card
    ⊢ False
  -/
  exact h _ (mem_powersetCard.2 ⟨hut.trans hts, rfl⟩) (ht.mono hut)
  /-
    🎉 no goals
  -/


/-- Arithmetic progressions can be pushed forward along bijective 2-Freiman homs. -/
@[to_additive "Arithmetic progressions can be pushed forward along bijective 2-Freiman homs."]
lemma IsMulFreimanHom.mulRothNumber_mono (hf : IsMulFreimanHom 2 A B f) (hf' : Set.BijOn f A B) :
    mulRothNumber B ≤ mulRothNumber A := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid α
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq β
    A : Finset α
    B : Finset β
    f : α → β
    hf : IsMulFreimanHom 2 (↑A) (↑B) f
    hf' : Set.BijOn f ↑A ↑B
    ⊢ LE.le (mulRothNumber B) (mulRothNumber A)
  -/
  obtain ⟨s, hsB, hcard, hs⟩ := mulRothNumber_spec B
  have hsA : invFunOn f A '' s ⊆ A :=
    (hf'.surjOn.mapsTo_invFunOn.mono (coe_subset.2 hsB) Subset.rfl).image_subset
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid α
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq β
    A : Finset α
    B : Finset β
    f : α → β
    hf : IsMulFreimanHom 2 (↑A) (↑B) f
    hf' : Set.BijOn f ↑A ↑B
    s : Finset β
    hsB : HasSubset.Subset s B
    hcard : Eq s.card (mulRothNumber B)
    hs : ThreeGPFree ↑s
    hsA : HasSubset.Subset (Set.image (Function.invFunOn f ↑A) ↑s) ↑A
    ⊢ LE.le (mulRothNumber B) (mulRothNumber A)
  -/
  have hfsA : Set.SurjOn f A s := hf'.surjOn.mono Subset.rfl (coe_subset.2 hsB)
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid α
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq β
    A : Finset α
    B : Finset β
    f : α → β
    hf : IsMulFreimanHom 2 (↑A) (↑B) f
    hf' : Set.BijOn f ↑A ↑B
    s : Finset β
    hsB : HasSubset.Subset s B
    hcard : Eq s.card (mulRothNumber B)
    hs : ThreeGPFree ↑s
    hsA : HasSubset.Subset (Set.image (Function.invFunOn f ↑A) ↑s) ↑A
    hfsA : Set.SurjOn f ↑A ↑s
    ⊢ LE.le (mulRothNumber B) (mulRothNumber A)
  -/
  rw [← hcard, ← s.card_image_of_injOn ((invFunOn_injOn_image f _).mono hfsA)]
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid α
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq β
    A : Finset α
    B : Finset β
    f : α → β
    hf : IsMulFreimanHom 2 (↑A) (↑B) f
    hf' : Set.BijOn f ↑A ↑B
    s : Finset β
    hsB : HasSubset.Subset s B
    hcard : Eq s.card (mulRothNumber B)
    hs : ThreeGPFree ↑s
    hsA : HasSubset.Subset (Set.image (Function.invFunOn f ↑A) ↑s) ↑A
    hfsA : Set.SurjOn f ↑A ↑s
    ⊢ LE.le (Finset.image (Function.invFunOn f ↑A) s).card (mulRothNumber A)
  -/
  refine ThreeGPFree.le_mulRothNumber ?_ (mod_cast hsA)
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid α
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq β
    A : Finset α
    B : Finset β
    f : α → β
    hf : IsMulFreimanHom 2 (↑A) (↑B) f
    hf' : Set.BijOn f ↑A ↑B
    s : Finset β
    hsB : HasSubset.Subset s B
    hcard : Eq s.card (mulRothNumber B)
    hs : ThreeGPFree ↑s
    hsA : HasSubset.Subset (Set.image (Function.invFunOn f ↑A) ↑s) ↑A
    hfsA : Set.SurjOn f ↑A ↑s
    ⊢ ThreeGPFree ↑(Finset.image (Function.invFunOn f ↑A) s)
  -/
  rw [coe_image]

  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid α
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq β
    A : Finset α
    B : Finset β
    f : α → β
    hf : IsMulFreimanHom 2 (↑A) (↑B) f
    hf' : Set.BijOn f ↑A ↑B
    s : Finset β
    hsB : HasSubset.Subset s B
    hcard : Eq s.card (mulRothNumber B)
    hs : ThreeGPFree ↑s
    hsA : HasSubset.Subset (Set.image (Function.invFunOn f ↑A) ↑s) ↑A
    hfsA : Set.SurjOn f ↑A ↑s
    ⊢ ThreeGPFree (Set.image (Function.invFunOn f ↑A) ↑s)
  -/
  simpa using (hf.subset hsA hfsA.bijOn_subset.mapsTo).threeGPFree (hf'.injOn.mono hsA) hs
  /-
    🎉 no goals
  -/


/-- Arithmetic progressions are preserved under 2-Freiman isos. -/
@[to_additive "Arithmetic progressions are preserved under 2-Freiman isos."]
lemma IsMulFreimanIso.mulRothNumber_congr (hf : IsMulFreimanIso 2 A B f) :
    mulRothNumber A = mulRothNumber B := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid α
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq β
    A : Finset α
    B : Finset β
    f : α → β
    hf : IsMulFreimanIso 2 (↑A) (↑B) f
    ⊢ Eq (mulRothNumber A) (mulRothNumber B)
  -/
  refine le_antisymm ?_ (hf.isMulFreimanHom.mulRothNumber_mono hf.bijOn)
  /-
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid α
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq β
    A : Finset α
    B : Finset β
    f : α → β
    hf : IsMulFreimanIso 2 (↑A) (↑B) f
    ⊢ LE.le (mulRothNumber A) (mulRothNumber B)
  -/
  obtain ⟨s, hsA, hcard, hs⟩ := mulRothNumber_spec A
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid α
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq β
    A : Finset α
    B : Finset β
    f : α → β
    hf : IsMulFreimanIso 2 (↑A) (↑B) f
    s : Finset α
    hsA : HasSubset.Subset s A
    hcard : Eq s.card (mulRothNumber A)
    hs : ThreeGPFree ↑s
    ⊢ LE.le (mulRothNumber A) (mulRothNumber B)
  -/
  rw [← coe_subset] at hsA
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid α
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq β
    A : Finset α
    B : Finset β
    f : α → β
    hf : IsMulFreimanIso 2 (↑A) (↑B) f
    s : Finset α
    hsA : HasSubset.Subset ↑s ↑A
    hcard : Eq s.card (mulRothNumber A)
    hs : ThreeGPFree ↑s
    ⊢ LE.le (mulRothNumber A) (mulRothNumber B)
  -/
  have hfs : Set.InjOn f s := hf.bijOn.injOn.mono hsA
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid α
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq β
    A : Finset α
    B : Finset β
    f : α → β
    hf : IsMulFreimanIso 2 (↑A) (↑B) f
    s : Finset α
    hsA : HasSubset.Subset ↑s ↑A
    hcard : Eq s.card (mulRothNumber A)
    hs : ThreeGPFree ↑s
    hfs : Set.InjOn f ↑s
    ⊢ LE.le (mulRothNumber A) (mulRothNumber B)
  -/
  have := (hf.subset hsA hfs.bijOn_image).threeGPFree_congr.1 hs
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid α
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq β
    A : Finset α
    B : Finset β
    f : α → β
    hf : IsMulFreimanIso 2 (↑A) (↑B) f
    s : Finset α
    hsA : HasSubset.Subset ↑s ↑A
    hcard : Eq s.card (mulRothNumber A)
    hs : ThreeGPFree ↑s
    hfs : Set.InjOn f ↑s
    this : ThreeGPFree (Set.image f ↑s)
    ⊢ LE.le (mulRothNumber A) (mulRothNumber B)
  -/
  rw [← coe_image] at this
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid α
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq β
    A : Finset α
    B : Finset β
    f : α → β
    hf : IsMulFreimanIso 2 (↑A) (↑B) f
    s : Finset α
    hsA : HasSubset.Subset ↑s ↑A
    hcard : Eq s.card (mulRothNumber A)
    hs : ThreeGPFree ↑s
    hfs : Set.InjOn f ↑s
    this : ThreeGPFree ↑(Finset.image f s)
    ⊢ LE.le (mulRothNumber A) (mulRothNumber B)
  -/
  rw [← hcard, ← Finset.card_image_of_injOn hfs]
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid α
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq β
    A : Finset α
    B : Finset β
    f : α → β
    hf : IsMulFreimanIso 2 (↑A) (↑B) f
    s : Finset α
    hsA : HasSubset.Subset ↑s ↑A
    hcard : Eq s.card (mulRothNumber A)
    hs : ThreeGPFree ↑s
    hfs : Set.InjOn f ↑s
    this : ThreeGPFree ↑(Finset.image f s)
    ⊢ LE.le (Finset.image f s).card (mulRothNumber B)
  -/
  refine this.le_mulRothNumber ?_
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid α
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq β
    A : Finset α
    B : Finset β
    f : α → β
    hf : IsMulFreimanIso 2 (↑A) (↑B) f
    s : Finset α
    hsA : HasSubset.Subset ↑s ↑A
    hcard : Eq s.card (mulRothNumber A)
    hs : ThreeGPFree ↑s
    hfs : Set.InjOn f ↑s
    this : ThreeGPFree ↑(Finset.image f s)
    ⊢ HasSubset.Subset (Finset.image f s) B
  -/
  rw [← coe_subset, coe_image]
  /-
    case intro.intro.intro
    α : Type u_2
    β : Type u_3
    inst✝³ : DecidableEq α
    inst✝² : CommMonoid α
    inst✝¹ : CommMonoid β
    inst✝ : DecidableEq β
    A : Finset α
    B : Finset β
    f : α → β
    hf : IsMulFreimanIso 2 (↑A) (↑B) f
    s : Finset α
    hsA : HasSubset.Subset ↑s ↑A
    hcard : Eq s.card (mulRothNumber A)
    hs : ThreeGPFree ↑s
    hfs : Set.InjOn f ↑s
    this : ThreeGPFree ↑(Finset.image f s)
    ⊢ HasSubset.Subset (Set.image f ↑s) ↑B
  -/
  exact (hf.bijOn.mapsTo.mono hsA Subset.rfl).image_subset
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem mulRothNumber_map_mul_left :
    mulRothNumber (s.map <| mulLeftEmbedding a) = mulRothNumber s := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : CancelCommMonoid α
    s : Finset α
    a : α
    ⊢ Eq (mulRothNumber (Finset.map (mulLeftEmbedding a) s)) (mulRothNumber s)
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : CancelCommMonoid α
      s : Finset α
      a : α
      ⊢ LE.le (mulRothNumber (Finset.map (mulLeftEmbedding a) s)) (mulRothNumber s)
    -/
  · obtain ⟨u, hus, hcard, hu⟩ := mulRothNumber_spec (s.map <| mulLeftEmbedding a)
    /-
      case refine_1.intro.intro.intro
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : CancelCommMonoid α
      s : Finset α
      a : α
      u : Finset α
      hus : HasSubset.Subset u (Finset.map (mulLeftEmbedding a) s)
      hcard : Eq u.card (mulRothNumber (Finset.map (mulLeftEmbedding a) s))
      hu : ThreeGPFree ↑u
      ⊢ LE.le (mulRothNumber (Finset.map (mulLeftEmbedding a) s)) (mulRothNumber s)
    -/
    rw [subset_map_iff] at hus
    /-
      case refine_1.intro.intro.intro
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : CancelCommMonoid α
      s : Finset α
      a : α
      u : Finset α
      hus : Exists fun u_1 => And (HasSubset.Subset u_1 s) (Eq u (Finset.map (mulLef …
      hcard : Eq u.card (mulRothNumber (Finset.map (mulLeftEmbedding a) s))
      hu : ThreeGPFree ↑u
      ⊢ LE.le (mulRothNumber (Finset.map (mulLeftEmbedding a) s)) (mulRothNumber s)
    -/
    obtain ⟨u, hus, rfl⟩ := hus
    /-
      case refine_1.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : CancelCommMonoid α
      s : Finset α
      a : α
      u : Finset α
      hus : HasSubset.Subset u s
      hcard : Eq (Finset.map (mulLeftEmbedding a) u).card (mulRothNumber (Finset.map …
      hu : ThreeGPFree ↑(Finset.map (mulLeftEmbedding a) u)
      ⊢ LE.le (mulRothNumber (Finset.map (mulLeftEmbedding a) s)) (mulRothNumber s)
    -/
    rw [coe_map] at hu
    /-
      case refine_1.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : CancelCommMonoid α
      s : Finset α
      a : α
      u : Finset α
      hus : HasSubset.Subset u s
      hcard : Eq (Finset.map (mulLeftEmbedding a) u).card (mulRothNumber (Finset.map …
      hu : ThreeGPFree (Set.image ⇑(mulLeftEmbedding a) ↑u)
      ⊢ LE.le (mulRothNumber (Finset.map (mulLeftEmbedding a) s)) (mulRothNumber s)
    -/
    rw [← hcard, card_map]
    /-
      case refine_1.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : CancelCommMonoid α
      s : Finset α
      a : α
      u : Finset α
      hus : HasSubset.Subset u s
      hcard : Eq (Finset.map (mulLeftEmbedding a) u).card (mulRothNumber (Finset.map …
      hu : ThreeGPFree (Set.image ⇑(mulLeftEmbedding a) ↑u)
      ⊢ LE.le u.card (mulRothNumber s)
    -/
    exact (threeGPFree_smul_set.1 hu).le_mulRothNumber hus
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : CancelCommMonoid α
      s : Finset α
      a : α
      ⊢ LE.le (mulRothNumber s) (mulRothNumber (Finset.map (mulLeftEmbedding a) s))
    -/
  · obtain ⟨u, hus, hcard, hu⟩ := mulRothNumber_spec s
    /-
      case refine_2.intro.intro.intro
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : CancelCommMonoid α
      s : Finset α
      a : α
      u : Finset α
      hus : HasSubset.Subset u s
      hcard : Eq u.card (mulRothNumber s)
      hu : ThreeGPFree ↑u
      ⊢ LE.le (mulRothNumber s) (mulRothNumber (Finset.map (mulLeftEmbedding a) s))
    -/
    have h : ThreeGPFree (u.map <| mulLeftEmbedding a : Set α) := by rw [coe_map]; exact hu.smul_set
    /-
      case refine_2.intro.intro.intro
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : CancelCommMonoid α
      s : Finset α
      a : α
      u : Finset α
      hus : HasSubset.Subset u s
      hcard : Eq u.card (mulRothNumber s)
      hu : ThreeGPFree ↑u
      h : ThreeGPFree ↑(Finset.map (mulLeftEmbedding a) u)
      ⊢ LE.le (mulRothNumber s) (mulRothNumber (Finset.map (mulLeftEmbedding a) s))
    -/
    convert h.le_mulRothNumber (map_subset_map.2 hus) using 1
    /-
      case h.e'_3
      α : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : CancelCommMonoid α
      s : Finset α
      a : α
      u : Finset α
      hus : HasSubset.Subset u s
      hcard : Eq u.card (mulRothNumber s)
      hu : ThreeGPFree ↑u
      h : ThreeGPFree ↑(Finset.map (mulLeftEmbedding a) u)
      ⊢ Eq (mulRothNumber s) (Finset.map (mulLeftEmbedding a) u).card
    -/
    rw [card_map, hcard]
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
theorem mulRothNumber_map_mul_right :
    mulRothNumber (s.map <| mulRightEmbedding a) = mulRothNumber s := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : CancelCommMonoid α
    s : Finset α
    a : α
    ⊢ Eq (mulRothNumber (Finset.map (mulRightEmbedding a) s)) (mulRothNumber s)
  -/
  rw [← mulLeftEmbedding_eq_mulRightEmbedding, mulRothNumber_map_mul_left s a]
  /-
    🎉 no goals
  -/


/-- The Roth number of a natural `N` is the largest integer `m` for which there is a subset of
`range N` of size `m` with no arithmetic progression of length 3.
Trivially, `rothNumberNat N ≤ N`, but Roth's theorem (proved in 1953) shows that
`rothNumberNat N = o(N)` and the construction by Behrend gives a lower bound of the form
`N * exp(-C sqrt(log(N))) ≤ rothNumberNat N`.
A significant refinement of Roth's theorem by Bloom and Sisask announced in 2020 gives
`rothNumberNat N = O(N / (log N)^(1+c))` for an absolute constant `c`. -/
def rothNumberNat : ℕ →o ℕ :=
  ⟨fun n => addRothNumber (range n), addRothNumber.mono.comp range_mono⟩


theorem rothNumberNat_def (n : ℕ) : rothNumberNat n = addRothNumber (range n) :=
  rfl


theorem rothNumberNat_le (N : ℕ) : rothNumberNat N ≤ N :=
  (addRothNumber_le _).trans (card_range _).le


theorem rothNumberNat_spec (n : ℕ) :
    ∃ t ⊆ range n, #t = rothNumberNat n ∧ ThreeAPFree (t : Set ℕ) :=
  addRothNumber_spec _


/-- A verbose specialization of `threeAPFree.le_addRothNumber`, sometimes convenient in
practice. -/
theorem ThreeAPFree.le_rothNumberNat (s : Finset ℕ) (hs : ThreeAPFree (s : Set ℕ))
    (hsn : ∀ x ∈ s, x < n) (hsk : #s = k) : k ≤ rothNumberNat n :=
  hsk.ge.trans <| hs.le_addRothNumber fun x hx => mem_range.2 <| hsn x hx


/-- The Roth number is a subadditive function. Note that by Fekete's lemma this shows that
the limit `rothNumberNat N / N` exists, but Roth's theorem gives the stronger result that this
limit is actually `0`. -/
theorem rothNumberNat_add_le (M N : ℕ) :
    rothNumberNat (M + N) ≤ rothNumberNat M + rothNumberNat N := by
  /-
    M N : Nat
    ⊢ LE.le (rothNumberNat (HAdd.hAdd M N)) (HAdd.hAdd (rothNumberNat M) (rothNumb …
  -/
  simp_rw [rothNumberNat_def]
  /-
    M N : Nat
    ⊢ LE.le (addRothNumber (Finset.range (HAdd.hAdd M N))) (HAdd.hAdd (addRothNumb …
  -/
  rw [range_add_eq_union, ← addRothNumber_map_add_left (range N) M]
  /-
    M N : Nat
    ⊢ LE.le (addRothNumber (Union.union (Finset.range M) (Finset.map (addLeftEmbed …
  -/
  exact addRothNumber_union_le _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem rothNumberNat_zero : rothNumberNat 0 = 0 :=
  rfl


theorem addRothNumber_Ico (a b : ℕ) : addRothNumber (Ico a b) = rothNumberNat (b - a) := by
  /-
    a b : Nat
    ⊢ Eq (addRothNumber (Finset.Ico a b)) (rothNumberNat (HSub.hSub b a))
  -/
  obtain h | h := le_total b a
    /-
      case inl
      a b : Nat
      h : LE.le b a
      ⊢ Eq (addRothNumber (Finset.Ico a b)) (rothNumberNat (HSub.hSub b a))
    -/
  · rw [tsub_eq_zero_of_le h, Ico_eq_empty_of_le h, rothNumberNat_zero, addRothNumber_empty]
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Nat
    h : LE.le a b
    ⊢ Eq (addRothNumber (Finset.Ico a b)) (rothNumberNat (HSub.hSub b a))
  -/
  convert addRothNumber_map_add_left _ a
  /-
    case h.e'_2.h.e'_6
    a b : Nat
    h : LE.le a b
    ⊢ Eq (Finset.Ico a b) (Finset.map (addLeftEmbedding a) (Finset.range (HSub.hSu …
  -/
  rw [range_eq_Ico, map_eq_image]
  /-
    case h.e'_2.h.e'_6
    a b : Nat
    h : LE.le a b
    ⊢ Eq (Finset.Ico a b) (Finset.image (⇑(addLeftEmbedding a)) (Finset.Ico 0 (HSu …
  -/
  convert (image_add_left_Ico 0 (b - a) _).symm
  /-
    case h.e'_2.h.e'_5
    a b : Nat
    h : LE.le a b
    ⊢ Eq b (HAdd.hAdd a (HSub.hSub b a))
  -/
  exact (add_tsub_cancel_of_le h).symm
  /-
    🎉 no goals
  -/


lemma Fin.addRothNumber_eq_rothNumberNat (hkn : 2 * k ≤ n) :
    addRothNumber (Iio k : Finset (Fin n.succ)) = rothNumberNat k :=
  IsAddFreimanIso.addRothNumber_congr <| mod_cast isAddFreimanIso_Iio two_ne_zero hkn


lemma Fin.addRothNumber_le_rothNumberNat (k n : ℕ) (hkn : k ≤ n) :
    addRothNumber (Iio k : Finset (Fin n.succ)) ≤ rothNumberNat k := by
  suffices h : Set.BijOn (Nat.cast : ℕ → Fin n.succ) (range k) (Iio k : Finset (Fin n.succ)) by
    exact (AddMonoidHomClass.isAddFreimanHom (Nat.castRingHom _) h.mapsTo).addRothNumber_mono h
  /-
    k n : Nat
    hkn : LE.le k n
    ⊢ Set.BijOn Nat.cast ↑(Finset.range k) ↑(Finset.Iio ↑k)
  -/
  refine ⟨?_, (CharP.natCast_injOn_Iio _ n.succ).mono (by simp; omega), ?_⟩
    /-
      case refine_1
      k n : Nat
      hkn : LE.le k n
      ⊢ Set.MapsTo Nat.cast ↑(Finset.range k) ↑(Finset.Iio ↑k)
    -/
  · simpa using fun x ↦ natCast_strictMono hkn
    /-
      🎉 no goals
    -/
  simp only [Set.SurjOn, coe_Iio, Set.subset_def, Set.mem_Iio, Set.mem_image, lt_iff_val_lt_val,
    val_cast_of_lt, Nat.lt_succ_iff.2 hkn, coe_range]
  /-
    case refine_2
    k n : Nat
    hkn : LE.le k n
    ⊢ ∀ (x : Fin n.succ), LT.lt (↑x) k → Exists fun x_1 => And (LT.lt x_1 k) (Eq ( …
  -/
  exact fun x hx ↦ ⟨x, hx, by simp⟩
  /-
    🎉 no goals
  -/


