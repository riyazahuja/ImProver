/-- `Set.mulAntidiagonal s t a` is the set of all pairs of an element in `s` and an element in `t`
that multiply to `a`. -/
@[to_additive
      "`Set.addAntidiagonal s t a` is the set of all pairs of an element in `s` and an
      element in `t` that add to `a`."]
def mulAntidiagonal (s t : Set α) (a : α) : Set (α × α) :=
  { x | x.1 ∈ s ∧ x.2 ∈ t ∧ x.1 * x.2 = a }


@[to_additive (attr := simp)]
theorem mem_mulAntidiagonal : x ∈ mulAntidiagonal s t a ↔ x.1 ∈ s ∧ x.2 ∈ t ∧ x.1 * x.2 = a :=
  Iff.rfl


@[to_additive]
theorem mulAntidiagonal_mono_left (h : s₁ ⊆ s₂) : mulAntidiagonal s₁ t a ⊆ mulAntidiagonal s₂ t a :=
  fun _ hx => ⟨h hx.1, hx.2.1, hx.2.2⟩


@[to_additive]
theorem mulAntidiagonal_mono_right (h : t₁ ⊆ t₂) :
    mulAntidiagonal s t₁ a ⊆ mulAntidiagonal s t₂ a := fun _ hx => ⟨hx.1, h hx.2.1, hx.2.2⟩


@[to_additive]
theorem swap_mem_mulAntidiagonal [CommSemigroup α] {s t : Set α} {a : α} {x : α × α} :
    x.swap ∈ Set.mulAntidiagonal s t a ↔ x ∈ Set.mulAntidiagonal t s a := by
  /-
    α : Type u_1
    inst✝ : CommSemigroup α
    s t : Set α
    a : α
    x : Prod α α
    ⊢ Iff (Membership.mem (s.mulAntidiagonal t a) x.swap) (Membership.mem (t.mulAn …
  -/
  simp [mul_comm, and_left_comm]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem swap_mem_mulAntidiagonal_aux [CommSemigroup α] {s t : Set α} {a : α} {x : α × α} :
    x.snd ∈ s ∧ x.fst ∈ t ∧ x.snd * x.fst = a
      ↔ x ∈ Set.mulAntidiagonal t s a := by
  /-
    α : Type u_1
    inst✝ : CommSemigroup α
    s t : Set α
    a : α
    x : Prod α α
    ⊢ Iff (And (Membership.mem s x.2) (And (Membership.mem t x.1) (Eq (HMul.hMul x …
  -/
  simp [mul_comm, and_left_comm]
  /-
    🎉 no goals
  -/



@[to_additive Set.AddAntidiagonal.fst_eq_fst_iff_snd_eq_snd]
theorem fst_eq_fst_iff_snd_eq_snd : (x : α × α).1 = (y : α × α).1 ↔ (x : α × α).2 = (y : α × α).2 :=
  ⟨fun h =>
    mul_left_cancel
      (y.2.2.2.trans <| by
          /-
            α : Type u_1
            inst✝ : CancelCommMonoid α
            s t : Set α
            a : α
            x y : ↑(s.mulAntidiagonal t a)
            h : Eq (↑x).1 (↑y).1
            ⊢ Eq a (HMul.hMul (↑y).1 (↑x).2)
          -/
          rw [← h]
          /-
            α : Type u_1
            inst✝ : CancelCommMonoid α
            s t : Set α
            a : α
            x y : ↑(s.mulAntidiagonal t a)
            h : Eq (↑x).1 (↑y).1
            ⊢ Eq a (HMul.hMul (↑x).1 (↑x).2)
          -/
          exact x.2.2.2.symm).symm,
          /-
            🎉 no goals
          -/
    fun h =>
    mul_right_cancel
      (y.2.2.2.trans <| by
          /-
            α : Type u_1
            inst✝ : CancelCommMonoid α
            s t : Set α
            a : α
            x y : ↑(s.mulAntidiagonal t a)
            h : Eq (↑x).2 (↑y).2
            ⊢ Eq a (HMul.hMul (↑x).1 (↑y).2)
          -/
          rw [← h]
          /-
            α : Type u_1
            inst✝ : CancelCommMonoid α
            s t : Set α
            a : α
            x y : ↑(s.mulAntidiagonal t a)
            h : Eq (↑x).2 (↑y).2
            ⊢ Eq a (HMul.hMul (↑x).1 (↑x).2)
          -/
          exact x.2.2.2.symm).symm⟩
          /-
            🎉 no goals
          -/


@[to_additive Set.AddAntidiagonal.eq_of_fst_eq_fst]
theorem eq_of_fst_eq_fst (h : (x : α × α).fst = (y : α × α).fst) : x = y :=
  Subtype.ext <| Prod.ext h <| fst_eq_fst_iff_snd_eq_snd.1 h


@[to_additive Set.AddAntidiagonal.eq_of_snd_eq_snd]
theorem eq_of_snd_eq_snd (h : (x : α × α).snd = (y : α × α).snd) : x = y :=
  Subtype.ext <| Prod.ext (fst_eq_fst_iff_snd_eq_snd.2 h) h


@[to_additive Set.AddAntidiagonal.eq_of_fst_le_fst_of_snd_le_snd]
theorem eq_of_fst_le_fst_of_snd_le_snd (h₁ : (x : α × α).1 ≤ (y : α × α).1)
    (h₂ : (x : α × α).2 ≤ (y : α × α).2) : x = y :=
  eq_of_fst_eq_fst <|
    h₁.eq_of_not_lt fun hlt =>
      (mul_lt_mul_of_lt_of_le hlt h₂).ne <|
        (mem_mulAntidiagonal.1 x.2).2.2.trans (mem_mulAntidiagonal.1 y.2).2.2.symm


@[to_additive Set.AddAntidiagonal.finite_of_isPWO]
theorem finite_of_isPWO (hs : s.IsPWO) (ht : t.IsPWO) (a) : (mulAntidiagonal s t a).Finite := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoid α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftMono α
    inst✝ : MulRightStrictMono α
    s t : Set α
    hs : s.IsPWO
    ht : t.IsPWO
    a : α
    ⊢ (s.mulAntidiagonal t a).Finite
  -/
  refine not_infinite.1 fun h => ?_
  have h1 : (mulAntidiagonal s t a).PartiallyWellOrderedOn (Prod.fst ⁻¹'o (· ≤ ·)) := fun f hf =>
    hs (Prod.fst ∘ f) fun n => (mem_mulAntidiagonal.1 (hf n)).1
  have h2 : (mulAntidiagonal s t a).PartiallyWellOrderedOn (Prod.snd ⁻¹'o (· ≤ ·)) := fun f hf =>
    ht (Prod.snd ∘ f) fun n => (mem_mulAntidiagonal.1 (hf n)).2.1
  obtain ⟨g, hg⟩ :=
    h1.exists_monotone_subseq (fun n => h.natEmbedding _ n) fun n => (h.natEmbedding _ n).2
  /-
    case intro
    α : Type u_1
    inst✝³ : CancelCommMonoid α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftMono α
    inst✝ : MulRightStrictMono α
    s t : Set α
    hs : s.IsPWO
    ht : t.IsPWO
    a : α
    h : (s.mulAntidiagonal t a).Infinite
    h1 : (s.mulAntidiagonal t a).PartiallyWellOrderedOn (Order.Preimage Prod.fst f …
    h2 : (s.mulAntidiagonal t a).PartiallyWellOrderedOn (Order.Preimage Prod.snd f …
    g : OrderEmbedding Nat Nat
    hg : ∀ (m n : Nat), LE.le m n → Order.Preimage Prod.fst (fun x1 x2 => LE.le x1 …
    ⊢ False
  -/
  obtain ⟨m, n, mn, h2'⟩ := h2 (fun x => (h.natEmbedding _) (g x)) fun n => (h.natEmbedding _ _).2
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝³ : CancelCommMonoid α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftMono α
    inst✝ : MulRightStrictMono α
    s t : Set α
    hs : s.IsPWO
    ht : t.IsPWO
    a : α
    h : (s.mulAntidiagonal t a).Infinite
    h1 : (s.mulAntidiagonal t a).PartiallyWellOrderedOn (Order.Preimage Prod.fst f …
    h2 : (s.mulAntidiagonal t a).PartiallyWellOrderedOn (Order.Preimage Prod.snd f …
    g : OrderEmbedding Nat Nat
    hg : ∀ (m n : Nat), LE.le m n → Order.Preimage Prod.fst (fun x1 x2 => LE.le x1 …
    m n : Nat
    mn : LT.lt m n
    h2' : Order.Preimage Prod.snd (fun x1 x2 => LE.le x1 x2) ↑((Set.Infinite.natEm …
    ⊢ False
  -/
  refine mn.ne (g.injective <| (h.natEmbedding _).injective ?_)
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝³ : CancelCommMonoid α
    inst✝² : PartialOrder α
    inst✝¹ : MulLeftMono α
    inst✝ : MulRightStrictMono α
    s t : Set α
    hs : s.IsPWO
    ht : t.IsPWO
    a : α
    h : (s.mulAntidiagonal t a).Infinite
    h1 : (s.mulAntidiagonal t a).PartiallyWellOrderedOn (Order.Preimage Prod.fst f …
    h2 : (s.mulAntidiagonal t a).PartiallyWellOrderedOn (Order.Preimage Prod.snd f …
    g : OrderEmbedding Nat Nat
    hg : ∀ (m n : Nat), LE.le m n → Order.Preimage Prod.fst (fun x1 x2 => LE.le x1 …
    m n : Nat
    mn : LT.lt m n
    h2' : Order.Preimage Prod.snd (fun x1 x2 => LE.le x1 x2) ↑((Set.Infinite.natEm …
    ⊢ Eq ((Set.Infinite.natEmbedding (s.mulAntidiagonal t a) h) (g m)) ((Set.Infin …
  -/
  exact eq_of_fst_le_fst_of_snd_le_snd _ _ _ (hg _ _ mn.le) h2'
  /-
    🎉 no goals
  -/


@[to_additive Set.AddAntidiagonal.finite_of_isWF]
theorem finite_of_isWF {s t : Set α} (hs : s.IsWF) (ht : t.IsWF)
    (a) : (mulAntidiagonal s t a).Finite :=
  finite_of_isPWO hs.isPWO ht.isPWO a


