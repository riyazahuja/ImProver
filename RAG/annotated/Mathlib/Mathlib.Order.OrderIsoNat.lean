/-- If `f` is a strictly `r`-increasing sequence, then this returns `f` as an order embedding. -/
def natLT (f : ℕ → α) (H : ∀ n : ℕ, r (f n) (f (n + 1))) : ((· < ·) : ℕ → ℕ → Prop) ↪r r :=
  ofMonotone f <| Nat.rel_of_forall_rel_succ_of_lt r H


@[simp]
theorem coe_natLT {f : ℕ → α} {H : ∀ n : ℕ, r (f n) (f (n + 1))} : ⇑(natLT f H) = f :=
  rfl


/-- If `f` is a strictly `r`-decreasing sequence, then this returns `f` as an order embedding. -/
def natGT (f : ℕ → α) (H : ∀ n : ℕ, r (f (n + 1)) (f n)) : ((· > ·) : ℕ → ℕ → Prop) ↪r r :=
  haveI := IsStrictOrder.swap r
  RelEmbedding.swap (natLT f H)


@[simp]
theorem coe_natGT {f : ℕ → α} {H : ∀ n : ℕ, r (f (n + 1)) (f n)} : ⇑(natGT f H) = f :=
  rfl


theorem exists_not_acc_lt_of_not_acc {a : α} {r} (h : ¬Acc r a) : ∃ b, ¬Acc r b ∧ r b a := by
  /-
    α : Type u_1
    a : α
    r : α → α → Prop
    h : Not (Acc r a)
    ⊢ Exists fun b => And (Not (Acc r b)) (r b a)
  -/
  contrapose! h
  /-
    α : Type u_1
    a : α
    r : α → α → Prop
    h : ∀ (b : α), Not (Acc r b) → Not (r b a)
    ⊢ Acc r a
  -/
  refine ⟨_, fun b hr => ?_⟩
  /-
    α : Type u_1
    a : α
    r : α → α → Prop
    h : ∀ (b : α), Not (Acc r b) → Not (r b a)
    b : α
    hr : r b a
    ⊢ Acc r b
  -/
  by_contra hb
  /-
    α : Type u_1
    a : α
    r : α → α → Prop
    h : ∀ (b : α), Not (Acc r b) → Not (r b a)
    b : α
    hr : r b a
    hb : Not (Acc r b)
    ⊢ False
  -/
  exact h b hb hr
  /-
    🎉 no goals
  -/


/-- A value is accessible iff it isn't contained in any infinite decreasing sequence. -/
theorem acc_iff_no_decreasing_seq {x} :
    Acc r x ↔ IsEmpty { f : ((· > ·) : ℕ → ℕ → Prop) ↪r r // x ∈ Set.range f } := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsStrictOrder α r
    x : α
    ⊢ Iff (Acc r x) (IsEmpty (Subtype fun f => Membership.mem (Set.range ⇑f) x))
  -/
  constructor
    /-
      case mp
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsStrictOrder α r
      x : α
      ⊢ Acc r x → IsEmpty (Subtype fun f => Membership.mem (Set.range ⇑f) x)
    -/
  · refine fun h => h.recOn fun x _ IH => ?_
    /-
      case mp
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsStrictOrder α r
      x✝¹ : α
      h : Acc r x✝¹
      x : α
      x✝ : ∀ (y : α), r y x → Acc r y
      IH : ∀ (y : α), r y x → IsEmpty (Subtype fun f => Membership.mem (Set.range ⇑f …
      ⊢ IsEmpty (Subtype fun f => Membership.mem (Set.range ⇑f) x)
    -/
    constructor
    /-
      case mp.false
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsStrictOrder α r
      x✝¹ : α
      h : Acc r x✝¹
      x : α
      x✝ : ∀ (y : α), r y x → Acc r y
      IH : ∀ (y : α), r y x → IsEmpty (Subtype fun f => Membership.mem (Set.range ⇑f …
      ⊢ (Subtype fun f => Membership.mem (Set.range ⇑f) x) → False
    -/
    rintro ⟨f, k, hf⟩
    /-
      case mp.false.mk.intro
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsStrictOrder α r
      x✝¹ : α
      h : Acc r x✝¹
      x : α
      x✝ : ∀ (y : α), r y x → Acc r y
      IH : ∀ (y : α), r y x → IsEmpty (Subtype fun f => Membership.mem (Set.range ⇑f …
      f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) r
      k : Nat
      hf : Eq (f k) x
      ⊢ False
    -/
    exact IsEmpty.elim' (IH (f (k + 1)) (hf ▸ f.map_rel_iff.2 (Nat.lt_succ_self _))) ⟨f, _, rfl⟩
    /-
      🎉 no goals
    -/
  · have : ∀ x : { a // ¬Acc r a }, ∃ y : { a // ¬Acc r a }, r y.1 x.1 := by
      rintro ⟨x, hx⟩
      cases exists_not_acc_lt_of_not_acc hx with
      | intro w h => exact ⟨⟨w, h.1⟩, h.2⟩
    /-
      case mpr
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsStrictOrder α r
      x : α
      this : ∀ (x : Subtype fun a => Not (Acc r a)), Exists fun y => r ↑y ↑x
      ⊢ IsEmpty (Subtype fun f => Membership.mem (Set.range ⇑f) x) → Acc r x
    -/
    choose f h using this
    refine fun E =>
      by_contradiction fun hx => E.elim' ⟨natGT (fun n => (f^[n] ⟨x, hx⟩).1) fun n => ?_, 0, rfl⟩
    /-
      case mpr
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsStrictOrder α r
      x : α
      f : (Subtype fun a => Not (Acc r a)) → Subtype fun a => Not (Acc r a)
      h : ∀ (x : Subtype fun a => Not (Acc r a)), r ↑(f x) ↑x
      E : IsEmpty (Subtype fun f => Membership.mem (Set.range ⇑f) x)
      hx : Not (Acc r x)
      n : Nat
      ⊢ r ((fun n => ↑(Nat.iterate f n ⟨x, hx⟩)) (HAdd.hAdd n 1)) ((fun n => ↑(Nat.i …
    -/
    simp only [Function.iterate_succ']
    /-
      case mpr
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsStrictOrder α r
      x : α
      f : (Subtype fun a => Not (Acc r a)) → Subtype fun a => Not (Acc r a)
      h : ∀ (x : Subtype fun a => Not (Acc r a)), r ↑(f x) ↑x
      E : IsEmpty (Subtype fun f => Membership.mem (Set.range ⇑f) x)
      hx : Not (Acc r x)
      n : Nat
      ⊢ r ↑(Function.comp f (Nat.iterate f n) ⟨x, hx⟩) ↑(Nat.iterate f n ⟨x, hx⟩)
    -/
    apply h
    /-
      🎉 no goals
    -/


theorem not_acc_of_decreasing_seq (f : ((· > ·) : ℕ → ℕ → Prop) ↪r r) (k : ℕ) : ¬Acc r (f k) := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsStrictOrder α r
    f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) r
    k : Nat
    ⊢ Not (Acc r (f k))
  -/
  rw [acc_iff_no_decreasing_seq, not_isEmpty_iff]
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsStrictOrder α r
    f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) r
    k : Nat
    ⊢ Nonempty (Subtype fun f_1 => Membership.mem (Set.range ⇑f_1) (f k))
  -/
  exact ⟨⟨f, k, rfl⟩⟩
  /-
    🎉 no goals
  -/


/-- A relation is well-founded iff it doesn't have any infinite decreasing sequence. -/
theorem wellFounded_iff_no_descending_seq :
    WellFounded r ↔ IsEmpty (((· > ·) : ℕ → ℕ → Prop) ↪r r) := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsStrictOrder α r
    ⊢ Iff (WellFounded r) (IsEmpty (RelEmbedding (fun x1 x2 => GT.gt x1 x2) r))
  -/
  constructor
    /-
      case mp
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsStrictOrder α r
      ⊢ WellFounded r → IsEmpty (RelEmbedding (fun x1 x2 => GT.gt x1 x2) r)
    -/
  · rintro ⟨h⟩
    /-
      case mp.intro
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsStrictOrder α r
      h : ∀ (a : α), Acc r a
      ⊢ IsEmpty (RelEmbedding (fun x1 x2 => GT.gt x1 x2) r)
    -/
    exact ⟨fun f => not_acc_of_decreasing_seq f 0 (h _)⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsStrictOrder α r
      ⊢ IsEmpty (RelEmbedding (fun x1 x2 => GT.gt x1 x2) r) → WellFounded r
    -/
  · intro h
    /-
      case mpr
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsStrictOrder α r
      h : IsEmpty (RelEmbedding (fun x1 x2 => GT.gt x1 x2) r)
      ⊢ WellFounded r
    -/
    exact ⟨fun x => acc_iff_no_decreasing_seq.2 inferInstance⟩
    /-
      🎉 no goals
    -/


theorem not_wellFounded_of_decreasing_seq (f : ((· > ·) : ℕ → ℕ → Prop) ↪r r) : ¬WellFounded r := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsStrictOrder α r
    f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) r
    ⊢ Not (WellFounded r)
  -/
  rw [wellFounded_iff_no_descending_seq, not_isEmpty_iff]
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsStrictOrder α r
    f : RelEmbedding (fun x1 x2 => GT.gt x1 x2) r
    ⊢ Nonempty (RelEmbedding (fun x1 x2 => GT.gt x1 x2) r)
  -/
  exact ⟨f⟩
  /-
    🎉 no goals
  -/


/-- An order embedding from `ℕ` to itself with a specified range -/
def orderEmbeddingOfSet [DecidablePred (· ∈ s)] : ℕ ↪o ℕ :=
  (RelEmbedding.orderEmbeddingOfLTEmbedding
    (RelEmbedding.natLT (Nat.Subtype.ofNat s) fun _ => Nat.Subtype.lt_succ_self _)).trans
    (OrderEmbedding.subtype s)


/-- `Nat.Subtype.ofNat` as an order isomorphism between `ℕ` and an infinite subset. See also
`Nat.Nth` for a version where the subset may be finite. -/
noncomputable def Subtype.orderIsoOfNat : ℕ ≃o s := by
  classical
  exact
    RelIso.ofSurjective
      (RelEmbedding.orderEmbeddingOfLTEmbedding
        (RelEmbedding.natLT (Nat.Subtype.ofNat s) fun n => Nat.Subtype.lt_succ_self _))
      Nat.Subtype.ofNat_surjective


@[simp]
theorem coe_orderEmbeddingOfSet [DecidablePred (· ∈ s)] :
    ⇑(orderEmbeddingOfSet s) = (↑) ∘ Subtype.ofNat s :=
  rfl


theorem orderEmbeddingOfSet_apply [DecidablePred (· ∈ s)] {n : ℕ} :
    orderEmbeddingOfSet s n = Subtype.ofNat s n :=
  rfl


@[simp]
theorem Subtype.orderIsoOfNat_apply [dP : DecidablePred (· ∈ s)] {n : ℕ} :
    Subtype.orderIsoOfNat s n = Subtype.ofNat s n := by
  /-
    s : Set Nat
    inst✝ : Infinite ↑s
    dP : DecidablePred fun x => Membership.mem s x
    n : Nat
    ⊢ Eq ((Nat.Subtype.orderIsoOfNat s) n) (Nat.Subtype.ofNat s n)
  -/
  simp [orderIsoOfNat]; congr!
                        /-
                          🎉 no goals
                        -/


theorem orderEmbeddingOfSet_range [DecidablePred (· ∈ s)] :
    Set.range (Nat.orderEmbeddingOfSet s) = s :=
  Subtype.coe_comp_ofNat_range


theorem exists_subseq_of_forall_mem_union {s t : Set α} (e : ℕ → α) (he : ∀ n, e n ∈ s ∪ t) :
    ∃ g : ℕ ↪o ℕ, (∀ n, e (g n) ∈ s) ∨ ∀ n, e (g n) ∈ t := by
  classical
    have : Infinite (e ⁻¹' s) ∨ Infinite (e ⁻¹' t) := by
      simp only [Set.infinite_coe_iff, ← Set.infinite_union, ← Set.preimage_union,
        Set.eq_univ_of_forall fun n => Set.mem_preimage.2 (he n), Set.infinite_univ]
    cases this
    exacts [⟨Nat.orderEmbeddingOfSet (e ⁻¹' s), Or.inl fun n => (Nat.Subtype.ofNat (e ⁻¹' s) _).2⟩,
      ⟨Nat.orderEmbeddingOfSet (e ⁻¹' t), Or.inr fun n => (Nat.Subtype.ofNat (e ⁻¹' t) _).2⟩]


theorem exists_increasing_or_nonincreasing_subseq' (r : α → α → Prop) (f : ℕ → α) :
    ∃ g : ℕ ↪o ℕ,
      (∀ n : ℕ, r (f (g n)) (f (g (n + 1)))) ∨ ∀ m n : ℕ, m < n → ¬r (f (g m)) (f (g n)) := by
  classical
    let bad : Set ℕ := { m | ∀ n, m < n → ¬r (f m) (f n) }
    by_cases hbad : Infinite bad
    · haveI := hbad
      refine ⟨Nat.orderEmbeddingOfSet bad, Or.intro_right _ fun m n mn => ?_⟩
      have h := @Set.mem_range_self _ _ ↑(Nat.orderEmbeddingOfSet bad) m
      rw [Nat.orderEmbeddingOfSet_range bad] at h
      exact h _ ((OrderEmbedding.lt_iff_lt _).2 mn)
    · rw [Set.infinite_coe_iff, Set.Infinite, not_not] at hbad
      obtain ⟨m, hm⟩ : ∃ m, ∀ n, m ≤ n → ¬n ∈ bad := by
        by_cases he : hbad.toFinset.Nonempty
        · refine
            ⟨(hbad.toFinset.max' he).succ, fun n hn nbad =>
              Nat.not_succ_le_self _
                (hn.trans (hbad.toFinset.le_max' n (hbad.mem_toFinset.2 nbad)))⟩
        · exact ⟨0, fun n _ nbad => he ⟨n, hbad.mem_toFinset.2 nbad⟩⟩
      have h : ∀ n : ℕ, ∃ n' : ℕ, n < n' ∧ r (f (n + m)) (f (n' + m)) := by
        intro n
        have h := hm _ (Nat.le_add_left m n)
        simp only [bad, exists_prop, not_not, Set.mem_setOf_eq, not_forall] at h
        obtain ⟨n', hn1, hn2⟩ := h
        refine ⟨n + n' - n - m, by omega, ?_⟩
        convert hn2
        omega
      let g' : ℕ → ℕ := @Nat.rec (fun _ => ℕ) m fun n gn => Nat.find (h gn)
      exact
        ⟨(RelEmbedding.natLT (fun n => g' n + m) fun n =>
              Nat.add_lt_add_right (Nat.find_spec (h (g' n))).1 m).orderEmbeddingOfLTEmbedding,
          Or.intro_left _ fun n => (Nat.find_spec (h (g' n))).2⟩


/-- This is the infinitary Erdős–Szekeres theorem, and an important lemma in the usual proof of
    Bolzano-Weierstrass for `ℝ`. -/
theorem exists_increasing_or_nonincreasing_subseq (r : α → α → Prop) [IsTrans α r] (f : ℕ → α) :
    ∃ g : ℕ ↪o ℕ,
      (∀ m n : ℕ, m < n → r (f (g m)) (f (g n))) ∨ ∀ m n : ℕ, m < n → ¬r (f (g m)) (f (g n)) := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsTrans α r
    f : Nat → α
    ⊢ Exists fun g => Or (∀ (m n : Nat), LT.lt m n → r (f (g m)) (f (g n))) (∀ (m  …
  -/
  obtain ⟨g, hr | hnr⟩ := exists_increasing_or_nonincreasing_subseq' r f
    /-
      case intro.inl
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsTrans α r
      f : Nat → α
      g : OrderEmbedding Nat Nat
      hr : ∀ (n : Nat), r (f (g n)) (f (g (HAdd.hAdd n 1)))
      ⊢ Exists fun g => Or (∀ (m n : Nat), LT.lt m n → r (f (g m)) (f (g n))) (∀ (m  …
    -/
  · refine ⟨g, Or.intro_left _ fun m n mn => ?_⟩
    /-
      case intro.inl
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsTrans α r
      f : Nat → α
      g : OrderEmbedding Nat Nat
      hr : ∀ (n : Nat), r (f (g n)) (f (g (HAdd.hAdd n 1)))
      m n : Nat
      mn : LT.lt m n
      ⊢ r (f (g m)) (f (g n))
    -/
    obtain ⟨x, rfl⟩ := Nat.exists_eq_add_of_le (Nat.succ_le_iff.2 mn)
    /-
      case intro.inl.intro
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsTrans α r
      f : Nat → α
      g : OrderEmbedding Nat Nat
      hr : ∀ (n : Nat), r (f (g n)) (f (g (HAdd.hAdd n 1)))
      m x : Nat
      mn : LT.lt m (HAdd.hAdd m.succ x)
      ⊢ r (f (g m)) (f (g (HAdd.hAdd m.succ x)))
    -/
    induction' x with x ih
      /-
        case intro.inl.intro.zero
        α : Type u_1
        r : α → α → Prop
        inst✝ : IsTrans α r
        f : Nat → α
        g : OrderEmbedding Nat Nat
        hr : ∀ (n : Nat), r (f (g n)) (f (g (HAdd.hAdd n 1)))
        m : Nat
        mn : LT.lt m (HAdd.hAdd m.succ 0)
        ⊢ r (f (g m)) (f (g (HAdd.hAdd m.succ 0)))
      -/
    · apply hr
      /-
        🎉 no goals
      -/
      /-
        case intro.inl.intro.succ
        α : Type u_1
        r : α → α → Prop
        inst✝ : IsTrans α r
        f : Nat → α
        g : OrderEmbedding Nat Nat
        hr : ∀ (n : Nat), r (f (g n)) (f (g (HAdd.hAdd n 1)))
        m x : Nat
        ih : LT.lt m (HAdd.hAdd m.succ x) → r (f (g m)) (f (g (HAdd.hAdd m.succ x)))
        mn : LT.lt m (HAdd.hAdd m.succ (HAdd.hAdd x 1))
        ⊢ r (f (g m)) (f (g (HAdd.hAdd m.succ (HAdd.hAdd x 1))))
      -/
    · apply IsTrans.trans _ _ _ _ (hr _)
      /-
        α : Type u_1
        r : α → α → Prop
        inst✝ : IsTrans α r
        f : Nat → α
        g : OrderEmbedding Nat Nat
        hr : ∀ (n : Nat), r (f (g n)) (f (g (HAdd.hAdd n 1)))
        m x : Nat
        ih : LT.lt m (HAdd.hAdd m.succ x) → r (f (g m)) (f (g (HAdd.hAdd m.succ x)))
        mn : LT.lt m (HAdd.hAdd m.succ (HAdd.hAdd x 1))
        ⊢ r (f (g m)) (f (g (m.succ.add x)))
      -/
      exact ih (lt_of_lt_of_le m.lt_succ_self (Nat.le_add_right _ _))
      /-
        🎉 no goals
      -/
    /-
      case intro.inr
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsTrans α r
      f : Nat → α
      g : OrderEmbedding Nat Nat
      hnr : ∀ (m n : Nat), LT.lt m n → Not (r (f (g m)) (f (g n)))
      ⊢ Exists fun g => Or (∀ (m n : Nat), LT.lt m n → r (f (g m)) (f (g n))) (∀ (m  …
    -/
  · exact ⟨g, Or.intro_right _ hnr⟩
    /-
      🎉 no goals
    -/


theorem WellFounded.monotone_chain_condition' [Preorder α] :
    WellFounded ((· > ·) : α → α → Prop) ↔ ∀ a : ℕ →o α, ∃ n, ∀ m, n ≤ m → ¬a n < a m := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    ⊢ Iff (WellFounded fun x1 x2 => GT.gt x1 x2) (∀ (a : OrderHom Nat α), Exists f …
  -/
  refine ⟨fun h a => ?_, fun h => ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : Preorder α
      h : WellFounded fun x1 x2 => GT.gt x1 x2
      a : OrderHom Nat α
      ⊢ Exists fun n => ∀ (m : Nat), LE.le n m → Not (LT.lt (a n) (a m))
    -/
  · have hne : (Set.range a).Nonempty := ⟨a 0, by simp⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : Preorder α
      h : WellFounded fun x1 x2 => GT.gt x1 x2
      a : OrderHom Nat α
      hne : (Set.range ⇑a).Nonempty
      ⊢ Exists fun n => ∀ (m : Nat), LE.le n m → Not (LT.lt (a n) (a m))
    -/
    obtain ⟨x, ⟨n, rfl⟩, H⟩ := h.has_min _ hne
    /-
      case refine_1.intro.intro.intro
      α : Type u_1
      inst✝ : Preorder α
      h : WellFounded fun x1 x2 => GT.gt x1 x2
      a : OrderHom Nat α
      hne : (Set.range ⇑a).Nonempty
      n : Nat
      H : ∀ (x : α), Membership.mem (Set.range ⇑a) x → Not (GT.gt x (a n))
      ⊢ Exists fun n => ∀ (m : Nat), LE.le n m → Not (LT.lt (a n) (a m))
    -/
    exact ⟨n, fun m _ => H _ (Set.mem_range_self _)⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : Preorder α
      h : ∀ (a : OrderHom Nat α), Exists fun n => ∀ (m : Nat), LE.le n m → Not (LT.l …
      ⊢ WellFounded fun x1 x2 => GT.gt x1 x2
    -/
  · refine RelEmbedding.wellFounded_iff_no_descending_seq.2 ⟨fun a => ?_⟩
    /-
      case refine_2
      α : Type u_1
      inst✝ : Preorder α
      h : ∀ (a : OrderHom Nat α), Exists fun n => ∀ (m : Nat), LE.le n m → Not (LT.l …
      a : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => GT.gt x1 x2
      ⊢ False
    -/
    obtain ⟨n, hn⟩ := h (a.swap : ((· < ·) : ℕ → ℕ → Prop) →r ((· < ·) : α → α → Prop)).toOrderHom
    /-
      case refine_2.intro
      α : Type u_1
      inst✝ : Preorder α
      h : ∀ (a : OrderHom Nat α), Exists fun n => ∀ (m : Nat), LE.le n m → Not (LT.l …
      a : RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => GT.gt x1 x2
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Not (LT.lt (a.swap.toRelHom.toOrderHom n) (a.swa …
      ⊢ False
    -/
    exact hn n.succ n.lt_succ_self.le ((RelEmbedding.map_rel_iff _).2 n.lt_succ_self)
    /-
      🎉 no goals
    -/


/-- The "monotone chain condition" below is sometimes a convenient form of well foundedness. -/
theorem WellFounded.monotone_chain_condition [PartialOrder α] :
    WellFounded ((· > ·) : α → α → Prop) ↔ ∀ a : ℕ →o α, ∃ n, ∀ m, n ≤ m → a n = a m :=
  WellFounded.monotone_chain_condition'.trans <| by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    ⊢ Iff (∀ (a : OrderHom Nat α), Exists fun n => ∀ (m : Nat), LE.le n m → Not (L …
  -/
  congrm ∀ a, ∃ n, ∀ m h, ?_
  /-
    case a
    α : Type u_1
    inst✝ : PartialOrder α
    a : OrderHom Nat α
    n m : Nat
    h : LE.le n m
    ⊢ Iff (Not (LT.lt (a n) (a m))) (Eq (a n) (a m))
  -/
  rw [lt_iff_le_and_ne]
  /-
    case a
    α : Type u_1
    inst✝ : PartialOrder α
    a : OrderHom Nat α
    n m : Nat
    h : LE.le n m
    ⊢ Iff (Not (And (LE.le (a n) (a m)) (Ne (a n) (a m)))) (Eq (a n) (a m))
  -/
  simp [a.mono h]
  /-
    🎉 no goals
  -/


/-- Given an eventually-constant monotone sequence `a₀ ≤ a₁ ≤ a₂ ≤ ...` in a partially-ordered
type, `monotonicSequenceLimitIndex a` is the least natural number `n` for which `aₙ` reaches the
constant value. For sequences that are not eventually constant, `monotonicSequenceLimitIndex a`
is defined, but is a junk value. -/
noncomputable def monotonicSequenceLimitIndex [Preorder α] (a : ℕ →o α) : ℕ :=
  sInf { n | ∀ m, n ≤ m → a n = a m }


/-- The constant value of an eventually-constant monotone sequence `a₀ ≤ a₁ ≤ a₂ ≤ ...` in a
partially-ordered type. -/
noncomputable def monotonicSequenceLimit [Preorder α] (a : ℕ →o α) :=
  a (monotonicSequenceLimitIndex a)


theorem WellFounded.iSup_eq_monotonicSequenceLimit [CompleteLattice α]
    (h : WellFounded ((· > ·) : α → α → Prop)) (a : ℕ →o α) :
    iSup a = monotonicSequenceLimit a := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    h : WellFounded fun x1 x2 => GT.gt x1 x2
    a : OrderHom Nat α
    ⊢ Eq (iSup ⇑a) (monotonicSequenceLimit a)
  -/
  refine (iSup_le fun m => ?_).antisymm (le_iSup a _)
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    h : WellFounded fun x1 x2 => GT.gt x1 x2
    a : OrderHom Nat α
    m : Nat
    ⊢ LE.le (a m) (monotonicSequenceLimit a)
  -/
  rcases le_or_lt m (monotonicSequenceLimitIndex a) with hm | hm
    /-
      case inl
      α : Type u_1
      inst✝ : CompleteLattice α
      h : WellFounded fun x1 x2 => GT.gt x1 x2
      a : OrderHom Nat α
      m : Nat
      hm : LE.le m (monotonicSequenceLimitIndex a)
      ⊢ LE.le (a m) (monotonicSequenceLimit a)
    -/
  · exact a.monotone hm
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : CompleteLattice α
      h : WellFounded fun x1 x2 => GT.gt x1 x2
      a : OrderHom Nat α
      m : Nat
      hm : LT.lt (monotonicSequenceLimitIndex a) m
      ⊢ LE.le (a m) (monotonicSequenceLimit a)
    -/
  · cases' WellFounded.monotone_chain_condition'.1 h a with n hn
    /-
      case inr.intro
      α : Type u_1
      inst✝ : CompleteLattice α
      h : WellFounded fun x1 x2 => GT.gt x1 x2
      a : OrderHom Nat α
      m : Nat
      hm : LT.lt (monotonicSequenceLimitIndex a) m
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Not (LT.lt (a n) (a m))
      ⊢ LE.le (a m) (monotonicSequenceLimit a)
    -/
    have : n ∈ {n | ∀ m, n ≤ m → a n = a m} := fun k hk => (a.mono hk).eq_of_not_lt (hn k hk)
    /-
      case inr.intro
      α : Type u_1
      inst✝ : CompleteLattice α
      h : WellFounded fun x1 x2 => GT.gt x1 x2
      a : OrderHom Nat α
      m : Nat
      hm : LT.lt (monotonicSequenceLimitIndex a) m
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Not (LT.lt (a n) (a m))
      this : Membership.mem (setOf fun n => ∀ (m : Nat), LE.le n m → Eq (a n) (a m)) n
      ⊢ LE.le (a m) (monotonicSequenceLimit a)
    -/
    exact (Nat.sInf_mem ⟨n, this⟩ m hm.le).ge
    /-
      🎉 no goals
    -/


theorem exists_covBy_seq_of_wellFoundedLT_wellFoundedGT (α) [Preorder α]
    [Nonempty α] [wfl : WellFoundedLT α] [wfg : WellFoundedGT α] :
    ∃ a : ℕ → α, IsMin (a 0) ∧ ∃ n, IsMax (a n) ∧ ∀ i < n, a i ⋖ a (i + 1) := by
  /-
    α : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    wfl : WellFoundedLT α
    wfg : WellFoundedGT α
    ⊢ Exists fun a => And (IsMin (a 0)) (Exists fun n => And (IsMax (a n)) (∀ (i : …
  -/
  choose next hnext using exists_covBy_of_wellFoundedLT (α := α)
  /-
    α : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    wfl : WellFoundedLT α
    wfg : WellFoundedGT α
    next : ⦃a : α⦄ → Not (IsMax a) → α
    hnext : ∀ ⦃a : α⦄ (h : Not (IsMax a)), CovBy a (next h)
    ⊢ Exists fun a => And (IsMin (a 0)) (Exists fun n => And (IsMax (a n)) (∀ (i : …
  -/
  have hα := Set.nonempty_iff_univ_nonempty.mp ‹_›
  classical
  let a : ℕ → α := Nat.rec (wfl.wf.min _ hα) fun _n a ↦ if ha : IsMax a then a else next ha
  refine ⟨a, isMin_iff_forall_not_lt.mpr fun _ ↦ wfl.wf.not_lt_min _ hα trivial, ?_⟩
  have cov n (hn : ¬ IsMax (a n)) : a n ⋖ a (n + 1) := by
    change a n ⋖ if ha : IsMax (a n) then a n else _
    rw [dif_neg hn]
    exact hnext hn
  have H : ∃ n, IsMax (a n) := by
    by_contra!
    exact (RelEmbedding.natGT a fun n ↦ (cov n (this n)).1).not_wellFounded_of_decreasing_seq wfg.wf
  exact ⟨_, wellFounded_lt.min_mem _ H, fun i h ↦ cov _ fun h' ↦ wellFounded_lt.not_lt_min _ H h' h⟩

