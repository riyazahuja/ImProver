local notation a " * " b => op a b


/-- `fold op b s` folds a commutative associative operation `op` over
  the multiset `s`. -/
def fold : α → Multiset α → α :=
  foldr op


theorem fold_eq_foldr (b : α) (s : Multiset α) :
    fold op b s = foldr op b s :=
  rfl


@[simp]
theorem coe_fold_r (b : α) (l : List α) : fold op b l = l.foldr op b :=
  rfl


theorem coe_fold_l (b : α) (l : List α) : fold op b l = l.foldl op b :=
                                      /-
                                        α : Type u_1
                                        op : α → α → α
                                        hc : Std.Commutative op
                                        ha : Std.Associative op
                                        b : α
                                        l : List α
                                        ⊢ Eq (List.foldl (fun x y => op y x) b l) (List.foldl op b l)
                                      -/
  (coe_foldr_swap op b l).trans <| by simp [hc.comm]
                                      /-
                                        🎉 no goals
                                      -/


theorem fold_eq_foldl (b : α) (s : Multiset α) :
    fold op b s = foldl op b s :=
  Quot.inductionOn s fun _ => coe_fold_l _ _ _


@[simp]
theorem fold_zero (b : α) : (0 : Multiset α).fold op b = b :=
  rfl


@[simp]
theorem fold_cons_left : ∀ (b a : α) (s : Multiset α), (a ::ₘ s).fold op b = a * s.fold op b :=
  foldr_cons _


theorem fold_cons_right (b a : α) (s : Multiset α) : (a ::ₘ s).fold op b = s.fold op b * a := by
  /-
    α : Type u_1
    op : α → α → α
    hc : Std.Commutative op
    ha : Std.Associative op
    b a : α
    s : Multiset α
    ⊢ Eq (Multiset.fold op b (Multiset.cons a s)) (op (Multiset.fold op b s) a)
  -/
  simp [hc.comm]
  /-
    🎉 no goals
  -/


theorem fold_cons'_right (b a : α) (s : Multiset α) : (a ::ₘ s).fold op b = s.fold op (b * a) := by
  /-
    α : Type u_1
    op : α → α → α
    hc : Std.Commutative op
    ha : Std.Associative op
    b a : α
    s : Multiset α
    ⊢ Eq (Multiset.fold op b (Multiset.cons a s)) (Multiset.fold op (op b a) s)
  -/
  rw [fold_eq_foldl, foldl_cons, ← fold_eq_foldl]
  /-
    🎉 no goals
  -/


theorem fold_cons'_left (b a : α) (s : Multiset α) : (a ::ₘ s).fold op b = s.fold op (a * b) := by
  /-
    α : Type u_1
    op : α → α → α
    hc : Std.Commutative op
    ha : Std.Associative op
    b a : α
    s : Multiset α
    ⊢ Eq (Multiset.fold op b (Multiset.cons a s)) (Multiset.fold op (op a b) s)
  -/
  rw [fold_cons'_right, hc.comm]
  /-
    🎉 no goals
  -/


theorem fold_add (b₁ b₂ : α) (s₁ s₂ : Multiset α) :
    (s₁ + s₂).fold op (b₁ * b₂) = s₁.fold op b₁ * s₂.fold op b₂ :=
                               /-
                                 α : Type u_1
                                 op : α → α → α
                                 hc : Std.Commutative op
                                 ha : Std.Associative op
                                 b₁ b₂ : α
                                 s₁ s₂ : Multiset α
                                 ⊢ Eq (Multiset.fold op (op b₁ b₂) (HAdd.hAdd s₁ 0)) (op (Multiset.fold op b₁ s …
                               -/
  Multiset.induction_on s₂ (by rw [add_zero, fold_zero, ← fold_cons'_right, ← fold_cons_right op])
                               /-
                                 🎉 no goals
                               -/
    (fun a b h => by rw [fold_cons_left, add_cons, fold_cons_left, h, ← ha.assoc, hc.comm a,
      ha.assoc])


theorem fold_bind {ι : Type*} (s : Multiset ι) (t : ι → Multiset α) (b : ι → α) (b₀ : α) :
    (s.bind t).fold op ((s.map b).fold op b₀) =
    (s.map fun i => (t i).fold op (b i)).fold op b₀ := by
  /-
    α : Type u_1
    op : α → α → α
    hc : Std.Commutative op
    ha : Std.Associative op
    ι : Type u_3
    s : Multiset ι
    t : ι → Multiset α
    b : ι → α
    b₀ : α
    ⊢ Eq (Multiset.fold op (Multiset.fold op b₀ (Multiset.map b s)) (s.bind t)) (M …
  -/
  induction' s using Multiset.induction_on with a ha ih
    /-
      case empty
      α : Type u_1
      op : α → α → α
      hc : Std.Commutative op
      ha : Std.Associative op
      ι : Type u_3
      t : ι → Multiset α
      b : ι → α
      b₀ : α
      ⊢ Eq (Multiset.fold op (Multiset.fold op b₀ (Multiset.map b 0)) (Multiset.bind …
    -/
  · rw [zero_bind, map_zero, map_zero, fold_zero]
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      op : α → α → α
      hc : Std.Commutative op
      ha✝ : Std.Associative op
      ι : Type u_3
      t : ι → Multiset α
      b : ι → α
      b₀ : α
      a : ι
      ha : Multiset ι
      ih : Eq (Multiset.fold op (Multiset.fold op b₀ (Multiset.map b ha)) (ha.bind t …
      ⊢ Eq (Multiset.fold op (Multiset.fold op b₀ (Multiset.map b (Multiset.cons a h …
    -/
  · rw [cons_bind, map_cons, map_cons, fold_cons_left, fold_cons_left, fold_add, ih]
    /-
      🎉 no goals
    -/


theorem fold_singleton (b a : α) : ({a} : Multiset α).fold op b = a * b :=
  foldr_singleton _ _ _


theorem fold_distrib {f g : β → α} (u₁ u₂ : α) (s : Multiset β) :
    (s.map fun x => f x * g x).fold op (u₁ * u₂) = (s.map f).fold op u₁ * (s.map g).fold op u₂ :=
                              /-
                                α : Type u_1
                                β : Type u_2
                                op : α → α → α
                                hc : Std.Commutative op
                                ha : Std.Associative op
                                f g : β → α
                                u₁ u₂ : α
                                s : Multiset β
                                ⊢ Eq (Multiset.fold op (op u₁ u₂) (Multiset.map (fun x => op (f x) (g x)) 0))  …
                              -/
  Multiset.induction_on s (by simp) (fun a b h => by
                              /-
                                🎉 no goals
                              -/
    rw [map_cons, fold_cons_left, h, map_cons, fold_cons_left, map_cons,
      fold_cons_right, ha.assoc, ← ha.assoc (g a), hc.comm (g a),
      ha.assoc, hc.comm (g a), ha.assoc])


theorem fold_hom {op' : β → β → β} [Std.Commutative op'] [Std.Associative op'] {m : α → β}
    (hm : ∀ x y, m (op x y) = op' (m x) (m y)) (b : α) (s : Multiset α) :
    (s.map m).fold op' (m b) = m (s.fold op b) :=
                              /-
                                α : Type u_1
                                β : Type u_2
                                op : α → α → α
                                hc : Std.Commutative op
                                ha : Std.Associative op
                                op' : β → β → β
                                inst✝¹ : Std.Commutative op'
                                inst✝ : Std.Associative op'
                                m : α → β
                                hm : ∀ (x y : α), Eq (m (op x y)) (op' (m x) (m y))
                                b : α
                                s : Multiset α
                                ⊢ Eq (Multiset.fold op' (m b) (Multiset.map m 0)) (m (Multiset.fold op b 0))
                              -/
                              /-
                                🎉 no goals
                              -/
  Multiset.induction_on s (by simp) (by simp +contextual [hm])
                                        /-
                                          🎉 no goals
                                        -/


theorem fold_union_inter [DecidableEq α] (s₁ s₂ : Multiset α) (b₁ b₂ : α) :
    ((s₁ ∪ s₂).fold op b₁ * (s₁ ∩ s₂).fold op b₂) = s₁.fold op b₁ * s₂.fold op b₂ := by
  /-
    α : Type u_1
    op : α → α → α
    hc : Std.Commutative op
    ha : Std.Associative op
    inst✝ : DecidableEq α
    s₁ s₂ : Multiset α
    b₁ b₂ : α
    ⊢ Eq (op (Multiset.fold op b₁ (Union.union s₁ s₂)) (Multiset.fold op b₂ (Inter …
  -/
  rw [← fold_add op, union_add_inter, fold_add op]
  /-
    🎉 no goals
  -/


@[simp]
theorem fold_dedup_idem [DecidableEq α] [hi : Std.IdempotentOp op] (s : Multiset α) (b : α) :
    (dedup s).fold op b = s.fold op b :=
                              /-
                                α : Type u_1
                                op : α → α → α
                                hc : Std.Commutative op
                                ha : Std.Associative op
                                inst✝ : DecidableEq α
                                hi : Std.IdempotentOp op
                                s : Multiset α
                                b : α
                                ⊢ Eq (Multiset.fold op b (Multiset.dedup 0)) (Multiset.fold op b 0)
                              -/
  Multiset.induction_on s (by simp) fun a s IH => by
                              /-
                                🎉 no goals
                              -/
    /-
      α : Type u_1
      op : α → α → α
      hc : Std.Commutative op
      ha : Std.Associative op
      inst✝ : DecidableEq α
      hi : Std.IdempotentOp op
      s✝ : Multiset α
      b a : α
      s : Multiset α
      IH : Eq (Multiset.fold op b s.dedup) (Multiset.fold op b s)
      ⊢ Eq (Multiset.fold op b (Multiset.cons a s).dedup) (Multiset.fold op b (Multi …
    -/
    by_cases h : a ∈ s <;> simp [IH, h]
                           /-
                             🎉 no goals
                           -/
    /-
      case pos
      α : Type u_1
      op : α → α → α
      hc : Std.Commutative op
      ha : Std.Associative op
      inst✝ : DecidableEq α
      hi : Std.IdempotentOp op
      s✝ : Multiset α
      b a : α
      s : Multiset α
      IH : Eq (Multiset.fold op b s.dedup) (Multiset.fold op b s)
      h : Membership.mem s a
      ⊢ Eq (Multiset.fold op b s) (op a (Multiset.fold op b s))
    -/
    show fold op b s = op a (fold op b s)
    /-
      case pos
      α : Type u_1
      op : α → α → α
      hc : Std.Commutative op
      ha : Std.Associative op
      inst✝ : DecidableEq α
      hi : Std.IdempotentOp op
      s✝ : Multiset α
      b a : α
      s : Multiset α
      IH : Eq (Multiset.fold op b s.dedup) (Multiset.fold op b s)
      h : Membership.mem s a
      ⊢ Eq (Multiset.fold op b s) (op a (Multiset.fold op b s))
    -/
    rw [← cons_erase h, fold_cons_left, ← ha.assoc, hi.idempotent]
    /-
      🎉 no goals
    -/


theorem le_smul_dedup [DecidableEq α] (s : Multiset α) : ∃ n : ℕ, s ≤ n • dedup s :=
  ⟨(s.map fun a => count a s).fold max 0,
    le_iff_count.2 fun a => by
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        s : Multiset α
        a : α
        ⊢ LE.le (Multiset.count a s) (Multiset.count a (HSMul.hSMul (Multiset.fold Max …
      -/
      rw [count_nsmul]; by_cases h : a ∈ s
        /-
          case pos
          α : Type u_1
          inst✝ : DecidableEq α
          s : Multiset α
          a : α
          h : Membership.mem s a
          ⊢ LE.le (Multiset.count a s) (HMul.hMul (Multiset.fold Max.max 0 (Multiset.map …
        -/
      · refine le_trans ?_ (Nat.mul_le_mul_left _ <| count_pos.2 <| mem_dedup.2 h)
        have : count a s ≤ fold max 0 (map (fun a => count a s) (a ::ₘ erase s a)) := by
          simp [le_max_left]
        /-
          case pos
          α : Type u_1
          inst✝ : DecidableEq α
          s : Multiset α
          a : α
          h : Membership.mem s a
          this : LE.le (Multiset.count a s) (Multiset.fold Max.max 0 (Multiset.map (fun  …
          ⊢ LE.le (Multiset.count a s) (HMul.hMul (Multiset.fold Max.max 0 (Multiset.map …
        -/
        rw [cons_erase h] at this
        /-
          case pos
          α : Type u_1
          inst✝ : DecidableEq α
          s : Multiset α
          a : α
          h : Membership.mem s a
          this : LE.le (Multiset.count a s) (Multiset.fold Max.max 0 (Multiset.map (fun  …
          ⊢ LE.le (Multiset.count a s) (HMul.hMul (Multiset.fold Max.max 0 (Multiset.map …
        -/
        simpa [mul_succ] using this
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          inst✝ : DecidableEq α
          s : Multiset α
          a : α
          h : Not (Membership.mem s a)
          ⊢ LE.le (Multiset.count a s) (HMul.hMul (Multiset.fold Max.max 0 (Multiset.map …
        -/
      · simp [count_eq_zero.2 h, Nat.zero_le]⟩
        /-
          🎉 no goals
        -/


