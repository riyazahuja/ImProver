theorem Perm.bagInter_right {l₁ l₂ : List α} (t : List α) (h : l₁ ~ l₂) :
    l₁.bagInter t ~ l₂.bagInter t := by
  induction h generalizing t with
  | nil => simp
  | cons x => by_cases x ∈ t <;> simp [*, Perm.cons]
  | swap x y =>
    by_cases h : x = y
    · simp [h]
    by_cases xt : x ∈ t <;> by_cases yt : y ∈ t
    · simp [xt, yt, mem_erase_of_ne h, mem_erase_of_ne (Ne.symm h), erase_comm, swap]
    · simp [xt, yt, mt mem_of_mem_erase, Perm.cons]
    · simp [xt, yt, mt mem_of_mem_erase, Perm.cons]
    · simp [xt, yt]
  | trans _ _ ih_1 ih_2 => exact (ih_1 _).trans (ih_2 _)


theorem Perm.bagInter_left (l : List α) {t₁ t₂ : List α} (p : t₁ ~ t₂) :
    l.bagInter t₁ = l.bagInter t₂ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l t₁ t₂ : List α
    p : t₁.Perm t₂
    ⊢ Eq (l.bagInter t₁) (l.bagInter t₂)
  -/
  induction' l with a l IH generalizing t₁ t₂ p; · simp
                                                   /-
                                                     🎉 no goals
                                                   -/
  /-
    case cons
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    l : List α
    IH : ∀ {t₁ t₂ : List α}, t₁.Perm t₂ → Eq (l.bagInter t₁) (l.bagInter t₂)
    t₁ t₂ : List α
    p : t₁.Perm t₂
    ⊢ Eq ((List.cons a l).bagInter t₁) ((List.cons a l).bagInter t₂)
  -/
  by_cases h : a ∈ t₁
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      l : List α
      IH : ∀ {t₁ t₂ : List α}, t₁.Perm t₂ → Eq (l.bagInter t₁) (l.bagInter t₂)
      t₁ t₂ : List α
      p : t₁.Perm t₂
      h : Membership.mem t₁ a
      ⊢ Eq ((List.cons a l).bagInter t₁) ((List.cons a l).bagInter t₂)
    -/
  · simp [h, p.subset h, IH (p.erase _)]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      l : List α
      IH : ∀ {t₁ t₂ : List α}, t₁.Perm t₂ → Eq (l.bagInter t₁) (l.bagInter t₂)
      t₁ t₂ : List α
      p : t₁.Perm t₂
      h : Not (Membership.mem t₁ a)
      ⊢ Eq ((List.cons a l).bagInter t₁) ((List.cons a l).bagInter t₂)
    -/
  · simp [h, mt p.mem_iff.2 h, IH p]
    /-
      🎉 no goals
    -/


theorem Perm.bagInter {l₁ l₂ t₁ t₂ : List α} (hl : l₁ ~ l₂) (ht : t₁ ~ t₂) :
    l₁.bagInter t₁ ~ l₂.bagInter t₂ :=
  ht.bagInter_left l₂ ▸ hl.bagInter_right _


theorem Perm.inter_append {l t₁ t₂ : List α} (h : Disjoint t₁ t₂) :
    l ∩ (t₁ ++ t₂) ~ l ∩ t₁ ++ l ∩ t₂ := by
  induction l with
  | nil => simp
  | cons x xs l_ih =>
    by_cases h₁ : x ∈ t₁
    · have h₂ : x ∉ t₂ := h h₁
      simp [*]
    by_cases h₂ : x ∈ t₂
    · simp only [*, inter_cons_of_not_mem, false_or, mem_append, inter_cons_of_mem,
        not_false_iff]
      refine Perm.trans (Perm.cons _ l_ih) ?_
      change [x] ++ xs ∩ t₁ ++ xs ∩ t₂ ~ xs ∩ t₁ ++ ([x] ++ xs ∩ t₂)
      rw [← List.append_assoc]
      solve_by_elim [Perm.append_right, perm_append_comm]
    · simp [*]


theorem Perm.take_inter {xs ys : List α} (n : ℕ) (h : xs ~ ys)
    (h' : ys.Nodup) : xs.take n ~ ys.inter (xs.take n) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs ys : List α
    n : Nat
    h : xs.Perm ys
    h' : ys.Nodup
    ⊢ (List.take n xs).Perm (ys.inter (List.take n xs))
  -/
  simp only [List.inter]
  exact Perm.trans (show xs.take n ~ xs.filter (xs.take n).elem by
      conv_lhs => rw [Nodup.take_eq_filter_mem ((Perm.nodup_iff h).2 h')])
    (Perm.filter _ h)


theorem Perm.drop_inter {xs ys : List α} (n : ℕ) (h : xs ~ ys) (h' : ys.Nodup) :
    xs.drop n ~ ys.inter (xs.drop n) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs ys : List α
    n : Nat
    h : xs.Perm ys
    h' : ys.Nodup
    ⊢ (List.drop n xs).Perm (ys.inter (List.drop n xs))
  -/
  by_cases h'' : n ≤ xs.length
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      xs ys : List α
      n : Nat
      h : xs.Perm ys
      h' : ys.Nodup
      h'' : LE.le n xs.length
      ⊢ (List.drop n xs).Perm (ys.inter (List.drop n xs))
    -/
  · let n' := xs.length - n
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      xs ys : List α
      n : Nat
      h : xs.Perm ys
      h' : ys.Nodup
      h'' : LE.le n xs.length
      n' : Nat := HSub.hSub xs.length n
      ⊢ (List.drop n xs).Perm (ys.inter (List.drop n xs))
    -/
    have h₀ : n = xs.length - n' := by rwa [Nat.sub_sub_self]
    have h₁ : xs.drop n = (xs.reverse.take n').reverse := by
      rw [take_reverse, h₀, reverse_reverse]
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      xs ys : List α
      n : Nat
      h : xs.Perm ys
      h' : ys.Nodup
      h'' : LE.le n xs.length
      n' : Nat := HSub.hSub xs.length n
      h₀ : Eq n (HSub.hSub xs.length n')
      h₁ : Eq (List.drop n xs) (List.take n' xs.reverse).reverse
      ⊢ (List.drop n xs).Perm (ys.inter (List.drop n xs))
    -/
    rw [h₁]
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      xs ys : List α
      n : Nat
      h : xs.Perm ys
      h' : ys.Nodup
      h'' : LE.le n xs.length
      n' : Nat := HSub.hSub xs.length n
      h₀ : Eq n (HSub.hSub xs.length n')
      h₁ : Eq (List.drop n xs) (List.take n' xs.reverse).reverse
      ⊢ (List.take n' xs.reverse).reverse.Perm (ys.inter (List.take n' xs.reverse).r …
    -/
    apply (reverse_perm _).trans
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      xs ys : List α
      n : Nat
      h : xs.Perm ys
      h' : ys.Nodup
      h'' : LE.le n xs.length
      n' : Nat := HSub.hSub xs.length n
      h₀ : Eq n (HSub.hSub xs.length n')
      h₁ : Eq (List.drop n xs) (List.take n' xs.reverse).reverse
      ⊢ (List.take n' xs.reverse).Perm (ys.inter (List.take n' xs.reverse).reverse)
    -/
    rw [inter_reverse]
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      xs ys : List α
      n : Nat
      h : xs.Perm ys
      h' : ys.Nodup
      h'' : LE.le n xs.length
      n' : Nat := HSub.hSub xs.length n
      h₀ : Eq n (HSub.hSub xs.length n')
      h₁ : Eq (List.drop n xs) (List.take n' xs.reverse).reverse
      ⊢ (List.take n' xs.reverse).Perm (ys.inter (List.take n' xs.reverse))
    -/
    apply Perm.take_inter _ _ h'
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      xs ys : List α
      n : Nat
      h : xs.Perm ys
      h' : ys.Nodup
      h'' : LE.le n xs.length
      n' : Nat := HSub.hSub xs.length n
      h₀ : Eq n (HSub.hSub xs.length n')
      h₁ : Eq (List.drop n xs) (List.take n' xs.reverse).reverse
      ⊢ xs.reverse.Perm ys
    -/
    apply (reverse_perm _).trans; assumption
                                  /-
                                    🎉 no goals
                                  -/
  · have : drop n xs = [] := by
      apply eq_nil_of_length_eq_zero
      rw [length_drop, Nat.sub_eq_zero_iff_le]
      apply le_of_not_ge h''
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      xs ys : List α
      n : Nat
      h : xs.Perm ys
      h' : ys.Nodup
      h'' : Not (LE.le n xs.length)
      this : Eq (List.drop n xs) List.nil
      ⊢ (List.drop n xs).Perm (ys.inter (List.drop n xs))
    -/
    simp [this, List.inter]
    /-
      🎉 no goals
    -/


theorem Perm.dropSlice_inter {xs ys : List α} (n m : ℕ) (h : xs ~ ys)
    (h' : ys.Nodup) : List.dropSlice n m xs ~ ys ∩ List.dropSlice n m xs := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs ys : List α
    n m : Nat
    h : xs.Perm ys
    h' : ys.Nodup
    ⊢ (List.dropSlice n m xs).Perm (Inter.inter ys (List.dropSlice n m xs))
  -/
  simp only [dropSlice_eq]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs ys : List α
    n m : Nat
    h : xs.Perm ys
    h' : ys.Nodup
    ⊢ (HAppend.hAppend (List.take n xs) (List.drop (HAdd.hAdd n m) xs)).Perm (Inte …
  -/
  have : n ≤ n + m := Nat.le_add_right _ _
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs ys : List α
    n m : Nat
    h : xs.Perm ys
    h' : ys.Nodup
    this : LE.le n (HAdd.hAdd n m)
    ⊢ (HAppend.hAppend (List.take n xs) (List.drop (HAdd.hAdd n m) xs)).Perm (Inte …
  -/
  have h₂ := h.nodup_iff.2 h'
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    xs ys : List α
    n m : Nat
    h : xs.Perm ys
    h' : ys.Nodup
    this : LE.le n (HAdd.hAdd n m)
    h₂ : xs.Nodup
    ⊢ (HAppend.hAppend (List.take n xs) (List.drop (HAdd.hAdd n m) xs)).Perm (Inte …
  -/
  apply Perm.trans _ (Perm.inter_append _).symm
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      xs ys : List α
      n m : Nat
      h : xs.Perm ys
      h' : ys.Nodup
      this : LE.le n (HAdd.hAdd n m)
      h₂ : xs.Nodup
      ⊢ (HAppend.hAppend (List.take n xs) (List.drop (HAdd.hAdd n m) xs)).Perm (HApp …
    -/
  · exact Perm.append (Perm.take_inter _ h h') (Perm.drop_inter _ h h')
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      xs ys : List α
      n m : Nat
      h : xs.Perm ys
      h' : ys.Nodup
      this : LE.le n (HAdd.hAdd n m)
      h₂ : xs.Nodup
      ⊢ (List.take n xs).Disjoint (List.drop (HAdd.hAdd n m) xs)
    -/
  · exact disjoint_take_drop h₂ this
    /-
      🎉 no goals
    -/


