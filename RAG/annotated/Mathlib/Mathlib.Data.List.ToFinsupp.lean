/-- Indexing into a `l : List M`, as a finitely-supported function,
where the support are all the indices within the length of the list
that index to a non-zero value. Indices beyond the end of the list are sent to 0.

This is a computable version of the `Finsupp.onFinset` construction.
-/
def toFinsupp : ℕ →₀ M where
  toFun i := getD l i 0
  support := (Finset.range l.length).filter fun i => getD l i 0 ≠ 0
  mem_support_toFun n := by
    /-
      M : Type u_1
      inst✝¹ : Zero M
      l : List M
      inst✝ : DecidablePred fun x => Ne (l.getD x 0) 0
      n✝ n : Nat
      ⊢ Iff (Membership.mem (Finset.filter (fun i => Ne (l.getD i 0) 0) (Finset.rang …
    -/
    simp only [Ne, Finset.mem_filter, Finset.mem_range, and_iff_right_iff_imp]
    /-
      M : Type u_1
      inst✝¹ : Zero M
      l : List M
      inst✝ : DecidablePred fun x => Ne (l.getD x 0) 0
      n✝ n : Nat
      ⊢ Not (Eq (l.getD n 0) 0) → LT.lt n l.length
    -/
    contrapose!
    /-
      M : Type u_1
      inst✝¹ : Zero M
      l : List M
      inst✝ : DecidablePred fun x => Ne (l.getD x 0) 0
      n✝ n : Nat
      ⊢ LE.le l.length n → Eq (l.getD n 0) 0
    -/
    exact getD_eq_default _ _
    /-
      🎉 no goals
    -/


@[norm_cast]
theorem coe_toFinsupp : (l.toFinsupp : ℕ → M) = (l.getD · 0) :=
  rfl


@[simp, norm_cast]
theorem toFinsupp_apply (i : ℕ) : (l.toFinsupp : ℕ → M) i = l.getD i 0 :=
  rfl


theorem toFinsupp_support :
    l.toFinsupp.support = (Finset.range l.length).filter (getD l · 0 ≠ 0) :=
  rfl


theorem toFinsupp_apply_lt (hn : n < l.length) : l.toFinsupp n = l[n] :=
  getD_eq_getElem _ _ hn


theorem toFinsupp_apply_fin (n : Fin l.length) : l.toFinsupp n = l[n] :=
  getD_eq_getElem _ _ n.isLt


theorem toFinsupp_apply_le (hn : l.length ≤ n) : l.toFinsupp n = 0 :=
  getD_eq_default _ _ hn


@[simp]
theorem toFinsupp_nil [DecidablePred fun i => getD ([] : List M) i 0 ≠ 0] :
    toFinsupp ([] : List M) = 0 := by
  /-
    M : Type u_1
    inst✝¹ : Zero M
    inst✝ : DecidablePred fun i => Ne (List.nil.getD i 0) 0
    ⊢ Eq List.nil.toFinsupp 0
  -/
  ext
  /-
    case h
    M : Type u_1
    inst✝¹ : Zero M
    inst✝ : DecidablePred fun i => Ne (List.nil.getD i 0) 0
    a✝ : Nat
    ⊢ Eq (List.nil.toFinsupp a✝) (0 a✝)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem toFinsupp_singleton (x : M) [DecidablePred (getD [x] · 0 ≠ 0)] :
    toFinsupp [x] = Finsupp.single 0 x := by
  /-
    M : Type u_1
    inst✝¹ : Zero M
    x : M
    inst✝ : DecidablePred fun x_1 => Ne ((List.cons x List.nil).getD x_1 0) 0
    ⊢ Eq (List.cons x List.nil).toFinsupp (Finsupp.single 0 x)
  -/
                  /-
                    🎉 no goals
                  -/
  ext ⟨_ | i⟩ <;> simp [Finsupp.single_apply, (Nat.zero_lt_succ _).ne]
                  /-
                    🎉 no goals
                  -/


@[deprecated "This lemma is unused, and can be proved by `simp`." (since := "2024-06-12")]
theorem toFinsupp_cons_apply_zero (x : M) (xs : List M)
    [DecidablePred (getD (x::xs) · 0 ≠ 0)] : (x::xs).toFinsupp 0 = x :=
  rfl


@[deprecated "This lemma is unused, and can be proved by `simp`." (since := "2024-06-12")]
theorem toFinsupp_cons_apply_succ (x : M) (xs : List M) (n : ℕ)
    [DecidablePred (getD (x::xs) · 0 ≠ 0)] [DecidablePred (getD xs · 0 ≠ 0)] :
    (x::xs).toFinsupp n.succ = xs.toFinsupp n :=
  rfl


theorem toFinsupp_append {R : Type*} [AddZeroClass R] (l₁ l₂ : List R)
    [DecidablePred (getD (l₁ ++ l₂) · 0 ≠ 0)] [DecidablePred (getD l₁ · 0 ≠ 0)]
    [DecidablePred (getD l₂ · 0 ≠ 0)] :
    toFinsupp (l₁ ++ l₂) =
      toFinsupp l₁ + (toFinsupp l₂).embDomain (addLeftEmbedding l₁.length) := by
  /-
    R : Type u_2
    inst✝³ : AddZeroClass R
    l₁ l₂ : List R
    inst✝² : DecidablePred fun x => Ne ((HAppend.hAppend l₁ l₂).getD x 0) 0
    inst✝¹ : DecidablePred fun x => Ne (l₁.getD x 0) 0
    inst✝ : DecidablePred fun x => Ne (l₂.getD x 0) 0
    ⊢ Eq (HAppend.hAppend l₁ l₂).toFinsupp (HAdd.hAdd l₁.toFinsupp (Finsupp.embDom …
  -/
  ext n
  /-
    case h
    R : Type u_2
    inst✝³ : AddZeroClass R
    l₁ l₂ : List R
    inst✝² : DecidablePred fun x => Ne ((HAppend.hAppend l₁ l₂).getD x 0) 0
    inst✝¹ : DecidablePred fun x => Ne (l₁.getD x 0) 0
    inst✝ : DecidablePred fun x => Ne (l₂.getD x 0) 0
    n : Nat
    ⊢ Eq ((HAppend.hAppend l₁ l₂).toFinsupp n) ((HAdd.hAdd l₁.toFinsupp (Finsupp.e …
  -/
  simp only [toFinsupp_apply, Finsupp.add_apply]
  cases lt_or_le n l₁.length with
  | inl h =>
    rw [getD_append _ _ _ _ h, Finsupp.embDomain_notin_range, add_zero]
    rintro ⟨k, rfl : length l₁ + k = n⟩
    omega
  | inr h =>
    rcases Nat.exists_eq_add_of_le h with ⟨k, rfl⟩
    rw [getD_append_right _ _ _ _ h, Nat.add_sub_cancel_left, getD_eq_default _ _ h, zero_add]
    exact Eq.symm (Finsupp.embDomain_apply _ _ _)


theorem toFinsupp_cons_eq_single_add_embDomain {R : Type*} [AddZeroClass R] (x : R) (xs : List R)
    [DecidablePred (getD (x::xs) · 0 ≠ 0)] [DecidablePred (getD xs · 0 ≠ 0)] :
    toFinsupp (x::xs) =
      Finsupp.single 0 x + (toFinsupp xs).embDomain ⟨Nat.succ, Nat.succ_injective⟩ := by
  classical
    convert toFinsupp_append [x] xs using 3
    · exact (toFinsupp_singleton x).symm
    · ext n
      exact add_comm n 1


theorem toFinsupp_concat_eq_toFinsupp_add_single {R : Type*} [AddZeroClass R] (x : R) (xs : List R)
    [DecidablePred fun i => getD (xs ++ [x]) i 0 ≠ 0] [DecidablePred fun i => getD xs i 0 ≠ 0] :
    toFinsupp (xs ++ [x]) = toFinsupp xs + Finsupp.single xs.length x := by
  classical rw [toFinsupp_append, toFinsupp_singleton, Finsupp.embDomain_single,
    addLeftEmbedding_apply, add_zero]



theorem toFinsupp_eq_sum_map_enum_single {R : Type*} [AddMonoid R] (l : List R)
    [DecidablePred (getD l · 0 ≠ 0)] :
    toFinsupp l = (l.enum.map fun nr : ℕ × R => Finsupp.single nr.1 nr.2).sum := by
  /- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: `induction` fails to substitute `l = []` in
  `[DecidablePred (getD l · 0 ≠ 0)]`, so we manually do some `revert`/`intro` as a workaround -/
  /-
    R : Type u_2
    inst✝¹ : AddMonoid R
    l : List R
    inst✝ : DecidablePred fun x => Ne (l.getD x 0) 0
    ⊢ Eq l.toFinsupp (List.map (fun nr => Finsupp.single nr.1 nr.2) l.enum).sum
  -/
  revert l; intro l
  induction l using List.reverseRecOn with
  | nil => exact toFinsupp_nil
  | append_singleton x xs ih =>
    classical simp [toFinsupp_concat_eq_toFinsupp_add_single, enum_append, ih]


