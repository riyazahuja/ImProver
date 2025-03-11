lemma prod_nonneg (h0 : ∀ i ∈ s, 0 ≤ f i) : 0 ≤ ∏ i ∈ s, f i :=
  prod_induction f (fun i ↦ 0 ≤ i) (fun _ _ ha hb ↦ mul_nonneg ha hb) zero_le_one h0


/-- If all `f i`, `i ∈ s`, are nonnegative and each `f i` is less than or equal to `g i`, then the
product of `f i` is less than or equal to the product of `g i`. See also `Finset.prod_le_prod'` for
the case of an ordered commutative multiplicative monoid. -/
@[gcongr]
lemma prod_le_prod (h0 : ∀ i ∈ s, 0 ≤ f i) (h1 : ∀ i ∈ s, f i ≤ g i) :
    ∏ i ∈ s, f i ≤ ∏ i ∈ s, g i := by
  induction s using Finset.cons_induction with
  | empty => simp
  | cons a s has ih =>
    simp only [prod_cons, forall_mem_cons] at h0 h1 ⊢
    have := posMulMono_iff_mulPosMono.1 ‹PosMulMono R›
    gcongr
    exacts [prod_nonneg h0.2, h0.1.trans h1.1, h1.1, ih h0.2 h1.2]


/-- If each `f i`, `i ∈ s` belongs to `[0, 1]`, then their product is less than or equal to one.
See also `Finset.prod_le_one'` for the case of an ordered commutative multiplicative monoid. -/
lemma prod_le_one (h0 : ∀ i ∈ s, 0 ≤ f i) (h1 : ∀ i ∈ s, f i ≤ 1) : ∏ i ∈ s, f i ≤ 1 := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝³ : CommMonoidWithZero R
    inst✝² : PartialOrder R
    inst✝¹ : ZeroLEOneClass R
    inst✝ : PosMulMono R
    f : ι → R
    s : Finset ι
    h0 : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
    h1 : ∀ (i : ι), Membership.mem s i → LE.le (f i) 1
    ⊢ LE.le (s.prod fun i => f i) 1
  -/
  convert ← prod_le_prod h0 h1
  /-
    case h.e'_4
    ι : Type u_1
    R : Type u_2
    inst✝³ : CommMonoidWithZero R
    inst✝² : PartialOrder R
    inst✝¹ : ZeroLEOneClass R
    inst✝ : PosMulMono R
    f : ι → R
    s : Finset ι
    h0 : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
    h1 : ∀ (i : ι), Membership.mem s i → LE.le (f i) 1
    ⊢ Eq (s.prod fun i => 1) 1
  -/
  exact Finset.prod_const_one
  /-
    🎉 no goals
  -/


lemma prod_pos (h0 : ∀ i ∈ s, 0 < f i) : 0 < ∏ i ∈ s, f i :=
  prod_induction f (fun x ↦ 0 < x) (fun _ _ ha hb ↦ mul_pos ha hb) zero_lt_one h0


lemma prod_lt_prod (hf : ∀ i ∈ s, 0 < f i) (hfg : ∀ i ∈ s, f i ≤ g i)
    (hlt : ∃ i ∈ s, f i < g i) :
    ∏ i ∈ s, f i < ∏ i ∈ s, g i := by
  classical
  obtain ⟨i, hi, hilt⟩ := hlt
  rw [← insert_erase hi, prod_insert (not_mem_erase _ _), prod_insert (not_mem_erase _ _)]
  have := posMulStrictMono_iff_mulPosStrictMono.1 ‹PosMulStrictMono R›
  refine mul_lt_mul_of_pos_of_nonneg' hilt ?_ ?_ ?_
  · exact prod_le_prod (fun j hj => le_of_lt (hf j (mem_of_mem_erase hj)))
      (fun _ hj ↦ hfg _ <| mem_of_mem_erase hj)
  · exact prod_pos fun j hj => hf j (mem_of_mem_erase hj)
  · exact (hf i hi).le.trans hilt.le


lemma prod_lt_prod_of_nonempty (hf : ∀ i ∈ s, 0 < f i) (hfg : ∀ i ∈ s, f i < g i)
    (h_ne : s.Nonempty) :
    ∏ i ∈ s, f i < ∏ i ∈ s, g i := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁴ : CommMonoidWithZero R
    inst✝³ : PartialOrder R
    inst✝² : ZeroLEOneClass R
    inst✝¹ : PosMulStrictMono R
    inst✝ : Nontrivial R
    f g : ι → R
    s : Finset ι
    hf : ∀ (i : ι), Membership.mem s i → LT.lt 0 (f i)
    hfg : ∀ (i : ι), Membership.mem s i → LT.lt (f i) (g i)
    h_ne : s.Nonempty
    ⊢ LT.lt (s.prod fun i => f i) (s.prod fun i => g i)
  -/
  apply prod_lt_prod hf fun i hi => le_of_lt (hfg i hi)
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁴ : CommMonoidWithZero R
    inst✝³ : PartialOrder R
    inst✝² : ZeroLEOneClass R
    inst✝¹ : PosMulStrictMono R
    inst✝ : Nontrivial R
    f g : ι → R
    s : Finset ι
    hf : ∀ (i : ι), Membership.mem s i → LT.lt 0 (f i)
    hfg : ∀ (i : ι), Membership.mem s i → LT.lt (f i) (g i)
    h_ne : s.Nonempty
    ⊢ Exists fun i => And (Membership.mem s i) (LT.lt (f i) (g i))
  -/
  obtain ⟨i, hi⟩ := h_ne
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    inst✝⁴ : CommMonoidWithZero R
    inst✝³ : PartialOrder R
    inst✝² : ZeroLEOneClass R
    inst✝¹ : PosMulStrictMono R
    inst✝ : Nontrivial R
    f g : ι → R
    s : Finset ι
    hf : ∀ (i : ι), Membership.mem s i → LT.lt 0 (f i)
    hfg : ∀ (i : ι), Membership.mem s i → LT.lt (f i) (g i)
    i : ι
    hi : Membership.mem s i
    ⊢ Exists fun i => And (Membership.mem s i) (LT.lt (f i) (g i))
  -/
  exact ⟨i, hi, hfg i hi⟩
  /-
    🎉 no goals
  -/


lemma sum_sq_le_sq_sum_of_nonneg (hf : ∀ i ∈ s, 0 ≤ f i) :
    ∑ i ∈ s, f i ^ 2 ≤ (∑ i ∈ s, f i) ^ 2 := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝ : OrderedSemiring R
    f : ι → R
    s : Finset ι
    hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
    ⊢ LE.le (s.sum fun i => HPow.hPow (f i) 2) (HPow.hPow (s.sum fun i => f i) 2)
  -/
  simp only [sq, sum_mul_sum]
  /-
    ι : Type u_1
    R : Type u_2
    inst✝ : OrderedSemiring R
    f : ι → R
    s : Finset ι
    hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
    ⊢ LE.le (s.sum fun x => HMul.hMul (f x) (f x)) (s.sum fun i => s.sum fun j =>  …
  -/
  refine sum_le_sum fun i hi ↦ ?_
  /-
    ι : Type u_1
    R : Type u_2
    inst✝ : OrderedSemiring R
    f : ι → R
    s : Finset ι
    hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
    i : ι
    hi : Membership.mem s i
    ⊢ LE.le (HMul.hMul (f i) (f i)) (s.sum fun j => HMul.hMul (f i) (f j))
  -/
  rw [← mul_sum]
  /-
    ι : Type u_1
    R : Type u_2
    inst✝ : OrderedSemiring R
    f : ι → R
    s : Finset ι
    hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
    i : ι
    hi : Membership.mem s i
    ⊢ LE.le (HMul.hMul (f i) (f i)) (HMul.hMul (f i) (s.sum fun i => f i))
  -/
  gcongr
    /-
      case a0
      ι : Type u_1
      R : Type u_2
      inst✝ : OrderedSemiring R
      f : ι → R
      s : Finset ι
      hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
      i : ι
      hi : Membership.mem s i
      ⊢ LE.le 0 (f i)
    -/
  · exact hf i hi
    /-
      🎉 no goals
    -/
    /-
      case h
      ι : Type u_1
      R : Type u_2
      inst✝ : OrderedSemiring R
      f : ι → R
      s : Finset ι
      hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
      i : ι
      hi : Membership.mem s i
      ⊢ LE.le (f i) (s.sum fun i => f i)
    -/
  · exact single_le_sum hf hi
    /-
      🎉 no goals
    -/


/-- If `g, h ≤ f` and `g i + h i ≤ f i`, then the product of `f` over `s` is at least the
  sum of the products of `g` and `h`. This is the version for `OrderedCommSemiring`. -/
lemma prod_add_prod_le {i : ι} {f g h : ι → R} (hi : i ∈ s) (h2i : g i + h i ≤ f i)
    (hgf : ∀ j ∈ s, j ≠ i → g j ≤ f j) (hhf : ∀ j ∈ s, j ≠ i → h j ≤ f j) (hg : ∀ i ∈ s, 0 ≤ g i)
    (hh : ∀ i ∈ s, 0 ≤ h i) : ((∏ i ∈ s, g i) + ∏ i ∈ s, h i) ≤ ∏ i ∈ s, f i := by
  classical
  simp_rw [prod_eq_mul_prod_diff_singleton hi]
  refine le_trans ?_ (mul_le_mul_of_nonneg_right h2i ?_)
  · rw [right_distrib]
    gcongr with j hj <;> aesop
  · apply prod_nonneg
    simp only [and_imp, mem_sdiff, mem_singleton]
    exact fun j hj hji ↦ le_trans (hg j hj) (hgf j hj hji)


theorem sum_mul_self_eq_zero_iff [LinearOrderedSemiring R] [ExistsAddOfLE R] (s : Finset ι)
    (f : ι → R) : ∑ i ∈ s, f i * f i = 0 ↔ ∀ i ∈ s, f i = 0 := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝¹ : LinearOrderedSemiring R
    inst✝ : ExistsAddOfLE R
    s : Finset ι
    f : ι → R
    ⊢ Iff (Eq (s.sum fun i => HMul.hMul (f i) (f i)) 0) (∀ (i : ι), Membership.mem …
  -/
  rw [sum_eq_zero_iff_of_nonneg fun _ _ ↦ mul_self_nonneg _]
  /-
    ι : Type u_1
    R : Type u_2
    inst✝¹ : LinearOrderedSemiring R
    inst✝ : ExistsAddOfLE R
    s : Finset ι
    f : ι → R
    ⊢ Iff (∀ (i : ι), Membership.mem s i → Eq (HMul.hMul (f i) (f i)) 0) (∀ (i : ι …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma abs_prod [LinearOrderedCommRing R] (s : Finset ι) (f : ι → R) :
    |∏ x ∈ s, f x| = ∏ x ∈ s, |f x| :=
  map_prod absHom _ _


@[simp, norm_cast]
theorem PNat.coe_prod {ι : Type*} (f : ι → ℕ+) (s : Finset ι) :
    ↑(∏ i ∈ s, f i) = (∏ i ∈ s, f i : ℕ) :=
  map_prod PNat.coeMonoidHom _ _


/-- Note that the name is to match `CanonicallyOrderedCommSemiring.mul_pos`. -/
@[simp] lemma _root_.CanonicallyOrderedCommSemiring.prod_pos [Nontrivial R] :
    0 < ∏ i ∈ s, f i ↔ (∀ i ∈ s, (0 : R) < f i) :=
  CanonicallyOrderedCommSemiring.multiset_prod_pos.trans Multiset.forall_mem_map_iff


/-- If `g, h ≤ f` and `g i + h i ≤ f i`, then the product of `f` over `s` is at least the
  sum of the products of `g` and `h`. This is the version for `CanonicallyOrderedCommSemiring`.
-/
lemma prod_add_prod_le' (hi : i ∈ s) (h2i : g i + h i ≤ f i) (hgf : ∀ j ∈ s, j ≠ i → g j ≤ f j)
    (hhf : ∀ j ∈ s, j ≠ i → h j ≤ f j) : ((∏ i ∈ s, g i) + ∏ i ∈ s, h i) ≤ ∏ i ∈ s, f i := by
  classical
  simp_rw [prod_eq_mul_prod_diff_singleton hi]
  refine le_trans ?_ (mul_le_mul_right' h2i _)
  rw [right_distrib]
  gcongr with j hj j hj <;> simp_all


/-- **Cauchy-Schwarz inequality** for finsets.

This is written in terms of sequences `f`, `g`, and `r`, where `r` is a stand-in for
`√(f i * g i)`. See `sum_mul_sq_le_sq_mul_sq` for the more usual form in terms of squared
sequences. -/
lemma sum_sq_le_sum_mul_sum_of_sq_eq_mul [LinearOrderedCommSemiring R] [ExistsAddOfLE R]
    (s : Finset ι) {r f g : ι → R} (hf : ∀ i ∈ s, 0 ≤ f i) (hg : ∀ i ∈ s, 0 ≤ g i)
    (ht : ∀ i ∈ s, r i ^ 2 = f i * g i) : (∑ i ∈ s, r i) ^ 2 ≤ (∑ i ∈ s, f i) * ∑ i ∈ s, g i := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝¹ : LinearOrderedCommSemiring R
    inst✝ : ExistsAddOfLE R
    s : Finset ι
    r f g : ι → R
    hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
    hg : ∀ (i : ι), Membership.mem s i → LE.le 0 (g i)
    ht : ∀ (i : ι), Membership.mem s i → Eq (HPow.hPow (r i) 2) (HMul.hMul (f i) ( …
    ⊢ LE.le (HPow.hPow (s.sum fun i => r i) 2) (HMul.hMul (s.sum fun i => f i) (s. …
  -/
  obtain h | h := (sum_nonneg hg).eq_or_gt
  · have ht' : ∑ i ∈ s, r i = 0 := sum_eq_zero fun i hi ↦ by
      simpa [(sum_eq_zero_iff_of_nonneg hg).1 h i hi] using ht i hi
    /-
      case inl
      ι : Type u_1
      R : Type u_2
      inst✝¹ : LinearOrderedCommSemiring R
      inst✝ : ExistsAddOfLE R
      s : Finset ι
      r f g : ι → R
      hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
      hg : ∀ (i : ι), Membership.mem s i → LE.le 0 (g i)
      ht : ∀ (i : ι), Membership.mem s i → Eq (HPow.hPow (r i) 2) (HMul.hMul (f i) ( …
      h : Eq (s.sum fun i => g i) 0
      ht' : Eq (s.sum fun i => r i) 0
      ⊢ LE.le (HPow.hPow (s.sum fun i => r i) 2) (HMul.hMul (s.sum fun i => f i) (s. …
    -/
    rw [h, ht']
    /-
      case inl
      ι : Type u_1
      R : Type u_2
      inst✝¹ : LinearOrderedCommSemiring R
      inst✝ : ExistsAddOfLE R
      s : Finset ι
      r f g : ι → R
      hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
      hg : ∀ (i : ι), Membership.mem s i → LE.le 0 (g i)
      ht : ∀ (i : ι), Membership.mem s i → Eq (HPow.hPow (r i) 2) (HMul.hMul (f i) ( …
      h : Eq (s.sum fun i => g i) 0
      ht' : Eq (s.sum fun i => r i) 0
      ⊢ LE.le (HPow.hPow 0 2) (HMul.hMul (s.sum fun i => f i) 0)
    -/
    simp
    /-
      🎉 no goals
    -/
  · refine le_of_mul_le_mul_of_pos_left
      (le_of_add_le_add_left (a := (∑ i ∈ s, g i) * (∑ i ∈ s, r i) ^ 2) ?_) h
    calc
      _ = ∑ i ∈ s, 2 * r i * (∑ j ∈ s, g j) * (∑ j ∈ s, r j) := by
          simp_rw [mul_assoc, ← mul_sum, ← sum_mul]; ring
      _ ≤ ∑ i ∈ s, (f i * (∑ j ∈ s, g j) ^ 2 + g i * (∑ j ∈ s, r j) ^ 2) := by
          gcongr with i hi
          have ht : (r i * (∑ j ∈ s, g j) * (∑ j ∈ s, r j)) ^ 2 =
              (f i * (∑ j ∈ s, g j) ^ 2) * (g i * (∑ j ∈ s, r j) ^ 2) := by
            conv_rhs => rw [mul_mul_mul_comm, ← ht i hi]
            ring
          refine le_of_eq_of_le ?_ (two_mul_le_add_of_sq_eq_mul
            (mul_nonneg (hf i hi) (sq_nonneg _)) (mul_nonneg (hg i hi) (sq_nonneg _)) ht)
          repeat rw [mul_assoc]
      _ = _ := by simp_rw [sum_add_distrib, ← sum_mul]; ring


/-- **Cauchy-Schwarz inequality** for finsets, squared version. -/
lemma sum_mul_sq_le_sq_mul_sq [LinearOrderedCommSemiring R] [ExistsAddOfLE R] (s : Finset ι)
    (f g : ι → R) : (∑ i ∈ s, f i * g i) ^ 2 ≤ (∑ i ∈ s, f i ^ 2) * ∑ i ∈ s, g i ^ 2 :=
  sum_sq_le_sum_mul_sum_of_sq_eq_mul s
    (fun _ _ ↦ sq_nonneg _) (fun _ _ ↦ sq_nonneg _) (fun _ _ ↦ mul_pow ..)


/-- **Sedrakyan's lemma**, aka **Titu's lemma** or **Engel's form**.

This is a specialization of the Cauchy-Schwarz inequality with the sequences `f n / √(g n)` and
`√(g n)`, though here it is proven without relying on square roots. -/
theorem sq_sum_div_le_sum_sq_div [LinearOrderedSemifield R] [ExistsAddOfLE R] (s : Finset ι)
    (f : ι → R) {g : ι → R} (hg : ∀ i ∈ s, 0 < g i) :
    (∑ i ∈ s, f i) ^ 2 / ∑ i ∈ s, g i ≤ ∑ i ∈ s, f i ^ 2 / g i := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : ExistsAddOfLE R
    s : Finset ι
    f g : ι → R
    hg : ∀ (i : ι), Membership.mem s i → LT.lt 0 (g i)
    ⊢ LE.le (HDiv.hDiv (HPow.hPow (s.sum fun i => f i) 2) (s.sum fun i => g i)) (s …
  -/
  have hg' : ∀ i ∈ s, 0 ≤ g i := fun i hi ↦ (hg i hi).le
  /-
    ι : Type u_1
    R : Type u_2
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : ExistsAddOfLE R
    s : Finset ι
    f g : ι → R
    hg : ∀ (i : ι), Membership.mem s i → LT.lt 0 (g i)
    hg' : ∀ (i : ι), Membership.mem s i → LE.le 0 (g i)
    ⊢ LE.le (HDiv.hDiv (HPow.hPow (s.sum fun i => f i) 2) (s.sum fun i => g i)) (s …
  -/
  have H : ∀ i ∈ s, 0 ≤ f i ^ 2 / g i := fun i hi ↦ div_nonneg (sq_nonneg _) (hg' i hi)
  refine div_le_of_le_mul₀ (sum_nonneg hg') (sum_nonneg H)
    (sum_sq_le_sum_mul_sum_of_sq_eq_mul _ H hg' fun i hi ↦ ?_)
  /-
    ι : Type u_1
    R : Type u_2
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : ExistsAddOfLE R
    s : Finset ι
    f g : ι → R
    hg : ∀ (i : ι), Membership.mem s i → LT.lt 0 (g i)
    hg' : ∀ (i : ι), Membership.mem s i → LE.le 0 (g i)
    H : ∀ (i : ι), Membership.mem s i → LE.le 0 (HDiv.hDiv (HPow.hPow (f i) 2) (g  …
    i : ι
    hi : Membership.mem s i
    ⊢ Eq (HPow.hPow (f i) 2) (HMul.hMul (HDiv.hDiv (HPow.hPow (f i) 2) (g i)) (g i))
  -/
  rw [div_mul_cancel₀]
  /-
    case h
    ι : Type u_1
    R : Type u_2
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : ExistsAddOfLE R
    s : Finset ι
    f g : ι → R
    hg : ∀ (i : ι), Membership.mem s i → LT.lt 0 (g i)
    hg' : ∀ (i : ι), Membership.mem s i → LE.le 0 (g i)
    H : ∀ (i : ι), Membership.mem s i → LE.le 0 (HDiv.hDiv (HPow.hPow (f i) 2) (g  …
    i : ι
    hi : Membership.mem s i
    ⊢ Ne (g i) 0
  -/
  exact (hg i hi).ne'
  /-
    🎉 no goals
  -/


lemma AbsoluteValue.sum_le [Semiring R] [OrderedSemiring S] (abv : AbsoluteValue R S)
    (s : Finset ι) (f : ι → R) : abv (∑ i ∈ s, f i) ≤ ∑ i ∈ s, abv (f i) :=
  Finset.le_sum_of_subadditive abv (map_zero _) abv.add_le _ _


lemma IsAbsoluteValue.abv_sum [Semiring R] [OrderedSemiring S] (abv : R → S) [IsAbsoluteValue abv]
    (f : ι → R) (s : Finset ι) : abv (∑ i ∈ s, f i) ≤ ∑ i ∈ s, abv (f i) :=
  (IsAbsoluteValue.toAbsoluteValue abv).sum_le _ _


@[deprecated (since := "2024-02-14")] alias abv_sum_le_sum_abv := IsAbsoluteValue.abv_sum


nonrec lemma AbsoluteValue.map_prod [CommSemiring R] [Nontrivial R] [LinearOrderedCommRing S]
    (abv : AbsoluteValue R S) (f : ι → R) (s : Finset ι) :
    abv (∏ i ∈ s, f i) = ∏ i ∈ s, abv (f i) :=
  map_prod abv f s


lemma IsAbsoluteValue.map_prod [CommSemiring R] [Nontrivial R] [LinearOrderedCommRing S]
    (abv : R → S) [IsAbsoluteValue abv] (f : ι → R) (s : Finset ι) :
    abv (∏ i ∈ s, f i) = ∏ i ∈ s, abv (f i) :=
  (IsAbsoluteValue.toAbsoluteValue abv).map_prod _ _


private alias ⟨_, prod_ne_zero⟩ := prod_ne_zero_iff


attribute [local instance] monadLiftOptionMetaM in
/-- The `positivity` extension which proves that `∏ i ∈ s, f i` is nonnegative if `f` is, and
positive if each `f i` is.

TODO: The following example does not work
```
example (s : Finset ℕ) (f : ℕ → ℤ) (hf : ∀ n, 0 ≤ f n) : 0 ≤ s.prod f := by positivity
```
because `compareHyp` can't look for assumptions behind binders.
-/
@[positivity Finset.prod _ _]
def evalFinsetProd : PositivityExt where eval {u α} zα pα e := do
  match e with
  | ~q(@Finset.prod $ι _ $instα $s $f) =>
    let i : Q($ι) ← mkFreshExprMVarQ q($ι) .syntheticOpaque
    have body : Q($α) := Expr.betaRev f #[i]
    let rbody ← core zα pα body
    let _instαmon ← synthInstanceQ q(CommMonoidWithZero $α)

    -- Try to show that the product is positive
    let p_pos : Option Q(0 < $e) := ← do
      let .positive pbody := rbody | pure none -- Fail if the body is not provably positive
      -- TODO(quote4#38): We must name the following, else `assertInstancesCommute` loops.
      let .some _instαzeroone ← trySynthInstanceQ q(ZeroLEOneClass $α) | pure none
      let .some _instαposmul ← trySynthInstanceQ q(PosMulStrictMono $α) | pure none
      let .some _instαnontriv ← trySynthInstanceQ q(Nontrivial $α) | pure none
      assertInstancesCommute
      let pr : Q(∀ i, 0 < $f i) ← mkLambdaFVars #[i] pbody (binderInfoForMVars := .default)
      return some q(prod_pos fun i _ ↦ $pr i)
    if let some p_pos := p_pos then return .positive p_pos

    -- Try to show that the product is nonnegative
    let p_nonneg : Option Q(0 ≤ $e) := ← do
      let .some pbody := rbody.toNonneg
        | return none -- Fail if the body is not provably nonnegative
      let pr : Q(∀ i, 0 ≤ $f i) ← mkLambdaFVars #[i] pbody (binderInfoForMVars := .default)
      -- TODO(quote4#38): We must name the following, else `assertInstancesCommute` loops.
      let .some _instαzeroone ← trySynthInstanceQ q(ZeroLEOneClass $α) | pure none
      let .some _instαposmul ← trySynthInstanceQ q(PosMulMono $α) | pure none
      assertInstancesCommute
      return some q(prod_nonneg fun i _ ↦ $pr i)
    if let some p_nonneg := p_nonneg then return .nonnegative p_nonneg

    -- Fall back to showing that the product is nonzero
    let pbody ← rbody.toNonzero
    let pr : Q(∀ i, $f i ≠ 0) ← mkLambdaFVars #[i] pbody (binderInfoForMVars := .default)
    -- TODO(quote4#38): We must name the following, else `assertInstancesCommute` loops.
    let _instαnontriv ← synthInstanceQ q(Nontrivial $α)
    let _instαnozerodiv ← synthInstanceQ q(NoZeroDivisors $α)
    assertInstancesCommute
    return .nonzero q(prod_ne_zero fun i _ ↦ $pr i)


