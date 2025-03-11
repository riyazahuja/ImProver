@[to_additive]
lemma mul_prod_Ico_eq_prod_Icc (h : a ≤ b) : f b * ∏ x ∈ Ico a b, f x = ∏ x ∈ Icc a b, f x := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid M
    f : α → M
    a b : α
    inst✝ : LocallyFiniteOrder α
    h : LE.le a b
    ⊢ Eq (HMul.hMul (f b) ((Finset.Ico a b).prod fun x => f x)) ((Finset.Icc a b). …
  -/
  rw [Icc_eq_cons_Ico h, prod_cons]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_Ico_mul_eq_prod_Icc (h : a ≤ b) : (∏ x ∈ Ico a b, f x) * f b = ∏ x ∈ Icc a b, f x := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid M
    f : α → M
    a b : α
    inst✝ : LocallyFiniteOrder α
    h : LE.le a b
    ⊢ Eq (HMul.hMul ((Finset.Ico a b).prod fun x => f x) (f b)) ((Finset.Icc a b). …
  -/
  rw [mul_comm, mul_prod_Ico_eq_prod_Icc h]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma mul_prod_Ioc_eq_prod_Icc (h : a ≤ b) : f a * ∏ x ∈ Ioc a b, f x = ∏ x ∈ Icc a b, f x := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid M
    f : α → M
    a b : α
    inst✝ : LocallyFiniteOrder α
    h : LE.le a b
    ⊢ Eq (HMul.hMul (f a) ((Finset.Ioc a b).prod fun x => f x)) ((Finset.Icc a b). …
  -/
  rw [Icc_eq_cons_Ioc h, prod_cons]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_Ioc_mul_eq_prod_Icc (h : a ≤ b) : (∏ x ∈ Ioc a b, f x) * f a = ∏ x ∈ Icc a b, f x := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid M
    f : α → M
    a b : α
    inst✝ : LocallyFiniteOrder α
    h : LE.le a b
    ⊢ Eq (HMul.hMul ((Finset.Ioc a b).prod fun x => f x) (f a)) ((Finset.Icc a b). …
  -/
  rw [mul_comm, mul_prod_Ioc_eq_prod_Icc h]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma mul_prod_Ioi_eq_prod_Ici (a : α) : f a * ∏ x ∈ Ioi a, f x = ∏ x ∈ Ici a, f x := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid M
    f : α → M
    inst✝ : LocallyFiniteOrderTop α
    a : α
    ⊢ Eq (HMul.hMul (f a) ((Finset.Ioi a).prod fun x => f x)) ((Finset.Ici a).prod …
  -/
  rw [Ici_eq_cons_Ioi, prod_cons]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_Ioi_mul_eq_prod_Ici (a : α) : (∏ x ∈ Ioi a, f x) * f a = ∏ x ∈ Ici a, f x := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid M
    f : α → M
    inst✝ : LocallyFiniteOrderTop α
    a : α
    ⊢ Eq (HMul.hMul ((Finset.Ioi a).prod fun x => f x) (f a)) ((Finset.Ici a).prod …
  -/
  rw [mul_comm, mul_prod_Ioi_eq_prod_Ici]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma mul_prod_Iio_eq_prod_Iic (a : α) : f a * ∏ x ∈ Iio a, f x = ∏ x ∈ Iic a, f x := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid M
    f : α → M
    inst✝ : LocallyFiniteOrderBot α
    a : α
    ⊢ Eq (HMul.hMul (f a) ((Finset.Iio a).prod fun x => f x)) ((Finset.Iic a).prod …
  -/
  rw [Iic_eq_cons_Iio, prod_cons]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_Iio_mul_eq_prod_Iic (a : α) : (∏ x ∈ Iio a, f x) * f a = ∏ x ∈ Iic a, f x := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝² : PartialOrder α
    inst✝¹ : CommMonoid M
    f : α → M
    inst✝ : LocallyFiniteOrderBot α
    a : α
    ⊢ Eq (HMul.hMul ((Finset.Iio a).prod fun x => f x) (f a)) ((Finset.Iic a).prod …
  -/
  rw [mul_comm, mul_prod_Iio_eq_prod_Iic]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_prod_Ioi_mul_eq_prod_prod_off_diag (f : α → α → M) :
    ∏ i, ∏ j ∈ Ioi i, f j i * f i j = ∏ i, ∏ j ∈ {i}ᶜ, f j i := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝⁴ : Fintype α
    inst✝³ : LinearOrder α
    inst✝² : LocallyFiniteOrderTop α
    inst✝¹ : LocallyFiniteOrderBot α
    inst✝ : CommMonoid M
    f : α → α → M
    ⊢ Eq (Finset.univ.prod fun i => (Finset.Ioi i).prod fun j => HMul.hMul (f j i) …
  -/
  simp_rw [← Ioi_disjUnion_Iio, prod_disjUnion, prod_mul_distrib]
  /-
    α : Type u_1
    M : Type u_2
    inst✝⁴ : Fintype α
    inst✝³ : LinearOrder α
    inst✝² : LocallyFiniteOrderTop α
    inst✝¹ : LocallyFiniteOrderBot α
    inst✝ : CommMonoid M
    f : α → α → M
    ⊢ Eq (HMul.hMul (Finset.univ.prod fun x => (Finset.Ioi x).prod fun x_1 => f x_ …
  -/
  congr 1
  /-
    case e_a
    α : Type u_1
    M : Type u_2
    inst✝⁴ : Fintype α
    inst✝³ : LinearOrder α
    inst✝² : LocallyFiniteOrderTop α
    inst✝¹ : LocallyFiniteOrderBot α
    inst✝ : CommMonoid M
    f : α → α → M
    ⊢ Eq (Finset.univ.prod fun x => (Finset.Ioi x).prod fun x_1 => f x x_1) (Finse …
  -/
  rw [prod_sigma', prod_sigma']
  /-
    case e_a
    α : Type u_1
    M : Type u_2
    inst✝⁴ : Fintype α
    inst✝³ : LinearOrder α
    inst✝² : LocallyFiniteOrderTop α
    inst✝¹ : LocallyFiniteOrderBot α
    inst✝ : CommMonoid M
    f : α → α → M
    ⊢ Eq ((Finset.univ.sigma Finset.Ioi).prod fun x => f x.fst x.snd) ((Finset.uni …
  -/
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
  refine prod_nbij' (fun i ↦ ⟨i.2, i.1⟩) (fun i ↦ ⟨i.2, i.1⟩) ?_ ?_ ?_ ?_ ?_ <;> simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


@[to_additive]
theorem prod_Ico_add' [OrderedCancelAddCommMonoid α] [ExistsAddOfLE α] [LocallyFiniteOrder α]
    (f : α → M) (a b c : α) : (∏ x ∈ Ico a b, f (x + c)) = ∏ x ∈ Ico (a + c) (b + c), f x := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝³ : CommMonoid M
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    f : α → M
    a b c : α
    ⊢ Eq ((Finset.Ico a b).prod fun x => f (HAdd.hAdd x c)) ((Finset.Ico (HAdd.hAd …
  -/
  rw [← map_add_right_Ico, prod_map]
  /-
    α : Type u_1
    M : Type u_2
    inst✝³ : CommMonoid M
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    f : α → M
    a b c : α
    ⊢ Eq ((Finset.Ico a b).prod fun x => f (HAdd.hAdd x c)) ((Finset.Ico a b).prod …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_Ico_add [OrderedCancelAddCommMonoid α] [ExistsAddOfLE α] [LocallyFiniteOrder α]
    (f : α → M) (a b c : α) : (∏ x ∈ Ico a b, f (c + x)) = ∏ x ∈ Ico (a + c) (b + c), f x := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝³ : CommMonoid M
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    f : α → M
    a b c : α
    ⊢ Eq ((Finset.Ico a b).prod fun x => f (HAdd.hAdd c x)) ((Finset.Ico (HAdd.hAd …
  -/
  convert prod_Ico_add' f a b c using 2
  /-
    case h.e'_2.a
    α : Type u_1
    M : Type u_2
    inst✝³ : CommMonoid M
    inst✝² : OrderedCancelAddCommMonoid α
    inst✝¹ : ExistsAddOfLE α
    inst✝ : LocallyFiniteOrder α
    f : α → M
    a b c x✝ : α
    a✝ : Membership.mem (Finset.Ico a b) x✝
    ⊢ Eq (f (HAdd.hAdd c x✝)) (f (HAdd.hAdd x✝ c))
  -/
  rw [add_comm]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem prod_Ico_add_right_sub_eq [OrderedCancelAddCommMonoid α] [ExistsAddOfLE α]
    [LocallyFiniteOrder α] [Sub α] [OrderedSub α] (a b c : α) :
    ∏ x ∈ Ico (a + c) (b + c), f (x - c) = ∏ x ∈ Ico a b, f x := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝⁵ : CommMonoid M
    f : α → M
    inst✝⁴ : OrderedCancelAddCommMonoid α
    inst✝³ : ExistsAddOfLE α
    inst✝² : LocallyFiniteOrder α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    a b c : α
    ⊢ Eq ((Finset.Ico (HAdd.hAdd a c) (HAdd.hAdd b c)).prod fun x => f (HSub.hSub  …
  -/
  simp only [← map_add_right_Ico, prod_map, addRightEmbedding_apply, add_tsub_cancel_right]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_Ico_succ_top {a b : ℕ} (hab : a ≤ b) (f : ℕ → M) :
    (∏ k ∈ Ico a (b + 1), f k) = (∏ k ∈ Ico a b, f k) * f b := by
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    a b : Nat
    hab : LE.le a b
    f : Nat → M
    ⊢ Eq ((Finset.Ico a (HAdd.hAdd b 1)).prod fun k => f k) (HMul.hMul ((Finset.Ic …
  -/
  rw [Nat.Ico_succ_right_eq_insert_Ico hab, prod_insert right_not_mem_Ico, mul_comm]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_eq_prod_Ico_succ_bot {a b : ℕ} (hab : a < b) (f : ℕ → M) :
    ∏ k ∈ Ico a b, f k = f a * ∏ k ∈ Ico (a + 1) b, f k := by
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    a b : Nat
    hab : LT.lt a b
    f : Nat → M
    ⊢ Eq ((Finset.Ico a b).prod fun k => f k) (HMul.hMul (f a) ((Finset.Ico (HAdd. …
  -/
  have ha : a ∉ Ico (a + 1) b := by simp
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    a b : Nat
    hab : LT.lt a b
    f : Nat → M
    ha : Not (Membership.mem (Finset.Ico (HAdd.hAdd a 1) b) a)
    ⊢ Eq ((Finset.Ico a b).prod fun k => f k) (HMul.hMul (f a) ((Finset.Ico (HAdd. …
  -/
  rw [← prod_insert ha, Nat.Ico_insert_succ_left hab]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_Ico_consecutive (f : ℕ → M) {m n k : ℕ} (hmn : m ≤ n) (hnk : n ≤ k) :
    ((∏ i ∈ Ico m n, f i) * ∏ i ∈ Ico n k, f i) = ∏ i ∈ Ico m k, f i :=
  Ico_union_Ico_eq_Ico hmn hnk ▸ Eq.symm (prod_union (Ico_disjoint_Ico_consecutive m n k))


@[to_additive]
theorem prod_Ioc_consecutive (f : ℕ → M) {m n k : ℕ} (hmn : m ≤ n) (hnk : n ≤ k) :
    ((∏ i ∈ Ioc m n, f i) * ∏ i ∈ Ioc n k, f i) = ∏ i ∈ Ioc m k, f i := by
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    f : Nat → M
    m n k : Nat
    hmn : LE.le m n
    hnk : LE.le n k
    ⊢ Eq (HMul.hMul ((Finset.Ioc m n).prod fun i => f i) ((Finset.Ioc n k).prod fu …
  -/
  rw [← Ioc_union_Ioc_eq_Ioc hmn hnk, prod_union]
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    f : Nat → M
    m n k : Nat
    hmn : LE.le m n
    hnk : LE.le n k
    ⊢ Disjoint (Finset.Ioc m n) (Finset.Ioc n k)
  -/
  apply disjoint_left.2 fun x hx h'x => _
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    f : Nat → M
    m n k : Nat
    hmn : LE.le m n
    hnk : LE.le n k
    ⊢ ∀ (x : Nat), Membership.mem (Finset.Ioc m n) x → Membership.mem (Finset.Ioc  …
  -/
  intros x hx h'x
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    f : Nat → M
    m n k : Nat
    hmn : LE.le m n
    hnk : LE.le n k
    x : Nat
    hx : Membership.mem (Finset.Ioc m n) x
    h'x : Membership.mem (Finset.Ioc n k) x
    ⊢ False
  -/
  exact lt_irrefl _ ((mem_Ioc.1 h'x).1.trans_le (mem_Ioc.1 hx).2)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_Ioc_succ_top {a b : ℕ} (hab : a ≤ b) (f : ℕ → M) :
    (∏ k ∈ Ioc a (b + 1), f k) = (∏ k ∈ Ioc a b, f k) * f (b + 1) := by
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    a b : Nat
    hab : LE.le a b
    f : Nat → M
    ⊢ Eq ((Finset.Ioc a (HAdd.hAdd b 1)).prod fun k => f k) (HMul.hMul ((Finset.Io …
  -/
  rw [← prod_Ioc_consecutive _ hab (Nat.le_succ b), Nat.Ioc_succ_singleton, prod_singleton]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_Icc_succ_top {a b : ℕ} (hab : a ≤ b + 1) (f : ℕ → M) :
    (∏ k in Icc a (b + 1), f k) = (∏ k in Icc a b, f k) * f (b + 1) := by
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    a b : Nat
    hab : LE.le a (HAdd.hAdd b 1)
    f : Nat → M
    ⊢ Eq ((Finset.Icc a (HAdd.hAdd b 1)).prod fun k => f k) (HMul.hMul ((Finset.Ic …
  -/
  rw [← Nat.Ico_succ_right, prod_Ico_succ_top hab, Nat.Ico_succ_right]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_range_mul_prod_Ico (f : ℕ → M) {m n : ℕ} (h : m ≤ n) :
    ((∏ k ∈ range m, f k) * ∏ k ∈ Ico m n, f k) = ∏ k ∈ range n, f k :=
  Nat.Ico_zero_eq_range ▸ Nat.Ico_zero_eq_range ▸ prod_Ico_consecutive f m.zero_le h


@[to_additive]
theorem prod_range_eq_mul_Ico (f : ℕ → M) {n : ℕ} (hn : 0 < n) :
    ∏ x ∈ Finset.range n, f x = f 0 * ∏ x ∈ Ico 1 n, f x :=
  Finset.range_eq_Ico ▸ Finset.prod_eq_prod_Ico_succ_bot hn f


@[to_additive]
theorem prod_Ico_eq_mul_inv {δ : Type*} [CommGroup δ] (f : ℕ → δ) {m n : ℕ} (h : m ≤ n) :
    ∏ k ∈ Ico m n, f k = (∏ k ∈ range n, f k) * (∏ k ∈ range m, f k)⁻¹ :=
                                 /-
                                   δ : Type u_3
                                   inst✝ : CommGroup δ
                                   f : Nat → δ
                                   m n : Nat
                                   h : LE.le m n
                                   ⊢ Eq (HMul.hMul ((Finset.Ico m n).prod fun k => f k) ((Finset.range m).prod fu …
                                 -/
  eq_mul_inv_iff_mul_eq.2 <| by (rw [mul_comm]; exact prod_range_mul_prod_Ico f h)
                                                /-
                                                  🎉 no goals
                                                -/


@[to_additive]
theorem prod_Ico_eq_div {δ : Type*} [CommGroup δ] (f : ℕ → δ) {m n : ℕ} (h : m ≤ n) :
    ∏ k ∈ Ico m n, f k = (∏ k ∈ range n, f k) / ∏ k ∈ range m, f k := by
  /-
    δ : Type u_3
    inst✝ : CommGroup δ
    f : Nat → δ
    m n : Nat
    h : LE.le m n
    ⊢ Eq ((Finset.Ico m n).prod fun k => f k) (HDiv.hDiv ((Finset.range n).prod fu …
  -/
  simpa only [div_eq_mul_inv] using prod_Ico_eq_mul_inv f h
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_range_div_prod_range {α : Type*} [CommGroup α] {f : ℕ → α} {n m : ℕ} (hnm : n ≤ m) :
    ((∏ k ∈ range m, f k) / ∏ k ∈ range n, f k) =
    ∏ k ∈ (range m).filter fun k => n ≤ k, f k := by
  /-
    α : Type u_3
    inst✝ : CommGroup α
    f : Nat → α
    n m : Nat
    hnm : LE.le n m
    ⊢ Eq (HDiv.hDiv ((Finset.range m).prod fun k => f k) ((Finset.range n).prod fu …
  -/
  rw [← prod_Ico_eq_div f hnm]
  /-
    α : Type u_3
    inst✝ : CommGroup α
    f : Nat → α
    n m : Nat
    hnm : LE.le n m
    ⊢ Eq ((Finset.Ico n m).prod fun k => f k) ((Finset.filter (fun k => LE.le n k) …
  -/
  congr
  /-
    case e_s
    α : Type u_3
    inst✝ : CommGroup α
    f : Nat → α
    n m : Nat
    hnm : LE.le n m
    ⊢ Eq (Finset.Ico n m) (Finset.filter (fun k => LE.le n k) (Finset.range m))
  -/
  apply Finset.ext
  /-
    case e_s.h
    α : Type u_3
    inst✝ : CommGroup α
    f : Nat → α
    n m : Nat
    hnm : LE.le n m
    ⊢ ∀ (a : Nat), Iff (Membership.mem (Finset.Ico n m) a) (Membership.mem (Finset …
  -/
  simp only [mem_Ico, mem_filter, mem_range, *]
  /-
    case e_s.h
    α : Type u_3
    inst✝ : CommGroup α
    f : Nat → α
    n m : Nat
    hnm : LE.le n m
    ⊢ ∀ (a : Nat), Iff (And (LE.le n a) (LT.lt a m)) (And (LT.lt a m) (LE.le n a))
  -/
  tauto
  /-
    🎉 no goals
  -/


/-- The two ways of summing over `(i, j)` in the range `a ≤ i ≤ j < b` are equal. -/
theorem sum_Ico_Ico_comm {M : Type*} [AddCommMonoid M] (a b : ℕ) (f : ℕ → ℕ → M) :
    (∑ i ∈ Finset.Ico a b, ∑ j ∈ Finset.Ico i b, f i j) =
      ∑ j ∈ Finset.Ico a b, ∑ i ∈ Finset.Ico a (j + 1), f i j := by
  /-
    M : Type u_3
    inst✝ : AddCommMonoid M
    a b : Nat
    f : Nat → Nat → M
    ⊢ Eq ((Finset.Ico a b).sum fun i => (Finset.Ico i b).sum fun j => f i j) ((Fin …
  -/
  rw [Finset.sum_sigma', Finset.sum_sigma']
  refine sum_nbij' (fun x ↦ ⟨x.2, x.1⟩) (fun x ↦ ⟨x.2, x.1⟩) ?_ ?_ (fun _ _ ↦ rfl) (fun _ _ ↦ rfl)
    (fun _ _ ↦ rfl) <;>
  /-
    case refine_1
    M : Type u_3
    inst✝ : AddCommMonoid M
    a b : Nat
    f : Nat → Nat → M
    ⊢ ∀ (a_1 : Sigma fun i => Nat), Membership.mem ((Finset.Ico a b).sigma fun i = …
  -/
  simp only [Finset.mem_Ico, Sigma.forall, Finset.mem_sigma] <;>
  /-
    case refine_1
    M : Type u_3
    inst✝ : AddCommMonoid M
    a b : Nat
    f : Nat → Nat → M
    ⊢ ∀ (a_1 b_1 : Nat), And (And (LE.le a a_1) (LT.lt a_1 b)) (And (LE.le a_1 b_1 …
  -/
  rintro a b ⟨⟨h₁, h₂⟩, ⟨h₃, h₄⟩⟩ <;>
  /-
    case refine_1.intro.intro.intro
    M : Type u_3
    inst✝ : AddCommMonoid M
    a✝ b✝ : Nat
    f : Nat → Nat → M
    a b : Nat
    h₁ : LE.le a✝ a
    h₂ : LT.lt a b✝
    h₃ : LE.le a b
    h₄ : LT.lt b b✝
    ⊢ And (And (LE.le a✝ b) (LT.lt b b✝)) (And (LE.le a✝ a) (LT.lt a (HAdd.hAdd b  …
  -/
  /-
    🎉 no goals
  -/
  omega
  /-
    🎉 no goals
  -/


/-- The two ways of summing over `(i, j)` in the range `a ≤ i < j < b` are equal. -/
theorem sum_Ico_Ico_comm' {M : Type*} [AddCommMonoid M] (a b : ℕ) (f : ℕ → ℕ → M) :
    (∑ i ∈ Finset.Ico a b, ∑ j ∈ Finset.Ico (i + 1) b, f i j) =
      ∑ j ∈ Finset.Ico a b, ∑ i ∈ Finset.Ico a j, f i j := by
  /-
    M : Type u_3
    inst✝ : AddCommMonoid M
    a b : Nat
    f : Nat → Nat → M
    ⊢ Eq ((Finset.Ico a b).sum fun i => (Finset.Ico (HAdd.hAdd i 1) b).sum fun j = …
  -/
  rw [Finset.sum_sigma', Finset.sum_sigma']
  refine sum_nbij' (fun x ↦ ⟨x.2, x.1⟩) (fun x ↦ ⟨x.2, x.1⟩) ?_ ?_ (fun _ _ ↦ rfl) (fun _ _ ↦ rfl)
    (fun _ _ ↦ rfl) <;>
  /-
    case refine_1
    M : Type u_3
    inst✝ : AddCommMonoid M
    a b : Nat
    f : Nat → Nat → M
    ⊢ ∀ (a_1 : Sigma fun i => Nat), Membership.mem ((Finset.Ico a b).sigma fun i = …
  -/
  simp only [Finset.mem_Ico, Sigma.forall, Finset.mem_sigma] <;>
  /-
    case refine_1
    M : Type u_3
    inst✝ : AddCommMonoid M
    a b : Nat
    f : Nat → Nat → M
    ⊢ ∀ (a_1 b_1 : Nat), And (And (LE.le a a_1) (LT.lt a_1 b)) (And (LE.le (HAdd.h …
  -/
  rintro a b ⟨⟨h₁, h₂⟩, ⟨h₃, h₄⟩⟩ <;>
  /-
    case refine_1.intro.intro.intro
    M : Type u_3
    inst✝ : AddCommMonoid M
    a✝ b✝ : Nat
    f : Nat → Nat → M
    a b : Nat
    h₁ : LE.le a✝ a
    h₂ : LT.lt a b✝
    h₃ : LE.le (HAdd.hAdd a 1) b
    h₄ : LT.lt b b✝
    ⊢ And (And (LE.le a✝ b) (LT.lt b b✝)) (And (LE.le a✝ a) (LT.lt a b))
  -/
  /-
    🎉 no goals
  -/
  omega
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_Ico_eq_prod_range (f : ℕ → M) (m n : ℕ) :
    ∏ k ∈ Ico m n, f k = ∏ k ∈ range (n - m), f (m + k) := by
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    f : Nat → M
    m n : Nat
    ⊢ Eq ((Finset.Ico m n).prod fun k => f k) ((Finset.range (HSub.hSub n m)).prod …
  -/
  by_cases h : m ≤ n
    /-
      case pos
      M : Type u_2
      inst✝ : CommMonoid M
      f : Nat → M
      m n : Nat
      h : LE.le m n
      ⊢ Eq ((Finset.Ico m n).prod fun k => f k) ((Finset.range (HSub.hSub n m)).prod …
    -/
  · rw [← Nat.Ico_zero_eq_range, prod_Ico_add, zero_add, tsub_add_cancel_of_le h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      M : Type u_2
      inst✝ : CommMonoid M
      f : Nat → M
      m n : Nat
      h : Not (LE.le m n)
      ⊢ Eq ((Finset.Ico m n).prod fun k => f k) ((Finset.range (HSub.hSub n m)).prod …
    -/
  · replace h : n ≤ m := le_of_not_ge h
    /-
      case neg
      M : Type u_2
      inst✝ : CommMonoid M
      f : Nat → M
      m n : Nat
      h : LE.le n m
      ⊢ Eq ((Finset.Ico m n).prod fun k => f k) ((Finset.range (HSub.hSub n m)).prod …
    -/
    rw [Ico_eq_empty_of_le h, tsub_eq_zero_iff_le.mpr h, range_zero, prod_empty, prod_empty]
    /-
      🎉 no goals
    -/


theorem prod_Ico_reflect (f : ℕ → M) (k : ℕ) {m n : ℕ} (h : m ≤ n + 1) :
    (∏ j ∈ Ico k m, f (n - j)) = ∏ j ∈ Ico (n + 1 - m) (n + 1 - k), f j := by
  have : ∀ i < m, i ≤ n := by
    intro i hi
    exact (add_le_add_iff_right 1).1 (le_trans (Nat.lt_iff_add_one_le.1 hi) h)
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    f : Nat → M
    k m n : Nat
    h : LE.le m (HAdd.hAdd n 1)
    this : ∀ (i : Nat), LT.lt i m → LE.le i n
    ⊢ Eq ((Finset.Ico k m).prod fun j => f (HSub.hSub n j)) ((Finset.Ico (HSub.hSu …
  -/
  rcases lt_or_le k m with hkm | hkm
    /-
      case inl
      M : Type u_2
      inst✝ : CommMonoid M
      f : Nat → M
      k m n : Nat
      h : LE.le m (HAdd.hAdd n 1)
      this : ∀ (i : Nat), LT.lt i m → LE.le i n
      hkm : LT.lt k m
      ⊢ Eq ((Finset.Ico k m).prod fun j => f (HSub.hSub n j)) ((Finset.Ico (HSub.hSu …
    -/
  · rw [← Nat.Ico_image_const_sub_eq_Ico (this _ hkm)]
    /-
      case inl
      M : Type u_2
      inst✝ : CommMonoid M
      f : Nat → M
      k m n : Nat
      h : LE.le m (HAdd.hAdd n 1)
      this : ∀ (i : Nat), LT.lt i m → LE.le i n
      hkm : LT.lt k m
      ⊢ Eq ((Finset.Ico k m).prod fun j => f (HSub.hSub n j)) ((Finset.image (fun x  …
    -/
    refine (prod_image ?_).symm
    /-
      case inl
      M : Type u_2
      inst✝ : CommMonoid M
      f : Nat → M
      k m n : Nat
      h : LE.le m (HAdd.hAdd n 1)
      this : ∀ (i : Nat), LT.lt i m → LE.le i n
      hkm : LT.lt k m
      ⊢ ∀ (x : Nat), Membership.mem (Finset.Ico k m) x → ∀ (y : Nat), Membership.mem …
    -/
    simp only [mem_Ico]
    /-
      case inl
      M : Type u_2
      inst✝ : CommMonoid M
      f : Nat → M
      k m n : Nat
      h : LE.le m (HAdd.hAdd n 1)
      this : ∀ (i : Nat), LT.lt i m → LE.le i n
      hkm : LT.lt k m
      ⊢ ∀ (x : Nat), And (LE.le k x) (LT.lt x m) → ∀ (y : Nat), And (LE.le k y) (LT. …
    -/
    rintro i ⟨_, im⟩ j ⟨_, jm⟩ Hij
    /-
      case inl.intro.intro
      M : Type u_2
      inst✝ : CommMonoid M
      f : Nat → M
      k m n : Nat
      h : LE.le m (HAdd.hAdd n 1)
      this : ∀ (i : Nat), LT.lt i m → LE.le i n
      hkm : LT.lt k m
      i : Nat
      left✝¹ : LE.le k i
      im : LT.lt i m
      j : Nat
      left✝ : LE.le k j
      jm : LT.lt j m
      Hij : Eq (HSub.hSub n i) (HSub.hSub n j)
      ⊢ Eq i j
    -/
    rw [← tsub_tsub_cancel_of_le (this _ im), Hij, tsub_tsub_cancel_of_le (this _ jm)]
    /-
      🎉 no goals
    -/
  · have : n + 1 - k ≤ n + 1 - m := by
      rw [tsub_le_tsub_iff_left h]
      exact hkm
    simp only [hkm, Ico_eq_empty_of_le, prod_empty, tsub_le_iff_right, Ico_eq_empty_of_le
      this]


theorem sum_Ico_reflect {δ : Type*} [AddCommMonoid δ] (f : ℕ → δ) (k : ℕ) {m n : ℕ}
    (h : m ≤ n + 1) : (∑ j ∈ Ico k m, f (n - j)) = ∑ j ∈ Ico (n + 1 - m) (n + 1 - k), f j :=
  @prod_Ico_reflect (Multiplicative δ) _ f k m n h


theorem prod_range_reflect (f : ℕ → M) (n : ℕ) :
    (∏ j ∈ range n, f (n - 1 - j)) = ∏ j ∈ range n, f j := by
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    f : Nat → M
    n : Nat
    ⊢ Eq ((Finset.range n).prod fun j => f (HSub.hSub (HSub.hSub n 1) j)) ((Finset …
  -/
  cases n
    /-
      case zero
      M : Type u_2
      inst✝ : CommMonoid M
      f : Nat → M
      ⊢ Eq ((Finset.range 0).prod fun j => f (HSub.hSub (HSub.hSub 0 1) j)) ((Finset …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      M : Type u_2
      inst✝ : CommMonoid M
      f : Nat → M
      n✝ : Nat
      ⊢ Eq ((Finset.range (HAdd.hAdd n✝ 1)).prod fun j => f (HSub.hSub (HSub.hSub (H …
    -/
  · simp only [← Nat.Ico_zero_eq_range, Nat.succ_sub_succ_eq_sub, tsub_zero]
    /-
      case succ
      M : Type u_2
      inst✝ : CommMonoid M
      f : Nat → M
      n✝ : Nat
      ⊢ Eq ((Finset.Ico 0 (HAdd.hAdd n✝ 1)).prod fun x => f (HSub.hSub n✝ x)) ((Fins …
    -/
    rw [prod_Ico_reflect _ _ le_rfl]
    /-
      case succ
      M : Type u_2
      inst✝ : CommMonoid M
      f : Nat → M
      n✝ : Nat
      ⊢ Eq ((Finset.Ico (HSub.hSub (HAdd.hAdd n✝ 1) (HAdd.hAdd n✝ 1)) (HSub.hSub (HA …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem sum_range_reflect {δ : Type*} [AddCommMonoid δ] (f : ℕ → δ) (n : ℕ) :
    (∑ j ∈ range n, f (n - 1 - j)) = ∑ j ∈ range n, f j :=
  @prod_range_reflect (Multiplicative δ) _ f n


@[simp]
theorem prod_Ico_id_eq_factorial : ∀ n : ℕ, (∏ x ∈ Ico 1 (n + 1), x) = n !
  | 0 => rfl
  | n + 1 => by
    rw [prod_Ico_succ_top <| Nat.succ_le_succ <| Nat.zero_le n, Nat.factorial_succ,
      prod_Ico_id_eq_factorial n, Nat.succ_eq_add_one, mul_comm]


@[simp]
theorem prod_range_add_one_eq_factorial : ∀ n : ℕ, (∏ x ∈ range n, (x + 1)) = n !
  | 0 => rfl
                /-
                  n : Nat
                  ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).prod fun x => HAdd.hAdd x 1) (HAdd.hAdd n …
                -/
  | n + 1 => by simp [factorial, Finset.range_succ, prod_range_add_one_eq_factorial n]
                /-
                  🎉 no goals
                -/


/-- Gauss' summation formula -/
theorem sum_range_id_mul_two (n : ℕ) : (∑ i ∈ range n, i) * 2 = n * (n - 1) :=
  calc
    (∑ i ∈ range n, i) * 2 = (∑ i ∈ range n, i) + ∑ i ∈ range n, (n - 1 - i) := by
      /-
        n : Nat
        ⊢ Eq (HMul.hMul ((Finset.range n).sum fun i => i) 2) (HAdd.hAdd ((Finset.range …
      -/
      rw [sum_range_reflect (fun i => i) n, mul_two]
      /-
        🎉 no goals
      -/
    _ = ∑ i ∈ range n, (i + (n - 1 - i)) := sum_add_distrib.symm
    _ = ∑ _ ∈ range n, (n - 1) :=
      sum_congr rfl fun _ hi => add_tsub_cancel_of_le <| Nat.le_sub_one_of_lt <| mem_range.1 hi
                          /-
                            n : Nat
                            ⊢ Eq ((Finset.range n).sum fun x => HSub.hSub n 1) (HMul.hMul n (HSub.hSub n 1))
                          -/
    _ = n * (n - 1) := by rw [sum_const, card_range, Nat.nsmul_eq_mul]
                          /-
                            🎉 no goals
                          -/


/-- Gauss' summation formula -/
theorem sum_range_id (n : ℕ) : ∑ i ∈ range n, i = n * (n - 1) / 2 := by
  /-
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun i => i) (HDiv.hDiv (HMul.hMul n (HSub.hSub n 1) …
  -/
  rw [← sum_range_id_mul_two n, Nat.mul_div_cancel _ zero_lt_two]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_range_diag_flip (n : ℕ) (f : ℕ → ℕ → M) :
    (∏ m ∈ range n, ∏ k ∈ range (m + 1), f k (m - k)) =
      ∏ m ∈ range n, ∏ k ∈ range (n - m), f m k := by
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    n : Nat
    f : Nat → Nat → M
    ⊢ Eq ((Finset.range n).prod fun m => (Finset.range (HAdd.hAdd m 1)).prod fun k …
  -/
  rw [prod_sigma', prod_sigma']
  /-
    M : Type u_2
    inst✝ : CommMonoid M
    n : Nat
    f : Nat → Nat → M
    ⊢ Eq (((Finset.range n).sigma fun m => Finset.range (HAdd.hAdd m 1)).prod fun  …
  -/
  refine prod_nbij' (fun a ↦ ⟨a.2, a.1 - a.2⟩) (fun a ↦ ⟨a.1 + a.2, a.1⟩) ?_ ?_ ?_ ?_ ?_ <;>
    simp +contextual only [mem_sigma, mem_range, lt_tsub_iff_left,
      Nat.lt_succ_iff, le_add_iff_nonneg_right, Nat.zero_le, and_true, and_imp, imp_self,
      implies_true, Sigma.forall, forall_const, add_tsub_cancel_of_le, Sigma.mk.inj_iff,
      add_tsub_cancel_left, heq_eq_eq]
  /-
    case refine_1
    M : Type u_2
    inst✝ : CommMonoid M
    n : Nat
    f : Nat → Nat → M
    ⊢ ∀ (a b : Nat), LT.lt a n → LE.le b a → LT.lt b n
  -/
  exact fun a b han hba ↦ lt_of_le_of_lt hba han
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_range_succ_div_prod : ((∏ i ∈ range (n + 1), f i) / ∏ i ∈ range n, f i) = f n :=
  div_eq_iff_eq_mul'.mpr <| prod_range_succ f n


@[to_additive]
theorem prod_range_succ_div_top : (∏ i ∈ range (n + 1), f i) / f n = ∏ i ∈ range n, f i :=
  div_eq_iff_eq_mul.mpr <| prod_range_succ f n


@[to_additive]
theorem prod_Ico_div_bot (hmn : m < n) : (∏ i ∈ Ico m n, f i) / f m = ∏ i ∈ Ico (m + 1) n, f i :=
  div_eq_iff_eq_mul'.mpr <| prod_eq_prod_Ico_succ_bot hmn _


@[to_additive]
theorem prod_Ico_succ_div_top (hmn : m ≤ n) :
    (∏ i ∈ Ico m (n + 1), f i) / f n = ∏ i ∈ Ico m n, f i :=
  div_eq_iff_eq_mul.mpr <| prod_Ico_succ_top hmn _


