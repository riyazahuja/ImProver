lemma natCast_card_filter (p) [DecidablePred p] (s : Finset ι) :
    (#{x ∈ s | p x} : α) = ∑ a ∈ s, if p a then (1 : α) else 0 := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝¹ : AddCommMonoidWithOne α
    p : ι → Prop
    inst✝ : DecidablePred p
    s : Finset ι
    ⊢ Eq (↑(Finset.filter (fun x => p x) s).card) (s.sum fun a => ite (p a) 1 0)
  -/
  rw [sum_ite, sum_const_zero, add_zero, sum_const, nsmul_one]
  /-
    🎉 no goals
  -/


@[simp] lemma sum_boole (p) [DecidablePred p] (s : Finset ι) :
    (∑ x ∈ s, if p x then 1 else 0 : α) = #{x ∈ s | p x} :=
  (natCast_card_filter _ _).symm


lemma sum_mul (s : Finset ι) (f : ι → α) (a : α) :
    (∑ i ∈ s, f i) * a = ∑ i ∈ s, f i * a := map_sum (AddMonoidHom.mulRight a) _ s


lemma mul_sum (s : Finset ι) (f : ι → α) (a : α) :
    a * ∑ i ∈ s, f i = ∑ i ∈ s, a * f i := map_sum (AddMonoidHom.mulLeft a) _ s


lemma sum_mul_sum {κ : Type*} (s : Finset ι) (t : Finset κ) (f : ι → α) (g : κ → α) :
    (∑ i ∈ s, f i) * ∑ j ∈ t, g j = ∑ i ∈ s, ∑ j ∈ t, f i * g j := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝ : NonUnitalNonAssocSemiring α
    κ : Type u_7
    s : Finset ι
    t : Finset κ
    f : ι → α
    g : κ → α
    ⊢ Eq (HMul.hMul (s.sum fun i => f i) (t.sum fun j => g j)) (s.sum fun i => t.s …
  -/
  simp_rw [sum_mul, ← mul_sum]
  /-
    🎉 no goals
  -/


lemma _root_.Fintype.sum_mul_sum {κ : Type*} [Fintype ι] [Fintype κ] (f : ι → α) (g : κ → α) :
    (∑ i, f i) * ∑ j, g j = ∑ i, ∑ j, f i * g j :=
  Finset.sum_mul_sum _ _ _ _


lemma _root_.Commute.sum_right (s : Finset ι) (f : ι → α) (b : α)
    (h : ∀ i ∈ s, Commute b (f i)) : Commute b (∑ i ∈ s, f i) :=
  (Commute.multiset_sum_right _ _) fun b hb => by
    /-
      ι : Type u_1
      α : Type u_3
      inst✝ : NonUnitalNonAssocSemiring α
      s : Finset ι
      f : ι → α
      b✝ : α
      h : ∀ (i : ι), Membership.mem s i → Commute b✝ (f i)
      b : α
      hb : Membership.mem (Multiset.map (fun i => f i) s.val) b
      ⊢ Commute b✝ b
    -/
    obtain ⟨i, hi, rfl⟩ := Multiset.mem_map.mp hb
    /-
      case intro.intro
      ι : Type u_1
      α : Type u_3
      inst✝ : NonUnitalNonAssocSemiring α
      s : Finset ι
      f : ι → α
      b : α
      h : ∀ (i : ι), Membership.mem s i → Commute b (f i)
      i : ι
      hi : Membership.mem s.val i
      hb : Membership.mem (Multiset.map (fun i => f i) s.val) (f i)
      ⊢ Commute b (f i)
    -/
    exact h _ hi
    /-
      🎉 no goals
    -/


lemma _root_.Commute.sum_left (s : Finset ι) (f : ι → α) (b : α)
    (h : ∀ i ∈ s, Commute (f i) b) : Commute (∑ i ∈ s, f i) b :=
  ((Commute.sum_right _ _ _) fun _i hi => (h _ hi).symm).symm


lemma sum_range_succ_mul_sum_range_succ (m n : ℕ) (f g : ℕ → α) :
    (∑ i ∈ range (m + 1), f i) * ∑ i ∈ range (n + 1), g i =
      (∑ i ∈ range m, f i) * ∑ i ∈ range n, g i +
        f m * ∑ i ∈ range n, g i + (∑ i ∈ range m, f i) * g n + f m * g n := by
  /-
    α : Type u_3
    inst✝ : NonUnitalNonAssocSemiring α
    m n : Nat
    f g : Nat → α
    ⊢ Eq (HMul.hMul ((Finset.range (HAdd.hAdd m 1)).sum fun i => f i) ((Finset.ran …
  -/
  simp only [add_mul, mul_add, add_assoc, sum_range_succ]
  /-
    🎉 no goals
  -/


lemma dvd_sum (h : ∀ i ∈ s, a ∣ f i) : a ∣ ∑ i ∈ s, f i :=
                                  /-
                                    ι : Type u_1
                                    α : Type u_3
                                    s : Finset ι
                                    a : α
                                    f : ι → α
                                    inst✝ : NonUnitalSemiring α
                                    h : ∀ (i : ι), Membership.mem s i → Dvd.dvd a (f i)
                                    y : α
                                    hy : Membership.mem (Multiset.map (fun i => f i) s.val) y
                                    ⊢ Dvd.dvd a y
                                  -/
  Multiset.dvd_sum fun y hy => by rcases Multiset.mem_map.1 hy with ⟨x, hx, rfl⟩; exact h x hx
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


lemma sum_mul_boole (s : Finset ι) (f : ι → α) (i : ι) :
                                                               /-
                                                                 ι : Type u_1
                                                                 α : Type u_3
                                                                 inst✝¹ : NonAssocSemiring α
                                                                 inst✝ : DecidableEq ι
                                                                 s : Finset ι
                                                                 f : ι → α
                                                                 i : ι
                                                                 ⊢ Eq (s.sum fun j => HMul.hMul (f j) (ite (Eq i j) 1 0)) (ite (Membership.mem  …
                                                               -/
    ∑ j ∈ s, f j * ite (i = j) 1 0 = ite (i ∈ s) (f i) 0 := by simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


lemma sum_boole_mul (s : Finset ι) (f : ι → α) (i : ι) :
                                                               /-
                                                                 ι : Type u_1
                                                                 α : Type u_3
                                                                 inst✝¹ : NonAssocSemiring α
                                                                 inst✝ : DecidableEq ι
                                                                 s : Finset ι
                                                                 f : ι → α
                                                                 i : ι
                                                                 ⊢ Eq (s.sum fun j => HMul.hMul (ite (Eq i j) 1 0) (f j)) (ite (Membership.mem  …
                                                               -/
    ∑ j ∈ s, ite (i = j) 1 0 * f j = ite (i ∈ s) (f i) 0 := by simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- If `f = g = h` everywhere but at `i`, where `f i = g i + h i`, then the product of `f` over `s`
  is the sum of the products of `g` and `h`. -/
theorem prod_add_prod_eq {s : Finset ι} {i : ι} {f g h : ι → α} (hi : i ∈ s)
    (h1 : g i + h i = f i) (h2 : ∀ j ∈ s, j ≠ i → g j = f j) (h3 : ∀ j ∈ s, j ≠ i → h j = f j) :
    (∏ i ∈ s, g i) + ∏ i ∈ s, h i = ∏ i ∈ s, f i := by
  classical
    simp_rw [prod_eq_mul_prod_diff_singleton hi, ← h1, right_distrib]
    congr 2 <;> apply prod_congr rfl <;> simpa


/-- The product over a sum can be written as a sum over the product of sets, `Finset.Pi`.
  `Finset.prod_univ_sum` is an alternative statement when the product is over `univ`. -/
lemma prod_sum (s : Finset ι) (t : ∀ i, Finset (κ i)) (f : ∀ i, κ i → α) :
    ∏ a ∈ s, ∑ b ∈ t a, f a b = ∑ p ∈ s.pi t, ∏ x ∈ s.attach, f x.1 (p x.1 x.2) := by
  classical
  induction s using Finset.induction with
  | empty => simp
  | insert ha ih =>
    rename_i a s
    have h₁ : ∀ x ∈ t a, ∀ y ∈ t a, x ≠ y →
      Disjoint (image (Pi.cons s a x) (pi s t)) (image (Pi.cons s a y) (pi s t)) := by
      intro x _ y _ h
      simp only [disjoint_iff_ne, mem_image]
      rintro _ ⟨p₂, _, eq₂⟩ _ ⟨p₃, _, eq₃⟩ eq
      have : Pi.cons s a x p₂ a (mem_insert_self _ _)
              = Pi.cons s a y p₃ a (mem_insert_self _ _) := by rw [eq₂, eq₃, eq]
      rw [Pi.cons_same, Pi.cons_same] at this
      exact h this
    rw [prod_insert ha, pi_insert ha, ih, sum_mul, sum_biUnion h₁]
    refine sum_congr rfl fun b _ => ?_
    have h₂ : ∀ p₁ ∈ pi s t, ∀ p₂ ∈ pi s t, Pi.cons s a b p₁ = Pi.cons s a b p₂ → p₁ = p₂ :=
      fun p₁ _ p₂ _ eq => Pi.cons_injective ha eq
    rw [sum_image h₂, mul_sum]
    refine sum_congr rfl fun g _ => ?_
    rw [attach_insert, prod_insert, prod_image]
    · simp only [Pi.cons_same]
      congr with ⟨v, hv⟩
      congr
      exact (Pi.cons_ne (by rintro rfl; exact ha hv)).symm
    · exact fun _ _ _ _ => Subtype.eq ∘ Subtype.mk.inj
    · simpa only [mem_image, mem_attach, Subtype.mk.injEq, true_and,
        Subtype.exists, exists_prop, exists_eq_right] using ha


/-- The product over `univ` of a sum can be written as a sum over the product of sets,
`Fintype.piFinset`. `Finset.prod_sum` is an alternative statement when the product is not
over `univ`. -/
lemma prod_univ_sum [Fintype ι] (t : ∀ i, Finset (κ i)) (f : ∀ i, κ i → α) :
    ∏ i, ∑ j ∈ t i, f i j = ∑ x ∈ piFinset t, ∏ i, f i (x i) := by
  /-
    ι : Type u_1
    α : Type u_3
    κ : ι → Type u_6
    inst✝² : CommSemiring α
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    t : (i : ι) → Finset (κ i)
    f : (i : ι) → κ i → α
    ⊢ Eq (Finset.univ.prod fun i => (t i).sum fun j => f i j) ((Fintype.piFinset t …
  -/
  simp only [prod_attach_univ, prod_sum, Finset.sum_univ_pi]
  /-
    🎉 no goals
  -/


lemma sum_prod_piFinset {κ : Type*} [Fintype ι] (s : Finset κ) (g : ι → κ → α) :
    ∑ f ∈ piFinset fun _ : ι ↦ s, ∏ i, g i (f i) = ∏ i, ∑ j ∈ s, g i j := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝² : CommSemiring α
    inst✝¹ : DecidableEq ι
    κ : Type u_7
    inst✝ : Fintype ι
    s : Finset κ
    g : ι → κ → α
    ⊢ Eq ((Fintype.piFinset fun x => s).sum fun f => Finset.univ.prod fun i => g i …
  -/
  rw [← prod_univ_sum]
  /-
    🎉 no goals
  -/


lemma sum_pow' (s : Finset ι') (f : ι' → α) (n : ℕ) :
    (∑ a ∈ s, f a) ^ n = ∑ p ∈ piFinset fun _i : Fin n ↦ s, ∏ i, f (p i) := by
  /-
    ι' : Type u_2
    α : Type u_3
    inst✝ : CommSemiring α
    s : Finset ι'
    f : ι' → α
    n : Nat
    ⊢ Eq (HPow.hPow (s.sum fun a => f a) n) ((Fintype.piFinset fun _i => s).sum fu …
  -/
  convert @prod_univ_sum (Fin n) _ _ _ _ _ (fun _i ↦ s) fun _i d ↦ f d; simp
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- The product of `f a + g a` over all of `s` is the sum over the powerset of `s` of the product of
`f` over a subset `t` times the product of `g` over the complement of `t`  -/
theorem prod_add (f g : ι → α) (s : Finset ι) :
    ∏ i ∈ s, (f i + g i) = ∑ t ∈ s.powerset, (∏ i ∈ t, f i) * ∏ i ∈ s \ t, g i := by
  classical
  calc
    ∏ i ∈ s, (f i + g i) =
        ∏ i ∈ s, ∑ p ∈ ({True, False} : Finset Prop), if p then f i else g i := by simp
    _ = ∑ p ∈ (s.pi fun _ => {True, False} : Finset (∀ a ∈ s, Prop)),
          ∏ a ∈ s.attach, if p a.1 a.2 then f a.1 else g a.1 := prod_sum _ _ _
    _ = ∑ t ∈ s.powerset, (∏ a ∈ t, f a) * ∏ a ∈ s \ t, g a :=
      sum_bij'
        (fun f _ ↦ {a ∈ s | ∃ h : a ∈ s, f a h})
        (fun t _ a _ => a ∈ t)
        (by simp)
        (by simp [Classical.em])
        (by simp_rw [mem_filter, funext_iff, eq_iff_iff, mem_pi, mem_insert]; tauto)
        (by simp_rw [Finset.ext_iff, @mem_filter _ _ (id _), mem_powerset]; tauto)
        (fun a _ ↦ by
          simp only [prod_ite, filter_attach', prod_map, Function.Embedding.coeFn_mk,
            Subtype.map_coe, id_eq, prod_attach, filter_congr_decidable]
          congr 2 with x
          simp only [mem_filter, mem_sdiff, not_and, not_exists, and_congr_right_iff]
          tauto)


theorem prod_one_add {f : ι → α} (s : Finset ι) :
    ∏ i ∈ s, (1 + f i) = ∑ t ∈ s.powerset, ∏ i ∈ t, f i := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝ : CommSemiring α
    f : ι → α
    s : Finset ι
    ⊢ Eq (s.prod fun i => HAdd.hAdd 1 (f i)) (s.powerset.sum fun t => t.prod fun i …
  -/
  classical simp only [add_comm (1 : α), prod_add, prod_const_one, mul_one]
  /-
    🎉 no goals
  -/


theorem prod_add_one {f : ι → α} (s : Finset ι) :
    ∏ i ∈ s, (f i + 1) = ∑ t ∈ s.powerset, ∏ i ∈ t, f i := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝ : CommSemiring α
    f : ι → α
    s : Finset ι
    ⊢ Eq (s.prod fun i => HAdd.hAdd (f i) 1) (s.powerset.sum fun t => t.prod fun i …
  -/
  classical simp only [prod_add, prod_const_one, mul_one]
  /-
    🎉 no goals
  -/


/-- `∏ i, (f i + g i) = (∏ i, f i) + ∑ i, g i * (∏ j < i, f j + g j) * (∏ j > i, f j)`. -/
theorem prod_add_ordered [LinearOrder ι] (s : Finset ι) (f g : ι → α) :
    ∏ i ∈ s, (f i + g i) =
      (∏ i ∈ s, f i) +
        ∑ i ∈ s, g i * (∏ j ∈ s with j < i, (f j + g j)) * ∏ j ∈ s with i < j, f j := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CommSemiring α
    inst✝ : LinearOrder ι
    s : Finset ι
    f g : ι → α
    ⊢ Eq (s.prod fun i => HAdd.hAdd (f i) (g i)) (HAdd.hAdd (s.prod fun i => f i)  …
  -/
  refine Finset.induction_on_max s (by simp) ?_
  /-
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CommSemiring α
    inst✝ : LinearOrder ι
    s : Finset ι
    f g : ι → α
    ⊢ ∀ (a : ι) (s : Finset ι), (∀ (x : ι), Membership.mem s x → LT.lt x a) → Eq ( …
  -/
  clear s
  /-
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CommSemiring α
    inst✝ : LinearOrder ι
    f g : ι → α
    ⊢ ∀ (a : ι) (s : Finset ι), (∀ (x : ι), Membership.mem s x → LT.lt x a) → Eq ( …
  -/
  intro a s ha ihs
  /-
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CommSemiring α
    inst✝ : LinearOrder ι
    f g : ι → α
    a : ι
    s : Finset ι
    ha : ∀ (x : ι), Membership.mem s x → LT.lt x a
    ihs : Eq (s.prod fun i => HAdd.hAdd (f i) (g i)) (HAdd.hAdd (s.prod fun i => f …
    ⊢ Eq ((Insert.insert a s).prod fun i => HAdd.hAdd (f i) (g i)) (HAdd.hAdd ((In …
  -/
  have ha' : a ∉ s := fun ha' => lt_irrefl a (ha a ha')
  rw [prod_insert ha', prod_insert ha', sum_insert ha', filter_insert, if_neg (lt_irrefl a),
    filter_true_of_mem ha, ihs, add_mul, mul_add, mul_add, add_assoc]
  /-
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CommSemiring α
    inst✝ : LinearOrder ι
    f g : ι → α
    a : ι
    s : Finset ι
    ha : ∀ (x : ι), Membership.mem s x → LT.lt x a
    ihs : Eq (s.prod fun i => HAdd.hAdd (f i) (g i)) (HAdd.hAdd (s.prod fun i => f …
    ha' : Not (Membership.mem s a)
    ⊢ Eq (HAdd.hAdd (HMul.hMul (f a) (s.prod fun i => f i)) (HAdd.hAdd (HMul.hMul  …
  -/
  congr 1
  /-
    case e_a
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CommSemiring α
    inst✝ : LinearOrder ι
    f g : ι → α
    a : ι
    s : Finset ι
    ha : ∀ (x : ι), Membership.mem s x → LT.lt x a
    ihs : Eq (s.prod fun i => HAdd.hAdd (f i) (g i)) (HAdd.hAdd (s.prod fun i => f …
    ha' : Not (Membership.mem s a)
    ⊢ Eq (HAdd.hAdd (HMul.hMul (f a) (s.sum fun i => HMul.hMul (HMul.hMul (g i) (( …
  -/
  rw [add_comm]
  /-
    case e_a
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CommSemiring α
    inst✝ : LinearOrder ι
    f g : ι → α
    a : ι
    s : Finset ι
    ha : ∀ (x : ι), Membership.mem s x → LT.lt x a
    ihs : Eq (s.prod fun i => HAdd.hAdd (f i) (g i)) (HAdd.hAdd (s.prod fun i => f …
    ha' : Not (Membership.mem s a)
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (g a) (s.prod fun i => f i)) (HMul.hMul  …
  -/
  congr 1
    /-
      case e_a.e_a
      ι : Type u_1
      α : Type u_3
      inst✝¹ : CommSemiring α
      inst✝ : LinearOrder ι
      f g : ι → α
      a : ι
      s : Finset ι
      ha : ∀ (x : ι), Membership.mem s x → LT.lt x a
      ihs : Eq (s.prod fun i => HAdd.hAdd (f i) (g i)) (HAdd.hAdd (s.prod fun i => f …
      ha' : Not (Membership.mem s a)
      ⊢ Eq (HAdd.hAdd (HMul.hMul (g a) (s.prod fun i => f i)) (HMul.hMul (g a) (s.su …
    -/
  · rw [filter_false_of_mem, prod_empty, mul_one]
    /-
      case e_a.e_a
      ι : Type u_1
      α : Type u_3
      inst✝¹ : CommSemiring α
      inst✝ : LinearOrder ι
      f g : ι → α
      a : ι
      s : Finset ι
      ha : ∀ (x : ι), Membership.mem s x → LT.lt x a
      ihs : Eq (s.prod fun i => HAdd.hAdd (f i) (g i)) (HAdd.hAdd (s.prod fun i => f …
      ha' : Not (Membership.mem s a)
      ⊢ ∀ (x : ι), Membership.mem (Insert.insert a s) x → Not (LT.lt a x)
    -/
    exact (forall_mem_insert _ _ _).2 ⟨lt_irrefl a, fun i hi => (ha i hi).not_lt⟩
    /-
      🎉 no goals
    -/
    /-
      case e_a.e_a
      ι : Type u_1
      α : Type u_3
      inst✝¹ : CommSemiring α
      inst✝ : LinearOrder ι
      f g : ι → α
      a : ι
      s : Finset ι
      ha : ∀ (x : ι), Membership.mem s x → LT.lt x a
      ihs : Eq (s.prod fun i => HAdd.hAdd (f i) (g i)) (HAdd.hAdd (s.prod fun i => f …
      ha' : Not (Membership.mem s a)
      ⊢ Eq (HMul.hMul (f a) (s.sum fun i => HMul.hMul (HMul.hMul (g i) ((Finset.filt …
    -/
  · rw [mul_sum]
    /-
      case e_a.e_a
      ι : Type u_1
      α : Type u_3
      inst✝¹ : CommSemiring α
      inst✝ : LinearOrder ι
      f g : ι → α
      a : ι
      s : Finset ι
      ha : ∀ (x : ι), Membership.mem s x → LT.lt x a
      ihs : Eq (s.prod fun i => HAdd.hAdd (f i) (g i)) (HAdd.hAdd (s.prod fun i => f …
      ha' : Not (Membership.mem s a)
      ⊢ Eq (s.sum fun i => HMul.hMul (f a) (HMul.hMul (HMul.hMul (g i) ((Finset.filt …
    -/
    refine sum_congr rfl fun i hi => ?_
    rw [filter_insert, if_neg (ha i hi).not_lt, filter_insert, if_pos (ha i hi), prod_insert,
      mul_left_comm]
    /-
      case e_a.e_a
      ι : Type u_1
      α : Type u_3
      inst✝¹ : CommSemiring α
      inst✝ : LinearOrder ι
      f g : ι → α
      a : ι
      s : Finset ι
      ha : ∀ (x : ι), Membership.mem s x → LT.lt x a
      ihs : Eq (s.prod fun i => HAdd.hAdd (f i) (g i)) (HAdd.hAdd (s.prod fun i => f …
      ha' : Not (Membership.mem s a)
      i : ι
      hi : Membership.mem s i
      ⊢ Not (Membership.mem (Finset.filter (fun j => LT.lt i j) s) a)
    -/
    exact mt (fun ha => (mem_filter.1 ha).1) ha'
    /-
      🎉 no goals
    -/


/-- Summing `a ^ #t * b ^ (n - #t)` over all finite subsets `t` of a finset `s`
gives `(a + b) ^ #s`. -/
theorem sum_pow_mul_eq_add_pow (a b : α) (s : Finset ι) :
    (∑ t ∈ s.powerset, a ^ #t * b ^ (#s - #t)) = (a + b) ^ #s := by
  classical
  rw [← prod_const, prod_add]
  refine Finset.sum_congr rfl fun t ht => ?_
  rw [prod_const, prod_const, ← card_sdiff (mem_powerset.1 ht)]


/-- Summing `a^#s * b^(n-#s)` over all finite subsets `s` of a fintype of cardinality `n`
gives `(a + b)^n`. The "good" proof involves expanding along all coordinates using the fact that
`x^n` is multilinear, but multilinear maps are only available now over rings, so we give instead
a proof reducing to the usual binomial theorem to have a result over semirings. -/
lemma _root_.Fintype.sum_pow_mul_eq_add_pow (ι : Type*) [Fintype ι] (a b : α) :
    ∑ s : Finset ι, a ^ #s * b ^ (Fintype.card ι - #s) = (a + b) ^ Fintype.card ι :=
  Finset.sum_pow_mul_eq_add_pow _ _ _


@[norm_cast]
theorem prod_natCast (s : Finset ι) (f : ι → ℕ) : ↑(∏ i ∈ s, f i : ℕ) = ∏ i ∈ s, (f i : α) :=
  map_prod (Nat.castRingHom α) f s


/-- The product of `f i - g i` over all of `s` is the sum over the powerset of `s` of the product of
`g` over a subset `t` times the product of `f` over the complement of `t` times `(-1) ^ #t`. -/
lemma prod_sub [DecidableEq ι] (f g : ι → α) (s : Finset ι) :
    ∏ i ∈ s, (f i - g i) = ∑ t ∈ s.powerset, (-1) ^ #t * (∏ i ∈ s \ t, f i) * ∏ i ∈ t, g i := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CommRing α
    inst✝ : DecidableEq ι
    f g : ι → α
    s : Finset ι
    ⊢ Eq (s.prod fun i => HSub.hSub (f i) (g i)) (s.powerset.sum fun t => HMul.hMu …
  -/
  simp [sub_eq_neg_add, prod_add, ← prod_const, ← prod_mul_distrib, mul_right_comm]
  /-
    🎉 no goals
  -/


/-- `∏ i, (f i - g i) = (∏ i, f i) - ∑ i, g i * (∏ j < i, f j - g j) * (∏ j > i, f j)`. -/
lemma prod_sub_ordered [LinearOrder ι] (s : Finset ι) (f g : ι → α) :
    ∏ i ∈ s, (f i - g i) =
      (∏ i ∈ s, f i) -
        ∑ i ∈ s, g i * (∏ j ∈ s with j < i, (f j - g j)) * ∏ j ∈ s with i < j, f j := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CommRing α
    inst✝ : LinearOrder ι
    s : Finset ι
    f g : ι → α
    ⊢ Eq (s.prod fun i => HSub.hSub (f i) (g i)) (HSub.hSub (s.prod fun i => f i)  …
  -/
  simp only [sub_eq_add_neg]
  /-
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CommRing α
    inst✝ : LinearOrder ι
    s : Finset ι
    f g : ι → α
    ⊢ Eq (s.prod fun x => HAdd.hAdd (f x) (Neg.neg (g x))) (HAdd.hAdd (s.prod fun  …
  -/
  convert prod_add_ordered s f fun i => -g i
  /-
    case h.e'_3.h.e'_6
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CommRing α
    inst✝ : LinearOrder ι
    s : Finset ι
    f g : ι → α
    ⊢ Eq (Neg.neg (s.sum fun x => HMul.hMul (HMul.hMul (g x) ((Finset.filter (fun  …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- `∏ i, (1 - f i) = 1 - ∑ i, f i * (∏ j < i, 1 - f j)`. This formula is useful in construction of
a partition of unity from a collection of “bump” functions. -/
theorem prod_one_sub_ordered [LinearOrder ι] (s : Finset ι) (f : ι → α) :
    ∏ i ∈ s, (1 - f i) = 1 - ∑ i ∈ s, f i * ∏ j ∈ s with j < i, (1 - f j) := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CommRing α
    inst✝ : LinearOrder ι
    s : Finset ι
    f : ι → α
    ⊢ Eq (s.prod fun i => HSub.hSub 1 (f i)) (HSub.hSub 1 (s.sum fun i => HMul.hMu …
  -/
  rw [prod_sub_ordered]
  /-
    ι : Type u_1
    α : Type u_3
    inst✝¹ : CommRing α
    inst✝ : LinearOrder ι
    s : Finset ι
    f : ι → α
    ⊢ Eq (HSub.hSub (s.prod fun i => 1) (s.sum fun i => HMul.hMul (HMul.hMul (f i) …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem prod_range_natCast_sub (n k : ℕ) :
    ∏ i ∈ range k, (n - i : α) = (∏ i ∈ range k, (n - i) : ℕ) := by
  /-
    α : Type u_3
    inst✝ : CommRing α
    n k : Nat
    ⊢ Eq ((Finset.range k).prod fun i => HSub.hSub ↑n ↑i) ↑((Finset.range k).prod  …
  -/
  rw [prod_natCast]
  /-
    α : Type u_3
    inst✝ : CommRing α
    n k : Nat
    ⊢ Eq ((Finset.range k).prod fun i => HSub.hSub ↑n ↑i) ((Finset.range k).prod f …
  -/
  rcases le_or_lt k n with hkn | hnk
    /-
      case inl
      α : Type u_3
      inst✝ : CommRing α
      n k : Nat
      hkn : LE.le k n
      ⊢ Eq ((Finset.range k).prod fun i => HSub.hSub ↑n ↑i) ((Finset.range k).prod f …
    -/
  · exact prod_congr rfl fun i hi => (Nat.cast_sub <| (mem_range.1 hi).le.trans hkn).symm
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_3
      inst✝ : CommRing α
      n k : Nat
      hnk : LT.lt n k
      ⊢ Eq ((Finset.range k).prod fun i => HSub.hSub ↑n ↑i) ((Finset.range k).prod f …
    -/
  · rw [← mem_range] at hnk
    /-
      case inr
      α : Type u_3
      inst✝ : CommRing α
      n k : Nat
      hnk : Membership.mem (Finset.range k) n
      ⊢ Eq ((Finset.range k).prod fun i => HSub.hSub ↑n ↑i) ((Finset.range k).prod f …
    -/
                                                /-
                                                  🎉 no goals
                                                -/
    rw [prod_eq_zero hnk, prod_eq_zero hnk] <;> simp
                                                /-
                                                  🎉 no goals
                                                -/


@[deprecated (since := "2024-05-27")] alias prod_range_cast_nat_sub := prod_range_natCast_sub



lemma _root_.Multiset.sum_map_div {s : Multiset ι} {f : ι → α} {a : α} :
    (s.map (fun x ↦ f x / a)).sum = (s.map f).sum / a := by
  /-
    ι : Type u_1
    α : Type u_3
    inst✝ : DivisionSemiring α
    s : Multiset ι
    f : ι → α
    a : α
    ⊢ Eq (Multiset.map (fun x => HDiv.hDiv (f x) a) s).sum (HDiv.hDiv (Multiset.ma …
  -/
  simp only [div_eq_mul_inv, Multiset.sum_map_mul_right]
  /-
    🎉 no goals
  -/


lemma sum_div (s : Finset ι) (f : ι → α) (a : α) :
                                                /-
                                                  ι : Type u_1
                                                  α : Type u_3
                                                  inst✝ : DivisionSemiring α
                                                  s : Finset ι
                                                  f : ι → α
                                                  a : α
                                                  ⊢ Eq (HDiv.hDiv (s.sum fun i => f i) a) (s.sum fun i => HDiv.hDiv (f i) a)
                                                -/
    (∑ i ∈ s, f i) / a = ∑ i ∈ s, f i / a := by simp only [div_eq_mul_inv, sum_mul]
                                                /-
                                                  🎉 no goals
                                                -/


lemma sum_pow (f : ι → α) (n : ℕ) : (∑ a, f a) ^ n = ∑ p : Fin n → ι, ∏ i, f (p i) := by
  /-
    ι : Type u_7
    α : Type u_9
    inst✝¹ : Fintype ι
    inst✝ : CommSemiring α
    f : ι → α
    n : Nat
    ⊢ Eq (HPow.hPow (Finset.univ.sum fun a => f a) n) (Finset.univ.sum fun p => Fi …
  -/
  simp [sum_pow']
  /-
    🎉 no goals
  -/


/-- A product of sums can be written as a sum of products. -/
lemma prod_sum {κ : ι → Type*} [∀ i, Fintype (κ i)] (f : ∀ i, κ i → α) :
    ∏ i, ∑ j, f i j = ∑ x : ∀ i, κ i, ∏ i, f i (x i) := Finset.prod_univ_sum _ _


lemma prod_add (f g : ι → α) : ∏ a, (f a + g a) = ∑ t, (∏ a ∈ t, f a) * ∏ a ∈ tᶜ, g a := by
  /-
    ι : Type u_7
    α : Type u_9
    inst✝² : Fintype ι
    inst✝¹ : CommSemiring α
    inst✝ : DecidableEq ι
    f g : ι → α
    ⊢ Eq (Finset.univ.prod fun a => HAdd.hAdd (f a) (g a)) (Finset.univ.sum fun t  …
  -/
  simpa [compl_eq_univ_sdiff] using Finset.prod_add f g univ
  /-
    🎉 no goals
  -/


protected lemma sum_div (hf : ∀ i ∈ s, n ∣ f i) : (∑ i ∈ s, f i) / n = ∑ i ∈ s, f i / n := by
  /-
    ι : Type u_7
    s : Finset ι
    f : ι → Nat
    n : Nat
    hf : ∀ (i : ι), Membership.mem s i → Dvd.dvd n (f i)
    ⊢ Eq (HDiv.hDiv (s.sum fun i => f i) n) (s.sum fun i => HDiv.hDiv (f i) n)
  -/
  obtain rfl | hn := n.eq_zero_or_pos
    /-
      case inl
      ι : Type u_7
      s : Finset ι
      f : ι → Nat
      hf : ∀ (i : ι), Membership.mem s i → Dvd.dvd 0 (f i)
      ⊢ Eq (HDiv.hDiv (s.sum fun i => f i) 0) (s.sum fun i => HDiv.hDiv (f i) 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Type u_7
    s : Finset ι
    f : ι → Nat
    n : Nat
    hf : ∀ (i : ι), Membership.mem s i → Dvd.dvd n (f i)
    hn : GT.gt n 0
    ⊢ Eq (HDiv.hDiv (s.sum fun i => f i) n) (s.sum fun i => HDiv.hDiv (f i) n)
  -/
  rw [Nat.div_eq_iff_eq_mul_left hn (dvd_sum hf), sum_mul]
  /-
    case inr
    ι : Type u_7
    s : Finset ι
    f : ι → Nat
    n : Nat
    hf : ∀ (i : ι), Membership.mem s i → Dvd.dvd n (f i)
    hn : GT.gt n 0
    ⊢ Eq (s.sum fun i => f i) (s.sum fun i => HMul.hMul (HDiv.hDiv (f i) n) n)
  -/
  refine sum_congr rfl fun s hs ↦ ?_
  /-
    case inr
    ι : Type u_7
    s✝ : Finset ι
    f : ι → Nat
    n : Nat
    hf : ∀ (i : ι), Membership.mem s✝ i → Dvd.dvd n (f i)
    hn : GT.gt n 0
    s : ι
    hs : Membership.mem s✝ s
    ⊢ Eq (f s) (HMul.hMul (HDiv.hDiv (f s) n) n)
  -/
  rw [Nat.div_mul_cancel (hf _ hs)]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
lemma cast_list_sum [AddMonoidWithOne β] (s : List ℕ) : (↑s.sum : β) = (s.map (↑)).sum :=
  map_list_sum (castAddMonoidHom β) _


@[simp, norm_cast]
lemma cast_list_prod [Semiring β] (s : List ℕ) : (↑s.prod : β) = (s.map (↑)).prod :=
  map_list_prod (castRingHom β) _


@[simp, norm_cast]
lemma cast_multiset_sum [AddCommMonoidWithOne β] (s : Multiset ℕ) :
    (↑s.sum : β) = (s.map (↑)).sum :=
  map_multiset_sum (castAddMonoidHom β) _


@[simp, norm_cast]
lemma cast_multiset_prod [CommSemiring β] (s : Multiset ℕ) : (↑s.prod : β) = (s.map (↑)).prod :=
  map_multiset_prod (castRingHom β) _


@[simp, norm_cast]
lemma cast_sum [AddCommMonoidWithOne β] (s : Finset α) (f : α → ℕ) :
    ↑(∑ x ∈ s, f x : ℕ) = ∑ x ∈ s, (f x : β) :=
  map_sum (castAddMonoidHom β) _ _


@[simp, norm_cast]
lemma cast_prod [CommSemiring β] (f : α → ℕ) (s : Finset α) :
    (↑(∏ i ∈ s, f i) : β) = ∏ i ∈ s, (f i : β) :=
  map_prod (castRingHom β) _ _


protected lemma sum_div (hf : ∀ i ∈ s, n ∣ f i) : (∑ i ∈ s, f i) / n = ∑ i ∈ s, f i / n := by
  /-
    ι : Type u_7
    s : Finset ι
    f : ι → Int
    n : Int
    hf : ∀ (i : ι), Membership.mem s i → Dvd.dvd n (f i)
    ⊢ Eq (HDiv.hDiv (s.sum fun i => f i) n) (s.sum fun i => HDiv.hDiv (f i) n)
  -/
  obtain rfl | hn := eq_or_ne n 0
    /-
      case inl
      ι : Type u_7
      s : Finset ι
      f : ι → Int
      hf : ∀ (i : ι), Membership.mem s i → Dvd.dvd 0 (f i)
      ⊢ Eq (HDiv.hDiv (s.sum fun i => f i) 0) (s.sum fun i => HDiv.hDiv (f i) 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Type u_7
    s : Finset ι
    f : ι → Int
    n : Int
    hf : ∀ (i : ι), Membership.mem s i → Dvd.dvd n (f i)
    hn : Ne n 0
    ⊢ Eq (HDiv.hDiv (s.sum fun i => f i) n) (s.sum fun i => HDiv.hDiv (f i) n)
  -/
  rw [Int.ediv_eq_iff_eq_mul_left hn (dvd_sum hf), sum_mul]
  /-
    case inr
    ι : Type u_7
    s : Finset ι
    f : ι → Int
    n : Int
    hf : ∀ (i : ι), Membership.mem s i → Dvd.dvd n (f i)
    hn : Ne n 0
    ⊢ Eq (s.sum fun i => f i) (s.sum fun i => HMul.hMul (HDiv.hDiv (f i) n) n)
  -/
  refine sum_congr rfl fun s hs ↦ ?_
  /-
    case inr
    ι : Type u_7
    s✝ : Finset ι
    f : ι → Int
    n : Int
    hf : ∀ (i : ι), Membership.mem s✝ i → Dvd.dvd n (f i)
    hn : Ne n 0
    s : ι
    hs : Membership.mem s✝ s
    ⊢ Eq (f s) (HMul.hMul (HDiv.hDiv (f s) n) n)
  -/
  rw [Int.ediv_mul_cancel (hf _ hs)]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
lemma cast_list_sum [AddGroupWithOne β] (s : List ℤ) : (↑s.sum : β) = (s.map (↑)).sum :=
  map_list_sum (castAddHom β) _


@[simp, norm_cast]
lemma cast_list_prod [Ring β] (s : List ℤ) : (↑s.prod : β) = (s.map (↑)).prod :=
  map_list_prod (castRingHom β) _


@[simp, norm_cast]
lemma cast_multiset_sum [AddCommGroupWithOne β] (s : Multiset ℤ) :
    (↑s.sum : β) = (s.map (↑)).sum :=
  map_multiset_sum (castAddHom β) _


@[simp, norm_cast]
lemma cast_multiset_prod {R : Type*} [CommRing R] (s : Multiset ℤ) :
    (↑s.prod : R) = (s.map (↑)).prod :=
  map_multiset_prod (castRingHom R) _


@[simp, norm_cast]
lemma cast_sum [AddCommGroupWithOne β] (s : Finset α) (f : α → ℤ) :
    ↑(∑ x ∈ s, f x : ℤ) = ∑ x ∈ s, (f x : β) :=
  map_sum (castAddHom β) _ _


@[simp, norm_cast]
lemma cast_prod {R : Type*} [CommRing R] (f : α → ℤ) (s : Finset α) :
    (↑(∏ i ∈ s, f i) : R) = ∏ i ∈ s, (f i : R) :=
  map_prod (Int.castRingHom R) _ _


