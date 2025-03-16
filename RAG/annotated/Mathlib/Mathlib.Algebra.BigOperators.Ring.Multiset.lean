@[simp] lemma prod_map_neg (s : Multiset α) : (s.map Neg.neg).prod = (-1) ^ card s * s.prod :=
                             /-
                               α : Type u_2
                               inst✝¹ : CommMonoid α
                               inst✝ : HasDistribNeg α
                               s : Multiset α
                               ⊢ ∀ (a : List α), Eq (Multiset.map Neg.neg (Quotient.mk (List.isSetoid α) a)). …
                             -/
  Quotient.inductionOn s (by simp)
                             /-
                               🎉 no goals
                             -/


lemma prod_eq_zero (h : (0 : α) ∈ s) : s.prod = 0 := by
  /-
    α : Type u_2
    inst✝ : CommMonoidWithZero α
    s : Multiset α
    h : Membership.mem s 0
    ⊢ Eq s.prod 0
  -/
  rcases Multiset.exists_cons_of_mem h with ⟨s', hs'⟩; simp [hs', Multiset.prod_cons]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp] lemma prod_eq_zero_iff : s.prod = 0 ↔ (0 : α) ∈ s :=
                                    /-
                                      α : Type u_2
                                      inst✝² : CommMonoidWithZero α
                                      inst✝¹ : NoZeroDivisors α
                                      inst✝ : Nontrivial α
                                      s : Multiset α
                                      l : List α
                                      ⊢ Iff (Eq (Multiset.prod (Quotient.mk (List.isSetoid α) l)) 0) (Membership.mem …
                                    -/
  Quotient.inductionOn s fun l ↦ by rw [quot_mk_to_coe, prod_coe]; exact List.prod_eq_zero_iff
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


lemma prod_ne_zero (h : (0 : α) ∉ s) : s.prod ≠ 0 := mt prod_eq_zero_iff.1 h


lemma sum_map_mul_left : sum (s.map fun i ↦ a * f i) = a * sum (s.map f) :=
                              /-
                                ι : Type u_1
                                α : Type u_2
                                inst✝ : NonUnitalNonAssocSemiring α
                                a : α
                                s : Multiset ι
                                f : ι → α
                                ⊢ Eq (Multiset.map (fun i => HMul.hMul a (f i)) 0).sum (HMul.hMul a (Multiset. …
                              -/
                              /-
                                🎉 no goals
                              -/
  Multiset.induction_on s (by simp) fun i s ih => by simp [ih, mul_add]
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma sum_map_mul_right : sum (s.map fun i ↦ f i * a) = sum (s.map f) * a :=
                              /-
                                ι : Type u_1
                                α : Type u_2
                                inst✝ : NonUnitalNonAssocSemiring α
                                a : α
                                s : Multiset ι
                                f : ι → α
                                ⊢ Eq (Multiset.map (fun i => HMul.hMul (f i) a) 0).sum (HMul.hMul (Multiset.ma …
                              -/
                              /-
                                🎉 no goals
                              -/
  Multiset.induction_on s (by simp) fun a s ih => by simp [ih, add_mul]
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma dvd_sum : (∀ x ∈ s, a ∣ x) → a ∣ s.sum :=
  Multiset.induction_on s (fun _ ↦ dvd_zero _) fun x s ih h ↦ by
    /-
      α : Type u_2
      inst✝ : NonUnitalSemiring α
      s✝ : Multiset α
      a x : α
      s : Multiset α
      ih : (∀ (x : α), Membership.mem s x → Dvd.dvd a x) → Dvd.dvd a s.sum
      h : ∀ (x_1 : α), Membership.mem (Multiset.cons x s) x_1 → Dvd.dvd a x_1
      ⊢ Dvd.dvd a (Multiset.cons x s).sum
    -/
    rw [sum_cons]
    /-
      α : Type u_2
      inst✝ : NonUnitalSemiring α
      s✝ : Multiset α
      a x : α
      s : Multiset α
      ih : (∀ (x : α), Membership.mem s x → Dvd.dvd a x) → Dvd.dvd a s.sum
      h : ∀ (x_1 : α), Membership.mem (Multiset.cons x s) x_1 → Dvd.dvd a x_1
      ⊢ Dvd.dvd a (HAdd.hAdd x s.sum)
    -/
    exact dvd_add (h _ (mem_cons_self _ _)) (ih fun y hy ↦ h _ <| mem_cons.2 <| Or.inr hy)
    /-
      🎉 no goals
    -/


lemma prod_map_sum {s : Multiset (Multiset α)} :
    prod (s.map sum) = sum ((Sections s).map prod) :=
                              /-
                                α : Type u_2
                                inst✝ : CommSemiring α
                                s : Multiset (Multiset α)
                                ⊢ Eq (Multiset.map Multiset.sum 0).prod (Multiset.map Multiset.prod (Multiset. …
                              -/
  Multiset.induction_on s (by simp) fun a s ih ↦ by
                              /-
                                🎉 no goals
                              -/
    /-
      α : Type u_2
      inst✝ : CommSemiring α
      s✝ : Multiset (Multiset α)
      a : Multiset α
      s : Multiset (Multiset α)
      ih : Eq (Multiset.map Multiset.sum s).prod (Multiset.map Multiset.prod s.Secti …
      ⊢ Eq (Multiset.map Multiset.sum (Multiset.cons a s)).prod (Multiset.map Multis …
    -/
    simp [ih, map_bind, sum_map_mul_left, sum_map_mul_right]
    /-
      🎉 no goals
    -/


lemma prod_map_add {s : Multiset ι} {f g : ι → α} :
    prod (s.map fun i ↦ f i + g i) =
      sum ((antidiagonal s).map fun p ↦ (p.1.map f).prod * (p.2.map g).prod) := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : CommSemiring α
    s : Multiset ι
    f g : ι → α
    ⊢ Eq (Multiset.map (fun i => HAdd.hAdd (f i) (g i)) s).prod (Multiset.map (fun …
  -/
  refine s.induction_on ?_ fun a s ih ↦ ?_
    /-
      case refine_1
      ι : Type u_1
      α : Type u_2
      inst✝ : CommSemiring α
      s : Multiset ι
      f g : ι → α
      ⊢ Eq (Multiset.map (fun i => HAdd.hAdd (f i) (g i)) 0).prod (Multiset.map (fun …
    -/
  · simp only [map_zero, prod_zero, antidiagonal_zero, map_singleton, mul_one, sum_singleton]
    /-
      🎉 no goals
    -/
  · simp only [map_cons, prod_cons, ih, sum_map_mul_left.symm, add_mul, mul_left_comm (f a),
      mul_left_comm (g a), sum_map_add, antidiagonal_cons, Prod.map_fst, Prod.map_snd,
      id_eq, map_add, map_map, Function.comp_apply, mul_assoc, sum_add]
    /-
      case refine_2
      ι : Type u_1
      α : Type u_2
      inst✝ : CommSemiring α
      s✝ : Multiset ι
      f g : ι → α
      a : ι
      s : Multiset ι
      ih : Eq (Multiset.map (fun i => HAdd.hAdd (f i) (g i)) s).prod (Multiset.map ( …
      ⊢ Eq (HAdd.hAdd (Multiset.map (fun i => HMul.hMul (Multiset.map f i.1).prod (H …
    -/
    exact add_comm _ _
    /-
      🎉 no goals
    -/


theorem multiset_sum_right (a : α) (h : ∀ b ∈ s, Commute a b) : Commute a s.sum := by
  /-
    α : Type u_2
    inst✝ : NonUnitalNonAssocSemiring α
    s : Multiset α
    a : α
    h : ∀ (b : α), Membership.mem s b → Commute a b
    ⊢ Commute a s.sum
  -/
  induction s using Quotient.inductionOn
  /-
    case h
    α : Type u_2
    inst✝ : NonUnitalNonAssocSemiring α
    s : Multiset α
    a : α
    a✝ : List α
    h : ∀ (b : α), Membership.mem (Quotient.mk (List.isSetoid α) a✝) b → Commute a b
    ⊢ Commute a (Multiset.sum (Quotient.mk (List.isSetoid α) a✝))
  -/
  rw [quot_mk_to_coe, sum_coe]
  /-
    case h
    α : Type u_2
    inst✝ : NonUnitalNonAssocSemiring α
    s : Multiset α
    a : α
    a✝ : List α
    h : ∀ (b : α), Membership.mem (Quotient.mk (List.isSetoid α) a✝) b → Commute a b
    ⊢ Commute a a✝.sum
  -/
  exact Commute.list_sum_right _ _ h
  /-
    🎉 no goals
  -/


theorem multiset_sum_left (b : α) (h : ∀ a ∈ s, Commute a b) : Commute s.sum b :=
  ((Commute.multiset_sum_right _ _) fun _ ha => (h _ ha).symm).symm


