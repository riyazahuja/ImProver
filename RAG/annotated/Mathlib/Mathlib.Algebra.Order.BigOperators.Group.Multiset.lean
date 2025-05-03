@[to_additive sum_nonneg]
lemma one_le_prod_of_one_le : (∀ x ∈ s, (1 : α) ≤ x) → 1 ≤ s.prod :=
                                        /-
                                          α : Type u_2
                                          inst✝ : OrderedCommMonoid α
                                          s : Multiset α
                                          l : List α
                                          hl : ∀ (x : α), Membership.mem (Quotient.mk (List.isSetoid α) l) x → LE.le 1 x
                                          ⊢ LE.le 1 (Multiset.prod (Quotient.mk (List.isSetoid α) l))
                                        -/
  Quotient.inductionOn s fun l hl => by simpa using List.one_le_prod_of_one_le hl
                                        /-
                                          🎉 no goals
                                        -/


@[to_additive]
lemma single_le_prod : (∀ x ∈ s, (1 : α) ≤ x) → ∀ x ∈ s, x ≤ s.prod :=
                                             /-
                                               α : Type u_2
                                               inst✝ : OrderedCommMonoid α
                                               s : Multiset α
                                               l : List α
                                               hl : ∀ (x : α), Membership.mem (Quotient.mk (List.isSetoid α) l) x → LE.le 1 x
                                               x : α
                                               hx : Membership.mem (Quotient.mk (List.isSetoid α) l) x
                                               ⊢ LE.le x (Multiset.prod (Quotient.mk (List.isSetoid α) l))
                                             -/
  Quotient.inductionOn s fun l hl x hx => by simpa using List.single_le_prod hl x hx
                                             /-
                                               🎉 no goals
                                             -/


@[to_additive sum_le_card_nsmul]
lemma prod_le_pow_card (s : Multiset α) (n : α) (h : ∀ x ∈ s, x ≤ n) : s.prod ≤ n ^ card s := by
  /-
    α : Type u_2
    inst✝ : OrderedCommMonoid α
    s : Multiset α
    n : α
    h : ∀ (x : α), Membership.mem s x → LE.le x n
    ⊢ LE.le s.prod (HPow.hPow n s.card)
  -/
  induction s using Quotient.inductionOn
  /-
    case h
    α : Type u_2
    inst✝ : OrderedCommMonoid α
    n : α
    a✝ : List α
    h : ∀ (x : α), Membership.mem (Quotient.mk (List.isSetoid α) a✝) x → LE.le x n
    ⊢ LE.le (Multiset.prod (Quotient.mk (List.isSetoid α) a✝)) (HPow.hPow n (Multi …
  -/
  simpa using List.prod_le_pow_card _ _ h
  /-
    🎉 no goals
  -/


@[to_additive all_zero_of_le_zero_le_of_sum_eq_zero]
lemma all_one_of_le_one_le_of_prod_eq_one :
    (∀ x ∈ s, (1 : α) ≤ x) → s.prod = 1 → ∀ x ∈ s, x = (1 : α) :=
  Quotient.inductionOn s (by
    /-
      α : Type u_2
      inst✝ : OrderedCommMonoid α
      s : Multiset α
      ⊢ ∀ (a : List α), (∀ (x : α), Membership.mem (Quotient.mk (List.isSetoid α) a) …
    -/
    simp only [quot_mk_to_coe, prod_coe, mem_coe]
    /-
      α : Type u_2
      inst✝ : OrderedCommMonoid α
      s : Multiset α
      ⊢ ∀ (a : List α), (∀ (x : α), Membership.mem a x → LE.le 1 x) → Eq a.prod 1 →  …
    -/
    exact fun l => List.all_one_of_le_one_le_of_prod_eq_one)
    /-
      🎉 no goals
    -/


@[to_additive]
lemma prod_le_prod_of_rel_le (h : s.Rel (· ≤ ·) t) : s.prod ≤ t.prod := by
  induction h with
  | zero => rfl
  | cons rh _ rt =>
    rw [prod_cons, prod_cons]
    exact mul_le_mul' rh rt


@[to_additive]
lemma prod_map_le_prod_map {s : Multiset ι} (f : ι → α) (g : ι → α) (h : ∀ i, i ∈ s → f i ≤ g i) :
    (s.map f).prod ≤ (s.map g).prod :=
  prod_le_prod_of_rel_le <| rel_map.2 <| rel_refl_of_refl_on h


@[to_additive]
lemma prod_map_le_prod (f : α → α) (h : ∀ x, x ∈ s → f x ≤ x) : (s.map f).prod ≤ s.prod :=
  prod_le_prod_of_rel_le <| rel_map_left.2 <| rel_refl_of_refl_on h


@[to_additive]
lemma prod_le_prod_map (f : α → α) (h : ∀ x, x ∈ s → x ≤ f x) : s.prod ≤ (s.map f).prod :=
  @prod_map_le_prod αᵒᵈ _ _ f h


@[to_additive card_nsmul_le_sum]
lemma pow_card_le_prod (h : ∀ x ∈ s, a ≤ x) : a ^ card s ≤ s.prod := by
  /-
    α : Type u_2
    inst✝ : OrderedCommMonoid α
    s : Multiset α
    a : α
    h : ∀ (x : α), Membership.mem s x → LE.le a x
    ⊢ LE.le (HPow.hPow a s.card) s.prod
  -/
  rw [← Multiset.prod_replicate, ← Multiset.map_const]
  /-
    α : Type u_2
    inst✝ : OrderedCommMonoid α
    s : Multiset α
    a : α
    h : ∀ (x : α), Membership.mem s x → LE.le a x
    ⊢ LE.le (Multiset.map (Function.const α a) s).prod s.prod
  -/
  exact prod_map_le_prod _ h
  /-
    🎉 no goals
  -/


@[to_additive le_sum_of_subadditive_on_pred]
lemma le_prod_of_submultiplicative_on_pred (f : α → β)
    (p : α → Prop) (h_one : f 1 = 1) (hp_one : p 1)
    (h_mul : ∀ a b, p a → p b → f (a * b) ≤ f a * f b) (hp_mul : ∀ a b, p a → p b → p (a * b))
    (s : Multiset α) (hps : ∀ a, a ∈ s → p a) : f s.prod ≤ (s.map f).prod := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : OrderedCommMonoid β
    f : α → β
    p : α → Prop
    h_one : Eq (f 1) 1
    hp_one : p 1
    h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
    hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    s : Multiset α
    hps : ∀ (a : α), Membership.mem s a → p a
    ⊢ LE.le (f s.prod) (Multiset.map f s).prod
  -/
  revert s
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : OrderedCommMonoid β
    f : α → β
    p : α → Prop
    h_one : Eq (f 1) 1
    hp_one : p 1
    h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
    hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    ⊢ ∀ (s : Multiset α), (∀ (a : α), Membership.mem s a → p a) → LE.le (f s.prod) …
  -/
  refine Multiset.induction ?_ ?_
    /-
      case refine_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : CommMonoid α
      inst✝ : OrderedCommMonoid β
      f : α → β
      p : α → Prop
      h_one : Eq (f 1) 1
      hp_one : p 1
      h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
      hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
      ⊢ (∀ (a : α), Membership.mem 0 a → p a) → LE.le (f (Multiset.prod 0)) (Multise …
    -/
  · simp [le_of_eq h_one]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : OrderedCommMonoid β
    f : α → β
    p : α → Prop
    h_one : Eq (f 1) 1
    hp_one : p 1
    h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
    hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    ⊢ ∀ (a : α) (s : Multiset α), ((∀ (a : α), Membership.mem s a → p a) → LE.le ( …
  -/
  intro a s hs hpsa
  /-
    case refine_2
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : OrderedCommMonoid β
    f : α → β
    p : α → Prop
    h_one : Eq (f 1) 1
    hp_one : p 1
    h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
    hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    a : α
    s : Multiset α
    hs : (∀ (a : α), Membership.mem s a → p a) → LE.le (f s.prod) (Multiset.map f  …
    hpsa : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → p a_1
    ⊢ LE.le (f (Multiset.cons a s).prod) (Multiset.map f (Multiset.cons a s)).prod
  -/
  have hps : ∀ x, x ∈ s → p x := fun x hx => hpsa x (mem_cons_of_mem hx)
  /-
    case refine_2
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : OrderedCommMonoid β
    f : α → β
    p : α → Prop
    h_one : Eq (f 1) 1
    hp_one : p 1
    h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
    hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    a : α
    s : Multiset α
    hs : (∀ (a : α), Membership.mem s a → p a) → LE.le (f s.prod) (Multiset.map f  …
    hpsa : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → p a_1
    hps : ∀ (x : α), Membership.mem s x → p x
    ⊢ LE.le (f (Multiset.cons a s).prod) (Multiset.map f (Multiset.cons a s)).prod
  -/
  have hp_prod : p s.prod := prod_induction p s hp_mul hp_one hps
  /-
    case refine_2
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : OrderedCommMonoid β
    f : α → β
    p : α → Prop
    h_one : Eq (f 1) 1
    hp_one : p 1
    h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
    hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    a : α
    s : Multiset α
    hs : (∀ (a : α), Membership.mem s a → p a) → LE.le (f s.prod) (Multiset.map f  …
    hpsa : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → p a_1
    hps : ∀ (x : α), Membership.mem s x → p x
    hp_prod : p s.prod
    ⊢ LE.le (f (Multiset.cons a s).prod) (Multiset.map f (Multiset.cons a s)).prod
  -/
  rw [prod_cons, map_cons, prod_cons]
  /-
    case refine_2
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : OrderedCommMonoid β
    f : α → β
    p : α → Prop
    h_one : Eq (f 1) 1
    hp_one : p 1
    h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
    hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    a : α
    s : Multiset α
    hs : (∀ (a : α), Membership.mem s a → p a) → LE.le (f s.prod) (Multiset.map f  …
    hpsa : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → p a_1
    hps : ∀ (x : α), Membership.mem s x → p x
    hp_prod : p s.prod
    ⊢ LE.le (f (HMul.hMul a s.prod)) (HMul.hMul (f a) (Multiset.map f s).prod)
  -/
  exact (h_mul a s.prod (hpsa a (mem_cons_self a s)) hp_prod).trans (mul_le_mul_left' (hs hps) _)
  /-
    🎉 no goals
  -/


@[to_additive le_sum_of_subadditive]
lemma le_prod_of_submultiplicative (f : α → β) (h_one : f 1 = 1)
    (h_mul : ∀ a b, f (a * b) ≤ f a * f b) (s : Multiset α) : f s.prod ≤ (s.map f).prod :=
  le_prod_of_submultiplicative_on_pred f (fun _ => True) h_one trivial (fun x y _ _ => h_mul x y)
        /-
          α : Type u_2
          β : Type u_3
          inst✝¹ : CommMonoid α
          inst✝ : OrderedCommMonoid β
          f : α → β
          h_one : Eq (f 1) 1
          h_mul : ∀ (a b : α), LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
          s : Multiset α
          ⊢ ∀ (a b : α), (fun x => True) a → (fun x => True) b → (fun x => True) (HMul.h …
        -/
        /-
          🎉 no goals
        -/
    (by simp) s (by simp)
                    /-
                      🎉 no goals
                    -/


@[to_additive le_sum_nonempty_of_subadditive_on_pred]
lemma le_prod_nonempty_of_submultiplicative_on_pred (f : α → β) (p : α → Prop)
    (h_mul : ∀ a b, p a → p b → f (a * b) ≤ f a * f b) (hp_mul : ∀ a b, p a → p b → p (a * b))
    (s : Multiset α) (hs_nonempty : s ≠ ∅) (hs : ∀ a, a ∈ s → p a) : f s.prod ≤ (s.map f).prod := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : OrderedCommMonoid β
    f : α → β
    p : α → Prop
    h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
    hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    s : Multiset α
    hs_nonempty : Ne s EmptyCollection.emptyCollection
    hs : ∀ (a : α), Membership.mem s a → p a
    ⊢ LE.le (f s.prod) (Multiset.map f s).prod
  -/
  revert s
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : OrderedCommMonoid β
    f : α → β
    p : α → Prop
    h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
    hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    ⊢ ∀ (s : Multiset α), Ne s EmptyCollection.emptyCollection → (∀ (a : α), Membe …
  -/
  refine Multiset.induction ?_ ?_
    /-
      case refine_1
      α : Type u_2
      β : Type u_3
      inst✝¹ : CommMonoid α
      inst✝ : OrderedCommMonoid β
      f : α → β
      p : α → Prop
      h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
      hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
      ⊢ Ne 0 EmptyCollection.emptyCollection → (∀ (a : α), Membership.mem 0 a → p a) …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : OrderedCommMonoid β
    f : α → β
    p : α → Prop
    h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
    hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    ⊢ ∀ (a : α) (s : Multiset α), (Ne s EmptyCollection.emptyCollection → (∀ (a :  …
  -/
  rintro a s hs - hsa_prop
  /-
    case refine_2
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : OrderedCommMonoid β
    f : α → β
    p : α → Prop
    h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
    hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    a : α
    s : Multiset α
    hs : Ne s EmptyCollection.emptyCollection → (∀ (a : α), Membership.mem s a → p …
    hsa_prop : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → p a_1
    ⊢ LE.le (f (Multiset.cons a s).prod) (Multiset.map f (Multiset.cons a s)).prod
  -/
  rw [prod_cons, map_cons, prod_cons]
  /-
    case refine_2
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : OrderedCommMonoid β
    f : α → β
    p : α → Prop
    h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
    hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    a : α
    s : Multiset α
    hs : Ne s EmptyCollection.emptyCollection → (∀ (a : α), Membership.mem s a → p …
    hsa_prop : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → p a_1
    ⊢ LE.le (f (HMul.hMul a s.prod)) (HMul.hMul (f a) (Multiset.map f s).prod)
  -/
  by_cases hs_empty : s = ∅
    /-
      case pos
      α : Type u_2
      β : Type u_3
      inst✝¹ : CommMonoid α
      inst✝ : OrderedCommMonoid β
      f : α → β
      p : α → Prop
      h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
      hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
      a : α
      s : Multiset α
      hs : Ne s EmptyCollection.emptyCollection → (∀ (a : α), Membership.mem s a → p …
      hsa_prop : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → p a_1
      hs_empty : Eq s EmptyCollection.emptyCollection
      ⊢ LE.le (f (HMul.hMul a s.prod)) (HMul.hMul (f a) (Multiset.map f s).prod)
    -/
  · simp [hs_empty]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : OrderedCommMonoid β
    f : α → β
    p : α → Prop
    h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
    hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    a : α
    s : Multiset α
    hs : Ne s EmptyCollection.emptyCollection → (∀ (a : α), Membership.mem s a → p …
    hsa_prop : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → p a_1
    hs_empty : Not (Eq s EmptyCollection.emptyCollection)
    ⊢ LE.le (f (HMul.hMul a s.prod)) (HMul.hMul (f a) (Multiset.map f s).prod)
  -/
  have hsa_restrict : ∀ x, x ∈ s → p x := fun x hx => hsa_prop x (mem_cons_of_mem hx)
  /-
    case neg
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : OrderedCommMonoid β
    f : α → β
    p : α → Prop
    h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
    hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    a : α
    s : Multiset α
    hs : Ne s EmptyCollection.emptyCollection → (∀ (a : α), Membership.mem s a → p …
    hsa_prop : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → p a_1
    hs_empty : Not (Eq s EmptyCollection.emptyCollection)
    hsa_restrict : ∀ (x : α), Membership.mem s x → p x
    ⊢ LE.le (f (HMul.hMul a s.prod)) (HMul.hMul (f a) (Multiset.map f s).prod)
  -/
  have hp_sup : p s.prod := prod_induction_nonempty p hp_mul hs_empty hsa_restrict
  /-
    case neg
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : OrderedCommMonoid β
    f : α → β
    p : α → Prop
    h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
    hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    a : α
    s : Multiset α
    hs : Ne s EmptyCollection.emptyCollection → (∀ (a : α), Membership.mem s a → p …
    hsa_prop : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → p a_1
    hs_empty : Not (Eq s EmptyCollection.emptyCollection)
    hsa_restrict : ∀ (x : α), Membership.mem s x → p x
    hp_sup : p s.prod
    ⊢ LE.le (f (HMul.hMul a s.prod)) (HMul.hMul (f a) (Multiset.map f s).prod)
  -/
  have hp_a : p a := hsa_prop a (mem_cons_self a s)
  /-
    case neg
    α : Type u_2
    β : Type u_3
    inst✝¹ : CommMonoid α
    inst✝ : OrderedCommMonoid β
    f : α → β
    p : α → Prop
    h_mul : ∀ (a b : α), p a → p b → LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f …
    hp_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    a : α
    s : Multiset α
    hs : Ne s EmptyCollection.emptyCollection → (∀ (a : α), Membership.mem s a → p …
    hsa_prop : ∀ (a_1 : α), Membership.mem (Multiset.cons a s) a_1 → p a_1
    hs_empty : Not (Eq s EmptyCollection.emptyCollection)
    hsa_restrict : ∀ (x : α), Membership.mem s x → p x
    hp_sup : p s.prod
    hp_a : p a
    ⊢ LE.le (f (HMul.hMul a s.prod)) (HMul.hMul (f a) (Multiset.map f s).prod)
  -/
  exact (h_mul a _ hp_a hp_sup).trans (mul_le_mul_left' (hs hs_empty hsa_restrict) _)
  /-
    🎉 no goals
  -/


@[to_additive le_sum_nonempty_of_subadditive]
lemma le_prod_nonempty_of_submultiplicative (f : α → β) (h_mul : ∀ a b, f (a * b) ≤ f a * f b)
    (s : Multiset α) (hs_nonempty : s ≠ ∅) : f s.prod ≤ (s.map f).prod :=
                                                                      /-
                                                                        α : Type u_2
                                                                        β : Type u_3
                                                                        inst✝¹ : CommMonoid α
                                                                        inst✝ : OrderedCommMonoid β
                                                                        f : α → β
                                                                        h_mul : ∀ (a b : α), LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
                                                                        s : Multiset α
                                                                        hs_nonempty : Ne s EmptyCollection.emptyCollection
                                                                        ⊢ ∀ (a b : α), (fun x => True) a → (fun x => True) b → LE.le (f (HMul.hMul a b …
                                                                      -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  le_prod_nonempty_of_submultiplicative_on_pred f (fun _ => True) (by simp [h_mul]) (by simp) s
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/
                    /-
                      α : Type u_2
                      β : Type u_3
                      inst✝¹ : CommMonoid α
                      inst✝ : OrderedCommMonoid β
                      f : α → β
                      h_mul : ∀ (a b : α), LE.le (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
                      s : Multiset α
                      hs_nonempty : Ne s EmptyCollection.emptyCollection
                      ⊢ ∀ (a : α), Membership.mem s a → (fun x => True) a
                    -/
    hs_nonempty (by simp)
                    /-
                      🎉 no goals
                    -/


@[to_additive sum_lt_sum]
lemma prod_lt_prod' (hle : ∀ i ∈ s, f i ≤ g i) (hlt : ∃ i ∈ s, f i < g i) :
    (s.map f).prod < (s.map g).prod := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : OrderedCancelCommMonoid α
    s : Multiset ι
    f g : ι → α
    hle : ∀ (i : ι), Membership.mem s i → LE.le (f i) (g i)
    hlt : Exists fun i => And (Membership.mem s i) (LT.lt (f i) (g i))
    ⊢ LT.lt (Multiset.map f s).prod (Multiset.map g s).prod
  -/
  obtain ⟨l⟩ := s
  /-
    case mk
    ι : Type u_1
    α : Type u_2
    inst✝ : OrderedCancelCommMonoid α
    s : Multiset ι
    f g : ι → α
    l : List ι
    hle : ∀ (i : ι), Membership.mem (Quot.mk (⇑(List.isSetoid ι)) l) i → LE.le (f  …
    hlt : Exists fun i => And (Membership.mem (Quot.mk (⇑(List.isSetoid ι)) l) i)  …
    ⊢ LT.lt (Multiset.map f (Quot.mk (⇑(List.isSetoid ι)) l)).prod (Multiset.map g …
  -/
  simp only [Multiset.quot_mk_to_coe'', Multiset.map_coe, Multiset.prod_coe]
  /-
    case mk
    ι : Type u_1
    α : Type u_2
    inst✝ : OrderedCancelCommMonoid α
    s : Multiset ι
    f g : ι → α
    l : List ι
    hle : ∀ (i : ι), Membership.mem (Quot.mk (⇑(List.isSetoid ι)) l) i → LE.le (f  …
    hlt : Exists fun i => And (Membership.mem (Quot.mk (⇑(List.isSetoid ι)) l) i)  …
    ⊢ LT.lt (List.map f l).prod (List.map g l).prod
  -/
  exact List.prod_lt_prod' f g hle hlt
  /-
    🎉 no goals
  -/


@[to_additive sum_lt_sum_of_nonempty]
lemma prod_lt_prod_of_nonempty' (hs : s ≠ ∅) (hfg : ∀ i ∈ s, f i < g i) :
    (s.map f).prod < (s.map g).prod := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : OrderedCancelCommMonoid α
    s : Multiset ι
    f g : ι → α
    hs : Ne s EmptyCollection.emptyCollection
    hfg : ∀ (i : ι), Membership.mem s i → LT.lt (f i) (g i)
    ⊢ LT.lt (Multiset.map f s).prod (Multiset.map g s).prod
  -/
  obtain ⟨i, hi⟩ := exists_mem_of_ne_zero hs
  /-
    case intro
    ι : Type u_1
    α : Type u_2
    inst✝ : OrderedCancelCommMonoid α
    s : Multiset ι
    f g : ι → α
    hs : Ne s EmptyCollection.emptyCollection
    hfg : ∀ (i : ι), Membership.mem s i → LT.lt (f i) (g i)
    i : ι
    hi : Membership.mem s i
    ⊢ LT.lt (Multiset.map f s).prod (Multiset.map g s).prod
  -/
  exact prod_lt_prod' (fun i hi => le_of_lt (hfg i hi)) ⟨i, hi, hfg i hi⟩
  /-
    🎉 no goals
  -/


@[to_additive] lemma prod_eq_one_iff : m.prod = 1 ↔ ∀ x ∈ m, x = (1 : α) :=
                                    /-
                                      α : Type u_2
                                      inst✝ : CanonicallyOrderedCommMonoid α
                                      m : Multiset α
                                      l : List α
                                      ⊢ Iff (Eq (Multiset.prod (Quotient.mk (List.isSetoid α) l)) 1) (∀ (x : α), Mem …
                                    -/
  Quotient.inductionOn m fun l ↦ by simpa using List.prod_eq_one_iff
                                    /-
                                      🎉 no goals
                                    -/


@[to_additive] lemma le_prod_of_mem (ha : a ∈ m) : a ≤ m.prod := by
  /-
    α : Type u_2
    inst✝ : CanonicallyOrderedCommMonoid α
    m : Multiset α
    a : α
    ha : Membership.mem m a
    ⊢ LE.le a m.prod
  -/
  obtain ⟨t, rfl⟩ := exists_cons_of_mem ha
  /-
    case intro
    α : Type u_2
    inst✝ : CanonicallyOrderedCommMonoid α
    a : α
    t : Multiset α
    ha : Membership.mem (Multiset.cons a t) a
    ⊢ LE.le a (Multiset.cons a t).prod
  -/
  rw [prod_cons]
  /-
    case intro
    α : Type u_2
    inst✝ : CanonicallyOrderedCommMonoid α
    a : α
    t : Multiset α
    ha : Membership.mem (Multiset.cons a t) a
    ⊢ LE.le a (HMul.hMul a t.prod)
  -/
  exact _root_.le_mul_right (le_refl a)
  /-
    🎉 no goals
  -/


lemma max_le_of_forall_le {α : Type*} [LinearOrder α] [OrderBot α] (l : Multiset α)
    (n : α) (h : ∀ x ∈ l, x ≤ n) : l.fold max ⊥ ≤ n := by
  /-
    α : Type u_4
    inst✝¹ : LinearOrder α
    inst✝ : OrderBot α
    l : Multiset α
    n : α
    h : ∀ (x : α), Membership.mem l x → LE.le x n
    ⊢ LE.le (Multiset.fold Max.max Bot.bot l) n
  -/
  induction l using Quotient.inductionOn
  /-
    case h
    α : Type u_4
    inst✝¹ : LinearOrder α
    inst✝ : OrderBot α
    n : α
    a✝ : List α
    h : ∀ (x : α), Membership.mem (Quotient.mk (List.isSetoid α) a✝) x → LE.le x n
    ⊢ LE.le (Multiset.fold Max.max Bot.bot (Quotient.mk (List.isSetoid α) a✝)) n
  -/
  simpa using List.max_le_of_forall_le _ _ h
  /-
    🎉 no goals
  -/


@[to_additive]
lemma max_prod_le [LinearOrderedCommMonoid α] {s : Multiset ι} {f g : ι → α} :
    max (s.map f).prod (s.map g).prod ≤ (s.map fun i ↦ max (f i) (g i)).prod := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : LinearOrderedCommMonoid α
    s : Multiset ι
    f g : ι → α
    ⊢ LE.le (Max.max (Multiset.map f s).prod (Multiset.map g s).prod) (Multiset.ma …
  -/
  obtain ⟨l⟩ := s
  /-
    case mk
    ι : Type u_1
    α : Type u_2
    inst✝ : LinearOrderedCommMonoid α
    s : Multiset ι
    f g : ι → α
    l : List ι
    ⊢ LE.le (Max.max (Multiset.map f (Quot.mk (⇑(List.isSetoid ι)) l)).prod (Multi …
  -/
  simp_rw [Multiset.quot_mk_to_coe'', Multiset.map_coe, Multiset.prod_coe]
  /-
    case mk
    ι : Type u_1
    α : Type u_2
    inst✝ : LinearOrderedCommMonoid α
    s : Multiset ι
    f g : ι → α
    l : List ι
    ⊢ LE.le (Max.max (List.map f l).prod (List.map g l).prod) (List.map (fun i =>  …
  -/
  apply List.max_prod_le
  /-
    🎉 no goals
  -/


@[to_additive]
lemma prod_min_le [LinearOrderedCommMonoid α] {s : Multiset ι} {f g : ι → α} :
    (s.map fun i ↦ min (f i) (g i)).prod ≤ min (s.map f).prod (s.map g).prod := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : LinearOrderedCommMonoid α
    s : Multiset ι
    f g : ι → α
    ⊢ LE.le (Multiset.map (fun i => Min.min (f i) (g i)) s).prod (Min.min (Multise …
  -/
  obtain ⟨l⟩ := s
  /-
    case mk
    ι : Type u_1
    α : Type u_2
    inst✝ : LinearOrderedCommMonoid α
    s : Multiset ι
    f g : ι → α
    l : List ι
    ⊢ LE.le (Multiset.map (fun i => Min.min (f i) (g i)) (Quot.mk (⇑(List.isSetoid …
  -/
  simp_rw [Multiset.quot_mk_to_coe'', Multiset.map_coe, Multiset.prod_coe]
  /-
    case mk
    ι : Type u_1
    α : Type u_2
    inst✝ : LinearOrderedCommMonoid α
    s : Multiset ι
    f g : ι → α
    l : List ι
    ⊢ LE.le (List.map (fun i => Min.min (f i) (g i)) l).prod (Min.min (List.map f  …
  -/
  apply List.prod_min_le
  /-
    🎉 no goals
  -/


lemma abs_sum_le_sum_abs [LinearOrderedAddCommGroup α] {s : Multiset α} :
    |s.sum| ≤ (s.map abs).sum :=
  le_sum_of_subadditive _ abs_zero abs_add s


