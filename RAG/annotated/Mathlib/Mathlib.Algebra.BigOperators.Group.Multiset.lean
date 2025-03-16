/-- Product of a multiset given a commutative monoid structure on `α`.
  `prod {a, b, c} = a * b * c` -/
@[to_additive
      "Sum of a multiset given a commutative additive monoid structure on `α`.
      `sum {a, b, c} = a + b + c`"]
def prod : Multiset α → α :=
  foldr (· * ·) 1


@[to_additive]
theorem prod_eq_foldr (s : Multiset α) :
    prod s = foldr (· * ·) 1 s :=
  rfl


@[to_additive]
theorem prod_eq_foldl (s : Multiset α) :
    prod s = foldl (· * ·) 1 s :=
                               /-
                                 α : Type u_3
                                 inst✝ : CommMonoid α
                                 s : Multiset α
                                 ⊢ Eq (Multiset.foldl (fun x y => HMul.hMul y x) 1 s) (Multiset.foldl (fun x1 x …
                               -/
  (foldr_swap _ _ _).trans (by simp [mul_comm])
                               /-
                                 🎉 no goals
                               -/


@[to_additive (attr := simp, norm_cast)]
theorem prod_coe (l : List α) : prod ↑l = l.prod := rfl


@[to_additive (attr := simp)]
theorem prod_toList (s : Multiset α) : s.toList.prod = s.prod := by
  /-
    α : Type u_3
    inst✝ : CommMonoid α
    s : Multiset α
    ⊢ Eq s.toList.prod s.prod
  -/
  conv_rhs => rw [← coe_toList s]
  /-
    α : Type u_3
    inst✝ : CommMonoid α
    s : Multiset α
    ⊢ Eq s.toList.prod (↑s.toList).prod
  -/
  rw [prod_coe]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem prod_zero : @prod α _ 0 = 1 :=
  rfl


@[to_additive (attr := simp)]
theorem prod_cons (a : α) (s) : prod (a ::ₘ s) = a * prod s :=
  foldr_cons _ _ _ _


@[to_additive (attr := simp)]
theorem prod_erase [DecidableEq α] (h : a ∈ s) : a * (s.erase a).prod = s.prod := by
  /-
    α : Type u_3
    inst✝¹ : CommMonoid α
    s : Multiset α
    a : α
    inst✝ : DecidableEq α
    h : Membership.mem s a
    ⊢ Eq (HMul.hMul a (s.erase a).prod) s.prod
  -/
  rw [← s.coe_toList, coe_erase, prod_coe, prod_coe, List.prod_erase (mem_toList.2 h)]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem prod_map_erase [DecidableEq ι] {a : ι} (h : a ∈ m) :
    f a * ((m.erase a).map f).prod = (m.map f).prod := by
  rw [← m.coe_toList, coe_erase, map_coe, map_coe, prod_coe, prod_coe,
    List.prod_map_erase f (mem_toList.2 h)]


@[to_additive (attr := simp)]
theorem prod_singleton (a : α) : prod {a} = a := by
  /-
    α : Type u_3
    inst✝ : CommMonoid α
    a : α
    ⊢ Eq (Singleton.singleton a).prod a
  -/
  simp only [mul_one, prod_cons, ← cons_zero, eq_self_iff_true, prod_zero]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_pair (a b : α) : ({a, b} : Multiset α).prod = a * b := by
  /-
    α : Type u_3
    inst✝ : CommMonoid α
    a b : α
    ⊢ Eq (Insert.insert a (Singleton.singleton b)).prod (HMul.hMul a b)
  -/
  rw [insert_eq_cons, prod_cons, prod_singleton]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem prod_add (s t : Multiset α) : prod (s + t) = prod s * prod t :=
                                            /-
                                              α : Type u_3
                                              inst✝ : CommMonoid α
                                              s t : Multiset α
                                              l₁ l₂ : List α
                                              ⊢ Eq (HAdd.hAdd (Quotient.mk (List.isSetoid α) l₁) (Quotient.mk (List.isSetoid …
                                            -/
  Quotient.inductionOn₂ s t fun l₁ l₂ => by simp
                                            /-
                                              🎉 no goals
                                            -/


@[to_additive]
theorem prod_nsmul (m : Multiset α) : ∀ n : ℕ, (n • m).prod = m.prod ^ n
  | 0 => by
    /-
      α : Type u_3
      inst✝ : CommMonoid α
      m : Multiset α
      ⊢ Eq (HSMul.hSMul 0 m).prod (HPow.hPow m.prod 0)
    -/
    rw [zero_nsmul, pow_zero]
    /-
      α : Type u_3
      inst✝ : CommMonoid α
      m : Multiset α
      ⊢ Eq (Multiset.prod 0) 1
    -/
    rfl
    /-
      🎉 no goals
    -/
                /-
                  α : Type u_3
                  inst✝ : CommMonoid α
                  m : Multiset α
                  n : Nat
                  ⊢ Eq (HSMul.hSMul (HAdd.hAdd n 1) m).prod (HPow.hPow m.prod (HAdd.hAdd n 1))
                -/
  | n + 1 => by rw [add_nsmul, one_nsmul, pow_add, pow_one, prod_add, prod_nsmul m n]
                /-
                  🎉 no goals
                -/


@[to_additive]
theorem prod_filter_mul_prod_filter_not (p) [DecidablePred p] :
    (s.filter p).prod * (s.filter (fun a ↦ ¬ p a)).prod = s.prod := by
  /-
    α : Type u_3
    inst✝¹ : CommMonoid α
    s : Multiset α
    p : α → Prop
    inst✝ : DecidablePred p
    ⊢ Eq (HMul.hMul (Multiset.filter p s).prod (Multiset.filter (fun a => Not (p a …
  -/
  rw [← prod_add, filter_add_not]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem prod_replicate (n : ℕ) (a : α) : (replicate n a).prod = a ^ n := by
  /-
    α : Type u_3
    inst✝ : CommMonoid α
    n : Nat
    a : α
    ⊢ Eq (Multiset.replicate n a).prod (HPow.hPow a n)
  -/
  simp [replicate, List.prod_replicate]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_map_eq_pow_single [DecidableEq ι] (i : ι)
    (hf : ∀ i' ≠ i, i' ∈ m → f i' = 1) : (m.map f).prod = f i ^ m.count i := by
  /-
    ι : Type u_2
    α : Type u_3
    inst✝¹ : CommMonoid α
    m : Multiset ι
    f : ι → α
    inst✝ : DecidableEq ι
    i : ι
    hf : ∀ (i' : ι), Ne i' i → Membership.mem m i' → Eq (f i') 1
    ⊢ Eq (Multiset.map f m).prod (HPow.hPow (f i) (Multiset.count i m))
  -/
  induction m using Quotient.inductionOn
  /-
    case h
    ι : Type u_2
    α : Type u_3
    inst✝¹ : CommMonoid α
    m : Multiset ι
    f : ι → α
    inst✝ : DecidableEq ι
    i : ι
    a✝ : List ι
    hf : ∀ (i' : ι), Ne i' i → Membership.mem (Quotient.mk (List.isSetoid ι) a✝) i …
    ⊢ Eq (Multiset.map f (Quotient.mk (List.isSetoid ι) a✝)).prod (HPow.hPow (f i) …
  -/
  simp [List.prod_map_eq_pow_single i f hf]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_eq_pow_single [DecidableEq α] (a : α) (h : ∀ a' ≠ a, a' ∈ s → a' = 1) :
    s.prod = a ^ s.count a := by
  /-
    α : Type u_3
    inst✝¹ : CommMonoid α
    s : Multiset α
    inst✝ : DecidableEq α
    a : α
    h : ∀ (a' : α), Ne a' a → Membership.mem s a' → Eq a' 1
    ⊢ Eq s.prod (HPow.hPow a (Multiset.count a s))
  -/
  induction s using Quotient.inductionOn; simp [List.prod_eq_pow_single a h]
                                          /-
                                            🎉 no goals
                                          -/


@[to_additive]
lemma prod_eq_one (h : ∀ x ∈ s, x = (1 : α)) : s.prod = 1 := by
  /-
    α : Type u_3
    inst✝ : CommMonoid α
    s : Multiset α
    h : ∀ (x : α), Membership.mem s x → Eq x 1
    ⊢ Eq s.prod 1
  -/
  induction s using Quotient.inductionOn; simp [List.prod_eq_one h]
                                          /-
                                            🎉 no goals
                                          -/


@[to_additive]
theorem pow_count [DecidableEq α] (a : α) : a ^ s.count a = (s.filter (Eq a)).prod := by
  /-
    α : Type u_3
    inst✝¹ : CommMonoid α
    s : Multiset α
    inst✝ : DecidableEq α
    a : α
    ⊢ Eq (HPow.hPow a (Multiset.count a s)) (Multiset.filter (Eq a) s).prod
  -/
  rw [filter_eq, prod_replicate]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_hom_ne_zero {s : Multiset α} (hs : s ≠ 0) {F : Type*} [FunLike F α β]
    [MulHomClass F α β] (f : F) :
    (s.map f).prod = f s.prod := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝³ : CommMonoid α
    inst✝² : CommMonoid β
    s : Multiset α
    hs : Ne s 0
    F : Type u_7
    inst✝¹ : FunLike F α β
    inst✝ : MulHomClass F α β
    f : F
    ⊢ Eq (Multiset.map (⇑f) s).prod (f s.prod)
  -/
  induction s using Quot.inductionOn; aesop (add simp List.prod_hom_nonempty)
                                      /-
                                        🎉 no goals
                                      -/


@[to_additive]
theorem prod_hom (s : Multiset α) {F : Type*} [FunLike F α β]
    [MonoidHomClass F α β] (f : F) :
    (s.map f).prod = f s.prod :=
                                     /-
                                       α : Type u_3
                                       β : Type u_4
                                       inst✝³ : CommMonoid α
                                       inst✝² : CommMonoid β
                                       s : Multiset α
                                       F : Type u_7
                                       inst✝¹ : FunLike F α β
                                       inst✝ : MonoidHomClass F α β
                                       f : F
                                       l : List α
                                       ⊢ Eq (Multiset.map (⇑f) (Quotient.mk (List.isSetoid α) l)).prod (f (Multiset.p …
                                     -/
  Quotient.inductionOn s fun l => by simp only [l.prod_hom f, quot_mk_to_coe, map_coe, prod_coe]
                                     /-
                                       🎉 no goals
                                     -/


@[to_additive]
theorem prod_hom' (s : Multiset ι) {F : Type*} [FunLike F α β]
    [MonoidHomClass F α β] (f : F)
    (g : ι → α) : (s.map fun i => f <| g i).prod = f (s.map g).prod := by
  /-
    ι : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝³ : CommMonoid α
    inst✝² : CommMonoid β
    s : Multiset ι
    F : Type u_7
    inst✝¹ : FunLike F α β
    inst✝ : MonoidHomClass F α β
    f : F
    g : ι → α
    ⊢ Eq (Multiset.map (fun i => f (g i)) s).prod (f (Multiset.map g s).prod)
  -/
  convert (s.map g).prod_hom f
  /-
    case h.e'_2.h.e'_3
    ι : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝³ : CommMonoid α
    inst✝² : CommMonoid β
    s : Multiset ι
    F : Type u_7
    inst✝¹ : FunLike F α β
    inst✝ : MonoidHomClass F α β
    f : F
    g : ι → α
    ⊢ Eq (Multiset.map (fun i => f (g i)) s) (Multiset.map (⇑f) (Multiset.map g s))
  -/
  exact (map_map _ _ _).symm
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_hom₂_ne_zero [CommMonoid γ] {s : Multiset ι} (hs : s ≠ 0) (f : α → β → γ)
    (hf : ∀ a b c d, f (a * b) (c * d) = f a c * f b d) (f₁ : ι → α) (f₂ : ι → β) :
    (s.map fun i => f (f₁ i) (f₂ i)).prod = f (s.map f₁).prod (s.map f₂).prod := by
  /-
    ι : Type u_2
    α : Type u_3
    β : Type u_4
    γ : Type u_6
    inst✝² : CommMonoid α
    inst✝¹ : CommMonoid β
    inst✝ : CommMonoid γ
    s : Multiset ι
    hs : Ne s 0
    f : α → β → γ
    hf : ∀ (a b : α) (c d : β), Eq (f (HMul.hMul a b) (HMul.hMul c d)) (HMul.hMul  …
    f₁ : ι → α
    f₂ : ι → β
    ⊢ Eq (Multiset.map (fun i => f (f₁ i) (f₂ i)) s).prod (f (Multiset.map f₁ s).p …
  -/
  induction s using Quotient.inductionOn; aesop (add simp List.prod_hom₂_nonempty)
                                          /-
                                            🎉 no goals
                                          -/


@[to_additive]
theorem prod_hom₂ [CommMonoid γ] (s : Multiset ι) (f : α → β → γ)
    (hf : ∀ a b c d, f (a * b) (c * d) = f a c * f b d) (hf' : f 1 1 = 1) (f₁ : ι → α)
    (f₂ : ι → β) : (s.map fun i => f (f₁ i) (f₂ i)).prod = f (s.map f₁).prod (s.map f₂).prod :=
  Quotient.inductionOn s fun l => by
    /-
      ι : Type u_2
      α : Type u_3
      β : Type u_4
      γ : Type u_6
      inst✝² : CommMonoid α
      inst✝¹ : CommMonoid β
      inst✝ : CommMonoid γ
      s : Multiset ι
      f : α → β → γ
      hf : ∀ (a b : α) (c d : β), Eq (f (HMul.hMul a b) (HMul.hMul c d)) (HMul.hMul  …
      hf' : Eq (f 1 1) 1
      f₁ : ι → α
      f₂ : ι → β
      l : List ι
      ⊢ Eq (Multiset.map (fun i => f (f₁ i) (f₂ i)) (Quotient.mk (List.isSetoid ι) l …
    -/
    simp only [l.prod_hom₂ f hf hf', quot_mk_to_coe, map_coe, prod_coe]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem prod_hom_rel (s : Multiset ι) {r : α → β → Prop} {f : ι → α} {g : ι → β}
    (h₁ : r 1 1) (h₂ : ∀ ⦃a b c⦄, r b c → r (f a * b) (g a * c)) :
    r (s.map f).prod (s.map g).prod :=
  Quotient.inductionOn s fun l => by
    /-
      ι : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝¹ : CommMonoid α
      inst✝ : CommMonoid β
      s : Multiset ι
      r : α → β → Prop
      f : ι → α
      g : ι → β
      h₁ : r 1 1
      h₂ : ∀ ⦃a : ι⦄ ⦃b : α⦄ ⦃c : β⦄, r b c → r (HMul.hMul (f a) b) (HMul.hMul (g a) …
      l : List ι
      ⊢ r (Multiset.map f (Quotient.mk (List.isSetoid ι) l)).prod (Multiset.map g (Q …
    -/
    simp only [l.prod_hom_rel h₁ h₂, quot_mk_to_coe, map_coe, prod_coe]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem prod_map_one : prod (m.map fun _ => (1 : α)) = 1 := by
  /-
    ι : Type u_2
    α : Type u_3
    inst✝ : CommMonoid α
    m : Multiset ι
    ⊢ Eq (Multiset.map (fun x => 1) m).prod 1
  -/
  rw [map_const', prod_replicate, one_pow]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem prod_map_mul : (m.map fun i => f i * g i).prod = (m.map f).prod * (m.map g).prod :=
  m.prod_hom₂ (· * ·) mul_mul_mul_comm (mul_one _) _ _


@[to_additive]
theorem prod_map_pow {n : ℕ} : (m.map fun i => f i ^ n).prod = (m.map f).prod ^ n :=
  m.prod_hom' (powMonoidHom n : α →* α) f


@[to_additive]
theorem prod_map_prod_map (m : Multiset β') (n : Multiset γ) {f : β' → γ → α} :
    prod (m.map fun a => prod <| n.map fun b => f a b) =
      prod (n.map fun b => prod <| m.map fun a => f a b) :=
                              /-
                                α : Type u_3
                                β' : Type u_5
                                γ : Type u_6
                                inst✝ : CommMonoid α
                                m : Multiset β'
                                n : Multiset γ
                                f : β' → γ → α
                                ⊢ Eq (Multiset.map (fun a => (Multiset.map (fun b => f a b) n).prod) 0).prod ( …
                              -/
                              /-
                                🎉 no goals
                              -/
  Multiset.induction_on m (by simp) fun a m ih => by simp [ih]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[to_additive]
theorem prod_induction (p : α → Prop) (s : Multiset α) (p_mul : ∀ a b, p a → p b → p (a * b))
    (p_one : p 1) (p_s : ∀ a ∈ s, p a) : p s.prod := by
  /-
    α : Type u_3
    inst✝ : CommMonoid α
    p : α → Prop
    s : Multiset α
    p_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    p_one : p 1
    p_s : ∀ (a : α), Membership.mem s a → p a
    ⊢ p s.prod
  -/
  rw [prod_eq_foldr]
  /-
    α : Type u_3
    inst✝ : CommMonoid α
    p : α → Prop
    s : Multiset α
    p_mul : ∀ (a b : α), p a → p b → p (HMul.hMul a b)
    p_one : p 1
    p_s : ∀ (a : α), Membership.mem s a → p a
    ⊢ p (Multiset.foldr (fun x1 x2 => HMul.hMul x1 x2) 1 s)
  -/
  exact foldr_induction (· * ·) 1 p s p_mul p_one p_s
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prod_induction_nonempty (p : α → Prop) (p_mul : ∀ a b, p a → p b → p (a * b)) (hs : s ≠ ∅)
    (p_s : ∀ a ∈ s, p a) : p s.prod := by
  induction s using Multiset.induction_on with
  | empty => simp at hs
  | cons a s hsa =>
    rw [prod_cons]
    by_cases hs_empty : s = ∅
    · simp [hs_empty, p_s a]
    have hps : ∀ x, x ∈ s → p x := fun x hxs => p_s x (mem_cons_of_mem hxs)
    exact p_mul a s.prod (p_s a (mem_cons_self a s)) (hsa hs_empty hps)


theorem prod_dvd_prod_of_le (h : s ≤ t) : s.prod ∣ t.prod := by
  /-
    α : Type u_3
    inst✝ : CommMonoid α
    s t : Multiset α
    h : LE.le s t
    ⊢ Dvd.dvd s.prod t.prod
  -/
  obtain ⟨z, rfl⟩ := exists_add_of_le h
  /-
    case intro
    α : Type u_3
    inst✝ : CommMonoid α
    s z : Multiset α
    h : LE.le s (HAdd.hAdd s z)
    ⊢ Dvd.dvd s.prod (HAdd.hAdd s z).prod
  -/
  simp only [prod_add, dvd_mul_right]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma _root_.map_multiset_prod [FunLike F α β] [MonoidHomClass F α β] (f : F) (s : Multiset α) :
    f s.prod = (s.map f).prod := (s.prod_hom f).symm


@[to_additive]
lemma _root_.map_multiset_ne_zero_prod [FunLike F α β] [MulHomClass F α β] (f : F)
    {s : Multiset α} (hs : s ≠ 0):
    f s.prod = (s.map f).prod := (s.prod_hom_ne_zero hs f).symm


@[to_additive]
protected lemma _root_.MonoidHom.map_multiset_prod (f : α →* β) (s : Multiset α) :
    f s.prod = (s.map f).prod := (s.prod_hom f).symm


@[to_additive]
protected lemma _root_.MulHom.map_multiset_ne_zero_prod (f : α →ₙ* β) (s : Multiset α)
    (hs : s ≠ 0) : f s.prod = (s.map f).prod := (s.prod_hom_ne_zero hs f).symm


lemma dvd_prod : a ∈ s → a ∣ s.prod :=
                                         /-
                                           α : Type u_3
                                           inst✝ : CommMonoid α
                                           s : Multiset α
                                           a✝ : α
                                           l : List α
                                           a : α
                                           h : Membership.mem (Quotient.mk (List.isSetoid α) l) a
                                           ⊢ Dvd.dvd a (Multiset.prod (Quotient.mk (List.isSetoid α) l))
                                         -/
  Quotient.inductionOn s (fun l a h ↦ by simpa using List.dvd_prod h) a
                                         /-
                                           🎉 no goals
                                         -/


@[to_additive] lemma fst_prod (s : Multiset (α × β)) : s.prod.1 = (s.map Prod.fst).prod :=
  map_multiset_prod (MonoidHom.fst _ _) _


@[to_additive] lemma snd_prod (s : Multiset (α × β)) : s.prod.2 = (s.map Prod.snd).prod :=
  map_multiset_prod (MonoidHom.snd _ _) _


theorem prod_dvd_prod_of_dvd [CommMonoid β] {S : Multiset α} (g1 g2 : α → β)
    (h : ∀ a ∈ S, g1 a ∣ g2 a) : (Multiset.map g1 S).prod ∣ (Multiset.map g2 S).prod := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    S : Multiset α
    g1 g2 : α → β
    h : ∀ (a : α), Membership.mem S a → Dvd.dvd (g1 a) (g2 a)
    ⊢ Dvd.dvd (Multiset.map g1 S).prod (Multiset.map g2 S).prod
  -/
  apply Multiset.induction_on' S
    /-
      case h₁
      α : Type u_3
      β : Type u_4
      inst✝ : CommMonoid β
      S : Multiset α
      g1 g2 : α → β
      h : ∀ (a : α), Membership.mem S a → Dvd.dvd (g1 a) (g2 a)
      ⊢ Dvd.dvd (Multiset.map g1 0).prod (Multiset.map g2 0).prod
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case h₂
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    S : Multiset α
    g1 g2 : α → β
    h : ∀ (a : α), Membership.mem S a → Dvd.dvd (g1 a) (g2 a)
    ⊢ ∀ {a : α} {s : Multiset α}, Membership.mem S a → HasSubset.Subset s S → Dvd. …
  -/
  intro a T haS _ IH
  /-
    case h₂
    α : Type u_3
    β : Type u_4
    inst✝ : CommMonoid β
    S : Multiset α
    g1 g2 : α → β
    h : ∀ (a : α), Membership.mem S a → Dvd.dvd (g1 a) (g2 a)
    a : α
    T : Multiset α
    haS : Membership.mem S a
    a✝ : HasSubset.Subset T S
    IH : Dvd.dvd (Multiset.map g1 T).prod (Multiset.map g2 T).prod
    ⊢ Dvd.dvd (Multiset.map g1 (Insert.insert a T)).prod (Multiset.map g2 (Insert. …
  -/
  simp [mul_dvd_mul (h a haS) IH]
  /-
    🎉 no goals
  -/


/-- `Multiset.sum`, the sum of the elements of a multiset, promoted to a morphism of
`AddCommMonoid`s. -/
def sumAddMonoidHom : Multiset α →+ α where
  toFun := sum
  map_zero' := sum_zero
  map_add' := sum_add


@[simp]
theorem coe_sumAddMonoidHom : (sumAddMonoidHom : Multiset α → α) = sum :=
  rfl


@[to_additive]
theorem prod_map_inv' (m : Multiset α) : (m.map Inv.inv).prod = m.prod⁻¹ :=
  m.prod_hom (invMonoidHom : α →* α)


@[to_additive (attr := simp)]
theorem prod_map_inv : (m.map fun i => (f i)⁻¹).prod = (m.map f).prod⁻¹ := by
  /-
    ι : Type u_2
    α : Type u_3
    inst✝ : DivisionCommMonoid α
    m : Multiset ι
    f : ι → α
    ⊢ Eq (Multiset.map (fun i => Inv.inv (f i)) m).prod (Inv.inv (Multiset.map f m …
  -/
  rw [← (m.map f).prod_map_inv', map_map, Function.comp_def]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem prod_map_div : (m.map fun i => f i / g i).prod = (m.map f).prod / (m.map g).prod :=
  m.prod_hom₂ (· / ·) mul_div_mul_comm (div_one _) _ _


@[to_additive]
theorem prod_map_zpow {n : ℤ} : (m.map fun i => f i ^ n).prod = (m.map f).prod ^ n := by
  /-
    ι : Type u_2
    α : Type u_3
    inst✝ : DivisionCommMonoid α
    m : Multiset ι
    f : ι → α
    n : Int
    ⊢ Eq (Multiset.map (fun i => HPow.hPow (f i) n) m).prod (HPow.hPow (Multiset.m …
  -/
  convert (m.map f).prod_hom (zpowGroupHom n : α →* α)
  /-
    case h.e'_2.h.e'_3
    ι : Type u_2
    α : Type u_3
    inst✝ : DivisionCommMonoid α
    m : Multiset ι
    f : ι → α
    n : Int
    ⊢ Eq (Multiset.map (fun i => HPow.hPow (f i) n) m) (Multiset.map (⇑(zpowGroupH …
  -/
  simp only [map_map, Function.comp_apply, zpowGroupHom_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem sum_map_singleton (s : Multiset α) : (s.map fun a => ({a} : Multiset α)).sum = s :=
                              /-
                                α : Type u_3
                                s : Multiset α
                                ⊢ Eq (Multiset.map (fun a => Singleton.singleton a) 0).sum 0
                              -/
                              /-
                                🎉 no goals
                              -/
  Multiset.induction_on s (by simp) (by simp)
                                        /-
                                          🎉 no goals
                                        -/


theorem sum_nat_mod (s : Multiset ℕ) (n : ℕ) : s.sum % n = (s.map (· % n)).sum % n := by
  /-
    s : Multiset Nat
    n : Nat
    ⊢ Eq (HMod.hMod s.sum n) (HMod.hMod (Multiset.map (fun x => HMod.hMod x n) s). …
  -/
                                           /-
                                             🎉 no goals
                                           -/
  induction s using Multiset.induction <;> simp [Nat.add_mod, *]
                                           /-
                                             🎉 no goals
                                           -/


theorem prod_nat_mod (s : Multiset ℕ) (n : ℕ) : s.prod % n = (s.map (· % n)).prod % n := by
  /-
    s : Multiset Nat
    n : Nat
    ⊢ Eq (HMod.hMod s.prod n) (HMod.hMod (Multiset.map (fun x => HMod.hMod x n) s) …
  -/
                                           /-
                                             🎉 no goals
                                           -/
  induction s using Multiset.induction <;> simp [Nat.mul_mod, *]
                                           /-
                                             🎉 no goals
                                           -/


theorem sum_int_mod (s : Multiset ℤ) (n : ℤ) : s.sum % n = (s.map (· % n)).sum % n := by
  /-
    s : Multiset Int
    n : Int
    ⊢ Eq (HMod.hMod s.sum n) (HMod.hMod (Multiset.map (fun x => HMod.hMod x n) s). …
  -/
                                           /-
                                             🎉 no goals
                                           -/
  induction s using Multiset.induction <;> simp [Int.add_emod, *]
                                           /-
                                             🎉 no goals
                                           -/


theorem prod_int_mod (s : Multiset ℤ) (n : ℤ) : s.prod % n = (s.map (· % n)).prod % n := by
  /-
    s : Multiset Int
    n : Int
    ⊢ Eq (HMod.hMod s.prod n) (HMod.hMod (Multiset.map (fun x => HMod.hMod x n) s) …
  -/
                                           /-
                                             🎉 no goals
                                           -/
  induction s using Multiset.induction <;> simp [Int.mul_emod, *]
                                           /-
                                             🎉 no goals
                                           -/


theorem sum_map_tsub [AddCommMonoid α] [PartialOrder α] [ExistsAddOfLE α]
    [CovariantClass α α (· + ·) (· ≤ ·)] [ContravariantClass α α (· + ·) (· ≤ ·)] [Sub α]
    [OrderedSub α] (l : Multiset ι) {f g : ι → α} (hfg : ∀ x ∈ l, g x ≤ f x) :
    (l.map fun x ↦ f x - g x).sum = (l.map f).sum - (l.map g).sum :=
  eq_tsub_of_add_eq <| by
    /-
      ι : Type u_2
      α : Type u_3
      inst✝⁶ : AddCommMonoid α
      inst✝⁵ : PartialOrder α
      inst✝⁴ : ExistsAddOfLE α
      inst✝³ : CovariantClass α α (fun x1 x2 => HAdd.hAdd x1 x2) fun x1 x2 => LE.le  …
      inst✝² : ContravariantClass α α (fun x1 x2 => HAdd.hAdd x1 x2) fun x1 x2 => LE …
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      l : Multiset ι
      f g : ι → α
      hfg : ∀ (x : ι), Membership.mem l x → LE.le (g x) (f x)
      ⊢ Eq (HAdd.hAdd (Multiset.map (fun x => HSub.hSub (f x) (g x)) l).sum (Multise …
    -/
    rw [← sum_map_add]
    /-
      ι : Type u_2
      α : Type u_3
      inst✝⁶ : AddCommMonoid α
      inst✝⁵ : PartialOrder α
      inst✝⁴ : ExistsAddOfLE α
      inst✝³ : CovariantClass α α (fun x1 x2 => HAdd.hAdd x1 x2) fun x1 x2 => LE.le  …
      inst✝² : ContravariantClass α α (fun x1 x2 => HAdd.hAdd x1 x2) fun x1 x2 => LE …
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      l : Multiset ι
      f g : ι → α
      hfg : ∀ (x : ι), Membership.mem l x → LE.le (g x) (f x)
      ⊢ Eq (Multiset.map (fun i => HAdd.hAdd (HSub.hSub (f i) (g i)) (g i)) l).sum ( …
    -/
    congr 1
    /-
      case e_a
      ι : Type u_2
      α : Type u_3
      inst✝⁶ : AddCommMonoid α
      inst✝⁵ : PartialOrder α
      inst✝⁴ : ExistsAddOfLE α
      inst✝³ : CovariantClass α α (fun x1 x2 => HAdd.hAdd x1 x2) fun x1 x2 => LE.le  …
      inst✝² : ContravariantClass α α (fun x1 x2 => HAdd.hAdd x1 x2) fun x1 x2 => LE …
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      l : Multiset ι
      f g : ι → α
      hfg : ∀ (x : ι), Membership.mem l x → LE.le (g x) (f x)
      ⊢ Eq (Multiset.map (fun i => HAdd.hAdd (HSub.hSub (f i) (g i)) (g i)) l) (Mult …
    -/
    exact map_congr rfl fun x hx => tsub_add_cancel_of_le <| hfg _ hx
    /-
      🎉 no goals
    -/


