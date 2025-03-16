local infixl:50 " ~ᵤ " => Associated


theorem exists_mem_multiset_dvd (hp : Prime p) {s : Multiset α} : p ∣ s.prod → ∃ a ∈ s, p ∣ a :=
  Multiset.induction_on s (fun h => (hp.not_dvd_one h).elim) fun a s ih h =>
                                /-
                                  α : Type u_1
                                  inst✝ : CommMonoidWithZero α
                                  p : α
                                  hp : Prime p
                                  s✝ : Multiset α
                                  a : α
                                  s : Multiset α
                                  ih : Dvd.dvd p s.prod → Exists fun a => And (Membership.mem s a) (Dvd.dvd p a)
                                  h : Dvd.dvd p (Multiset.cons a s).prod
                                  ⊢ Dvd.dvd p (HMul.hMul a s.prod)
                                -/
    have : p ∣ a * s.prod := by simpa using h
                                /-
                                  🎉 no goals
                                -/
    match hp.dvd_or_dvd this with
    | Or.inl h => ⟨a, Multiset.mem_cons_self a s, h⟩
    | Or.inr h =>
      let ⟨a, has, h⟩ := ih h
      ⟨a, Multiset.mem_cons_of_mem has, h⟩


theorem exists_mem_multiset_map_dvd (hp : Prime p) {s : Multiset β} {f : β → α} :
    p ∣ (s.map f).prod → ∃ a ∈ s, p ∣ f a := fun h => by
  simpa only [exists_prop, Multiset.mem_map, exists_exists_and_eq_and] using
    hp.exists_mem_multiset_dvd h


theorem exists_mem_finset_dvd (hp : Prime p) {s : Finset β} {f : β → α} :
    p ∣ s.prod f → ∃ i ∈ s, p ∣ f i :=
  hp.exists_mem_multiset_map_dvd


theorem Prod.associated_iff {M N : Type*} [Monoid M] [Monoid N] {x z : M × N} :
    x ~ᵤ z ↔ x.1 ~ᵤ z.1 ∧ x.2 ~ᵤ z.2 :=
  ⟨fun ⟨u, hu⟩ => ⟨⟨(MulEquiv.prodUnits.toFun u).1, (Prod.eq_iff_fst_eq_snd_eq.1 hu).1⟩,
    ⟨(MulEquiv.prodUnits.toFun u).2, (Prod.eq_iff_fst_eq_snd_eq.1 hu).2⟩⟩,
  fun ⟨⟨u₁, h₁⟩, ⟨u₂, h₂⟩⟩ =>
    ⟨MulEquiv.prodUnits.invFun (u₁, u₂), Prod.eq_iff_fst_eq_snd_eq.2 ⟨h₁, h₂⟩⟩⟩


theorem Associated.prod {M : Type*} [CommMonoid M] {ι : Type*} (s : Finset ι) (f : ι → M)
    (g : ι → M) (h : ∀ i, i ∈ s → (f i) ~ᵤ (g i)) : (∏ i ∈ s, f i) ~ᵤ (∏ i ∈ s, g i) := by
  induction s using Finset.induction with
  | empty =>
    simp only [Finset.prod_empty]
    rfl
  | @insert j s hjs IH =>
    classical
    convert_to (∏ i ∈ insert j s, f i) ~ᵤ (∏ i ∈ insert j s, g i)
    rw [Finset.prod_insert hjs, Finset.prod_insert hjs]
    exact Associated.mul_mul (h j (Finset.mem_insert_self j s))
      (IH (fun i hi ↦ h i (Finset.mem_insert_of_mem hi)))


theorem exists_associated_mem_of_dvd_prod [CancelCommMonoidWithZero α] {p : α} (hp : Prime p)
    {s : Multiset α} : (∀ r ∈ s, Prime r) → p ∣ s.prod → ∃ q ∈ s, p ~ᵤ q :=
                              /-
                                α : Type u_1
                                inst✝ : CancelCommMonoidWithZero α
                                p : α
                                hp : Prime p
                                s : Multiset α
                                ⊢ (∀ (r : α), Membership.mem 0 r → Prime r) → Dvd.dvd p (Multiset.prod 0) → Ex …
                              -/
  Multiset.induction_on s (by simp [mt isUnit_iff_dvd_one.2 hp.not_unit]) fun a s ih hs hps => by
                              /-
                                🎉 no goals
                              -/
    /-
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      p : α
      hp : Prime p
      s✝ : Multiset α
      a : α
      s : Multiset α
      ih : (∀ (r : α), Membership.mem s r → Prime r) → Dvd.dvd p s.prod → Exists fun …
      hs : ∀ (r : α), Membership.mem (Multiset.cons a s) r → Prime r
      hps : Dvd.dvd p (Multiset.cons a s).prod
      ⊢ Exists fun q => And (Membership.mem (Multiset.cons a s) q) (Associated p q)
    -/
    rw [Multiset.prod_cons] at hps
    /-
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      p : α
      hp : Prime p
      s✝ : Multiset α
      a : α
      s : Multiset α
      ih : (∀ (r : α), Membership.mem s r → Prime r) → Dvd.dvd p s.prod → Exists fun …
      hs : ∀ (r : α), Membership.mem (Multiset.cons a s) r → Prime r
      hps : Dvd.dvd p (HMul.hMul a s.prod)
      ⊢ Exists fun q => And (Membership.mem (Multiset.cons a s) q) (Associated p q)
    -/
    rcases hp.dvd_or_dvd hps with h | h
      /-
        case inl
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        p : α
        hp : Prime p
        s✝ : Multiset α
        a : α
        s : Multiset α
        ih : (∀ (r : α), Membership.mem s r → Prime r) → Dvd.dvd p s.prod → Exists fun …
        hs : ∀ (r : α), Membership.mem (Multiset.cons a s) r → Prime r
        hps : Dvd.dvd p (HMul.hMul a s.prod)
        h : Dvd.dvd p a
        ⊢ Exists fun q => And (Membership.mem (Multiset.cons a s) q) (Associated p q)
      -/
    · have hap := hs a (Multiset.mem_cons.2 (Or.inl rfl))
      /-
        case inl
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        p : α
        hp : Prime p
        s✝ : Multiset α
        a : α
        s : Multiset α
        ih : (∀ (r : α), Membership.mem s r → Prime r) → Dvd.dvd p s.prod → Exists fun …
        hs : ∀ (r : α), Membership.mem (Multiset.cons a s) r → Prime r
        hps : Dvd.dvd p (HMul.hMul a s.prod)
        h : Dvd.dvd p a
        hap : Prime a
        ⊢ Exists fun q => And (Membership.mem (Multiset.cons a s) q) (Associated p q)
      -/
      exact ⟨a, Multiset.mem_cons_self a _, hp.associated_of_dvd hap h⟩
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        p : α
        hp : Prime p
        s✝ : Multiset α
        a : α
        s : Multiset α
        ih : (∀ (r : α), Membership.mem s r → Prime r) → Dvd.dvd p s.prod → Exists fun …
        hs : ∀ (r : α), Membership.mem (Multiset.cons a s) r → Prime r
        hps : Dvd.dvd p (HMul.hMul a s.prod)
        h : Dvd.dvd p s.prod
        ⊢ Exists fun q => And (Membership.mem (Multiset.cons a s) q) (Associated p q)
      -/
    · rcases ih (fun r hr => hs _ (Multiset.mem_cons.2 (Or.inr hr))) h with ⟨q, hq₁, hq₂⟩
      /-
        case inr.intro.intro
        α : Type u_1
        inst✝ : CancelCommMonoidWithZero α
        p : α
        hp : Prime p
        s✝ : Multiset α
        a : α
        s : Multiset α
        ih : (∀ (r : α), Membership.mem s r → Prime r) → Dvd.dvd p s.prod → Exists fun …
        hs : ∀ (r : α), Membership.mem (Multiset.cons a s) r → Prime r
        hps : Dvd.dvd p (HMul.hMul a s.prod)
        h : Dvd.dvd p s.prod
        q : α
        hq₁ : Membership.mem s q
        hq₂ : Associated p q
        ⊢ Exists fun q => And (Membership.mem (Multiset.cons a s) q) (Associated p q)
      -/
      exact ⟨q, Multiset.mem_cons.2 (Or.inr hq₁), hq₂⟩
      /-
        🎉 no goals
      -/


open Submonoid in
/-- Let x, y ∈ α. If x * y can be written as a product of units and prime elements, then x can be
written as a product of units and prime elements. -/
theorem divisor_closure_eq_closure [CancelCommMonoidWithZero α]
    (x y : α) (hxy : x * y ∈ closure { r : α | IsUnit r ∨ Prime r}) :
    x ∈ closure { r : α | IsUnit r ∨ Prime r} := by
  /-
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    x y : α
    hxy : Membership.mem (Submonoid.closure (setOf fun r => Or (IsUnit r) (Prime r …
    ⊢ Membership.mem (Submonoid.closure (setOf fun r => Or (IsUnit r) (Prime r))) x
  -/
  obtain ⟨m, hm, hprod⟩ := exists_multiset_of_mem_closure hxy
  induction m using Multiset.induction generalizing x y with
  | empty =>
    apply subset_closure
    simp only [Set.mem_setOf]
    simp only [Multiset.prod_zero] at hprod
    left; exact isUnit_of_mul_eq_one _ _ hprod.symm
  | @cons c s hind =>
    simp only [Multiset.mem_cons, forall_eq_or_imp, Set.mem_setOf] at hm
    simp only [Multiset.prod_cons] at hprod
    simp only [Set.mem_setOf_eq] at hind
    obtain ⟨ha₁ | ha₂, hs⟩ := hm
    · rcases ha₁.exists_right_inv with ⟨k, hk⟩
      refine hind x (y*k) ?_ hs ?_
      · simp only [← mul_assoc, ← hprod, ← Multiset.prod_cons, mul_comm]
        refine multiset_prod_mem _ _ (Multiset.forall_mem_cons.2 ⟨subset_closure (Set.mem_def.2 ?_),
          Multiset.forall_mem_cons.2 ⟨subset_closure (Set.mem_def.2 ?_), (fun t ht =>
          subset_closure (hs t ht))⟩⟩)
        · left; exact isUnit_of_mul_eq_one_right _ _ hk
        · left; exact ha₁
      · rw [← mul_one s.prod, ← hk, ← mul_assoc, ← mul_assoc, mul_eq_mul_right_iff, mul_comm]
        left; exact hprod
    · rcases ha₂.dvd_mul.1 (Dvd.intro _ hprod) with ⟨c, hc⟩ | ⟨c, hc⟩
      · rw [hc]; rw [hc, mul_assoc] at hprod
        refine Submonoid.mul_mem _ (subset_closure (Set.mem_def.2 ?_))
          (hind _ _ ?_ hs (mul_left_cancel₀ ha₂.ne_zero hprod))
        · right; exact ha₂
        rw [← mul_left_cancel₀ ha₂.ne_zero hprod]
        exact multiset_prod_mem _ _ (fun t ht => subset_closure (hs t ht))
      rw [hc, mul_comm x _, mul_assoc, mul_comm c _] at hprod
      refine hind x c ?_ hs (mul_left_cancel₀ ha₂.ne_zero hprod)
      rw [← mul_left_cancel₀ ha₂.ne_zero hprod]
      exact multiset_prod_mem _ _ (fun t ht => subset_closure (hs t ht))


theorem Multiset.prod_primes_dvd [CancelCommMonoidWithZero α]
    [∀ a : α, DecidablePred (Associated a)] {s : Multiset α} (n : α) (h : ∀ a ∈ s, Prime a)
    (div : ∀ a ∈ s, a ∣ n) (uniq : ∀ a, s.countP (Associated a) ≤ 1) : s.prod ∣ n := by
  induction s using Multiset.induction_on generalizing n with
  | empty => simp only [Multiset.prod_zero, one_dvd]
  | cons a s induct =>
    rw [Multiset.prod_cons]
    obtain ⟨k, rfl⟩ : a ∣ n := div a (Multiset.mem_cons_self a s)
    apply mul_dvd_mul_left a
    refine induct _ (fun a ha => h a (Multiset.mem_cons_of_mem ha)) (fun b b_in_s => ?_)
      fun a => (Multiset.countP_le_of_le _ (Multiset.le_cons_self _ _)).trans (uniq a)
    have b_div_n := div b (Multiset.mem_cons_of_mem b_in_s)
    have a_prime := h a (Multiset.mem_cons_self a s)
    have b_prime := h b (Multiset.mem_cons_of_mem b_in_s)
    refine (b_prime.dvd_or_dvd b_div_n).resolve_left fun b_div_a => ?_
    have assoc := b_prime.associated_of_dvd a_prime b_div_a
    have := uniq a
    rw [Multiset.countP_cons_of_pos _ (Associated.refl _), Nat.succ_le_succ_iff, ← not_lt,
      Multiset.countP_pos] at this
    exact this ⟨b, b_in_s, assoc.symm⟩


theorem Finset.prod_primes_dvd [CancelCommMonoidWithZero α] [Subsingleton αˣ] {s : Finset α} (n : α)
    (h : ∀ a ∈ s, Prime a) (div : ∀ a ∈ s, a ∣ n) : (∏ p ∈ s, p) ∣ n := by
  classical
    exact
      Multiset.prod_primes_dvd n (by simpa only [Multiset.map_id', Finset.mem_def] using h)
        (by simpa only [Multiset.map_id', Finset.mem_def] using div)
        (by
          simp only [Multiset.map_id', associated_eq_eq, Multiset.countP_eq_card_filter,
            ← s.val.count_eq_card_filter_eq, ← Multiset.nodup_iff_count_le_one, s.nodup])


theorem prod_mk {p : Multiset α} : (p.map Associates.mk).prod = Associates.mk p.prod :=
                              /-
                                α : Type u_1
                                inst✝ : CommMonoid α
                                p : Multiset α
                                ⊢ Eq (Multiset.map Associates.mk 0).prod (Associates.mk (Multiset.prod 0))
                              -/
                              /-
                                🎉 no goals
                              -/
  Multiset.induction_on p (by simp) fun a s ih => by simp [ih, Associates.mk_mul_mk]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem finset_prod_mk {p : Finset β} {f : β → α} :
    (∏ i ∈ p, Associates.mk (f i)) = Associates.mk (∏ i ∈ p, f i) := by
  -- Porting note: added
  have : (fun i => Associates.mk (f i)) = Associates.mk ∘ f :=
    funext fun x => Function.comp_apply
  rw [Finset.prod_eq_multiset_prod, this, ← Multiset.map_map, prod_mk,
    ← Finset.prod_eq_multiset_prod]


theorem rel_associated_iff_map_eq_map {p q : Multiset α} :
    Multiset.Rel Associated p q ↔ p.map Associates.mk = q.map Associates.mk := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    p q : Multiset α
    ⊢ Iff (Multiset.Rel Associated p q) (Eq (Multiset.map Associates.mk p) (Multis …
  -/
  rw [← Multiset.rel_eq, Multiset.rel_map]
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    p q : Multiset α
    ⊢ Iff (Multiset.Rel Associated p q) (Multiset.Rel (fun a b => Eq (Associates.m …
  -/
  simp only [mk_eq_mk_iff_associated]
  /-
    🎉 no goals
  -/


theorem prod_eq_one_iff {p : Multiset (Associates α)} :
    p.prod = 1 ↔ ∀ a ∈ p, (a : Associates α) = 1 :=
                              /-
                                α : Type u_1
                                inst✝ : CommMonoid α
                                p : Multiset (Associates α)
                                ⊢ Iff (Eq (Multiset.prod 0) 1) (∀ (a : Associates α), Membership.mem 0 a → Eq  …
                              -/
  Multiset.induction_on p (by simp)
                              /-
                                🎉 no goals
                              -/
        /-
          α : Type u_1
          inst✝ : CommMonoid α
          p : Multiset (Associates α)
          ⊢ ∀ (a : Associates α) (s : Multiset (Associates α)), Iff (Eq s.prod 1) (∀ (a  …
        -/
    (by simp +contextual [mul_eq_one, or_imp, forall_and])
        /-
          🎉 no goals
        -/


theorem prod_le_prod {p q : Multiset (Associates α)} (h : p ≤ q) : p.prod ≤ q.prod := by
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    p q : Multiset (Associates α)
    h : LE.le p q
    ⊢ LE.le p.prod q.prod
  -/
  haveI := Classical.decEq (Associates α)
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    p q : Multiset (Associates α)
    h : LE.le p q
    this : DecidableEq (Associates α)
    ⊢ LE.le p.prod q.prod
  -/
  haveI := Classical.decEq α
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    p q : Multiset (Associates α)
    h : LE.le p q
    this✝ : DecidableEq (Associates α)
    this : DecidableEq α
    ⊢ LE.le p.prod q.prod
  -/
  suffices p.prod ≤ (p + (q - p)).prod by rwa [add_tsub_cancel_of_le h] at this
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    p q : Multiset (Associates α)
    h : LE.le p q
    this✝ : DecidableEq (Associates α)
    this : DecidableEq α
    ⊢ LE.le p.prod (HAdd.hAdd p (HSub.hSub q p)).prod
  -/
  suffices p.prod * 1 ≤ p.prod * (q - p).prod by simpa
  /-
    α : Type u_1
    inst✝ : CommMonoid α
    p q : Multiset (Associates α)
    h : LE.le p q
    this✝ : DecidableEq (Associates α)
    this : DecidableEq α
    ⊢ LE.le (HMul.hMul p.prod 1) (HMul.hMul p.prod (HSub.hSub q p).prod)
  -/
  exact mul_mono (le_refl p.prod) one_le
  /-
    🎉 no goals
  -/


theorem exists_mem_multiset_le_of_prime {s : Multiset (Associates α)} {p : Associates α}
    (hp : Prime p) : p ≤ s.prod → ∃ a ∈ s, p ≤ a :=
  Multiset.induction_on s (fun ⟨_, Eq⟩ => (hp.ne_one (mul_eq_one.1 Eq.symm).1).elim)
    fun a s ih h =>
                                /-
                                  α : Type u_1
                                  inst✝ : CancelCommMonoidWithZero α
                                  s✝ : Multiset (Associates α)
                                  p : Associates α
                                  hp : Prime p
                                  a : Associates α
                                  s : Multiset (Associates α)
                                  ih : LE.le p s.prod → Exists fun a => And (Membership.mem s a) (LE.le p a)
                                  h : LE.le p (Multiset.cons a s).prod
                                  ⊢ LE.le p (HMul.hMul a s.prod)
                                -/
    have : p ≤ a * s.prod := by simpa using h
                                /-
                                  🎉 no goals
                                -/
    match Prime.le_or_le hp this with
    | Or.inl h => ⟨a, Multiset.mem_cons_self a s, h⟩
    | Or.inr h =>
      let ⟨a, has, h⟩ := ih h
      ⟨a, Multiset.mem_cons_of_mem has, h⟩


theorem prod_ne_zero_of_prime [CancelCommMonoidWithZero α] [Nontrivial α] (s : Multiset α)
    (h : ∀ x ∈ s, Prime x) : s.prod ≠ 0 :=
  Multiset.prod_ne_zero fun h0 => Prime.ne_zero (h 0 h0) rfl


theorem Prime.dvd_finset_prod_iff {S : Finset α} {p : M} (pp : Prime p) (g : α → M) :
    p ∣ S.prod g ↔ ∃ a ∈ S, p ∣ g a :=
  ⟨pp.exists_mem_finset_dvd, fun ⟨_, ha1, ha2⟩ => dvd_trans ha2 (dvd_prod_of_mem g ha1)⟩


theorem Prime.not_dvd_finset_prod {S : Finset α} {p : M} (pp : Prime p) {g : α → M}
    (hS : ∀ a ∈ S, ¬p ∣ g a) : ¬p ∣ S.prod g := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝ : CommMonoidWithZero M
    S : Finset α
    p : M
    pp : Prime p
    g : α → M
    hS : ∀ (a : α), Membership.mem S a → Not (Dvd.dvd p (g a))
    ⊢ Not (Dvd.dvd p (S.prod g))
  -/
  exact mt (Prime.dvd_finset_prod_iff pp _).1 <| not_exists.2 fun a => not_and.2 (hS a)
  /-
    🎉 no goals
  -/


theorem Prime.dvd_finsupp_prod_iff {f : α →₀ M} {g : α → M → ℕ} {p : ℕ} (pp : Prime p) :
    p ∣ f.prod g ↔ ∃ a ∈ f.support, p ∣ g a (f a) :=
  Prime.dvd_finset_prod_iff pp _


theorem Prime.not_dvd_finsupp_prod {f : α →₀ M} {g : α → M → ℕ} {p : ℕ} (pp : Prime p)
    (hS : ∀ a ∈ f.support, ¬p ∣ g a (f a)) : ¬p ∣ f.prod g :=
  Prime.not_dvd_finset_prod pp hS


