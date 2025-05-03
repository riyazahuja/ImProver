/-- If `x * y = a * ∏ i ∈ s, p i` where `p i` is always prime, then
  `x` and `y` can both be written as a divisor of `a` multiplied by
  a product over a subset of `s`  -/
theorem mul_eq_mul_prime_prod {α : Type*} [DecidableEq α] {x y a : R} {s : Finset α} {p : α → R}
    (hp : ∀ i ∈ s, Prime (p i)) (hx : x * y = a * ∏ i ∈ s, p i) :
    ∃ (t u : Finset α) (b c : R),
      t ∪ u = s ∧ Disjoint t u ∧ a = b * c ∧ (x = b * ∏ i ∈ t, p i) ∧ y = c * ∏ i ∈ u, p i := by
  /-
    R : Type u_1
    inst✝¹ : CancelCommMonoidWithZero R
    α : Type u_2
    inst✝ : DecidableEq α
    x y a : R
    s : Finset α
    p : α → R
    hp : ∀ (i : α), Membership.mem s i → Prime (p i)
    hx : Eq (HMul.hMul x y) (HMul.hMul a (s.prod fun i => p i))
    ⊢ Exists fun t => Exists fun u => Exists fun b => Exists fun c => And (Eq (Uni …
  -/
  induction' s using Finset.induction with i s his ih generalizing x y a
    /-
      case empty
      R : Type u_1
      inst✝¹ : CancelCommMonoidWithZero R
      α : Type u_2
      inst✝ : DecidableEq α
      p : α → R
      x y a : R
      hp : ∀ (i : α), Membership.mem EmptyCollection.emptyCollection i → Prime (p i)
      hx : Eq (HMul.hMul x y) (HMul.hMul a (EmptyCollection.emptyCollection.prod fun …
      ⊢ Exists fun t => Exists fun u => Exists fun b => Exists fun c => And (Eq (Uni …
    -/
  · exact ⟨∅, ∅, x, y, by simp [hx]⟩
    /-
      🎉 no goals
    -/
    /-
      case insert
      R : Type u_1
      inst✝¹ : CancelCommMonoidWithZero R
      α : Type u_2
      inst✝ : DecidableEq α
      p : α → R
      i : α
      s : Finset α
      his : Not (Membership.mem s i)
      ih : ∀ {x y a : R}, (∀ (i : α), Membership.mem s i → Prime (p i)) → Eq (HMul.h …
      x y a : R
      hp : ∀ (i_1 : α), Membership.mem (Insert.insert i s) i_1 → Prime (p i_1)
      hx : Eq (HMul.hMul x y) (HMul.hMul a ((Insert.insert i s).prod fun i => p i))
      ⊢ Exists fun t => Exists fun u => Exists fun b => Exists fun c => And (Eq (Uni …
    -/
  · rw [prod_insert his, ← mul_assoc] at hx
    /-
      case insert
      R : Type u_1
      inst✝¹ : CancelCommMonoidWithZero R
      α : Type u_2
      inst✝ : DecidableEq α
      p : α → R
      i : α
      s : Finset α
      his : Not (Membership.mem s i)
      ih : ∀ {x y a : R}, (∀ (i : α), Membership.mem s i → Prime (p i)) → Eq (HMul.h …
      x y a : R
      hp : ∀ (i_1 : α), Membership.mem (Insert.insert i s) i_1 → Prime (p i_1)
      hx : Eq (HMul.hMul x y) (HMul.hMul (HMul.hMul a (p i)) (s.prod fun x => p x))
      ⊢ Exists fun t => Exists fun u => Exists fun b => Exists fun c => And (Eq (Uni …
    -/
    have hpi : Prime (p i) := hp i (mem_insert_self _ _)
    rcases ih (fun i hi ↦ hp i (mem_insert_of_mem hi)) hx with
      ⟨t, u, b, c, htus, htu, hbc, rfl, rfl⟩
    /-
      case insert.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝¹ : CancelCommMonoidWithZero R
      α : Type u_2
      inst✝ : DecidableEq α
      p : α → R
      i : α
      s : Finset α
      his : Not (Membership.mem s i)
      ih : ∀ {x y a : R}, (∀ (i : α), Membership.mem s i → Prime (p i)) → Eq (HMul.h …
      a : R
      hp : ∀ (i_1 : α), Membership.mem (Insert.insert i s) i_1 → Prime (p i_1)
      hpi : Prime (p i)
      t u : Finset α
      b c : R
      htus : Eq (Union.union t u) s
      htu : Disjoint t u
      hbc : Eq (HMul.hMul a (p i)) (HMul.hMul b c)
      hx : Eq (HMul.hMul (HMul.hMul b (t.prod fun i => p i)) (HMul.hMul c (u.prod fu …
      ⊢ Exists fun t_1 => Exists fun u_1 => Exists fun b_1 => Exists fun c_1 => And  …
    -/
    have hit : i ∉ t := fun hit ↦ his (htus ▸ mem_union_left _ hit)
    /-
      case insert.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝¹ : CancelCommMonoidWithZero R
      α : Type u_2
      inst✝ : DecidableEq α
      p : α → R
      i : α
      s : Finset α
      his : Not (Membership.mem s i)
      ih : ∀ {x y a : R}, (∀ (i : α), Membership.mem s i → Prime (p i)) → Eq (HMul.h …
      a : R
      hp : ∀ (i_1 : α), Membership.mem (Insert.insert i s) i_1 → Prime (p i_1)
      hpi : Prime (p i)
      t u : Finset α
      b c : R
      htus : Eq (Union.union t u) s
      htu : Disjoint t u
      hbc : Eq (HMul.hMul a (p i)) (HMul.hMul b c)
      hx : Eq (HMul.hMul (HMul.hMul b (t.prod fun i => p i)) (HMul.hMul c (u.prod fu …
      hit : Not (Membership.mem t i)
      ⊢ Exists fun t_1 => Exists fun u_1 => Exists fun b_1 => Exists fun c_1 => And  …
    -/
    have hiu : i ∉ u := fun hiu ↦ his (htus ▸ mem_union_right _ hiu)
    /-
      case insert.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝¹ : CancelCommMonoidWithZero R
      α : Type u_2
      inst✝ : DecidableEq α
      p : α → R
      i : α
      s : Finset α
      his : Not (Membership.mem s i)
      ih : ∀ {x y a : R}, (∀ (i : α), Membership.mem s i → Prime (p i)) → Eq (HMul.h …
      a : R
      hp : ∀ (i_1 : α), Membership.mem (Insert.insert i s) i_1 → Prime (p i_1)
      hpi : Prime (p i)
      t u : Finset α
      b c : R
      htus : Eq (Union.union t u) s
      htu : Disjoint t u
      hbc : Eq (HMul.hMul a (p i)) (HMul.hMul b c)
      hx : Eq (HMul.hMul (HMul.hMul b (t.prod fun i => p i)) (HMul.hMul c (u.prod fu …
      hit : Not (Membership.mem t i)
      hiu : Not (Membership.mem u i)
      ⊢ Exists fun t_1 => Exists fun u_1 => Exists fun b_1 => Exists fun c_1 => And  …
    -/
    obtain ⟨d, rfl⟩ | ⟨d, rfl⟩ : p i ∣ b ∨ p i ∣ c := hpi.dvd_or_dvd ⟨a, by rw [← hbc, mul_comm]⟩
      /-
        case insert.intro.intro.intro.intro.intro.intro.intro.intro.inl.intro
        R : Type u_1
        inst✝¹ : CancelCommMonoidWithZero R
        α : Type u_2
        inst✝ : DecidableEq α
        p : α → R
        i : α
        s : Finset α
        his : Not (Membership.mem s i)
        ih : ∀ {x y a : R}, (∀ (i : α), Membership.mem s i → Prime (p i)) → Eq (HMul.h …
        a : R
        hp : ∀ (i_1 : α), Membership.mem (Insert.insert i s) i_1 → Prime (p i_1)
        hpi : Prime (p i)
        t u : Finset α
        c : R
        htus : Eq (Union.union t u) s
        htu : Disjoint t u
        hit : Not (Membership.mem t i)
        hiu : Not (Membership.mem u i)
        d : R
        hbc : Eq (HMul.hMul a (p i)) (HMul.hMul (HMul.hMul (p i) d) c)
        hx : Eq (HMul.hMul (HMul.hMul (HMul.hMul (p i) d) (t.prod fun i => p i)) (HMul …
        ⊢ Exists fun t_1 => Exists fun u_1 => Exists fun b => Exists fun c_1 => And (E …
      -/
    · rw [mul_assoc, mul_comm a, mul_right_inj' hpi.ne_zero] at hbc
      exact ⟨insert i t, u, d, c, by rw [insert_union, htus], disjoint_insert_left.2 ⟨hiu, htu⟩, by
          simp [hbc, prod_insert hit, mul_assoc, mul_comm, mul_left_comm]⟩
      /-
        case insert.intro.intro.intro.intro.intro.intro.intro.intro.inr.intro
        R : Type u_1
        inst✝¹ : CancelCommMonoidWithZero R
        α : Type u_2
        inst✝ : DecidableEq α
        p : α → R
        i : α
        s : Finset α
        his : Not (Membership.mem s i)
        ih : ∀ {x y a : R}, (∀ (i : α), Membership.mem s i → Prime (p i)) → Eq (HMul.h …
        a : R
        hp : ∀ (i_1 : α), Membership.mem (Insert.insert i s) i_1 → Prime (p i_1)
        hpi : Prime (p i)
        t u : Finset α
        b : R
        htus : Eq (Union.union t u) s
        htu : Disjoint t u
        hit : Not (Membership.mem t i)
        hiu : Not (Membership.mem u i)
        d : R
        hbc : Eq (HMul.hMul a (p i)) (HMul.hMul b (HMul.hMul (p i) d))
        hx : Eq (HMul.hMul (HMul.hMul b (t.prod fun i => p i)) (HMul.hMul (HMul.hMul ( …
        ⊢ Exists fun t_1 => Exists fun u_1 => Exists fun b_1 => Exists fun c => And (E …
      -/
    · rw [← mul_assoc, mul_right_comm b, mul_left_inj' hpi.ne_zero] at hbc
      exact ⟨t, insert i u, b, d, by rw [union_insert, htus], disjoint_insert_right.2 ⟨hit, htu⟩, by
          simp [← hbc, prod_insert hiu, mul_assoc, mul_comm, mul_left_comm]⟩


/-- If `x * y = a * p ^ n` where `p` is prime, then `x` and `y` can both be written
  as the product of a power of `p` and a divisor of `a`. -/
theorem mul_eq_mul_prime_pow {x y a p : R} {n : ℕ} (hp : Prime p) (hx : x * y = a * p ^ n) :
    ∃ (i j : ℕ) (b c : R), i + j = n ∧ a = b * c ∧ x = b * p ^ i ∧ y = c * p ^ j := by
  rcases mul_eq_mul_prime_prod (fun _ _ ↦ hp)
    (show x * y = a * (range n).prod fun _ ↦ p by simpa) with
      ⟨t, u, b, c, htus, htu, rfl, rfl, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    inst✝ : CancelCommMonoidWithZero R
    p : R
    n : Nat
    hp : Prime p
    t u : Finset Nat
    b c : R
    htus : Eq (Union.union t u) (Finset.range n)
    htu : Disjoint t u
    hx : Eq (HMul.hMul (HMul.hMul b (t.prod fun i => p)) (HMul.hMul c (u.prod fun  …
    ⊢ Exists fun i => Exists fun j => Exists fun b_1 => Exists fun c_1 => And (Eq  …
  -/
  exact ⟨#t, #u, b, c, by rw [← card_union_of_disjoint htu, htus, card_range], by simp⟩
  /-
    🎉 no goals
  -/


theorem Prime.neg {p : α} (hp : Prime p) : Prime (-p) := by
  /-
    α : Type u_1
    inst✝ : CommRing α
    p : α
    hp : Prime p
    ⊢ Prime (Neg.neg p)
  -/
  obtain ⟨h1, h2, h3⟩ := hp
  /-
    case intro.intro
    α : Type u_1
    inst✝ : CommRing α
    p : α
    h1 : Ne p 0
    h2 : Not (IsUnit p)
    h3 : ∀ (a b : α), Dvd.dvd p (HMul.hMul a b) → Or (Dvd.dvd p a) (Dvd.dvd p b)
    ⊢ Prime (Neg.neg p)
  -/
  exact ⟨neg_ne_zero.mpr h1, by rwa [IsUnit.neg_iff], by simpa [neg_dvd] using h3⟩
  /-
    🎉 no goals
  -/


theorem Prime.abs [LinearOrder α] {p : α} (hp : Prime p) : Prime (abs p) := by
  /-
    α : Type u_1
    inst✝¹ : CommRing α
    inst✝ : LinearOrder α
    p : α
    hp : Prime p
    ⊢ Prime (_root_.abs p)
  -/
  obtain h | h := abs_choice p <;> rw [h]
    /-
      case inl
      α : Type u_1
      inst✝¹ : CommRing α
      inst✝ : LinearOrder α
      p : α
      hp : Prime p
      h : Eq (_root_.abs p) p
      ⊢ Prime p
    -/
  · exact hp
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝¹ : CommRing α
      inst✝ : LinearOrder α
      p : α
      hp : Prime p
      h : Eq (_root_.abs p) (Neg.neg p)
      ⊢ Prime (Neg.neg p)
    -/
  · exact hp.neg
    /-
      🎉 no goals
    -/


