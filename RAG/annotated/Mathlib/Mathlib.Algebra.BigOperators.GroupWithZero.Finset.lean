lemma prod_eq_zero (hi : i ∈ s) (h : f i = 0) : ∏ j ∈ s, f j = 0 := by
  /-
    ι : Type u_1
    M₀ : Type u_4
    inst✝ : CommMonoidWithZero M₀
    f : ι → M₀
    s : Finset ι
    i : ι
    hi : Membership.mem s i
    h : Eq (f i) 0
    ⊢ Eq (s.prod fun j => f j) 0
  -/
  classical rw [← prod_erase_mul _ _ hi, h, mul_zero]
  /-
    🎉 no goals
  -/


lemma prod_ite_zero :
    (∏ i ∈ s, if p i then f i else 0) = if ∀ i ∈ s, p i then ∏ i ∈ s, f i else 0 := by
  /-
    ι : Type u_1
    M₀ : Type u_4
    inst✝¹ : CommMonoidWithZero M₀
    p : ι → Prop
    inst✝ : DecidablePred p
    f : ι → M₀
    s : Finset ι
    ⊢ Eq (s.prod fun i => ite (p i) (f i) 0) (ite (∀ (i : ι), Membership.mem s i → …
  -/
  split_ifs with h
    /-
      case pos
      ι : Type u_1
      M₀ : Type u_4
      inst✝¹ : CommMonoidWithZero M₀
      p : ι → Prop
      inst✝ : DecidablePred p
      f : ι → M₀
      s : Finset ι
      h : ∀ (i : ι), Membership.mem s i → p i
      ⊢ Eq (s.prod fun i => ite (p i) (f i) 0) (s.prod fun i => f i)
    -/
  · exact prod_congr rfl fun i hi => by simp [h i hi]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      M₀ : Type u_4
      inst✝¹ : CommMonoidWithZero M₀
      p : ι → Prop
      inst✝ : DecidablePred p
      f : ι → M₀
      s : Finset ι
      h : Not (∀ (i : ι), Membership.mem s i → p i)
      ⊢ Eq (s.prod fun i => ite (p i) (f i) 0) 0
    -/
  · push_neg at h
    /-
      case neg
      ι : Type u_1
      M₀ : Type u_4
      inst✝¹ : CommMonoidWithZero M₀
      p : ι → Prop
      inst✝ : DecidablePred p
      f : ι → M₀
      s : Finset ι
      h : Exists fun i => And (Membership.mem s i) (Not (p i))
      ⊢ Eq (s.prod fun i => ite (p i) (f i) 0) 0
    -/
    rcases h with ⟨i, hi, hq⟩
    /-
      case neg.intro.intro
      ι : Type u_1
      M₀ : Type u_4
      inst✝¹ : CommMonoidWithZero M₀
      p : ι → Prop
      inst✝ : DecidablePred p
      f : ι → M₀
      s : Finset ι
      i : ι
      hi : Membership.mem s i
      hq : Not (p i)
      ⊢ Eq (s.prod fun i => ite (p i) (f i) 0) 0
    -/
    exact prod_eq_zero hi (by simp [hq])
    /-
      🎉 no goals
    -/


lemma prod_boole : ∏ i ∈ s, (ite (p i) 1 0 : M₀) = ite (∀ i ∈ s, p i) 1 0 := by
  /-
    ι : Type u_1
    M₀ : Type u_4
    inst✝¹ : CommMonoidWithZero M₀
    p : ι → Prop
    inst✝ : DecidablePred p
    s : Finset ι
    ⊢ Eq (s.prod fun i => ite (p i) 1 0) (ite (∀ (i : ι), Membership.mem s i → p i …
  -/
  rw [prod_ite_zero, prod_const_one]
  /-
    🎉 no goals
  -/


lemma support_prod_subset (s : Finset ι) (f : ι → κ → M₀) :
    support (fun x ↦ ∏ i ∈ s, f i x) ⊆ ⋂ i ∈ s, support (f i) :=
  fun _ hx ↦ Set.mem_iInter₂.2 fun _ hi H ↦ hx <| prod_eq_zero hi H


lemma prod_eq_zero_iff : ∏ x ∈ s, f x = 0 ↔ ∃ a ∈ s, f a = 0 := by
  classical
    induction s using Finset.induction_on with
    | empty => exact ⟨Not.elim one_ne_zero, fun ⟨_, H, _⟩ => by simp at H⟩
    | insert ha ih => rw [prod_insert ha, mul_eq_zero, exists_mem_insert, ih]


lemma prod_ne_zero_iff : ∏ x ∈ s, f x ≠ 0 ↔ ∀ a ∈ s, f a ≠ 0 := by
  /-
    ι : Type u_1
    M₀ : Type u_4
    inst✝² : CommMonoidWithZero M₀
    f : ι → M₀
    s : Finset ι
    inst✝¹ : Nontrivial M₀
    inst✝ : NoZeroDivisors M₀
    ⊢ Iff (Ne (s.prod fun x => f x) 0) (∀ (a : ι), Membership.mem s a → Ne (f a) 0)
  -/
  rw [Ne, prod_eq_zero_iff]
  /-
    ι : Type u_1
    M₀ : Type u_4
    inst✝² : CommMonoidWithZero M₀
    f : ι → M₀
    s : Finset ι
    inst✝¹ : Nontrivial M₀
    inst✝ : NoZeroDivisors M₀
    ⊢ Iff (Not (Exists fun a => And (Membership.mem s a) (Eq (f a) 0))) (∀ (a : ι) …
  -/
  push_neg; rfl
            /-
              🎉 no goals
            -/


lemma support_prod (s : Finset ι) (f : ι → κ → M₀) :
    support (fun j ↦ ∏ i ∈ s, f i j) = ⋂ i ∈ s, support (f i) :=
                     /-
                       ι : Type u_1
                       κ : Type u_2
                       M₀ : Type u_4
                       inst✝² : CommMonoidWithZero M₀
                       inst✝¹ : Nontrivial M₀
                       inst✝ : NoZeroDivisors M₀
                       s : Finset ι
                       f : ι → κ → M₀
                       x : κ
                       ⊢ Iff (Membership.mem (Function.support fun j => s.prod fun i => f i j) x) (Me …
                     -/
  Set.ext fun x ↦ by simp [support, prod_eq_zero_iff]
                     /-
                       🎉 no goals
                     -/


lemma prod_ite_zero : (∏ i, if p i then f i else 0) = if ∀ i, p i then ∏ i, f i else 0 := by
  /-
    ι : Type u_1
    M₀ : Type u_4
    inst✝² : Fintype ι
    inst✝¹ : CommMonoidWithZero M₀
    p : ι → Prop
    inst✝ : DecidablePred p
    f : ι → M₀
    ⊢ Eq (Finset.univ.prod fun i => ite (p i) (f i) 0) (ite (∀ (i : ι), p i) (Fins …
  -/
  simp [Finset.prod_ite_zero]
  /-
    🎉 no goals
  -/


                                                                        /-
                                                                          ι : Type u_1
                                                                          M₀ : Type u_4
                                                                          inst✝² : Fintype ι
                                                                          inst✝¹ : CommMonoidWithZero M₀
                                                                          p : ι → Prop
                                                                          inst✝ : DecidablePred p
                                                                          ⊢ Eq (Finset.univ.prod fun i => ite (p i) 1 0) (ite (∀ (i : ι), p i) 1 0)
                                                                        -/
lemma prod_boole : ∏ i, (ite (p i) 1 0 : M₀) = ite (∀ i, p i) 1 0 := by simp [Finset.prod_boole]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


lemma Units.mk0_prod [CommGroupWithZero G₀] (s : Finset ι) (f : ι → G₀) (h) :
    Units.mk0 (∏ i ∈ s, f i) h =
      ∏ i ∈ s.attach, Units.mk0 (f i) fun hh ↦ h (Finset.prod_eq_zero i.2 hh) := by
  /-
    ι : Type u_1
    G₀ : Type u_3
    inst✝ : CommGroupWithZero G₀
    s : Finset ι
    f : ι → G₀
    h : Ne (s.prod fun i => f i) 0
    ⊢ Eq (Units.mk0 (s.prod fun i => f i) h) (s.attach.prod fun i => Units.mk0 (f  …
  -/
  classical induction s using Finset.induction_on <;> simp [*]
  /-
    🎉 no goals
  -/

