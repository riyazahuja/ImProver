/-- Prime `p` divides the product of a list `L` iff it divides some `a ∈ L` -/
theorem Prime.dvd_prod_iff {p : M} {L : List M} (pp : Prime p) : p ∣ L.prod ↔ ∃ a ∈ L, p ∣ a := by
  /-
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    p : M
    L : List M
    pp : Prime p
    ⊢ Iff (Dvd.dvd p L.prod) (Exists fun a => And (Membership.mem L a) (Dvd.dvd p  …
  -/
  constructor
    /-
      case mp
      M : Type u_1
      inst✝ : CommMonoidWithZero M
      p : M
      L : List M
      pp : Prime p
      ⊢ Dvd.dvd p L.prod → Exists fun a => And (Membership.mem L a) (Dvd.dvd p a)
    -/
  · intro h
    /-
      case mp
      M : Type u_1
      inst✝ : CommMonoidWithZero M
      p : M
      L : List M
      pp : Prime p
      h : Dvd.dvd p L.prod
      ⊢ Exists fun a => And (Membership.mem L a) (Dvd.dvd p a)
    -/
    induction' L with L_hd L_tl L_ih
      /-
        case mp.nil
        M : Type u_1
        inst✝ : CommMonoidWithZero M
        p : M
        pp : Prime p
        h : Dvd.dvd p List.nil.prod
        ⊢ Exists fun a => And (Membership.mem List.nil a) (Dvd.dvd p a)
      -/
    · rw [prod_nil] at h
      /-
        case mp.nil
        M : Type u_1
        inst✝ : CommMonoidWithZero M
        p : M
        pp : Prime p
        h : Dvd.dvd p 1
        ⊢ Exists fun a => And (Membership.mem List.nil a) (Dvd.dvd p a)
      -/
      exact absurd h pp.not_dvd_one
      /-
        🎉 no goals
      -/
      /-
        case mp.cons
        M : Type u_1
        inst✝ : CommMonoidWithZero M
        p : M
        pp : Prime p
        L_hd : M
        L_tl : List M
        L_ih : Dvd.dvd p L_tl.prod → Exists fun a => And (Membership.mem L_tl a) (Dvd. …
        h : Dvd.dvd p (List.cons L_hd L_tl).prod
        ⊢ Exists fun a => And (Membership.mem (List.cons L_hd L_tl) a) (Dvd.dvd p a)
      -/
    · rw [prod_cons] at h
      /-
        case mp.cons
        M : Type u_1
        inst✝ : CommMonoidWithZero M
        p : M
        pp : Prime p
        L_hd : M
        L_tl : List M
        L_ih : Dvd.dvd p L_tl.prod → Exists fun a => And (Membership.mem L_tl a) (Dvd. …
        h : Dvd.dvd p (HMul.hMul L_hd L_tl.prod)
        ⊢ Exists fun a => And (Membership.mem (List.cons L_hd L_tl) a) (Dvd.dvd p a)
      -/
      cases' pp.dvd_or_dvd h with hd hd
        /-
          case mp.cons.inl
          M : Type u_1
          inst✝ : CommMonoidWithZero M
          p : M
          pp : Prime p
          L_hd : M
          L_tl : List M
          L_ih : Dvd.dvd p L_tl.prod → Exists fun a => And (Membership.mem L_tl a) (Dvd. …
          h : Dvd.dvd p (HMul.hMul L_hd L_tl.prod)
          hd : Dvd.dvd p L_hd
          ⊢ Exists fun a => And (Membership.mem (List.cons L_hd L_tl) a) (Dvd.dvd p a)
        -/
      · exact ⟨L_hd, mem_cons_self L_hd L_tl, hd⟩
        /-
          🎉 no goals
        -/
        /-
          case mp.cons.inr
          M : Type u_1
          inst✝ : CommMonoidWithZero M
          p : M
          pp : Prime p
          L_hd : M
          L_tl : List M
          L_ih : Dvd.dvd p L_tl.prod → Exists fun a => And (Membership.mem L_tl a) (Dvd. …
          h : Dvd.dvd p (HMul.hMul L_hd L_tl.prod)
          hd : Dvd.dvd p L_tl.prod
          ⊢ Exists fun a => And (Membership.mem (List.cons L_hd L_tl) a) (Dvd.dvd p a)
        -/
      · obtain ⟨x, hx1, hx2⟩ := L_ih hd
        /-
          case mp.cons.inr.intro.intro
          M : Type u_1
          inst✝ : CommMonoidWithZero M
          p : M
          pp : Prime p
          L_hd : M
          L_tl : List M
          L_ih : Dvd.dvd p L_tl.prod → Exists fun a => And (Membership.mem L_tl a) (Dvd. …
          h : Dvd.dvd p (HMul.hMul L_hd L_tl.prod)
          hd : Dvd.dvd p L_tl.prod
          x : M
          hx1 : Membership.mem L_tl x
          hx2 : Dvd.dvd p x
          ⊢ Exists fun a => And (Membership.mem (List.cons L_hd L_tl) a) (Dvd.dvd p a)
        -/
        exact ⟨x, mem_cons_of_mem L_hd hx1, hx2⟩
        /-
          🎉 no goals
        -/
    /-
      case mpr
      M : Type u_1
      inst✝ : CommMonoidWithZero M
      p : M
      L : List M
      pp : Prime p
      ⊢ (Exists fun a => And (Membership.mem L a) (Dvd.dvd p a)) → Dvd.dvd p L.prod
    -/
  · exact fun ⟨a, ha1, ha2⟩ => dvd_trans ha2 (dvd_prod ha1)
    /-
      🎉 no goals
    -/


theorem Prime.not_dvd_prod {p : M} {L : List M} (pp : Prime p) (hL : ∀ a ∈ L, ¬p ∣ a) :
    ¬p ∣ L.prod :=
  mt (Prime.dvd_prod_iff pp).1 <| not_exists.2 fun a => not_and.2 (hL a)


theorem mem_list_primes_of_dvd_prod {p : M} (hp : Prime p) {L : List M} (hL : ∀ q ∈ L, Prime q)
    (hpL : p ∣ L.prod) : p ∈ L := by
  /-
    M : Type u_1
    inst✝¹ : CancelCommMonoidWithZero M
    inst✝ : Subsingleton (Units M)
    p : M
    hp : Prime p
    L : List M
    hL : ∀ (q : M), Membership.mem L q → Prime q
    hpL : Dvd.dvd p L.prod
    ⊢ Membership.mem L p
  -/
  obtain ⟨x, hx1, hx2⟩ := hp.dvd_prod_iff.mp hpL
  /-
    case intro.intro
    M : Type u_1
    inst✝¹ : CancelCommMonoidWithZero M
    inst✝ : Subsingleton (Units M)
    p : M
    hp : Prime p
    L : List M
    hL : ∀ (q : M), Membership.mem L q → Prime q
    hpL : Dvd.dvd p L.prod
    x : M
    hx1 : Membership.mem L x
    hx2 : Dvd.dvd p x
    ⊢ Membership.mem L p
  -/
  rwa [(prime_dvd_prime_iff_eq hp (hL x hx1)).mp hx2]
  /-
    🎉 no goals
  -/


theorem perm_of_prod_eq_prod :
    ∀ {l₁ l₂ : List M}, l₁.prod = l₂.prod → (∀ p ∈ l₁, Prime p) → (∀ p ∈ l₂, Prime p) → Perm l₁ l₂
  | [], [], _, _, _ => Perm.nil
  | [], a :: l, h₁, _, h₃ =>
    have ha : a ∣ 1 := prod_nil (M := M) ▸ h₁.symm ▸ (prod_cons (l := l)).symm ▸ dvd_mul_right _ _
    absurd ha (Prime.not_dvd_one (h₃ a (mem_cons_self _ _)))
  | a :: l, [], h₁, h₂, _ =>
    have ha : a ∣ 1 := prod_nil (M := M) ▸ h₁ ▸ (prod_cons (l := l)).symm ▸ dvd_mul_right _ _
    absurd ha (Prime.not_dvd_one (h₂ a (mem_cons_self _ _)))
  | a :: l₁, b :: l₂, h, hl₁, hl₂ => by
    classical
      have hl₁' : ∀ p ∈ l₁, Prime p := fun p hp => hl₁ p (mem_cons_of_mem _ hp)
      have hl₂' : ∀ p ∈ (b :: l₂).erase a, Prime p := fun p hp => hl₂ p (mem_of_mem_erase hp)
      have ha : a ∈ b :: l₂ :=
        mem_list_primes_of_dvd_prod (hl₁ a (mem_cons_self _ _)) hl₂
          (h ▸ by rw [prod_cons]; exact dvd_mul_right _ _)
      have hb : b :: l₂ ~ a :: (b :: l₂).erase a := perm_cons_erase ha
      have hl : prod l₁ = prod ((b :: l₂).erase a) :=
        (mul_right_inj' (hl₁ a (mem_cons_self _ _)).ne_zero).1 <| by
          rwa [← prod_cons, ← prod_cons, ← hb.prod_eq]
      exact Perm.trans ((perm_of_prod_eq_prod hl hl₁' hl₂').cons _) hb.symm


