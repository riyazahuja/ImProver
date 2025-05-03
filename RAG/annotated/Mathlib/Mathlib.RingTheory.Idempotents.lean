theorem isIdempotentElem_one_sub_one_sub_pow_pow
    (x : R) (n : ℕ) (hx : (x - x ^ 2) ^ n = 0) :
    IsIdempotentElem (1 - (1 - x ^ n) ^ n) := by
  /-
    R : Type u_1
    inst✝ : Ring R
    x : R
    n : Nat
    hx : Eq (HPow.hPow (HSub.hSub x (HPow.hPow x 2)) n) 0
    ⊢ IsIdempotentElem (HSub.hSub 1 (HPow.hPow (HSub.hSub 1 (HPow.hPow x n)) n))
  -/
  let P : Polynomial ℤ := 1 - (1 - .X ^ n) ^ n
  have : (.X - .X ^ 2) ^ n ∣ P - P ^ 2 := by
    have H₁ : .X ^ n ∣ P := by
      have := sub_dvd_pow_sub_pow 1 ((1 : Polynomial ℤ) - Polynomial.X ^ n) n
      rwa [sub_sub_cancel, one_pow] at this
    have H₂ : (1 - .X) ^ n ∣ 1 - P := by
      simp only [sub_sub_cancel, P]
      simpa using pow_dvd_pow_of_dvd (sub_dvd_pow_sub_pow (α := Polynomial ℤ) 1 Polynomial.X n) n
    have := mul_dvd_mul H₁ H₂
    simpa only [← mul_pow, mul_sub, mul_one, ← pow_two] using this
  /-
    R : Type u_1
    inst✝ : Ring R
    x : R
    n : Nat
    hx : Eq (HPow.hPow (HSub.hSub x (HPow.hPow x 2)) n) 0
    P : Polynomial Int := HSub.hSub 1 (HPow.hPow (HSub.hSub 1 (HPow.hPow Polynomia …
    this : Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (HPow.hPow Polynomial.X 2))  …
    ⊢ IsIdempotentElem (HSub.hSub 1 (HPow.hPow (HSub.hSub 1 (HPow.hPow x n)) n))
  -/
  have := map_dvd (Polynomial.aeval x) this
  /-
    R : Type u_1
    inst✝ : Ring R
    x : R
    n : Nat
    hx : Eq (HPow.hPow (HSub.hSub x (HPow.hPow x 2)) n) 0
    P : Polynomial Int := HSub.hSub 1 (HPow.hPow (HSub.hSub 1 (HPow.hPow Polynomia …
    this✝ : Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (HPow.hPow Polynomial.X 2)) …
    this : Dvd.dvd ((Polynomial.aeval x) (HPow.hPow (HSub.hSub Polynomial.X (HPow. …
    ⊢ IsIdempotentElem (HSub.hSub 1 (HPow.hPow (HSub.hSub 1 (HPow.hPow x n)) n))
  -/
  simp only [map_pow, map_sub, Polynomial.aeval_X, hx, map_one, zero_dvd_iff, P] at this
  /-
    R : Type u_1
    inst✝ : Ring R
    x : R
    n : Nat
    hx : Eq (HPow.hPow (HSub.hSub x (HPow.hPow x 2)) n) 0
    P : Polynomial Int := HSub.hSub 1 (HPow.hPow (HSub.hSub 1 (HPow.hPow Polynomia …
    this✝ : Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (HPow.hPow Polynomial.X 2)) …
    this : Eq (HSub.hSub (HSub.hSub 1 (HPow.hPow (HSub.hSub 1 (HPow.hPow x n)) n)) …
    ⊢ IsIdempotentElem (HSub.hSub 1 (HPow.hPow (HSub.hSub 1 (HPow.hPow x n)) n))
  -/
  rwa [sub_eq_zero, eq_comm, pow_two] at this
  /-
    🎉 no goals
  -/


theorem exists_isIdempotentElem_mul_eq_zero_of_ker_isNilpotent_aux
    (h : ∀ x ∈ RingHom.ker f, IsNilpotent x)
    (e₁ : S) (he : e₁ ∈ f.range) (he₁ : IsIdempotentElem e₁)
    (e₂ : R) (he₂ : IsIdempotentElem e₂) (he₁e₂ : e₁ * f e₂ = 0) :
    ∃ e' : R, IsIdempotentElem e' ∧ f e' = e₁ ∧ e' * e₂ = 0 := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    e₁ : S
    he : Membership.mem f.range e₁
    he₁ : IsIdempotentElem e₁
    e₂ : R
    he₂ : IsIdempotentElem e₂
    he₁e₂ : Eq (HMul.hMul e₁ (f e₂)) 0
    ⊢ Exists fun e' => And (IsIdempotentElem e') (And (Eq (f e') e₁) (Eq (HMul.hMu …
  -/
  obtain ⟨e₁, rfl⟩ := he
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    e₂ : R
    he₂ : IsIdempotentElem e₂
    e₁ : R
    he₁ : IsIdempotentElem (f e₁)
    he₁e₂ : Eq (HMul.hMul (f e₁) (f e₂)) 0
    ⊢ Exists fun e' => And (IsIdempotentElem e') (And (Eq (f e') (f e₁)) (Eq (HMul …
  -/
  cases subsingleton_or_nontrivial R
    /-
      case intro.inl
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      e₂ : R
      he₂ : IsIdempotentElem e₂
      e₁ : R
      he₁ : IsIdempotentElem (f e₁)
      he₁e₂ : Eq (HMul.hMul (f e₁) (f e₂)) 0
      h✝ : Subsingleton R
      ⊢ Exists fun e' => And (IsIdempotentElem e') (And (Eq (f e') (f e₁)) (Eq (HMul …
    -/
  · exact ⟨_, Subsingleton.elim _ _, rfl, Subsingleton.elim _ _⟩
    /-
      🎉 no goals
    -/
  /-
    case intro.inr
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    e₂ : R
    he₂ : IsIdempotentElem e₂
    e₁ : R
    he₁ : IsIdempotentElem (f e₁)
    he₁e₂ : Eq (HMul.hMul (f e₁) (f e₂)) 0
    h✝ : Nontrivial R
    ⊢ Exists fun e' => And (IsIdempotentElem e') (And (Eq (f e') (f e₁)) (Eq (HMul …
  -/
  let a := e₁ - e₁ * e₂
  /-
    case intro.inr
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    e₂ : R
    he₂ : IsIdempotentElem e₂
    e₁ : R
    he₁ : IsIdempotentElem (f e₁)
    he₁e₂ : Eq (HMul.hMul (f e₁) (f e₂)) 0
    h✝ : Nontrivial R
    a : R := HSub.hSub e₁ (HMul.hMul e₁ e₂)
    ⊢ Exists fun e' => And (IsIdempotentElem e') (And (Eq (f e') (f e₁)) (Eq (HMul …
  -/
  have ha : f a = f e₁ := by rw [map_sub, map_mul, he₁e₂, sub_zero]
  /-
    case intro.inr
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    e₂ : R
    he₂ : IsIdempotentElem e₂
    e₁ : R
    he₁ : IsIdempotentElem (f e₁)
    he₁e₂ : Eq (HMul.hMul (f e₁) (f e₂)) 0
    h✝ : Nontrivial R
    a : R := HSub.hSub e₁ (HMul.hMul e₁ e₂)
    ha : Eq (f a) (f e₁)
    ⊢ Exists fun e' => And (IsIdempotentElem e') (And (Eq (f e') (f e₁)) (Eq (HMul …
  -/
  have ha' : a * e₂ = 0 := by rw [sub_mul, mul_assoc, he₂.eq, sub_self]
  have hx' : a - a ^ 2 ∈ RingHom.ker f := by
    simp [RingHom.mem_ker, mul_sub, pow_two, ha, he₁.eq]
  /-
    case intro.inr
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    e₂ : R
    he₂ : IsIdempotentElem e₂
    e₁ : R
    he₁ : IsIdempotentElem (f e₁)
    he₁e₂ : Eq (HMul.hMul (f e₁) (f e₂)) 0
    h✝ : Nontrivial R
    a : R := HSub.hSub e₁ (HMul.hMul e₁ e₂)
    ha : Eq (f a) (f e₁)
    ha' : Eq (HMul.hMul a e₂) 0
    hx' : Membership.mem (RingHom.ker f) (HSub.hSub a (HPow.hPow a 2))
    ⊢ Exists fun e' => And (IsIdempotentElem e') (And (Eq (f e') (f e₁)) (Eq (HMul …
  -/
  obtain ⟨n, hn⟩ := h _ hx'
  /-
    case intro.inr.intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    e₂ : R
    he₂ : IsIdempotentElem e₂
    e₁ : R
    he₁ : IsIdempotentElem (f e₁)
    he₁e₂ : Eq (HMul.hMul (f e₁) (f e₂)) 0
    h✝ : Nontrivial R
    a : R := HSub.hSub e₁ (HMul.hMul e₁ e₂)
    ha : Eq (f a) (f e₁)
    ha' : Eq (HMul.hMul a e₂) 0
    hx' : Membership.mem (RingHom.ker f) (HSub.hSub a (HPow.hPow a 2))
    n : Nat
    hn : Eq (HPow.hPow (HSub.hSub a (HPow.hPow a 2)) n) 0
    ⊢ Exists fun e' => And (IsIdempotentElem e') (And (Eq (f e') (f e₁)) (Eq (HMul …
  -/
  refine ⟨_, isIdempotentElem_one_sub_one_sub_pow_pow _ _ hn, ?_, ?_⟩
    /-
      case intro.inr.intro.refine_1
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      e₂ : R
      he₂ : IsIdempotentElem e₂
      e₁ : R
      he₁ : IsIdempotentElem (f e₁)
      he₁e₂ : Eq (HMul.hMul (f e₁) (f e₂)) 0
      h✝ : Nontrivial R
      a : R := HSub.hSub e₁ (HMul.hMul e₁ e₂)
      ha : Eq (f a) (f e₁)
      ha' : Eq (HMul.hMul a e₂) 0
      hx' : Membership.mem (RingHom.ker f) (HSub.hSub a (HPow.hPow a 2))
      n : Nat
      hn : Eq (HPow.hPow (HSub.hSub a (HPow.hPow a 2)) n) 0
      ⊢ Eq (f (HSub.hSub 1 (HPow.hPow (HSub.hSub 1 (HPow.hPow a n)) n))) (f e₁)
    -/
  · cases' n with n
      /-
        case intro.inr.intro.refine_1.zero
        R : Type u_1
        S : Type u_2
        inst✝¹ : Ring R
        inst✝ : Ring S
        f : RingHom R S
        h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
        e₂ : R
        he₂ : IsIdempotentElem e₂
        e₁ : R
        he₁ : IsIdempotentElem (f e₁)
        he₁e₂ : Eq (HMul.hMul (f e₁) (f e₂)) 0
        h✝ : Nontrivial R
        a : R := HSub.hSub e₁ (HMul.hMul e₁ e₂)
        ha : Eq (f a) (f e₁)
        ha' : Eq (HMul.hMul a e₂) 0
        hx' : Membership.mem (RingHom.ker f) (HSub.hSub a (HPow.hPow a 2))
        hn : Eq (HPow.hPow (HSub.hSub a (HPow.hPow a 2)) 0) 0
        ⊢ Eq (f (HSub.hSub 1 (HPow.hPow (HSub.hSub 1 (HPow.hPow a 0)) 0))) (f e₁)
      -/
    · simp at hn
      /-
        🎉 no goals
      -/
    simp only [map_sub, map_one, map_pow, ha, he₁.pow_succ_eq,
      he₁.one_sub.pow_succ_eq, sub_sub_cancel]
    /-
      case intro.inr.intro.refine_2
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      e₂ : R
      he₂ : IsIdempotentElem e₂
      e₁ : R
      he₁ : IsIdempotentElem (f e₁)
      he₁e₂ : Eq (HMul.hMul (f e₁) (f e₂)) 0
      h✝ : Nontrivial R
      a : R := HSub.hSub e₁ (HMul.hMul e₁ e₂)
      ha : Eq (f a) (f e₁)
      ha' : Eq (HMul.hMul a e₂) 0
      hx' : Membership.mem (RingHom.ker f) (HSub.hSub a (HPow.hPow a 2))
      n : Nat
      hn : Eq (HPow.hPow (HSub.hSub a (HPow.hPow a 2)) n) 0
      ⊢ Eq (HMul.hMul (HSub.hSub 1 (HPow.hPow (HSub.hSub 1 (HPow.hPow a n)) n)) e₂) 0
    -/
  · obtain ⟨k, hk⟩ := (Commute.one_left (MulOpposite.op <| 1 - a ^ n)).sub_dvd_pow_sub_pow n
    /-
      case intro.inr.intro.refine_2.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      e₂ : R
      he₂ : IsIdempotentElem e₂
      e₁ : R
      he₁ : IsIdempotentElem (f e₁)
      he₁e₂ : Eq (HMul.hMul (f e₁) (f e₂)) 0
      h✝ : Nontrivial R
      a : R := HSub.hSub e₁ (HMul.hMul e₁ e₂)
      ha : Eq (f a) (f e₁)
      ha' : Eq (HMul.hMul a e₂) 0
      hx' : Membership.mem (RingHom.ker f) (HSub.hSub a (HPow.hPow a 2))
      n : Nat
      hn : Eq (HPow.hPow (HSub.hSub a (HPow.hPow a 2)) n) 0
      k : MulOpposite R
      hk : Eq (HSub.hSub (HPow.hPow 1 n) (HPow.hPow (MulOpposite.op (HSub.hSub 1 (HP …
      ⊢ Eq (HMul.hMul (HSub.hSub 1 (HPow.hPow (HSub.hSub 1 (HPow.hPow a n)) n)) e₂) 0
    -/
    apply_fun MulOpposite.unop at hk
    /-
      case intro.inr.intro.refine_2.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      e₂ : R
      he₂ : IsIdempotentElem e₂
      e₁ : R
      he₁ : IsIdempotentElem (f e₁)
      he₁e₂ : Eq (HMul.hMul (f e₁) (f e₂)) 0
      h✝ : Nontrivial R
      a : R := HSub.hSub e₁ (HMul.hMul e₁ e₂)
      ha : Eq (f a) (f e₁)
      ha' : Eq (HMul.hMul a e₂) 0
      hx' : Membership.mem (RingHom.ker f) (HSub.hSub a (HPow.hPow a 2))
      n : Nat
      hn : Eq (HPow.hPow (HSub.hSub a (HPow.hPow a 2)) n) 0
      k : MulOpposite R
      hk : Eq (MulOpposite.unop (HSub.hSub (HPow.hPow 1 n) (HPow.hPow (MulOpposite.o …
      ⊢ Eq (HMul.hMul (HSub.hSub 1 (HPow.hPow (HSub.hSub 1 (HPow.hPow a n)) n)) e₂) 0
    -/
    have : 1 - (1 - a ^ n) ^ n = MulOpposite.unop k * a ^ n := by simpa using hk
    /-
      case intro.inr.intro.refine_2.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      e₂ : R
      he₂ : IsIdempotentElem e₂
      e₁ : R
      he₁ : IsIdempotentElem (f e₁)
      he₁e₂ : Eq (HMul.hMul (f e₁) (f e₂)) 0
      h✝ : Nontrivial R
      a : R := HSub.hSub e₁ (HMul.hMul e₁ e₂)
      ha : Eq (f a) (f e₁)
      ha' : Eq (HMul.hMul a e₂) 0
      hx' : Membership.mem (RingHom.ker f) (HSub.hSub a (HPow.hPow a 2))
      n : Nat
      hn : Eq (HPow.hPow (HSub.hSub a (HPow.hPow a 2)) n) 0
      k : MulOpposite R
      hk : Eq (MulOpposite.unop (HSub.hSub (HPow.hPow 1 n) (HPow.hPow (MulOpposite.o …
      this : Eq (HSub.hSub 1 (HPow.hPow (HSub.hSub 1 (HPow.hPow a n)) n)) (HMul.hMul …
      ⊢ Eq (HMul.hMul (HSub.hSub 1 (HPow.hPow (HSub.hSub 1 (HPow.hPow a n)) n)) e₂) 0
    -/
    rw [this, mul_assoc]
    /-
      case intro.inr.intro.refine_2.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      e₂ : R
      he₂ : IsIdempotentElem e₂
      e₁ : R
      he₁ : IsIdempotentElem (f e₁)
      he₁e₂ : Eq (HMul.hMul (f e₁) (f e₂)) 0
      h✝ : Nontrivial R
      a : R := HSub.hSub e₁ (HMul.hMul e₁ e₂)
      ha : Eq (f a) (f e₁)
      ha' : Eq (HMul.hMul a e₂) 0
      hx' : Membership.mem (RingHom.ker f) (HSub.hSub a (HPow.hPow a 2))
      n : Nat
      hn : Eq (HPow.hPow (HSub.hSub a (HPow.hPow a 2)) n) 0
      k : MulOpposite R
      hk : Eq (MulOpposite.unop (HSub.hSub (HPow.hPow 1 n) (HPow.hPow (MulOpposite.o …
      this : Eq (HSub.hSub 1 (HPow.hPow (HSub.hSub 1 (HPow.hPow a n)) n)) (HMul.hMul …
      ⊢ Eq (HMul.hMul (MulOpposite.unop k) (HMul.hMul (HPow.hPow a n) e₂)) 0
    -/
    cases' n with n
      /-
        case intro.inr.intro.refine_2.intro.zero
        R : Type u_1
        S : Type u_2
        inst✝¹ : Ring R
        inst✝ : Ring S
        f : RingHom R S
        h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
        e₂ : R
        he₂ : IsIdempotentElem e₂
        e₁ : R
        he₁ : IsIdempotentElem (f e₁)
        he₁e₂ : Eq (HMul.hMul (f e₁) (f e₂)) 0
        h✝ : Nontrivial R
        a : R := HSub.hSub e₁ (HMul.hMul e₁ e₂)
        ha : Eq (f a) (f e₁)
        ha' : Eq (HMul.hMul a e₂) 0
        hx' : Membership.mem (RingHom.ker f) (HSub.hSub a (HPow.hPow a 2))
        k : MulOpposite R
        hn : Eq (HPow.hPow (HSub.hSub a (HPow.hPow a 2)) 0) 0
        hk : Eq (MulOpposite.unop (HSub.hSub (HPow.hPow 1 0) (HPow.hPow (MulOpposite.o …
        this : Eq (HSub.hSub 1 (HPow.hPow (HSub.hSub 1 (HPow.hPow a 0)) 0)) (HMul.hMul …
        ⊢ Eq (HMul.hMul (MulOpposite.unop k) (HMul.hMul (HPow.hPow a 0) e₂)) 0
      -/
    · simp at hn
      /-
        🎉 no goals
      -/
    /-
      case intro.inr.intro.refine_2.intro.succ
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      e₂ : R
      he₂ : IsIdempotentElem e₂
      e₁ : R
      he₁ : IsIdempotentElem (f e₁)
      he₁e₂ : Eq (HMul.hMul (f e₁) (f e₂)) 0
      h✝ : Nontrivial R
      a : R := HSub.hSub e₁ (HMul.hMul e₁ e₂)
      ha : Eq (f a) (f e₁)
      ha' : Eq (HMul.hMul a e₂) 0
      hx' : Membership.mem (RingHom.ker f) (HSub.hSub a (HPow.hPow a 2))
      k : MulOpposite R
      n : Nat
      hn : Eq (HPow.hPow (HSub.hSub a (HPow.hPow a 2)) (HAdd.hAdd n 1)) 0
      hk : Eq (MulOpposite.unop (HSub.hSub (HPow.hPow 1 (HAdd.hAdd n 1)) (HPow.hPow  …
      this : Eq (HSub.hSub 1 (HPow.hPow (HSub.hSub 1 (HPow.hPow a (HAdd.hAdd n 1)))  …
      ⊢ Eq (HMul.hMul (MulOpposite.unop k) (HMul.hMul (HPow.hPow a (HAdd.hAdd n 1))  …
    -/
    rw [pow_succ, mul_assoc, ha', mul_zero, mul_zero]
    /-
      🎉 no goals
    -/


/-- Orthogonal idempotents lift along nil ideals. -/
theorem exists_isIdempotentElem_mul_eq_zero_of_ker_isNilpotent
    (h : ∀ x ∈ RingHom.ker f, IsNilpotent x)
    (e₁ : S) (he : e₁ ∈ f.range) (he₁ : IsIdempotentElem e₁)
    (e₂ : R) (he₂ : IsIdempotentElem e₂) (he₁e₂ : e₁ * f e₂ = 0) (he₂e₁ : f e₂ * e₁ = 0) :
    ∃ e' : R, IsIdempotentElem e' ∧ f e' = e₁ ∧ e' * e₂ = 0 ∧ e₂ * e' = 0 := by
  obtain ⟨e', h₁, rfl, h₂⟩ := exists_isIdempotentElem_mul_eq_zero_of_ker_isNilpotent_aux
    f h e₁ he he₁ e₂ he₂ he₁e₂
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    e₂ : R
    he₂ : IsIdempotentElem e₂
    e' : R
    h₁ : IsIdempotentElem e'
    h₂ : Eq (HMul.hMul e' e₂) 0
    he : Membership.mem f.range (f e')
    he₁ : IsIdempotentElem (f e')
    he₁e₂ : Eq (HMul.hMul (f e') (f e₂)) 0
    he₂e₁ : Eq (HMul.hMul (f e₂) (f e')) 0
    ⊢ Exists fun e'_1 => And (IsIdempotentElem e'_1) (And (Eq (f e'_1) (f e')) (An …
  -/
  refine ⟨(1 - e₂) * e', ?_, ?_, ?_, ?_⟩
    /-
      case intro.intro.intro.refine_1
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      e₂ : R
      he₂ : IsIdempotentElem e₂
      e' : R
      h₁ : IsIdempotentElem e'
      h₂ : Eq (HMul.hMul e' e₂) 0
      he : Membership.mem f.range (f e')
      he₁ : IsIdempotentElem (f e')
      he₁e₂ : Eq (HMul.hMul (f e') (f e₂)) 0
      he₂e₁ : Eq (HMul.hMul (f e₂) (f e')) 0
      ⊢ IsIdempotentElem (HMul.hMul (HSub.hSub 1 e₂) e')
    -/
  · rw [IsIdempotentElem, mul_assoc, ← mul_assoc e', mul_sub, mul_one, h₂, sub_zero, h₁.eq]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      e₂ : R
      he₂ : IsIdempotentElem e₂
      e' : R
      h₁ : IsIdempotentElem e'
      h₂ : Eq (HMul.hMul e' e₂) 0
      he : Membership.mem f.range (f e')
      he₁ : IsIdempotentElem (f e')
      he₁e₂ : Eq (HMul.hMul (f e') (f e₂)) 0
      he₂e₁ : Eq (HMul.hMul (f e₂) (f e')) 0
      ⊢ Eq (f (HMul.hMul (HSub.hSub 1 e₂) e')) (f e')
    -/
  · rw [map_mul, map_sub, map_one, sub_mul, one_mul, he₂e₁, sub_zero]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_3
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      e₂ : R
      he₂ : IsIdempotentElem e₂
      e' : R
      h₁ : IsIdempotentElem e'
      h₂ : Eq (HMul.hMul e' e₂) 0
      he : Membership.mem f.range (f e')
      he₁ : IsIdempotentElem (f e')
      he₁e₂ : Eq (HMul.hMul (f e') (f e₂)) 0
      he₂e₁ : Eq (HMul.hMul (f e₂) (f e')) 0
      ⊢ Eq (HMul.hMul (HMul.hMul (HSub.hSub 1 e₂) e') e₂) 0
    -/
  · rw [mul_assoc, h₂, mul_zero]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_4
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      e₂ : R
      he₂ : IsIdempotentElem e₂
      e' : R
      h₁ : IsIdempotentElem e'
      h₂ : Eq (HMul.hMul e' e₂) 0
      he : Membership.mem f.range (f e')
      he₁ : IsIdempotentElem (f e')
      he₁e₂ : Eq (HMul.hMul (f e') (f e₂)) 0
      he₂e₁ : Eq (HMul.hMul (f e₂) (f e')) 0
      ⊢ Eq (HMul.hMul e₂ (HMul.hMul (HSub.hSub 1 e₂) e')) 0
    -/
  · rw [← mul_assoc, mul_sub, mul_one, he₂.eq, sub_self, zero_mul]
    /-
      🎉 no goals
    -/


/-- Idempotents lift along nil ideals. -/
theorem exists_isIdempotentElem_eq_of_ker_isNilpotent (h : ∀ x ∈ RingHom.ker f, IsNilpotent x)
    (e : S) (he : e ∈ f.range) (he' : IsIdempotentElem e) :
    ∃ e' : R, IsIdempotentElem e' ∧ f e' = e := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    e : S
    he : Membership.mem f.range e
    he' : IsIdempotentElem e
    ⊢ Exists fun e' => And (IsIdempotentElem e') (Eq (f e') e)
  -/
  simpa using exists_isIdempotentElem_mul_eq_zero_of_ker_isNilpotent f h e he he' 0 .zero (by simp)
  /-
    🎉 no goals
  -/


/-- A family `{ eᵢ }` of idempotent elements is orthogonal if `eᵢ * eⱼ = 0` for all `i ≠ j`. -/
@[mk_iff]
structure OrthogonalIdempotents : Prop where
  idem : ∀ i, IsIdempotentElem (e i)
  ortho : Pairwise (e · * e · = 0)


lemma OrthogonalIdempotents.mul_eq [DecidableEq I] (he : OrthogonalIdempotents e) (i j) :
    e i * e j = if i = j then e i else 0 := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    I : Type u_3
    e : I → R
    inst✝ : DecidableEq I
    he : OrthogonalIdempotents e
    i j : I
    ⊢ Eq (HMul.hMul (e i) (e j)) (ite (Eq i j) (e i) 0)
  -/
  split
    /-
      case isTrue
      R : Type u_1
      inst✝¹ : Ring R
      I : Type u_3
      e : I → R
      inst✝ : DecidableEq I
      he : OrthogonalIdempotents e
      i j : I
      h✝ : Eq i j
      ⊢ Eq (HMul.hMul (e i) (e j)) (e i)
    -/
  · simp [*, (he.idem j).eq]
    /-
      🎉 no goals
    -/
    /-
      case isFalse
      R : Type u_1
      inst✝¹ : Ring R
      I : Type u_3
      e : I → R
      inst✝ : DecidableEq I
      he : OrthogonalIdempotents e
      i j : I
      h✝ : Not (Eq i j)
      ⊢ Eq (HMul.hMul (e i) (e j)) 0
    -/
  · exact he.ortho ‹_›
    /-
      🎉 no goals
    -/


lemma OrthogonalIdempotents.iff_mul_eq [DecidableEq I] :
    OrthogonalIdempotents e ↔ ∀ i j, e i * e j = if i = j then e i else 0 :=
                               /-
                                 R : Type u_1
                                 inst✝¹ : Ring R
                                 I : Type u_3
                                 e : I → R
                                 inst✝ : DecidableEq I
                                 H : ∀ (i j : I), Eq (HMul.hMul (e i) (e j)) (ite (Eq i j) (e i) 0)
                                 i : I
                                 ⊢ IsIdempotentElem (e i)
                               -/
                               /-
                                 🎉 no goals
                               -/
  ⟨mul_eq, fun H ↦ ⟨fun i ↦ by simpa using H i i, fun i j e ↦ by simpa [e] using H i j⟩⟩
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


lemma OrthogonalIdempotents.isIdempotentElem_sum (he : OrthogonalIdempotents e) {s : Finset I} :
    IsIdempotentElem (∑ i ∈ s, e i) := by
  classical
  simp [IsIdempotentElem, Finset.sum_mul, Finset.mul_sum, he.mul_eq]


lemma OrthogonalIdempotents.mul_sum_of_mem (he : OrthogonalIdempotents e)
    {i : I} {s : Finset I} (h : i ∈ s) : e i * ∑ j ∈ s, e j = e i := by
  classical
  simp [Finset.mul_sum, he.mul_eq, h]


lemma OrthogonalIdempotents.mul_sum_of_not_mem (he : OrthogonalIdempotents e)
    {i : I} {s : Finset I} (h : i ∉ s) : e i * ∑ j ∈ s, e j = 0 := by
  classical
  simp [Finset.mul_sum, he.mul_eq, h]


lemma OrthogonalIdempotents.map (he : OrthogonalIdempotents e) :
    OrthogonalIdempotents (f ∘ e) := by
  classical
  simp [iff_mul_eq, he.mul_eq, ← map_mul f, apply_ite f]


lemma OrthogonalIdempotents.map_injective_iff (hf : Function.Injective f) :
    OrthogonalIdempotents (f ∘ e) ↔ OrthogonalIdempotents e := by
  classical
  simp [iff_mul_eq, ← hf.eq_iff, apply_ite]


lemma OrthogonalIdempotents.embedding (he : OrthogonalIdempotents e) {J} (i : J ↪ I) :
    OrthogonalIdempotents (e ∘ i) := by
  classical
  simp [iff_mul_eq, he.mul_eq]


lemma OrthogonalIdempotents.equiv {J} (i : J ≃ I) :
    OrthogonalIdempotents (e ∘ i) ↔ OrthogonalIdempotents e := by
  classical
  simp [iff_mul_eq, i.forall_congr_left]


lemma OrthogonalIdempotents.unique [Unique I] :
    OrthogonalIdempotents e ↔ IsIdempotentElem (e default) := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    I : Type u_3
    e : I → R
    inst✝ : Unique I
    ⊢ Iff (OrthogonalIdempotents e) (IsIdempotentElem (e Inhabited.default))
  -/
  simp only [orthogonalIdempotents_iff, Unique.forall_iff, Subsingleton.pairwise, and_true]
  /-
    🎉 no goals
  -/


lemma OrthogonalIdempotents.option (he : OrthogonalIdempotents e) [Fintype I] (x)
    (hx : IsIdempotentElem x) (hx₁ : x * ∑ i, e i = 0) (hx₂ : (∑ i, e i) * x = 0) :
    OrthogonalIdempotents (Option.elim · x e) where
  idem i := i.rec hx he.idem
  ortho i j ne := by
    classical
    cases' i with i <;> cases' j with j
    · cases ne rfl
    · simpa only [mul_assoc, Finset.sum_mul, he.mul_eq, Finset.sum_ite_eq', Finset.mem_univ,
        ↓reduceIte, zero_mul] using congr_arg (· * e j) hx₁
    · simpa only [Option.elim_some, Option.elim_none, ← mul_assoc, Finset.mul_sum, he.mul_eq,
        Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte, mul_zero] using congr_arg (e i * ·) hx₂
    · exact he.ortho (Option.some_inj.ne.mp ne)


lemma OrthogonalIdempotents.lift_of_isNilpotent_ker_aux
    (h : ∀ x ∈ RingHom.ker f, IsNilpotent x)
    {n} {e : Fin n → S} (he : OrthogonalIdempotents e) (he' : ∀ i, e i ∈ f.range) :
    ∃ e' : Fin n → R, OrthogonalIdempotents e' ∧ f ∘ e' = e := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    n : Nat
    e : Fin n → S
    he : OrthogonalIdempotents e
    he' : ∀ (i : Fin n), Membership.mem f.range (e i)
    ⊢ Exists fun e' => And (OrthogonalIdempotents e') (Eq (Function.comp (⇑f) e') e)
  -/
  induction' n with n IH
    /-
      case zero
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      e : Fin 0 → S
      he : OrthogonalIdempotents e
      he' : ∀ (i : Fin 0), Membership.mem f.range (e i)
      ⊢ Exists fun e' => And (OrthogonalIdempotents e') (Eq (Function.comp (⇑f) e') e)
    -/
  · refine ⟨0, ⟨finZeroElim, finZeroElim⟩, funext finZeroElim⟩
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      n : Nat
      IH : ∀ {e : Fin n → S}, OrthogonalIdempotents e → (∀ (i : Fin n), Membership.m …
      e : Fin (HAdd.hAdd n 1) → S
      he : OrthogonalIdempotents e
      he' : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem f.range (e i)
      ⊢ Exists fun e' => And (OrthogonalIdempotents e') (Eq (Function.comp (⇑f) e') e)
    -/
  · obtain ⟨e', h₁, h₂⟩ := IH (he.embedding (Fin.succEmb n)) (fun i ↦ he' _)
    /-
      case succ.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      n : Nat
      IH : ∀ {e : Fin n → S}, OrthogonalIdempotents e → (∀ (i : Fin n), Membership.m …
      e : Fin (HAdd.hAdd n 1) → S
      he : OrthogonalIdempotents e
      he' : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem f.range (e i)
      e' : Fin n → R
      h₁ : OrthogonalIdempotents e'
      h₂ : Eq (Function.comp (⇑f) e') (Function.comp e ⇑(Fin.succEmb n))
      ⊢ Exists fun e' => And (OrthogonalIdempotents e') (Eq (Function.comp (⇑f) e') e)
    -/
    have h₂' (i) : f (e' i) = e i.succ := congr_fun h₂ i
    obtain ⟨e₀, h₃, h₄, h₅, h₆⟩ :=
      exists_isIdempotentElem_mul_eq_zero_of_ker_isNilpotent f h _ (he' 0) (he.idem 0) _
      h₁.isIdempotentElem_sum
      (by simp [Finset.mul_sum, h₂', he.mul_eq, Fin.succ_ne_zero, eq_comm])
      (by simp [Finset.sum_mul, h₂', he.mul_eq, Fin.succ_ne_zero])
    /-
      case succ.intro.intro.intro.intro.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      n : Nat
      IH : ∀ {e : Fin n → S}, OrthogonalIdempotents e → (∀ (i : Fin n), Membership.m …
      e : Fin (HAdd.hAdd n 1) → S
      he : OrthogonalIdempotents e
      he' : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem f.range (e i)
      e' : Fin n → R
      h₁ : OrthogonalIdempotents e'
      h₂ : Eq (Function.comp (⇑f) e') (Function.comp e ⇑(Fin.succEmb n))
      h₂' : ∀ (i : Fin n), Eq (f (e' i)) (e i.succ)
      e₀ : R
      h₃ : IsIdempotentElem e₀
      h₄ : Eq (f e₀) (e 0)
      h₅ : Eq (HMul.hMul e₀ (Finset.sum ?m.111702 fun i => e' i)) 0
      h₆ : Eq (HMul.hMul (Finset.sum ?m.111702 fun i => e' i) e₀) 0
      ⊢ Exists fun e' => And (OrthogonalIdempotents e') (Eq (Function.comp (⇑f) e') e)
    -/
    refine ⟨_, (h₁.option _ h₃ h₅ h₆).embedding (finSuccEquiv n).toEmbedding, funext fun i ↦ ?_⟩
    /-
      case succ.intro.intro.intro.intro.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      n : Nat
      IH : ∀ {e : Fin n → S}, OrthogonalIdempotents e → (∀ (i : Fin n), Membership.m …
      e : Fin (HAdd.hAdd n 1) → S
      he : OrthogonalIdempotents e
      he' : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem f.range (e i)
      e' : Fin n → R
      h₁ : OrthogonalIdempotents e'
      h₂ : Eq (Function.comp (⇑f) e') (Function.comp e ⇑(Fin.succEmb n))
      h₂' : ∀ (i : Fin n), Eq (f (e' i)) (e i.succ)
      e₀ : R
      h₃ : IsIdempotentElem e₀
      h₄ : Eq (f e₀) (e 0)
      h₅ : Eq (HMul.hMul e₀ (Finset.univ.sum fun i => e' i)) 0
      h₆ : Eq (HMul.hMul (Finset.univ.sum fun i => e' i) e₀) 0
      i : Fin (HAdd.hAdd n 1)
      ⊢ Eq (Function.comp (⇑f) (Function.comp (fun x => x.elim e₀ e') ⇑(finSuccEquiv …
    -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    obtain ⟨_ | i, rfl⟩ := (finSuccEquiv n).symm.surjective i <;> simp [*]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- A family of orthogonal idempotents lift along nil ideals. -/
lemma OrthogonalIdempotents.lift_of_isNilpotent_ker [Finite I]
    (h : ∀ x ∈ RingHom.ker f, IsNilpotent x)
    {e : I → S} (he : OrthogonalIdempotents e) (he' : ∀ i, e i ∈ f.range) :
    ∃ e' : I → R, OrthogonalIdempotents e' ∧ f ∘ e' = e := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : Ring R
    inst✝¹ : Ring S
    f : RingHom R S
    I : Type u_3
    inst✝ : Finite I
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    e : I → S
    he : OrthogonalIdempotents e
    he' : ∀ (i : I), Membership.mem f.range (e i)
    ⊢ Exists fun e' => And (OrthogonalIdempotents e') (Eq (Function.comp (⇑f) e') e)
  -/
  cases nonempty_fintype I
  obtain ⟨e', h₁, h₂⟩ := lift_of_isNilpotent_ker_aux f h
    (he.embedding (Fintype.equivFin I).symm.toEmbedding) (fun _ ↦ he' _)
  refine ⟨_, h₁.embedding (Fintype.equivFin I).toEmbedding,
    by ext x; simpa using congr_fun h₂ (Fintype.equivFin I x)⟩


/--
A family `{ eᵢ }` of idempotent elements is complete orthogonal if
1. (orthogonal) `eᵢ * eⱼ = 0` for all `i ≠ j`.
2. (complete) `∑ eᵢ = 1`
-/
@[mk_iff]
structure CompleteOrthogonalIdempotents (e : I → R) extends OrthogonalIdempotents e : Prop where
  complete : ∑ i, e i = 1


lemma CompleteOrthogonalIdempotents.unique_iff [Unique I] :
    CompleteOrthogonalIdempotents e ↔ e default = 1 := by
  rw [completeOrthogonalIdempotents_iff, OrthogonalIdempotents.unique, Fintype.sum_unique,
    and_iff_right_iff_imp]
  /-
    R : Type u_1
    inst✝² : Ring R
    I : Type u_3
    e : I → R
    inst✝¹ : Fintype I
    inst✝ : Unique I
    ⊢ Eq (e Inhabited.default) 1 → IsIdempotentElem (e Inhabited.default)
  -/
  exact (· ▸ IsIdempotentElem.one)
  /-
    🎉 no goals
  -/


lemma CompleteOrthogonalIdempotents.pair_iff {x y : R} :
    CompleteOrthogonalIdempotents ![x, y] ↔ IsIdempotentElem x ∧ y = 1 - x := by
  /-
    R : Type u_1
    inst✝ : Ring R
    x y : R
    ⊢ Iff (CompleteOrthogonalIdempotents (Matrix.vecCons x (Matrix.vecCons y Matri …
  -/
  rw [completeOrthogonalIdempotents_iff, orthogonalIdempotents_iff, and_assoc, Pairwise]
  simp only [Nat.succ_eq_add_one, Nat.reduceAdd, Fin.forall_fin_two, Fin.isValue,
    Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, ne_eq, not_true_eq_false,
    false_implies, zero_ne_one, not_false_eq_true, true_implies, true_and, one_ne_zero,
    and_true, and_self, Fin.sum_univ_two, eq_sub_iff_add_eq, @add_comm _ _ y]
  /-
    R : Type u_1
    inst✝ : Ring R
    x y : R
    ⊢ Iff (And (And (IsIdempotentElem x) (IsIdempotentElem y)) (And (And (Eq (HMul …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝ : Ring R
      x y : R
      ⊢ And (And (IsIdempotentElem x) (IsIdempotentElem y)) (And (And (Eq (HMul.hMul …
    -/
  · exact fun h ↦ ⟨h.1.1, h.2.2⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝ : Ring R
      x y : R
      ⊢ And (IsIdempotentElem x) (Eq (HAdd.hAdd x y) 1) → And (And (IsIdempotentElem …
    -/
  · rintro ⟨h₁, h₂⟩
    /-
      case mpr.intro
      R : Type u_1
      inst✝ : Ring R
      x y : R
      h₁ : IsIdempotentElem x
      h₂ : Eq (HAdd.hAdd x y) 1
      ⊢ And (And (IsIdempotentElem x) (IsIdempotentElem y)) (And (And (Eq (HMul.hMul …
    -/
    obtain rfl := eq_sub_iff_add_eq'.mpr h₂
    /-
      case mpr.intro
      R : Type u_1
      inst✝ : Ring R
      x : R
      h₁ : IsIdempotentElem x
      h₂ : Eq (HAdd.hAdd x (HSub.hSub 1 x)) 1
      ⊢ And (And (IsIdempotentElem x) (IsIdempotentElem (HSub.hSub 1 x))) (And (And  …
    -/
    exact ⟨⟨h₁, h₁.one_sub⟩, ⟨by simp [mul_sub, h₁.eq], by simp [sub_mul, h₁.eq]⟩, h₂⟩
    /-
      🎉 no goals
    -/


lemma CompleteOrthogonalIdempotents.of_isIdempotentElem {e : R} (he : IsIdempotentElem e) :
    CompleteOrthogonalIdempotents ![e, 1 - e] :=
  pair_iff.mpr ⟨he, rfl⟩


lemma CompleteOrthogonalIdempotents.single {I : Type*} [Fintype I] [DecidableEq I]
    (R : I → Type*) [∀ i, Ring (R i)] :
    CompleteOrthogonalIdempotents (Pi.single (f := R) · 1) := by
  /-
    I : Type u_4
    inst✝² : Fintype I
    inst✝¹ : DecidableEq I
    R : I → Type u_5
    inst✝ : (i : I) → Ring (R i)
    ⊢ CompleteOrthogonalIdempotents fun x => Pi.single x 1
  -/
  refine ⟨⟨by simp [IsIdempotentElem, ← Pi.single_mul], ?_⟩, Finset.univ_sum_single 1⟩
  /-
    I : Type u_4
    inst✝² : Fintype I
    inst✝¹ : DecidableEq I
    R : I → Type u_5
    inst✝ : (i : I) → Ring (R i)
    ⊢ Pairwise fun x1 x2 => Eq (HMul.hMul (Pi.single x1 1) (Pi.single x2 1)) 0
  -/
  intros i j hij
  /-
    I : Type u_4
    inst✝² : Fintype I
    inst✝¹ : DecidableEq I
    R : I → Type u_5
    inst✝ : (i : I) → Ring (R i)
    i j : I
    hij : Ne i j
    ⊢ Eq (HMul.hMul (Pi.single i 1) (Pi.single j 1)) 0
  -/
  ext k
  /-
    case h
    I : Type u_4
    inst✝² : Fintype I
    inst✝¹ : DecidableEq I
    R : I → Type u_5
    inst✝ : (i : I) → Ring (R i)
    i j : I
    hij : Ne i j
    k : I
    ⊢ Eq (HMul.hMul (Pi.single i 1) (Pi.single j 1) k) (0 k)
  -/
  by_cases hi : i = k
    /-
      case pos
      I : Type u_4
      inst✝² : Fintype I
      inst✝¹ : DecidableEq I
      R : I → Type u_5
      inst✝ : (i : I) → Ring (R i)
      i j : I
      hij : Ne i j
      k : I
      hi : Eq i k
      ⊢ Eq (HMul.hMul (Pi.single i 1) (Pi.single j 1) k) (0 k)
    -/
  · subst hi; simp [hij]
              /-
                🎉 no goals
              -/
    /-
      case neg
      I : Type u_4
      inst✝² : Fintype I
      inst✝¹ : DecidableEq I
      R : I → Type u_5
      inst✝ : (i : I) → Ring (R i)
      i j : I
      hij : Ne i j
      k : I
      hi : Not (Eq i k)
      ⊢ Eq (HMul.hMul (Pi.single i 1) (Pi.single j 1) k) (0 k)
    -/
  · simp [hi]
    /-
      🎉 no goals
    -/


lemma CompleteOrthogonalIdempotents.map (he : CompleteOrthogonalIdempotents e) :
    CompleteOrthogonalIdempotents (f ∘ e) where
  __ := he.toOrthogonalIdempotents.map f
                 /-
                   R : Type u_1
                   S : Type u_2
                   inst✝² : Ring R
                   inst✝¹ : Ring S
                   f : RingHom R S
                   I : Type u_3
                   e : I → R
                   inst✝ : Fintype I
                   he : CompleteOrthogonalIdempotents e
                   ⊢ Eq (Finset.univ.sum fun i => Function.comp (⇑f) e i) 1
                 -/
  complete := by simp only [Function.comp_apply, ← map_sum, he.complete, map_one]
                 /-
                   🎉 no goals
                 -/


lemma CompleteOrthogonalIdempotents.map_injective_iff (hf : Function.Injective f) :
    CompleteOrthogonalIdempotents (f ∘ e) ↔ CompleteOrthogonalIdempotents e := by
  simp [completeOrthogonalIdempotents_iff, ← hf.eq_iff, apply_ite,
    OrthogonalIdempotents.map_injective_iff f hf]


lemma CompleteOrthogonalIdempotents.equiv {J} [Fintype J] (i : J ≃ I) :
    CompleteOrthogonalIdempotents (e ∘ i) ↔ CompleteOrthogonalIdempotents e := by
  simp only [completeOrthogonalIdempotents_iff, OrthogonalIdempotents.equiv, Function.comp_apply,
    and_congr_right_iff, Fintype.sum_equiv i _ e (fun _ ↦ rfl)]


lemma CompleteOrthogonalIdempotents.option (he : OrthogonalIdempotents e) :
    CompleteOrthogonalIdempotents (Option.elim · (1 - ∑ i, e i) e) where
  __ := he.option _ he.isIdempotentElem_sum.one_sub
        /-
          R : Type u_1
          inst✝¹ : Ring R
          I : Type u_3
          e : I → R
          inst✝ : Fintype I
          he : OrthogonalIdempotents e
          ⊢ Eq (HMul.hMul (HSub.hSub 1 (Finset.univ.sum fun i => e i)) (Finset.univ.sum  …
        -/
        /-
          🎉 no goals
        -/
    (by simp [sub_mul, he.isIdempotentElem_sum.eq]) (by simp [mul_sub, he.isIdempotentElem_sum.eq])
                                                        /-
                                                          🎉 no goals
                                                        -/
  complete := by
    /-
      R : Type u_1
      inst✝¹ : Ring R
      I : Type u_3
      e : I → R
      inst✝ : Fintype I
      he : OrthogonalIdempotents e
      ⊢ Eq (Finset.univ.sum fun i => i.elim (HSub.hSub 1 (Finset.univ.sum fun i => e …
    -/
    rw [Fintype.sum_option]
    /-
      R : Type u_1
      inst✝¹ : Ring R
      I : Type u_3
      e : I → R
      inst✝ : Fintype I
      he : OrthogonalIdempotents e
      ⊢ Eq (HAdd.hAdd (Option.none.elim (HSub.hSub 1 (Finset.univ.sum fun i => e i)) …
    -/
    exact sub_add_cancel _ _
    /-
      🎉 no goals
    -/


@[nontriviality]
lemma CompleteOrthogonalIdempotents.of_subsingleton [Subsingleton R] :
    CompleteOrthogonalIdempotents e :=
  ⟨⟨fun _ ↦ Subsingleton.elim _ _, fun _ _ _ ↦ Subsingleton.elim _ _⟩, Subsingleton.elim _ _⟩


lemma CompleteOrthogonalIdempotents.lift_of_isNilpotent_ker_aux
    (h : ∀ x ∈ RingHom.ker f, IsNilpotent x)
    {n} {e : Fin n → S} (he : CompleteOrthogonalIdempotents e) (he' : ∀ i, e i ∈ f.range) :
    ∃ e' : Fin n → R, CompleteOrthogonalIdempotents e' ∧ f ∘ e' = e := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    n : Nat
    e : Fin n → S
    he : CompleteOrthogonalIdempotents e
    he' : ∀ (i : Fin n), Membership.mem f.range (e i)
    ⊢ Exists fun e' => And (CompleteOrthogonalIdempotents e') (Eq (Function.comp ( …
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      n : Nat
      e : Fin n → S
      he : CompleteOrthogonalIdempotents e
      he' : ∀ (i : Fin n), Membership.mem f.range (e i)
      h✝ : Subsingleton R
      ⊢ Exists fun e' => And (CompleteOrthogonalIdempotents e') (Eq (Function.comp ( …
    -/
  · choose e' he' using he'
    /-
      case inl
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      n : Nat
      e : Fin n → S
      he : CompleteOrthogonalIdempotents e
      h✝ : Subsingleton R
      e' : Fin n → R
      he' : ∀ (i : Fin n), Eq (f (e' i)) (e i)
      ⊢ Exists fun e' => And (CompleteOrthogonalIdempotents e') (Eq (Function.comp ( …
    -/
    exact ⟨e', .of_subsingleton, funext he'⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    n : Nat
    e : Fin n → S
    he : CompleteOrthogonalIdempotents e
    he' : ∀ (i : Fin n), Membership.mem f.range (e i)
    h✝ : Nontrivial R
    ⊢ Exists fun e' => And (CompleteOrthogonalIdempotents e') (Eq (Function.comp ( …
  -/
  cases subsingleton_or_nontrivial S
    /-
      case inr.inl
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      n : Nat
      e : Fin n → S
      he : CompleteOrthogonalIdempotents e
      he' : ∀ (i : Fin n), Membership.mem f.range (e i)
      h✝¹ : Nontrivial R
      h✝ : Subsingleton S
      ⊢ Exists fun e' => And (CompleteOrthogonalIdempotents e') (Eq (Function.comp ( …
    -/
  · obtain ⟨n, hn⟩ := h 1 (Subsingleton.elim _ _)
    /-
      case inr.inl.intro
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      n✝ : Nat
      e : Fin n✝ → S
      he : CompleteOrthogonalIdempotents e
      he' : ∀ (i : Fin n✝), Membership.mem f.range (e i)
      h✝¹ : Nontrivial R
      h✝ : Subsingleton S
      n : Nat
      hn : Eq (HPow.hPow 1 n) 0
      ⊢ Exists fun e' => And (CompleteOrthogonalIdempotents e') (Eq (Function.comp ( …
    -/
    simp at hn
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    n : Nat
    e : Fin n → S
    he : CompleteOrthogonalIdempotents e
    he' : ∀ (i : Fin n), Membership.mem f.range (e i)
    h✝¹ : Nontrivial R
    h✝ : Nontrivial S
    ⊢ Exists fun e' => And (CompleteOrthogonalIdempotents e') (Eq (Function.comp ( …
  -/
  cases' n with n
    /-
      case inr.inr.zero
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      h✝¹ : Nontrivial R
      h✝ : Nontrivial S
      e : Fin 0 → S
      he : CompleteOrthogonalIdempotents e
      he' : ∀ (i : Fin 0), Membership.mem f.range (e i)
      ⊢ Exists fun e' => And (CompleteOrthogonalIdempotents e') (Eq (Function.comp ( …
    -/
  · simpa using he.complete
    /-
      🎉 no goals
    -/
  /-
    case inr.inr.succ
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    h✝¹ : Nontrivial R
    h✝ : Nontrivial S
    n : Nat
    e : Fin (HAdd.hAdd n 1) → S
    he : CompleteOrthogonalIdempotents e
    he' : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem f.range (e i)
    ⊢ Exists fun e' => And (CompleteOrthogonalIdempotents e') (Eq (Function.comp ( …
  -/
  obtain ⟨e', h₁, h₂⟩ := OrthogonalIdempotents.lift_of_isNilpotent_ker f h he.1 he'
  refine ⟨_, (equiv (finSuccEquiv n)).mpr
    (CompleteOrthogonalIdempotents.option (h₁.embedding (Fin.succEmb _))), funext fun i ↦ ?_⟩
  /-
    case inr.inr.succ.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    h✝¹ : Nontrivial R
    h✝ : Nontrivial S
    n : Nat
    e : Fin (HAdd.hAdd n 1) → S
    he : CompleteOrthogonalIdempotents e
    he' : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem f.range (e i)
    e' : Fin (HAdd.hAdd n 1) → R
    h₁ : OrthogonalIdempotents e'
    h₂ : Eq (Function.comp (⇑f) e') e
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Function.comp (⇑f) (Function.comp (fun x => x.elim (HSub.hSub 1 (Finset. …
  -/
  have (i) : f (e' i) = e i := congr_fun h₂ i
  /-
    case inr.inr.succ.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : Ring S
    f : RingHom R S
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    h✝¹ : Nontrivial R
    h✝ : Nontrivial S
    n : Nat
    e : Fin (HAdd.hAdd n 1) → S
    he : CompleteOrthogonalIdempotents e
    he' : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem f.range (e i)
    e' : Fin (HAdd.hAdd n 1) → R
    h₁ : OrthogonalIdempotents e'
    h₂ : Eq (Function.comp (⇑f) e') e
    i : Fin (HAdd.hAdd n 1)
    this : ∀ (i : Fin (HAdd.hAdd n 1)), Eq (f (e' i)) (e i)
    ⊢ Eq (Function.comp (⇑f) (Function.comp (fun x => x.elim (HSub.hSub 1 (Finset. …
  -/
  obtain ⟨_ | i, rfl⟩ := (finSuccEquiv n).symm.surjective i
  · simp only [Fin.val_succEmb, Function.comp_apply, finSuccEquiv_symm_none, finSuccEquiv_zero,
      Option.elim_none, map_sub, map_one, map_sum, this, ← he.complete, sub_eq_iff_eq_add,
      Fin.sum_univ_succ]
    /-
      case inr.inr.succ.intro.intro.intro.some
      R : Type u_1
      S : Type u_2
      inst✝¹ : Ring R
      inst✝ : Ring S
      f : RingHom R S
      h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
      h✝¹ : Nontrivial R
      h✝ : Nontrivial S
      n : Nat
      e : Fin (HAdd.hAdd n 1) → S
      he : CompleteOrthogonalIdempotents e
      he' : ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem f.range (e i)
      e' : Fin (HAdd.hAdd n 1) → R
      h₁ : OrthogonalIdempotents e'
      h₂ : Eq (Function.comp (⇑f) e') e
      this : ∀ (i : Fin (HAdd.hAdd n 1)), Eq (f (e' i)) (e i)
      i : Fin n
      ⊢ Eq (Function.comp (⇑f) (Function.comp (fun x => x.elim (HSub.hSub 1 (Finset. …
    -/
  · simp [this]
    /-
      🎉 no goals
    -/


/-- A system of complete orthogonal idempotents lift along nil ideals. -/
lemma CompleteOrthogonalIdempotents.lift_of_isNilpotent_ker
    (h : ∀ x ∈ RingHom.ker f, IsNilpotent x)
    {e : I → S} (he : CompleteOrthogonalIdempotents e) (he' : ∀ i, e i ∈ f.range) :
    ∃ e' : I → R, CompleteOrthogonalIdempotents e' ∧ f ∘ e' = e := by
  obtain ⟨e', h₁, h₂⟩ := lift_of_isNilpotent_ker_aux f h
    ((equiv (Fintype.equivFin I).symm).mpr he) (fun _ ↦ he' _)
  refine ⟨_, ((equiv (Fintype.equivFin I)).mpr h₁),
    by ext x; simpa using congr_fun h₂ (Fintype.equivFin I x)⟩


theorem eq_of_isNilpotent_sub_of_isIdempotentElem_of_commute {e₁ e₂ : R}
    (he₁ : IsIdempotentElem e₁) (he₂ : IsIdempotentElem e₂) (H : IsNilpotent (e₁ - e₂))
    (H' : Commute e₁ e₂) :
    e₁ = e₂ := by
  have : (e₁ - e₂) ^ 3 = (e₁ - e₂) := by
    simp only [pow_succ, pow_zero, mul_sub, one_mul, sub_mul, he₁.eq, he₂.eq,
      H'.eq, mul_assoc]
    simp only [← mul_assoc, he₁.eq, he₂.eq]
    abel
  /-
    R : Type u_1
    inst✝ : Ring R
    e₁ e₂ : R
    he₁ : IsIdempotentElem e₁
    he₂ : IsIdempotentElem e₂
    H : IsNilpotent (HSub.hSub e₁ e₂)
    H' : Commute e₁ e₂
    this : Eq (HPow.hPow (HSub.hSub e₁ e₂) 3) (HSub.hSub e₁ e₂)
    ⊢ Eq e₁ e₂
  -/
  obtain ⟨n, hn⟩ := H
  have : (e₁ - e₂) ^ (2 * n + 1) = (e₁ - e₂) := by
    clear hn; induction n <;> simp [mul_add, add_assoc, pow_add _ (2 * _) 3, this, ← pow_succ, *]
  /-
    case intro
    R : Type u_1
    inst✝ : Ring R
    e₁ e₂ : R
    he₁ : IsIdempotentElem e₁
    he₂ : IsIdempotentElem e₂
    H' : Commute e₁ e₂
    this✝ : Eq (HPow.hPow (HSub.hSub e₁ e₂) 3) (HSub.hSub e₁ e₂)
    n : Nat
    hn : Eq (HPow.hPow (HSub.hSub e₁ e₂) n) 0
    this : Eq (HPow.hPow (HSub.hSub e₁ e₂) (HAdd.hAdd (HMul.hMul 2 n) 1)) (HSub.hS …
    ⊢ Eq e₁ e₂
  -/
  rwa [pow_succ, two_mul, pow_add, hn, zero_mul, zero_mul, eq_comm, sub_eq_zero] at this
  /-
    🎉 no goals
  -/


theorem CompleteOrthogonalIdempotents.of_ker_isNilpotent_of_isMulCentral
    (h : ∀ x ∈ RingHom.ker f, IsNilpotent x)
    (he : ∀ i, IsIdempotentElem (e i))
    (he' : ∀ i, IsMulCentral (e i))
    (he'' : CompleteOrthogonalIdempotents (f ∘ e)) :
    CompleteOrthogonalIdempotents e := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : Ring R
    inst✝¹ : Ring S
    f : RingHom R S
    I : Type u_3
    e : I → R
    inst✝ : Fintype I
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    he : ∀ (i : I), IsIdempotentElem (e i)
    he' : ∀ (i : I), IsMulCentral (e i)
    he'' : CompleteOrthogonalIdempotents (Function.comp (⇑f) e)
    ⊢ CompleteOrthogonalIdempotents e
  -/
  obtain ⟨e', h₁, h₂⟩ := lift_of_isNilpotent_ker f h he'' (fun _ ↦ ⟨_, rfl⟩)
  obtain rfl : e = e' := by
    ext i
    refine eq_of_isNilpotent_sub_of_isIdempotentElem_of_commute
      (he _) (h₁.idem _) (h _ ?_) ((he' i).comm _)
    simpa [RingHom.mem_ker, sub_eq_zero] using congr_fun h₂.symm i
  /-
    case intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : Ring R
    inst✝¹ : Ring S
    f : RingHom R S
    I : Type u_3
    e : I → R
    inst✝ : Fintype I
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    he : ∀ (i : I), IsIdempotentElem (e i)
    he' : ∀ (i : I), IsMulCentral (e i)
    he'' : CompleteOrthogonalIdempotents (Function.comp (⇑f) e)
    h₁ : CompleteOrthogonalIdempotents e
    h₂ : Eq (Function.comp (⇑f) e) (Function.comp (⇑f) e)
    ⊢ CompleteOrthogonalIdempotents e
  -/
  exact h₁
  /-
    🎉 no goals
  -/


theorem eq_of_isNilpotent_sub_of_isIdempotentElem {e₁ e₂ : R}
    (he₁ : IsIdempotentElem e₁) (he₂ : IsIdempotentElem e₂) (H : IsNilpotent (e₁ - e₂)) :
    e₁ = e₂ :=
  eq_of_isNilpotent_sub_of_isIdempotentElem_of_commute he₁ he₂ H (.all _ _)


@[stacks 00J9]
theorem existsUnique_isIdempotentElem_eq_of_ker_isNilpotent (h : ∀ x ∈ RingHom.ker f, IsNilpotent x)
    (e : S) (he : e ∈ f.range) (he' : IsIdempotentElem e) :
    ∃! e' : R, IsIdempotentElem e' ∧ f e' = e := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : Ring S
    f : RingHom R S
    h : ∀ (x : R), Membership.mem (RingHom.ker f) x → IsNilpotent x
    e : S
    he : Membership.mem f.range e
    he' : IsIdempotentElem e
    ⊢ ExistsUnique fun e' => And (IsIdempotentElem e') (Eq (f e') e)
  -/
  obtain ⟨e', he₂, rfl⟩ := exists_isIdempotentElem_eq_of_ker_isNilpotent f h e he he'
  exact ⟨e', ⟨he₂, rfl⟩, fun x ⟨hx, hx'⟩ ↦
    eq_of_isNilpotent_sub_of_isIdempotentElem hx he₂
      (h _ (by rw [RingHom.mem_ker, map_sub, hx', sub_self]))⟩


/-- A family of orthogonal idempotents induces an surjection `R ≃+* ∏ R ⧸ ⟨1 - eᵢ⟩` -/
lemma OrthogonalIdempotents.surjective_pi {I : Type*} [Finite I] {e : I → R}
    (he : OrthogonalIdempotents e) :
    Function.Surjective (Pi.ringHom fun i ↦ Ideal.Quotient.mk (Ideal.span {1 - e i})) := by
  suffices Pairwise fun i j ↦ IsCoprime (Ideal.span {1 - e i}) (Ideal.span {1 - e j}) by
    intro x
    obtain ⟨x, rfl⟩ := Ideal.quotientInfToPiQuotient_surj this x
    obtain ⟨x, rfl⟩ := Ideal.Quotient.mk_surjective x
    exact ⟨x, by ext i; simp [Ideal.quotientInfToPiQuotient]⟩
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    I : Type u_3
    inst✝ : Finite I
    e : I → R
    he : OrthogonalIdempotents e
    ⊢ Pairwise fun i j => IsCoprime (Ideal.span (Singleton.singleton (HSub.hSub 1  …
  -/
  intros i j hij
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    I : Type u_3
    inst✝ : Finite I
    e : I → R
    he : OrthogonalIdempotents e
    i j : I
    hij : Ne i j
    ⊢ IsCoprime (Ideal.span (Singleton.singleton (HSub.hSub 1 (e i)))) (Ideal.span …
  -/
  rw [Ideal.isCoprime_span_singleton_iff]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    I : Type u_3
    inst✝ : Finite I
    e : I → R
    he : OrthogonalIdempotents e
    i j : I
    hij : Ne i j
    ⊢ IsCoprime (HSub.hSub 1 (e i)) (HSub.hSub 1 (e j))
  -/
  exact ⟨1, e i, by simp [mul_sub, sub_mul, he.ortho hij]⟩
  /-
    🎉 no goals
  -/


lemma OrthogonalIdempotents.prod_one_sub {I : Type*} {e : I → R}
    (he : OrthogonalIdempotents e) (s : Finset I) :
    ∏ i ∈ s, (1 - e i) = 1 - ∑ i ∈ s, e i := by
  induction s using Finset.cons_induction with
  | empty => simp
  | cons a s has ih =>
    simp [ih, sub_mul, mul_sub, he.mul_sum_of_not_mem has, sub_sub]


theorem CompleteOrthogonalIdempotents.of_ker_isNilpotent (h : ∀ x ∈ RingHom.ker f, IsNilpotent x)
    (he : ∀ i, IsIdempotentElem (e i))
    (he' : CompleteOrthogonalIdempotents (f ∘ e)) :
    CompleteOrthogonalIdempotents e :=
  of_ker_isNilpotent_of_isMulCentral f h he
    (fun _ ↦ Semigroup.mem_center_iff.mpr (mul_comm · _)) he'


lemma CompleteOrthogonalIdempotents.prod_one_sub
    (he : CompleteOrthogonalIdempotents e) :
    ∏ i, (1 - e i) = 0 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    I : Type u_3
    inst✝ : Fintype I
    e : I → R
    he : CompleteOrthogonalIdempotents e
    ⊢ Eq (Finset.univ.prod fun i => HSub.hSub 1 (e i)) 0
  -/
  rw [he.1.prod_one_sub, he.complete, sub_self]
  /-
    🎉 no goals
  -/


lemma CompleteOrthogonalIdempotents.of_prod_one_sub
    (he : OrthogonalIdempotents e) (he' : ∏ i, (1 - e i) = 0) :
    CompleteOrthogonalIdempotents e where
  __ := he
                 /-
                   R : Type u_1
                   inst✝¹ : CommRing R
                   I : Type u_3
                   inst✝ : Fintype I
                   e : I → R
                   he : OrthogonalIdempotents e
                   he' : Eq (Finset.univ.prod fun i => HSub.hSub 1 (e i)) 0
                   ⊢ Eq (Finset.univ.sum fun i => e i) 1
                 -/
  complete := by rwa [he.prod_one_sub, sub_eq_zero, eq_comm] at he'
                 /-
                   🎉 no goals
                 -/


/-- A family of complete orthogonal idempotents induces an isomorphism `R ≃+* ∏ R ⧸ ⟨1 - eᵢ⟩` -/
lemma CompleteOrthogonalIdempotents.bijective_pi (he : CompleteOrthogonalIdempotents e) :
    Function.Bijective (Pi.ringHom fun i ↦ Ideal.Quotient.mk (Ideal.span {1 - e i})) := by
  classical
  refine ⟨?_, he.1.surjective_pi⟩
  rw [injective_iff_map_eq_zero]
  intro x hx
  simp [funext_iff, Ideal.Quotient.eq_zero_iff_mem, Ideal.mem_span_singleton] at hx
  suffices ∀ s : Finset I, (∏ i in s, (1 - e i)) * x = x by
    rw [← this Finset.univ, he.prod_one_sub, zero_mul]
  refine fun s ↦ Finset.induction_on s (by simp) ?_
  intros a s has e'
  suffices (1 - e a) * x = x by simp [has, mul_assoc, e', this]
  obtain ⟨c, rfl⟩ := hx a
  rw [← mul_assoc, (he.idem a).one_sub.eq]


lemma CompleteOrthogonalIdempotents.bijective_pi' (he : CompleteOrthogonalIdempotents (1 - e ·)) :
    Function.Bijective (Pi.ringHom fun i ↦ Ideal.Quotient.mk (Ideal.span {e i})) := by
  obtain ⟨e', rfl, h⟩ : ∃ e' : I → R, (e' = e) ∧ Function.Bijective (Pi.ringHom fun i ↦
      Ideal.Quotient.mk (Ideal.span {e' i})) := ⟨_, funext (by simp), he.bijective_pi⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    I : Type u_3
    inst✝ : Fintype I
    e' : I → R
    h : Function.Bijective ⇑(Pi.ringHom fun i => Ideal.Quotient.mk (Ideal.span (Si …
    he : CompleteOrthogonalIdempotents fun x => HSub.hSub 1 (e' x)
    ⊢ Function.Bijective ⇑(Pi.ringHom fun i => Ideal.Quotient.mk (Ideal.span (Sing …
  -/
  exact h
  /-
    🎉 no goals
  -/


lemma bijective_pi_of_isIdempotentElem (e : I → R)
    (he : ∀ i, IsIdempotentElem (e i))
    (he₁ : ∀ i j, i ≠ j → (1 - e i) * (1 - e j) = 0) (he₂ : ∏ i, e i = 0) :
    Function.Bijective (Pi.ringHom fun i ↦ Ideal.Quotient.mk (Ideal.span {e i})) :=
  (CompleteOrthogonalIdempotents.of_prod_one_sub
                                        /-
                                          R : Type u_1
                                          inst✝¹ : CommRing R
                                          I : Type u_3
                                          inst✝ : Fintype I
                                          e : I → R
                                          he : ∀ (i : I), IsIdempotentElem (e i)
                                          he₁ : ∀ (i j : I), Ne i j → Eq (HMul.hMul (HSub.hSub 1 (e i)) (HSub.hSub 1 (e  …
                                          he₂ : Eq (Finset.univ.prod fun i => e i) 0
                                          ⊢ Eq (Finset.univ.prod fun i => HSub.hSub 1 (HSub.hSub 1 (e i))) 0
                                        -/
      ⟨fun i ↦ (he i).one_sub, he₁⟩ (by simpa using he₂)).bijective_pi'
                                        /-
                                          🎉 no goals
                                        -/


