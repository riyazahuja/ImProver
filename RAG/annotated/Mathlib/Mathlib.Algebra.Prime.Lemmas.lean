theorem comap_prime (hinv : ∀ a, g (f a : N) = a) (hp : Prime (f p)) : Prime p :=
                       /-
                         M : Type u_1
                         N : Type u_2
                         inst✝⁵ : CommMonoidWithZero M
                         inst✝⁴ : CommMonoidWithZero N
                         F : Type u_3
                         G : Type u_4
                         inst✝³ : FunLike F M N
                         inst✝² : MonoidWithZeroHomClass F M N
                         inst✝¹ : FunLike G N M
                         inst✝ : MulHomClass G N M
                         f : F
                         g : G
                         p : M
                         hinv : ∀ (a : M), Eq (g (f a)) a
                         hp : Prime (f p)
                         h : Eq p 0
                         ⊢ Eq (f p) 0
                       -/
  ⟨fun h => hp.1 <| by simp [h], fun h => hp.2.1 <| h.map f, fun a b h => by
                       /-
                         🎉 no goals
                       -/
    refine
        (hp.2.2 (f a) (f b) <| by
              convert map_dvd f h
              simp).imp
          ?_ ?_ <;>
        /-
          case refine_1
          M : Type u_1
          N : Type u_2
          inst✝⁵ : CommMonoidWithZero M
          inst✝⁴ : CommMonoidWithZero N
          F : Type u_3
          G : Type u_4
          inst✝³ : FunLike F M N
          inst✝² : MonoidWithZeroHomClass F M N
          inst✝¹ : FunLike G N M
          inst✝ : MulHomClass G N M
          f : F
          g : G
          p : M
          hinv : ∀ (a : M), Eq (g (f a)) a
          hp : Prime (f p)
          a b : M
          h : Dvd.dvd p (HMul.hMul a b)
          ⊢ Dvd.dvd (f p) (f a) → Dvd.dvd p a
        -/
        /-
          case refine_1
          M : Type u_1
          N : Type u_2
          inst✝⁵ : CommMonoidWithZero M
          inst✝⁴ : CommMonoidWithZero N
          F : Type u_3
          G : Type u_4
          inst✝³ : FunLike F M N
          inst✝² : MonoidWithZeroHomClass F M N
          inst✝¹ : FunLike G N M
          inst✝ : MulHomClass G N M
          f : F
          g : G
          p : M
          hinv : ∀ (a : M), Eq (g (f a)) a
          hp : Prime (f p)
          a b : M
          h✝ : Dvd.dvd p (HMul.hMul a b)
          h : Dvd.dvd (f p) (f a)
          ⊢ Dvd.dvd p a
        -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
        /-
          case refine_2
          M : Type u_1
          N : Type u_2
          inst✝⁵ : CommMonoidWithZero M
          inst✝⁴ : CommMonoidWithZero N
          F : Type u_3
          G : Type u_4
          inst✝³ : FunLike F M N
          inst✝² : MonoidWithZeroHomClass F M N
          inst✝¹ : FunLike G N M
          inst✝ : MulHomClass G N M
          f : F
          g : G
          p : M
          hinv : ∀ (a : M), Eq (g (f a)) a
          hp : Prime (f p)
          a b : M
          h✝ : Dvd.dvd p (HMul.hMul a b)
          h : Dvd.dvd (f p) (f b)
          ⊢ Dvd.dvd p b
        -/
                                  /-
                                    🎉 no goals
                                  -/
        convert ← map_dvd g h <;> apply hinv⟩
                                  /-
                                    🎉 no goals
                                  -/


theorem MulEquiv.prime_iff {E : Type*} [EquivLike E M N] [MulEquivClass E M N] (e : E) :
    Prime (e p) ↔ Prime p := by
  /-
    M : Type u_1
    N : Type u_2
    inst✝³ : CommMonoidWithZero M
    inst✝² : CommMonoidWithZero N
    p : M
    E : Type u_5
    inst✝¹ : EquivLike E M N
    inst✝ : MulEquivClass E M N
    e : E
    ⊢ Iff (Prime (e p)) (Prime p)
  -/
  let e := MulEquivClass.toMulEquiv e
  exact ⟨comap_prime e e.symm fun a => by simp,
    fun h => (comap_prime e.symm e fun a => by simp) <| (e.symm_apply_apply p).substr h⟩


theorem Prime.left_dvd_or_dvd_right_of_dvd_mul [CancelCommMonoidWithZero M] {p : M} (hp : Prime p)
    {a b : M} : a ∣ p * b → p ∣ a ∨ a ∣ b := by
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p : M
    hp : Prime p
    a b : M
    ⊢ Dvd.dvd a (HMul.hMul p b) → Or (Dvd.dvd p a) (Dvd.dvd a b)
  -/
  rintro ⟨c, hc⟩
  /-
    case intro
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p : M
    hp : Prime p
    a b c : M
    hc : Eq (HMul.hMul p b) (HMul.hMul a c)
    ⊢ Or (Dvd.dvd p a) (Dvd.dvd a b)
  -/
  rcases hp.2.2 a c (hc ▸ dvd_mul_right _ _) with (h | ⟨x, rfl⟩)
    /-
      case intro.inl
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p : M
      hp : Prime p
      a b c : M
      hc : Eq (HMul.hMul p b) (HMul.hMul a c)
      h : Dvd.dvd p a
      ⊢ Or (Dvd.dvd p a) (Dvd.dvd a b)
    -/
  · exact Or.inl h
    /-
      🎉 no goals
    -/
    /-
      case intro.inr.intro
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p : M
      hp : Prime p
      a b x : M
      hc : Eq (HMul.hMul p b) (HMul.hMul a (HMul.hMul p x))
      ⊢ Or (Dvd.dvd p a) (Dvd.dvd a b)
    -/
  · rw [mul_left_comm, mul_right_inj' hp.ne_zero] at hc
    /-
      case intro.inr.intro
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p : M
      hp : Prime p
      a b x : M
      hc : Eq b (HMul.hMul a x)
      ⊢ Or (Dvd.dvd p a) (Dvd.dvd a b)
    -/
    exact Or.inr (hc.symm ▸ dvd_mul_right _ _)
    /-
      🎉 no goals
    -/


theorem Prime.pow_dvd_of_dvd_mul_left [CancelCommMonoidWithZero M] {p a b : M} (hp : Prime p)
    (n : ℕ) (h : ¬p ∣ a) (h' : p ^ n ∣ a * b) : p ^ n ∣ b := by
  induction n with
  | zero =>
    rw [pow_zero]
    exact one_dvd b
  | succ n ih =>
    obtain ⟨c, rfl⟩ := ih (dvd_trans (pow_dvd_pow p n.le_succ) h')
    rw [pow_succ]
    apply mul_dvd_mul_left _ ((hp.dvd_or_dvd _).resolve_left h)
    rwa [← mul_dvd_mul_iff_left (pow_ne_zero n hp.ne_zero), ← pow_succ, mul_left_comm]


theorem Prime.pow_dvd_of_dvd_mul_right [CancelCommMonoidWithZero M] {p a b : M} (hp : Prime p)
    (n : ℕ) (h : ¬p ∣ b) (h' : p ^ n ∣ a * b) : p ^ n ∣ a := by
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p a b : M
    hp : Prime p
    n : Nat
    h : Not (Dvd.dvd p b)
    h' : Dvd.dvd (HPow.hPow p n) (HMul.hMul a b)
    ⊢ Dvd.dvd (HPow.hPow p n) a
  -/
  rw [mul_comm] at h'
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p a b : M
    hp : Prime p
    n : Nat
    h : Not (Dvd.dvd p b)
    h' : Dvd.dvd (HPow.hPow p n) (HMul.hMul b a)
    ⊢ Dvd.dvd (HPow.hPow p n) a
  -/
  exact hp.pow_dvd_of_dvd_mul_left n h h'
  /-
    🎉 no goals
  -/


theorem Prime.dvd_of_pow_dvd_pow_mul_pow_of_square_not_dvd [CancelCommMonoidWithZero M] {p a b : M}
    {n : ℕ} (hp : Prime p) (hpow : p ^ n.succ ∣ a ^ n.succ * b ^ n) (hb : ¬p ^ 2 ∣ b) : p ∣ a := by
  -- Suppose `p ∣ b`, write `b = p * x` and `hy : a ^ n.succ * b ^ n = p ^ n.succ * y`.
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p a b : M
    n : Nat
    hp : Prime p
    hpow : Dvd.dvd (HPow.hPow p n.succ) (HMul.hMul (HPow.hPow a n.succ) (HPow.hPow …
    hb : Not (Dvd.dvd (HPow.hPow p 2) b)
    ⊢ Dvd.dvd p a
  -/
  rcases hp.dvd_or_dvd ((dvd_pow_self p (Nat.succ_ne_zero n)).trans hpow) with H | hbdiv
    /-
      case inl
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p a b : M
      n : Nat
      hp : Prime p
      hpow : Dvd.dvd (HPow.hPow p n.succ) (HMul.hMul (HPow.hPow a n.succ) (HPow.hPow …
      hb : Not (Dvd.dvd (HPow.hPow p 2) b)
      H : Dvd.dvd p (HPow.hPow a n.succ)
      ⊢ Dvd.dvd p a
    -/
  · exact hp.dvd_of_dvd_pow H
    /-
      🎉 no goals
    -/
  /-
    case inr
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p a b : M
    n : Nat
    hp : Prime p
    hpow : Dvd.dvd (HPow.hPow p n.succ) (HMul.hMul (HPow.hPow a n.succ) (HPow.hPow …
    hb : Not (Dvd.dvd (HPow.hPow p 2) b)
    hbdiv : Dvd.dvd p (HPow.hPow b n)
    ⊢ Dvd.dvd p a
  -/
  obtain ⟨x, rfl⟩ := hp.dvd_of_dvd_pow hbdiv
  /-
    case inr.intro
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p a : M
    n : Nat
    hp : Prime p
    x : M
    hpow : Dvd.dvd (HPow.hPow p n.succ) (HMul.hMul (HPow.hPow a n.succ) (HPow.hPow …
    hb : Not (Dvd.dvd (HPow.hPow p 2) (HMul.hMul p x))
    hbdiv : Dvd.dvd p (HPow.hPow (HMul.hMul p x) n)
    ⊢ Dvd.dvd p a
  -/
  obtain ⟨y, hy⟩ := hpow
  -- Then we can divide out a common factor of `p ^ n` from the equation `hy`.
  have : a ^ n.succ * x ^ n = p * y := by
    refine mul_left_cancel₀ (pow_ne_zero n hp.ne_zero) ?_
    rw [← mul_assoc _ p, ← pow_succ, ← hy, mul_pow, ← mul_assoc (a ^ n.succ), mul_comm _ (p ^ n),
      mul_assoc]
  -- So `p ∣ a` (and we're done) or `p ∣ x`, which can't be the case since it implies `p^2 ∣ b`.
  /-
    case inr.intro.intro
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p a : M
    n : Nat
    hp : Prime p
    x : M
    hb : Not (Dvd.dvd (HPow.hPow p 2) (HMul.hMul p x))
    hbdiv : Dvd.dvd p (HPow.hPow (HMul.hMul p x) n)
    y : M
    hy : Eq (HMul.hMul (HPow.hPow a n.succ) (HPow.hPow (HMul.hMul p x) n)) (HMul.h …
    this : Eq (HMul.hMul (HPow.hPow a n.succ) (HPow.hPow x n)) (HMul.hMul p y)
    ⊢ Dvd.dvd p a
  -/
  refine hp.dvd_of_dvd_pow ((hp.dvd_or_dvd ⟨_, this⟩).resolve_right fun hdvdx => hb ?_)
  /-
    case inr.intro.intro
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p a : M
    n : Nat
    hp : Prime p
    x : M
    hb : Not (Dvd.dvd (HPow.hPow p 2) (HMul.hMul p x))
    hbdiv : Dvd.dvd p (HPow.hPow (HMul.hMul p x) n)
    y : M
    hy : Eq (HMul.hMul (HPow.hPow a n.succ) (HPow.hPow (HMul.hMul p x) n)) (HMul.h …
    this : Eq (HMul.hMul (HPow.hPow a n.succ) (HPow.hPow x n)) (HMul.hMul p y)
    hdvdx : Dvd.dvd p (HPow.hPow x n)
    ⊢ Dvd.dvd (HPow.hPow p 2) (HMul.hMul p x)
  -/
  obtain ⟨z, rfl⟩ := hp.dvd_of_dvd_pow hdvdx
  /-
    case inr.intro.intro.intro
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p a : M
    n : Nat
    hp : Prime p
    y z : M
    hb : Not (Dvd.dvd (HPow.hPow p 2) (HMul.hMul p (HMul.hMul p z)))
    hbdiv : Dvd.dvd p (HPow.hPow (HMul.hMul p (HMul.hMul p z)) n)
    hy : Eq (HMul.hMul (HPow.hPow a n.succ) (HPow.hPow (HMul.hMul p (HMul.hMul p z …
    this : Eq (HMul.hMul (HPow.hPow a n.succ) (HPow.hPow (HMul.hMul p z) n)) (HMul …
    hdvdx : Dvd.dvd p (HPow.hPow (HMul.hMul p z) n)
    ⊢ Dvd.dvd (HPow.hPow p 2) (HMul.hMul p (HMul.hMul p z))
  -/
  rw [pow_two, ← mul_assoc]
  /-
    case inr.intro.intro.intro
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p a : M
    n : Nat
    hp : Prime p
    y z : M
    hb : Not (Dvd.dvd (HPow.hPow p 2) (HMul.hMul p (HMul.hMul p z)))
    hbdiv : Dvd.dvd p (HPow.hPow (HMul.hMul p (HMul.hMul p z)) n)
    hy : Eq (HMul.hMul (HPow.hPow a n.succ) (HPow.hPow (HMul.hMul p (HMul.hMul p z …
    this : Eq (HMul.hMul (HPow.hPow a n.succ) (HPow.hPow (HMul.hMul p z) n)) (HMul …
    hdvdx : Dvd.dvd p (HPow.hPow (HMul.hMul p z) n)
    ⊢ Dvd.dvd (HMul.hMul p p) (HMul.hMul (HMul.hMul p p) z)
  -/
  exact dvd_mul_right _ _
  /-
    🎉 no goals
  -/


theorem prime_pow_succ_dvd_mul {M : Type*} [CancelCommMonoidWithZero M] {p x y : M} (h : Prime p)
    {i : ℕ} (hxy : p ^ (i + 1) ∣ x * y) : p ^ (i + 1) ∣ x ∨ p ∣ y := by
  /-
    M : Type u_3
    inst✝ : CancelCommMonoidWithZero M
    p x y : M
    h : Prime p
    i : Nat
    hxy : Dvd.dvd (HPow.hPow p (HAdd.hAdd i 1)) (HMul.hMul x y)
    ⊢ Or (Dvd.dvd (HPow.hPow p (HAdd.hAdd i 1)) x) (Dvd.dvd p y)
  -/
  rw [or_iff_not_imp_right]
  /-
    M : Type u_3
    inst✝ : CancelCommMonoidWithZero M
    p x y : M
    h : Prime p
    i : Nat
    hxy : Dvd.dvd (HPow.hPow p (HAdd.hAdd i 1)) (HMul.hMul x y)
    ⊢ Not (Dvd.dvd p y) → Dvd.dvd (HPow.hPow p (HAdd.hAdd i 1)) x
  -/
  intro hy
  induction i generalizing x with
  | zero => rw [pow_one] at hxy ⊢; exact (h.dvd_or_dvd hxy).resolve_right hy
  | succ i ih =>
    rw [pow_succ'] at hxy ⊢
    obtain ⟨x', rfl⟩ := (h.dvd_or_dvd (dvd_of_mul_right_dvd hxy)).resolve_right hy
    rw [mul_assoc] at hxy
    exact mul_dvd_mul_left p (ih ((mul_dvd_mul_iff_left h.ne_zero).mp hxy))


theorem not_irreducible_pow {M} [Monoid M] {x : M} {n : ℕ} (hn : n ≠ 1) :
    ¬ Irreducible (x ^ n) := by
  cases n with
  | zero => simp
  | succ n =>
    intro ⟨h₁, h₂⟩
    have := h₂ _ _ (pow_succ _ _)
    rw [isUnit_pow_iff (Nat.succ_ne_succ.mp hn), or_self] at this
    exact h₁ (this.pow _)


theorem Irreducible.of_map {F : Type*} [Monoid M] [Monoid N] [FunLike F M N] [MonoidHomClass F M N]
    {f : F} [IsLocalHom f] {x} (hfx : Irreducible (f x)) : Irreducible x :=
  ⟨fun hu ↦ hfx.not_unit <| hu.map f,
      /-
        M : Type u_1
        N : Type u_2
        F : Type u_3
        inst✝⁴ : Monoid M
        inst✝³ : Monoid N
        inst✝² : FunLike F M N
        inst✝¹ : MonoidHomClass F M N
        f : F
        inst✝ : IsLocalHom f
        x : M
        hfx : Irreducible (f x)
        ⊢ ∀ (a b : M), Eq x (HMul.hMul a b) → Or (IsUnit a) (IsUnit b)
      -/
   by rintro p q rfl
      /-
        M : Type u_1
        N : Type u_2
        F : Type u_3
        inst✝⁴ : Monoid M
        inst✝³ : Monoid N
        inst✝² : FunLike F M N
        inst✝¹ : MonoidHomClass F M N
        f : F
        inst✝ : IsLocalHom f
        p q : M
        hfx : Irreducible (f (HMul.hMul p q))
        ⊢ Or (IsUnit p) (IsUnit q)
      -/
      exact (hfx.isUnit_or_isUnit <| map_mul f p q).imp (.of_map f _) (.of_map f _)⟩
      /-
        🎉 no goals
      -/


theorem irreducible_units_mul (a : Mˣ) (b : M) : Irreducible (↑a * b) ↔ Irreducible b := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    a : Units M
    b : M
    ⊢ Iff (Irreducible (HMul.hMul (↑a) b)) (Irreducible b)
  -/
  simp only [irreducible_iff, Units.isUnit_units_mul, and_congr_right_iff]
  /-
    M : Type u_1
    inst✝ : Monoid M
    a : Units M
    b : M
    ⊢ Not (IsUnit b) → Iff (∀ (a_2 b_1 : M), Eq (HMul.hMul (↑a) b) (HMul.hMul a_2  …
  -/
  refine fun _ => ⟨fun h A B HAB => ?_, fun h A B HAB => ?_⟩
    /-
      case refine_1
      M : Type u_1
      inst✝ : Monoid M
      a : Units M
      b : M
      x✝ : Not (IsUnit b)
      h : ∀ (a_1 b_1 : M), Eq (HMul.hMul (↑a) b) (HMul.hMul a_1 b_1) → Or (IsUnit a_ …
      A B : M
      HAB : Eq b (HMul.hMul A B)
      ⊢ Or (IsUnit A) (IsUnit B)
    -/
  · rw [← a.isUnit_units_mul]
    /-
      case refine_1
      M : Type u_1
      inst✝ : Monoid M
      a : Units M
      b : M
      x✝ : Not (IsUnit b)
      h : ∀ (a_1 b_1 : M), Eq (HMul.hMul (↑a) b) (HMul.hMul a_1 b_1) → Or (IsUnit a_ …
      A B : M
      HAB : Eq b (HMul.hMul A B)
      ⊢ Or (IsUnit (HMul.hMul (↑a) A)) (IsUnit B)
    -/
    apply h
    /-
      case refine_1.a
      M : Type u_1
      inst✝ : Monoid M
      a : Units M
      b : M
      x✝ : Not (IsUnit b)
      h : ∀ (a_1 b_1 : M), Eq (HMul.hMul (↑a) b) (HMul.hMul a_1 b_1) → Or (IsUnit a_ …
      A B : M
      HAB : Eq b (HMul.hMul A B)
      ⊢ Eq (HMul.hMul (↑a) b) (HMul.hMul (HMul.hMul (↑a) A) B)
    -/
    rw [mul_assoc, ← HAB]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      M : Type u_1
      inst✝ : Monoid M
      a : Units M
      b : M
      x✝ : Not (IsUnit b)
      h : ∀ (a b_1 : M), Eq b (HMul.hMul a b_1) → Or (IsUnit a) (IsUnit b_1)
      A B : M
      HAB : Eq (HMul.hMul (↑a) b) (HMul.hMul A B)
      ⊢ Or (IsUnit A) (IsUnit B)
    -/
  · rw [← a⁻¹.isUnit_units_mul]
    /-
      case refine_2
      M : Type u_1
      inst✝ : Monoid M
      a : Units M
      b : M
      x✝ : Not (IsUnit b)
      h : ∀ (a b_1 : M), Eq b (HMul.hMul a b_1) → Or (IsUnit a) (IsUnit b_1)
      A B : M
      HAB : Eq (HMul.hMul (↑a) b) (HMul.hMul A B)
      ⊢ Or (IsUnit (HMul.hMul (↑(Inv.inv a)) A)) (IsUnit B)
    -/
    apply h
    /-
      case refine_2.a
      M : Type u_1
      inst✝ : Monoid M
      a : Units M
      b : M
      x✝ : Not (IsUnit b)
      h : ∀ (a b_1 : M), Eq b (HMul.hMul a b_1) → Or (IsUnit a) (IsUnit b_1)
      A B : M
      HAB : Eq (HMul.hMul (↑a) b) (HMul.hMul A B)
      ⊢ Eq b (HMul.hMul (HMul.hMul (↑(Inv.inv a)) A) B)
    -/
    rw [mul_assoc, ← HAB, Units.inv_mul_cancel_left]
    /-
      🎉 no goals
    -/


theorem irreducible_isUnit_mul {a b : M} (h : IsUnit a) : Irreducible (a * b) ↔ Irreducible b :=
  let ⟨a, ha⟩ := h
  ha ▸ irreducible_units_mul a b


theorem irreducible_mul_units (a : Mˣ) (b : M) : Irreducible (b * ↑a) ↔ Irreducible b := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    a : Units M
    b : M
    ⊢ Iff (Irreducible (HMul.hMul b ↑a)) (Irreducible b)
  -/
  simp only [irreducible_iff, Units.isUnit_mul_units, and_congr_right_iff]
  /-
    M : Type u_1
    inst✝ : Monoid M
    a : Units M
    b : M
    ⊢ Not (IsUnit b) → Iff (∀ (a_2 b_1 : M), Eq (HMul.hMul b ↑a) (HMul.hMul a_2 b_ …
  -/
  refine fun _ => ⟨fun h A B HAB => ?_, fun h A B HAB => ?_⟩
    /-
      case refine_1
      M : Type u_1
      inst✝ : Monoid M
      a : Units M
      b : M
      x✝ : Not (IsUnit b)
      h : ∀ (a_1 b_1 : M), Eq (HMul.hMul b ↑a) (HMul.hMul a_1 b_1) → Or (IsUnit a_1) …
      A B : M
      HAB : Eq b (HMul.hMul A B)
      ⊢ Or (IsUnit A) (IsUnit B)
    -/
  · rw [← Units.isUnit_mul_units B a]
    /-
      case refine_1
      M : Type u_1
      inst✝ : Monoid M
      a : Units M
      b : M
      x✝ : Not (IsUnit b)
      h : ∀ (a_1 b_1 : M), Eq (HMul.hMul b ↑a) (HMul.hMul a_1 b_1) → Or (IsUnit a_1) …
      A B : M
      HAB : Eq b (HMul.hMul A B)
      ⊢ Or (IsUnit A) (IsUnit (HMul.hMul B ↑a))
    -/
    apply h
    /-
      case refine_1.a
      M : Type u_1
      inst✝ : Monoid M
      a : Units M
      b : M
      x✝ : Not (IsUnit b)
      h : ∀ (a_1 b_1 : M), Eq (HMul.hMul b ↑a) (HMul.hMul a_1 b_1) → Or (IsUnit a_1) …
      A B : M
      HAB : Eq b (HMul.hMul A B)
      ⊢ Eq (HMul.hMul b ↑a) (HMul.hMul A (HMul.hMul B ↑a))
    -/
    rw [← mul_assoc, ← HAB]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      M : Type u_1
      inst✝ : Monoid M
      a : Units M
      b : M
      x✝ : Not (IsUnit b)
      h : ∀ (a b_1 : M), Eq b (HMul.hMul a b_1) → Or (IsUnit a) (IsUnit b_1)
      A B : M
      HAB : Eq (HMul.hMul b ↑a) (HMul.hMul A B)
      ⊢ Or (IsUnit A) (IsUnit B)
    -/
  · rw [← Units.isUnit_mul_units B a⁻¹]
    /-
      case refine_2
      M : Type u_1
      inst✝ : Monoid M
      a : Units M
      b : M
      x✝ : Not (IsUnit b)
      h : ∀ (a b_1 : M), Eq b (HMul.hMul a b_1) → Or (IsUnit a) (IsUnit b_1)
      A B : M
      HAB : Eq (HMul.hMul b ↑a) (HMul.hMul A B)
      ⊢ Or (IsUnit A) (IsUnit (HMul.hMul B ↑(Inv.inv a)))
    -/
    apply h
    /-
      case refine_2.a
      M : Type u_1
      inst✝ : Monoid M
      a : Units M
      b : M
      x✝ : Not (IsUnit b)
      h : ∀ (a b_1 : M), Eq b (HMul.hMul a b_1) → Or (IsUnit a) (IsUnit b_1)
      A B : M
      HAB : Eq (HMul.hMul b ↑a) (HMul.hMul A B)
      ⊢ Eq b (HMul.hMul A (HMul.hMul B ↑(Inv.inv a)))
    -/
    rw [← mul_assoc, ← HAB, Units.mul_inv_cancel_right]
    /-
      🎉 no goals
    -/


theorem irreducible_mul_isUnit {a b : M} (h : IsUnit a) : Irreducible (b * a) ↔ Irreducible b :=
  let ⟨a, ha⟩ := h
  ha ▸ irreducible_mul_units a b


theorem irreducible_mul_iff {a b : M} :
    Irreducible (a * b) ↔ Irreducible a ∧ IsUnit b ∨ Irreducible b ∧ IsUnit a := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    a b : M
    ⊢ Iff (Irreducible (HMul.hMul a b)) (Or (And (Irreducible a) (IsUnit b)) (And  …
  -/
  constructor
    /-
      case mp
      M : Type u_1
      inst✝ : Monoid M
      a b : M
      ⊢ Irreducible (HMul.hMul a b) → Or (And (Irreducible a) (IsUnit b)) (And (Irre …
    -/
  · refine fun h => Or.imp (fun h' => ⟨?_, h'⟩) (fun h' => ⟨?_, h'⟩) (h.isUnit_or_isUnit rfl).symm
      /-
        case mp.refine_1
        M : Type u_1
        inst✝ : Monoid M
        a b : M
        h : Irreducible (HMul.hMul a b)
        h' : IsUnit b
        ⊢ Irreducible a
      -/
    · rwa [irreducible_mul_isUnit h'] at h
      /-
        🎉 no goals
      -/
      /-
        case mp.refine_2
        M : Type u_1
        inst✝ : Monoid M
        a b : M
        h : Irreducible (HMul.hMul a b)
        h' : IsUnit a
        ⊢ Irreducible b
      -/
    · rwa [irreducible_isUnit_mul h'] at h
      /-
        🎉 no goals
      -/
    /-
      case mpr
      M : Type u_1
      inst✝ : Monoid M
      a b : M
      ⊢ Or (And (Irreducible a) (IsUnit b)) (And (Irreducible b) (IsUnit a)) → Irred …
    -/
  · rintro (⟨ha, hb⟩ | ⟨hb, ha⟩)
      /-
        case mpr.inl.intro
        M : Type u_1
        inst✝ : Monoid M
        a b : M
        ha : Irreducible a
        hb : IsUnit b
        ⊢ Irreducible (HMul.hMul a b)
      -/
    · rwa [irreducible_mul_isUnit hb]
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.intro
        M : Type u_1
        inst✝ : Monoid M
        a b : M
        hb : Irreducible b
        ha : IsUnit a
        ⊢ Irreducible (HMul.hMul a b)
      -/
    · rwa [irreducible_isUnit_mul ha]
      /-
        🎉 no goals
      -/


/--
Irreducibility is preserved by multiplicative equivalences.
Note that surjective + local hom is not enough. Consider the additive monoids `M = ℕ ⊕ ℕ`, `N = ℕ`,
with a surjective local (additive) hom `f : M →+ N` sending `(m, n)` to `2m + n`.
It is local because the only add unit in `N` is `0`, with preimage `{(0, 0)}` also an add unit.
Then `x = (1, 0)` is irreducible in `M`, but `f x = 2 = 1 + 1` is not irreducible in `N`.
-/
theorem Irreducible.map {x : M} (h : Irreducible x) : Irreducible (f x) :=
  ⟨fun g ↦ h.not_unit g.of_map, fun a b g ↦
    let f := MulEquivClass.toMulEquiv f
    (h.isUnit_or_isUnit (symm_apply_apply f x ▸ map_mul f.symm a b ▸ congrArg f.symm g)).imp
      (·.of_map) (·.of_map)⟩


theorem MulEquiv.irreducible_iff (f : F) {a : M} :
    Irreducible (f a) ↔ Irreducible a :=
  ⟨Irreducible.of_map, Irreducible.map f⟩


theorem Irreducible.not_square (ha : Irreducible a) : ¬IsSquare a := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    a : M
    ha : Irreducible a
    ⊢ Not (IsSquare a)
  -/
  rw [isSquare_iff_exists_sq]
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    a : M
    ha : Irreducible a
    ⊢ Not (Exists fun c => Eq a (HPow.hPow c 2))
  -/
  rintro ⟨b, rfl⟩
  /-
    case intro
    M : Type u_1
    inst✝ : CommMonoid M
    b : M
    ha : Irreducible (HPow.hPow b 2)
    ⊢ False
  -/
  exact not_irreducible_pow (by decide) ha
  /-
    🎉 no goals
  -/


theorem IsSquare.not_irreducible (ha : IsSquare a) : ¬Irreducible a := fun h => h.not_square ha


theorem succ_dvd_or_succ_dvd_of_succ_sum_dvd_mul (hp : Prime p) {a b : M} {k l : ℕ} :
    p ^ k ∣ a → p ^ l ∣ b → p ^ (k + l + 1) ∣ a * b → p ^ (k + 1) ∣ a ∨ p ^ (l + 1) ∣ b :=
  fun ⟨x, hx⟩ ⟨y, hy⟩ ⟨z, hz⟩ =>
  have h : p ^ (k + l) * (x * y) = p ^ (k + l) * (p * z) := by
    /-
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p : M
      hp : Prime p
      a b : M
      k l : Nat
      x✝² : Dvd.dvd (HPow.hPow p k) a
      x✝¹ : Dvd.dvd (HPow.hPow p l) b
      x✝ : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd k l) 1)) (HMul.hMul a b)
      x : M
      hx : Eq a (HMul.hMul (HPow.hPow p k) x)
      y : M
      hy : Eq b (HMul.hMul (HPow.hPow p l) y)
      z : M
      hz : Eq (HMul.hMul a b) (HMul.hMul (HPow.hPow p (HAdd.hAdd (HAdd.hAdd k l) 1)) …
      ⊢ Eq (HMul.hMul (HPow.hPow p (HAdd.hAdd k l)) (HMul.hMul x y)) (HMul.hMul (HPo …
    -/
    simpa [mul_comm, pow_add, hx, hy, mul_assoc, mul_left_comm] using hz
    /-
      🎉 no goals
    -/
  have hp0 : p ^ (k + l) ≠ 0 := pow_ne_zero _ hp.ne_zero
                                 /-
                                   M : Type u_1
                                   inst✝ : CancelCommMonoidWithZero M
                                   p : M
                                   hp : Prime p
                                   a b : M
                                   k l : Nat
                                   x✝² : Dvd.dvd (HPow.hPow p k) a
                                   x✝¹ : Dvd.dvd (HPow.hPow p l) b
                                   x✝ : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd k l) 1)) (HMul.hMul a b)
                                   x : M
                                   hx : Eq a (HMul.hMul (HPow.hPow p k) x)
                                   y : M
                                   hy : Eq b (HMul.hMul (HPow.hPow p l) y)
                                   z : M
                                   hz : Eq (HMul.hMul a b) (HMul.hMul (HPow.hPow p (HAdd.hAdd (HAdd.hAdd k l) 1)) …
                                   h : Eq (HMul.hMul (HPow.hPow p (HAdd.hAdd k l)) (HMul.hMul x y)) (HMul.hMul (H …
                                   hp0 : Ne (HPow.hPow p (HAdd.hAdd k l)) 0
                                   ⊢ Eq (HMul.hMul x y) (HMul.hMul p z)
                                 -/
  have hpd : p ∣ x * y := ⟨z, by rwa [mul_right_inj' hp0] at h⟩
                                 /-
                                   🎉 no goals
                                 -/
  (hp.dvd_or_dvd hpd).elim
                                  /-
                                    M : Type u_1
                                    inst✝ : CancelCommMonoidWithZero M
                                    p : M
                                    hp : Prime p
                                    a b : M
                                    k l : Nat
                                    x✝³ : Dvd.dvd (HPow.hPow p k) a
                                    x✝² : Dvd.dvd (HPow.hPow p l) b
                                    x✝¹ : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd k l) 1)) (HMul.hMul a b)
                                    x : M
                                    hx : Eq a (HMul.hMul (HPow.hPow p k) x)
                                    y : M
                                    hy : Eq b (HMul.hMul (HPow.hPow p l) y)
                                    z : M
                                    hz : Eq (HMul.hMul a b) (HMul.hMul (HPow.hPow p (HAdd.hAdd (HAdd.hAdd k l) 1)) …
                                    h : Eq (HMul.hMul (HPow.hPow p (HAdd.hAdd k l)) (HMul.hMul x y)) (HMul.hMul (H …
                                    hp0 : Ne (HPow.hPow p (HAdd.hAdd k l)) 0
                                    hpd : Dvd.dvd p (HMul.hMul x y)
                                    x✝ : Dvd.dvd p x
                                    d : M
                                    hd : Eq x (HMul.hMul p d)
                                    ⊢ Eq a (HMul.hMul (HPow.hPow p (HAdd.hAdd k 1)) d)
                                  -/
    (fun ⟨d, hd⟩ => Or.inl ⟨d, by simp [*, pow_succ, mul_comm, mul_left_comm, mul_assoc]⟩)
                                  /-
                                    🎉 no goals
                                  -/
                                 /-
                                   M : Type u_1
                                   inst✝ : CancelCommMonoidWithZero M
                                   p : M
                                   hp : Prime p
                                   a b : M
                                   k l : Nat
                                   x✝³ : Dvd.dvd (HPow.hPow p k) a
                                   x✝² : Dvd.dvd (HPow.hPow p l) b
                                   x✝¹ : Dvd.dvd (HPow.hPow p (HAdd.hAdd (HAdd.hAdd k l) 1)) (HMul.hMul a b)
                                   x : M
                                   hx : Eq a (HMul.hMul (HPow.hPow p k) x)
                                   y : M
                                   hy : Eq b (HMul.hMul (HPow.hPow p l) y)
                                   z : M
                                   hz : Eq (HMul.hMul a b) (HMul.hMul (HPow.hPow p (HAdd.hAdd (HAdd.hAdd k l) 1)) …
                                   h : Eq (HMul.hMul (HPow.hPow p (HAdd.hAdd k l)) (HMul.hMul x y)) (HMul.hMul (H …
                                   hp0 : Ne (HPow.hPow p (HAdd.hAdd k l)) 0
                                   hpd : Dvd.dvd p (HMul.hMul x y)
                                   x✝ : Dvd.dvd p y
                                   d : M
                                   hd : Eq y (HMul.hMul p d)
                                   ⊢ Eq b (HMul.hMul (HPow.hPow p (HAdd.hAdd l 1)) d)
                                 -/
    fun ⟨d, hd⟩ => Or.inr ⟨d, by simp [*, pow_succ, mul_comm, mul_left_comm, mul_assoc]⟩
                                 /-
                                   🎉 no goals
                                 -/


theorem Prime.not_square (hp : Prime p) : ¬IsSquare p :=
  hp.irreducible.not_square


theorem IsSquare.not_prime (ha : IsSquare a) : ¬Prime a := fun h => h.not_square ha


theorem not_prime_pow {n : ℕ} (hn : n ≠ 1) : ¬Prime (a ^ n) := fun hp =>
  not_irreducible_pow hn hp.irreducible


theorem DvdNotUnit.isUnit_of_irreducible_right [CommMonoidWithZero M] {p q : M}
    (h : DvdNotUnit p q) (hq : Irreducible q) : IsUnit p := by
  /-
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    p q : M
    h : DvdNotUnit p q
    hq : Irreducible q
    ⊢ IsUnit p
  -/
  obtain ⟨_, x, hx, hx'⟩ := h
  /-
    case intro.intro.intro
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    p q : M
    hq : Irreducible q
    left✝ : Ne p 0
    x : M
    hx : Not (IsUnit x)
    hx' : Eq q (HMul.hMul p x)
    ⊢ IsUnit p
  -/
  exact Or.resolve_right ((irreducible_iff.1 hq).right p x hx') hx
  /-
    🎉 no goals
  -/


theorem not_irreducible_of_not_unit_dvdNotUnit [CommMonoidWithZero M] {p q : M} (hp : ¬IsUnit p)
    (h : DvdNotUnit p q) : ¬Irreducible q :=
  mt h.isUnit_of_irreducible_right hp


theorem DvdNotUnit.not_unit [CommMonoidWithZero M] {p q : M} (hp : DvdNotUnit p q) : ¬IsUnit q := by
  /-
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    p q : M
    hp : DvdNotUnit p q
    ⊢ Not (IsUnit q)
  -/
  obtain ⟨-, x, hx, rfl⟩ := hp
  /-
    case intro.intro.intro
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    p x : M
    hx : Not (IsUnit x)
    ⊢ Not (IsUnit (HMul.hMul p x))
  -/
  exact fun hc => hx (isUnit_iff_dvd_one.mpr (dvd_of_mul_left_dvd (isUnit_iff_dvd_one.mp hc)))
  /-
    🎉 no goals
  -/


theorem DvdNotUnit.ne [CancelCommMonoidWithZero M] {p q : M} (h : DvdNotUnit p q) : p ≠ q := by
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p q : M
    h : DvdNotUnit p q
    ⊢ Ne p q
  -/
  by_contra hcontra
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p q : M
    h : DvdNotUnit p q
    hcontra : Eq p q
    ⊢ False
  -/
  obtain ⟨hp, x, hx', hx''⟩ := h
  /-
    case intro.intro.intro
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p q : M
    hcontra : Eq p q
    hp : Ne p 0
    x : M
    hx' : Not (IsUnit x)
    hx'' : Eq q (HMul.hMul p x)
    ⊢ False
  -/
  conv_lhs at hx'' => rw [← hcontra, ← mul_one p]
  /-
    case intro.intro.intro
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p q : M
    hcontra : Eq p q
    hp : Ne p 0
    x : M
    hx' : Not (IsUnit x)
    hx'' : Eq (HMul.hMul p 1) (HMul.hMul p x)
    ⊢ False
  -/
  rw [(mul_left_cancel₀ hp hx'').symm] at hx'
  /-
    case intro.intro.intro
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p q : M
    hcontra : Eq p q
    hp : Ne p 0
    x : M
    hx' : Not (IsUnit 1)
    hx'' : Eq (HMul.hMul p 1) (HMul.hMul p x)
    ⊢ False
  -/
  exact hx' isUnit_one
  /-
    🎉 no goals
  -/


theorem pow_injective_of_not_isUnit [CancelCommMonoidWithZero M] {q : M} (hq : ¬IsUnit q)
    (hq' : q ≠ 0) : Function.Injective fun n : ℕ => q ^ n := by
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    q : M
    hq : Not (IsUnit q)
    hq' : Ne q 0
    ⊢ Function.Injective fun n => HPow.hPow q n
  -/
  refine injective_of_lt_imp_ne fun n m h => DvdNotUnit.ne ⟨pow_ne_zero n hq', q ^ (m - n), ?_, ?_⟩
    /-
      case refine_1
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      q : M
      hq : Not (IsUnit q)
      hq' : Ne q 0
      n m : Nat
      h : LT.lt n m
      ⊢ Not (IsUnit (HPow.hPow q (HSub.hSub m n)))
    -/
  · exact not_isUnit_of_not_isUnit_dvd hq (dvd_pow (dvd_refl _) (Nat.sub_pos_of_lt h).ne')
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      q : M
      hq : Not (IsUnit q)
      hq' : Ne q 0
      n m : Nat
      h : LT.lt n m
      ⊢ Eq (HPow.hPow q m) (HMul.hMul (HPow.hPow q n) (HPow.hPow q (HSub.hSub m n)))
    -/
  · exact (pow_mul_pow_sub q h.le).symm
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-09-22")]
alias pow_injective_of_not_unit := pow_injective_of_not_isUnit


theorem pow_inj_of_not_isUnit [CancelCommMonoidWithZero M] {q : M} (hq : ¬IsUnit q)
    (hq' : q ≠ 0) {m n : ℕ} : q ^ m = q ^ n ↔ m = n :=
  (pow_injective_of_not_isUnit hq hq').eq_iff


