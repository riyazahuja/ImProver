/-- The divided power structure on an ideal I of a commutative ring A -/
structure DividedPowers where
  /-- The divided power function underlying a divided power structure -/
  dpow : ℕ → A → A
  dpow_null : ∀ {n x} (_ : x ∉ I), dpow n x = 0
  dpow_zero : ∀ {x} (_ : x ∈ I), dpow 0 x = 1
  dpow_one : ∀ {x} (_ : x ∈ I), dpow 1 x = x
  dpow_mem : ∀ {n x} (_ : n ≠ 0) (_ : x ∈ I), dpow n x ∈ I
  dpow_add : ∀ (n) {x y} (_ : x ∈ I) (_ : y ∈ I),
    dpow n (x + y) = (antidiagonal n).sum fun k ↦ dpow k.1 x * dpow k.2 y
  dpow_mul : ∀ (n) {a : A} {x} (_ : x ∈ I),
    dpow n (a * x) = a ^ n * dpow n x
  mul_dpow : ∀ (m n) {x} (_ : x ∈ I),
    dpow m x * dpow n x = choose (m + n) m * dpow (m + n) x
  dpow_comp : ∀ (m) {n x} (_ : n ≠ 0) (_ : x ∈ I),
    dpow m (dpow n x) = uniformBell m n * dpow (m * n) x


variable (A) in
/-- The canonical `DividedPowers` structure on the zero ideal -/
def dividedPowersBot [DecidableEq A] : DividedPowers (⊥ : Ideal A) where
  dpow n a := ite (a = 0 ∧ n = 0) 1 0
  dpow_null {n a} ha := by
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      n : Nat
      a : A
      ha : Not (Membership.mem Bot.bot a)
      ⊢ Eq ((fun n a => ite (And (Eq a 0) (Eq n 0)) 1 0) n a) 0
    -/
    simp only [mem_bot] at ha
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      n : Nat
      a : A
      ha : Not (Eq a 0)
      ⊢ Eq ((fun n a => ite (And (Eq a 0) (Eq n 0)) 1 0) n a) 0
    -/
    dsimp
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      n : Nat
      a : A
      ha : Not (Eq a 0)
      ⊢ Eq (ite (And (Eq a 0) (Eq n 0)) 1 0) 0
    -/
    rw [if_neg]
    /-
      case hnc
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      n : Nat
      a : A
      ha : Not (Eq a 0)
      ⊢ Not (And (Eq a 0) (Eq n 0))
    -/
    exact not_and_of_not_left (n = 0) ha
    /-
      🎉 no goals
    -/
  dpow_zero {a} ha := by
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      a : A
      ha : Membership.mem Bot.bot a
      ⊢ Eq ((fun n a => ite (And (Eq a 0) (Eq n 0)) 1 0) 0 a) 1
    -/
    rw [mem_bot.mp ha]
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      a : A
      ha : Membership.mem Bot.bot a
      ⊢ Eq ((fun n a => ite (And (Eq a 0) (Eq n 0)) 1 0) 0 0) 1
    -/
    simp only [and_self, ite_true]
    /-
      🎉 no goals
    -/
  dpow_one {a} ha := by
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      a : A
      ha : Membership.mem Bot.bot a
      ⊢ Eq ((fun n a => ite (And (Eq a 0) (Eq n 0)) 1 0) 1 a) a
    -/
    simp [mem_bot.mp ha]
    /-
      🎉 no goals
    -/
  dpow_mem {n a} hn _ := by
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      n : Nat
      a : A
      hn : Ne n 0
      x✝ : Membership.mem Bot.bot a
      ⊢ Membership.mem Bot.bot ((fun n a => ite (And (Eq a 0) (Eq n 0)) 1 0) n a)
    -/
    simp only [mem_bot, ite_eq_right_iff, and_imp]
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      n : Nat
      a : A
      hn : Ne n 0
      x✝ : Membership.mem Bot.bot a
      ⊢ Eq a 0 → Eq n 0 → Eq 1 0
    -/
    exact fun _ a ↦ False.elim (hn a)
    /-
      🎉 no goals
    -/
  dpow_add n a b ha hb := by
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      n : Nat
      a b : A
      ha : Membership.mem Bot.bot a
      hb : Membership.mem Bot.bot b
      ⊢ Eq ((fun n a => ite (And (Eq a 0) (Eq n 0)) 1 0) n (HAdd.hAdd a b)) ((Finset …
    -/
    rw [mem_bot.mp ha, mem_bot.mp hb, add_zero]
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      n : Nat
      a b : A
      ha : Membership.mem Bot.bot a
      hb : Membership.mem Bot.bot b
      ⊢ Eq ((fun n a => ite (And (Eq a 0) (Eq n 0)) 1 0) n 0) ((Finset.HasAntidiagon …
    -/
    simp only [true_and, ge_iff_le, tsub_eq_zero_iff_le, mul_ite, mul_one, mul_zero]
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      n : Nat
      a b : A
      ha : Membership.mem Bot.bot a
      hb : Membership.mem Bot.bot b
      ⊢ Eq (ite (Eq n 0) 1 0) ((Finset.HasAntidiagonal.antidiagonal n).sum fun x =>  …
    -/
    split_ifs with h
      /-
        case pos
        A : Type u_1
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : DecidableEq A
        n : Nat
        a b : A
        ha : Membership.mem Bot.bot a
        hb : Membership.mem Bot.bot b
        h : Eq n 0
        ⊢ Eq 1 ((Finset.HasAntidiagonal.antidiagonal n).sum fun x => ite (Eq x.2 0) (i …
      -/
    · simp [h]
      /-
        🎉 no goals
      -/
      /-
        case neg
        A : Type u_1
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : DecidableEq A
        n : Nat
        a b : A
        ha : Membership.mem Bot.bot a
        hb : Membership.mem Bot.bot b
        h : Not (Eq n 0)
        ⊢ Eq 0 ((Finset.HasAntidiagonal.antidiagonal n).sum fun x => ite (Eq x.2 0) (i …
      -/
    · symm
      /-
        case neg
        A : Type u_1
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : DecidableEq A
        n : Nat
        a b : A
        ha : Membership.mem Bot.bot a
        hb : Membership.mem Bot.bot b
        h : Not (Eq n 0)
        ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal n).sum fun x => ite (Eq x.2 0) (ite …
      -/
      apply sum_eq_zero
      /-
        case neg.h
        A : Type u_1
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : DecidableEq A
        n : Nat
        a b : A
        ha : Membership.mem Bot.bot a
        hb : Membership.mem Bot.bot b
        h : Not (Eq n 0)
        ⊢ ∀ (x : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal n) …
      -/
      intro i hi
      /-
        case neg.h
        A : Type u_1
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : DecidableEq A
        n : Nat
        a b : A
        ha : Membership.mem Bot.bot a
        hb : Membership.mem Bot.bot b
        h : Not (Eq n 0)
        i : Prod Nat Nat
        hi : Membership.mem (Finset.HasAntidiagonal.antidiagonal n) i
        ⊢ Eq (ite (Eq i.2 0) (ite (Eq i.1 0) 1 0) 0) 0
      -/
      simp only [mem_antidiagonal] at hi
      /-
        case neg.h
        A : Type u_1
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : DecidableEq A
        n : Nat
        a b : A
        ha : Membership.mem Bot.bot a
        hb : Membership.mem Bot.bot b
        h : Not (Eq n 0)
        i : Prod Nat Nat
        hi : Eq (HAdd.hAdd i.1 i.2) n
        ⊢ Eq (ite (Eq i.2 0) (ite (Eq i.1 0) 1 0) 0) 0
      -/
      split_ifs with h2 h1
        /-
          case pos
          A : Type u_1
          inst✝¹ : CommSemiring A
          I : Ideal A
          inst✝ : DecidableEq A
          n : Nat
          a b : A
          ha : Membership.mem Bot.bot a
          hb : Membership.mem Bot.bot b
          h : Not (Eq n 0)
          i : Prod Nat Nat
          hi : Eq (HAdd.hAdd i.1 i.2) n
          h2 : Eq i.2 0
          h1 : Eq i.1 0
          ⊢ Eq 1 0
        -/
      · rw [h1, h2, add_zero] at hi
        /-
          case pos
          A : Type u_1
          inst✝¹ : CommSemiring A
          I : Ideal A
          inst✝ : DecidableEq A
          n : Nat
          a b : A
          ha : Membership.mem Bot.bot a
          hb : Membership.mem Bot.bot b
          h : Not (Eq n 0)
          i : Prod Nat Nat
          hi : Eq 0 n
          h2 : Eq i.2 0
          h1 : Eq i.1 0
          ⊢ Eq 1 0
        -/
        exfalso
        /-
          case pos
          A : Type u_1
          inst✝¹ : CommSemiring A
          I : Ideal A
          inst✝ : DecidableEq A
          n : Nat
          a b : A
          ha : Membership.mem Bot.bot a
          hb : Membership.mem Bot.bot b
          h : Not (Eq n 0)
          i : Prod Nat Nat
          hi : Eq 0 n
          h2 : Eq i.2 0
          h1 : Eq i.1 0
          ⊢ False
        -/
        exact h hi.symm
        /-
          🎉 no goals
        -/
        /-
          case neg
          A : Type u_1
          inst✝¹ : CommSemiring A
          I : Ideal A
          inst✝ : DecidableEq A
          n : Nat
          a b : A
          ha : Membership.mem Bot.bot a
          hb : Membership.mem Bot.bot b
          h : Not (Eq n 0)
          i : Prod Nat Nat
          hi : Eq (HAdd.hAdd i.1 i.2) n
          h2 : Eq i.2 0
          h1 : Not (Eq i.1 0)
          ⊢ Eq 0 0
        -/
      · rfl
        /-
          🎉 no goals
        -/
        /-
          case neg
          A : Type u_1
          inst✝¹ : CommSemiring A
          I : Ideal A
          inst✝ : DecidableEq A
          n : Nat
          a b : A
          ha : Membership.mem Bot.bot a
          hb : Membership.mem Bot.bot b
          h : Not (Eq n 0)
          i : Prod Nat Nat
          hi : Eq (HAdd.hAdd i.1 i.2) n
          h2 : Not (Eq i.2 0)
          ⊢ Eq 0 0
        -/
      · rfl
        /-
          🎉 no goals
        -/
  dpow_mul n {a x} hx := by
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      n : Nat
      a x : A
      hx : Membership.mem Bot.bot x
      ⊢ Eq ((fun n a => ite (And (Eq a 0) (Eq n 0)) 1 0) n (HMul.hMul a x)) (HMul.hM …
    -/
    rw [mem_bot.mp hx]
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      n : Nat
      a x : A
      hx : Membership.mem Bot.bot x
      ⊢ Eq ((fun n a => ite (And (Eq a 0) (Eq n 0)) 1 0) n (HMul.hMul a 0)) (HMul.hM …
    -/
    simp only [mul_zero, true_and, mul_ite, mul_one]
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      n : Nat
      a x : A
      hx : Membership.mem Bot.bot x
      ⊢ Eq (ite (Eq n 0) 1 0) (ite (Eq n 0) (HPow.hPow a n) 0)
    -/
    by_cases hn : n = 0
      /-
        case pos
        A : Type u_1
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : DecidableEq A
        n : Nat
        a x : A
        hx : Membership.mem Bot.bot x
        hn : Eq n 0
        ⊢ Eq (ite (Eq n 0) 1 0) (ite (Eq n 0) (HPow.hPow a n) 0)
      -/
    · rw [if_pos hn, hn, if_pos rfl, _root_.pow_zero]
      /-
        🎉 no goals
      -/
      /-
        case neg
        A : Type u_1
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : DecidableEq A
        n : Nat
        a x : A
        hx : Membership.mem Bot.bot x
        hn : Not (Eq n 0)
        ⊢ Eq (ite (Eq n 0) 1 0) (ite (Eq n 0) (HPow.hPow a n) 0)
      -/
    · simp only [if_neg hn]
      /-
        🎉 no goals
      -/
  mul_dpow m n {x} hx := by
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      m n : Nat
      x : A
      hx : Membership.mem Bot.bot x
      ⊢ Eq (HMul.hMul ((fun n a => ite (And (Eq a 0) (Eq n 0)) 1 0) m x) ((fun n a = …
    -/
    rw [mem_bot.mp hx]
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      m n : Nat
      x : A
      hx : Membership.mem Bot.bot x
      ⊢ Eq (HMul.hMul ((fun n a => ite (And (Eq a 0) (Eq n 0)) 1 0) m 0) ((fun n a = …
    -/
    simp only [true_and, mul_ite, mul_one, mul_zero, add_eq_zero]
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      m n : Nat
      x : A
      hx : Membership.mem Bot.bot x
      ⊢ Eq (ite (Eq n 0) (ite (Eq m 0) 1 0) 0) (ite (And (Eq m 0) (Eq n 0)) (↑((HAdd …
    -/
    by_cases hn : n = 0
      /-
        case pos
        A : Type u_1
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : DecidableEq A
        m n : Nat
        x : A
        hx : Membership.mem Bot.bot x
        hn : Eq n 0
        ⊢ Eq (ite (Eq n 0) (ite (Eq m 0) 1 0) 0) (ite (And (Eq m 0) (Eq n 0)) (↑((HAdd …
      -/
    · simp only [hn, ite_true, and_true, add_zero, choose_self, cast_one]
      /-
        🎉 no goals
      -/
      /-
        case neg
        A : Type u_1
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : DecidableEq A
        m n : Nat
        x : A
        hx : Membership.mem Bot.bot x
        hn : Not (Eq n 0)
        ⊢ Eq (ite (Eq n 0) (ite (Eq m 0) 1 0) 0) (ite (And (Eq m 0) (Eq n 0)) (↑((HAdd …
      -/
    · rw [if_neg hn, if_neg]
      /-
        case neg.hnc
        A : Type u_1
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : DecidableEq A
        m n : Nat
        x : A
        hx : Membership.mem Bot.bot x
        hn : Not (Eq n 0)
        ⊢ Not (And (Eq m 0) (Eq n 0))
      -/
      exact not_and_of_not_right (m = 0) hn
      /-
        🎉 no goals
      -/
  dpow_comp m {n a} hn ha := by
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      m n : Nat
      a : A
      hn : Ne n 0
      ha : Membership.mem Bot.bot a
      ⊢ Eq ((fun n a => ite (And (Eq a 0) (Eq n 0)) 1 0) m ((fun n a => ite (And (Eq …
    -/
    rw [mem_bot.mp ha]
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      m n : Nat
      a : A
      hn : Ne n 0
      ha : Membership.mem Bot.bot a
      ⊢ Eq ((fun n a => ite (And (Eq a 0) (Eq n 0)) 1 0) m ((fun n a => ite (And (Eq …
    -/
    simp only [true_and, ite_eq_right_iff, _root_.mul_eq_zero, mul_ite, mul_one, mul_zero]
    /-
      A : Type u_1
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : DecidableEq A
      m n : Nat
      a : A
      hn : Ne n 0
      ha : Membership.mem Bot.bot a
      ⊢ Eq (ite (And (Eq n 0 → Eq 1 0) (Eq m 0)) 1 0) (ite (Or (Eq m 0) (Eq n 0)) (↑ …
    -/
    by_cases hm: m = 0
      /-
        case pos
        A : Type u_1
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : DecidableEq A
        m n : Nat
        a : A
        hn : Ne n 0
        ha : Membership.mem Bot.bot a
        hm : Eq m 0
        ⊢ Eq (ite (And (Eq n 0 → Eq 1 0) (Eq m 0)) 1 0) (ite (Or (Eq m 0) (Eq n 0)) (↑ …
      -/
    · simp only [hm, and_true, true_or, ite_true, uniformBell_zero_left, cast_one]
      /-
        case pos
        A : Type u_1
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : DecidableEq A
        m n : Nat
        a : A
        hn : Ne n 0
        ha : Membership.mem Bot.bot a
        hm : Eq m 0
        ⊢ Eq (ite (Eq n 0 → Eq 1 0) 1 0) 1
      -/
      rw [if_pos]
      /-
        case pos.hc
        A : Type u_1
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : DecidableEq A
        m n : Nat
        a : A
        hn : Ne n 0
        ha : Membership.mem Bot.bot a
        hm : Eq m 0
        ⊢ Eq n 0 → Eq 1 0
      -/
      exact fun h ↦ False.elim (hn h)
      /-
        🎉 no goals
      -/
      /-
        case neg
        A : Type u_1
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : DecidableEq A
        m n : Nat
        a : A
        hn : Ne n 0
        ha : Membership.mem Bot.bot a
        hm : Not (Eq m 0)
        ⊢ Eq (ite (And (Eq n 0 → Eq 1 0) (Eq m 0)) 1 0) (ite (Or (Eq m 0) (Eq n 0)) (↑ …
      -/
    · simp only [hm, and_false, ite_false, false_or, if_neg hn]
      /-
        🎉 no goals
      -/


instance [DecidableEq A] : Inhabited (DividedPowers (⊥ : Ideal A)) :=
  ⟨dividedPowersBot A⟩


/-- The coercion from the divided powers structures to functions -/
instance : CoeFun (DividedPowers I) fun _ ↦ ℕ → A → A := ⟨fun hI ↦ hI.dpow⟩


variable {I} in
@[ext]
theorem DividedPowers.ext (hI : DividedPowers I) (hI' : DividedPowers I)
    (h_eq : ∀ (n : ℕ) {x : A} (_ : x ∈ I), hI.dpow n x = hI'.dpow n x) :
    hI = hI' := by
  /-
    A : Type u_1
    inst✝ : CommSemiring A
    I : Ideal A
    hI hI' : DividedPowers I
    h_eq : ∀ (n : Nat) {x : A}, Membership.mem I x → Eq (hI.dpow n x) (hI'.dpow n x)
    ⊢ Eq hI hI'
  -/
  obtain ⟨hI, h₀, _⟩ := hI
  /-
    case mk
    A : Type u_1
    inst✝ : CommSemiring A
    I : Ideal A
    hI' : DividedPowers I
    hI : Nat → A → A
    h₀ : ∀ {n : Nat} {x : A}, Not (Membership.mem I x) → Eq (hI n x) 0
    dpow_zero✝ : ∀ {x : A}, Membership.mem I x → Eq (hI 0 x) 1
    dpow_one✝ : ∀ {x : A}, Membership.mem I x → Eq (hI 1 x) x
    dpow_mem✝ : ∀ {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Membership.mem  …
    dpow_add✝ : ∀ (n : Nat) {x y : A}, Membership.mem I x → Membership.mem I y → E …
    dpow_mul✝ : ∀ (n : Nat) {a x : A}, Membership.mem I x → Eq (hI n (HMul.hMul a  …
    mul_dpow✝ : ∀ (m n : Nat) {x : A}, Membership.mem I x → Eq (HMul.hMul (hI m x) …
    dpow_comp✝ : ∀ (m : Nat) {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Eq ( …
    h_eq : ∀ (n : Nat) {x : A}, Membership.mem I x → Eq ({ dpow := hI, dpow_null : …
    ⊢ Eq { dpow := hI, dpow_null := h₀, dpow_zero := dpow_zero✝, dpow_one := dpow_ …
  -/
  obtain ⟨hI', h₀', _⟩ := hI'
  /-
    case mk.mk
    A : Type u_1
    inst✝ : CommSemiring A
    I : Ideal A
    hI : Nat → A → A
    h₀ : ∀ {n : Nat} {x : A}, Not (Membership.mem I x) → Eq (hI n x) 0
    dpow_zero✝¹ : ∀ {x : A}, Membership.mem I x → Eq (hI 0 x) 1
    dpow_one✝¹ : ∀ {x : A}, Membership.mem I x → Eq (hI 1 x) x
    dpow_mem✝¹ : ∀ {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Membership.mem …
    dpow_add✝¹ : ∀ (n : Nat) {x y : A}, Membership.mem I x → Membership.mem I y →  …
    dpow_mul✝¹ : ∀ (n : Nat) {a x : A}, Membership.mem I x → Eq (hI n (HMul.hMul a …
    mul_dpow✝¹ : ∀ (m n : Nat) {x : A}, Membership.mem I x → Eq (HMul.hMul (hI m x …
    dpow_comp✝¹ : ∀ (m : Nat) {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Eq  …
    hI' : Nat → A → A
    h₀' : ∀ {n : Nat} {x : A}, Not (Membership.mem I x) → Eq (hI' n x) 0
    dpow_zero✝ : ∀ {x : A}, Membership.mem I x → Eq (hI' 0 x) 1
    dpow_one✝ : ∀ {x : A}, Membership.mem I x → Eq (hI' 1 x) x
    dpow_mem✝ : ∀ {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Membership.mem  …
    dpow_add✝ : ∀ (n : Nat) {x y : A}, Membership.mem I x → Membership.mem I y → E …
    dpow_mul✝ : ∀ (n : Nat) {a x : A}, Membership.mem I x → Eq (hI' n (HMul.hMul a …
    mul_dpow✝ : ∀ (m n : Nat) {x : A}, Membership.mem I x → Eq (HMul.hMul (hI' m x …
    dpow_comp✝ : ∀ (m : Nat) {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Eq ( …
    h_eq : ∀ (n : Nat) {x : A}, Membership.mem I x → Eq ({ dpow := hI, dpow_null : …
    ⊢ Eq { dpow := hI, dpow_null := h₀, dpow_zero := dpow_zero✝¹, dpow_one := dpow …
  -/
  simp only [mk.injEq]
  /-
    case mk.mk
    A : Type u_1
    inst✝ : CommSemiring A
    I : Ideal A
    hI : Nat → A → A
    h₀ : ∀ {n : Nat} {x : A}, Not (Membership.mem I x) → Eq (hI n x) 0
    dpow_zero✝¹ : ∀ {x : A}, Membership.mem I x → Eq (hI 0 x) 1
    dpow_one✝¹ : ∀ {x : A}, Membership.mem I x → Eq (hI 1 x) x
    dpow_mem✝¹ : ∀ {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Membership.mem …
    dpow_add✝¹ : ∀ (n : Nat) {x y : A}, Membership.mem I x → Membership.mem I y →  …
    dpow_mul✝¹ : ∀ (n : Nat) {a x : A}, Membership.mem I x → Eq (hI n (HMul.hMul a …
    mul_dpow✝¹ : ∀ (m n : Nat) {x : A}, Membership.mem I x → Eq (HMul.hMul (hI m x …
    dpow_comp✝¹ : ∀ (m : Nat) {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Eq  …
    hI' : Nat → A → A
    h₀' : ∀ {n : Nat} {x : A}, Not (Membership.mem I x) → Eq (hI' n x) 0
    dpow_zero✝ : ∀ {x : A}, Membership.mem I x → Eq (hI' 0 x) 1
    dpow_one✝ : ∀ {x : A}, Membership.mem I x → Eq (hI' 1 x) x
    dpow_mem✝ : ∀ {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Membership.mem  …
    dpow_add✝ : ∀ (n : Nat) {x y : A}, Membership.mem I x → Membership.mem I y → E …
    dpow_mul✝ : ∀ (n : Nat) {a x : A}, Membership.mem I x → Eq (hI' n (HMul.hMul a …
    mul_dpow✝ : ∀ (m n : Nat) {x : A}, Membership.mem I x → Eq (HMul.hMul (hI' m x …
    dpow_comp✝ : ∀ (m : Nat) {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Eq ( …
    h_eq : ∀ (n : Nat) {x : A}, Membership.mem I x → Eq ({ dpow := hI, dpow_null : …
    ⊢ Eq hI hI'
  -/
  ext n x
  /-
    case mk.mk.h.h
    A : Type u_1
    inst✝ : CommSemiring A
    I : Ideal A
    hI : Nat → A → A
    h₀ : ∀ {n : Nat} {x : A}, Not (Membership.mem I x) → Eq (hI n x) 0
    dpow_zero✝¹ : ∀ {x : A}, Membership.mem I x → Eq (hI 0 x) 1
    dpow_one✝¹ : ∀ {x : A}, Membership.mem I x → Eq (hI 1 x) x
    dpow_mem✝¹ : ∀ {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Membership.mem …
    dpow_add✝¹ : ∀ (n : Nat) {x y : A}, Membership.mem I x → Membership.mem I y →  …
    dpow_mul✝¹ : ∀ (n : Nat) {a x : A}, Membership.mem I x → Eq (hI n (HMul.hMul a …
    mul_dpow✝¹ : ∀ (m n : Nat) {x : A}, Membership.mem I x → Eq (HMul.hMul (hI m x …
    dpow_comp✝¹ : ∀ (m : Nat) {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Eq  …
    hI' : Nat → A → A
    h₀' : ∀ {n : Nat} {x : A}, Not (Membership.mem I x) → Eq (hI' n x) 0
    dpow_zero✝ : ∀ {x : A}, Membership.mem I x → Eq (hI' 0 x) 1
    dpow_one✝ : ∀ {x : A}, Membership.mem I x → Eq (hI' 1 x) x
    dpow_mem✝ : ∀ {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Membership.mem  …
    dpow_add✝ : ∀ (n : Nat) {x y : A}, Membership.mem I x → Membership.mem I y → E …
    dpow_mul✝ : ∀ (n : Nat) {a x : A}, Membership.mem I x → Eq (hI' n (HMul.hMul a …
    mul_dpow✝ : ∀ (m n : Nat) {x : A}, Membership.mem I x → Eq (HMul.hMul (hI' m x …
    dpow_comp✝ : ∀ (m : Nat) {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Eq ( …
    h_eq : ∀ (n : Nat) {x : A}, Membership.mem I x → Eq ({ dpow := hI, dpow_null : …
    n : Nat
    x : A
    ⊢ Eq (hI n x) (hI' n x)
  -/
  by_cases hx : x ∈ I
    /-
      case pos
      A : Type u_1
      inst✝ : CommSemiring A
      I : Ideal A
      hI : Nat → A → A
      h₀ : ∀ {n : Nat} {x : A}, Not (Membership.mem I x) → Eq (hI n x) 0
      dpow_zero✝¹ : ∀ {x : A}, Membership.mem I x → Eq (hI 0 x) 1
      dpow_one✝¹ : ∀ {x : A}, Membership.mem I x → Eq (hI 1 x) x
      dpow_mem✝¹ : ∀ {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Membership.mem …
      dpow_add✝¹ : ∀ (n : Nat) {x y : A}, Membership.mem I x → Membership.mem I y →  …
      dpow_mul✝¹ : ∀ (n : Nat) {a x : A}, Membership.mem I x → Eq (hI n (HMul.hMul a …
      mul_dpow✝¹ : ∀ (m n : Nat) {x : A}, Membership.mem I x → Eq (HMul.hMul (hI m x …
      dpow_comp✝¹ : ∀ (m : Nat) {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Eq  …
      hI' : Nat → A → A
      h₀' : ∀ {n : Nat} {x : A}, Not (Membership.mem I x) → Eq (hI' n x) 0
      dpow_zero✝ : ∀ {x : A}, Membership.mem I x → Eq (hI' 0 x) 1
      dpow_one✝ : ∀ {x : A}, Membership.mem I x → Eq (hI' 1 x) x
      dpow_mem✝ : ∀ {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Membership.mem  …
      dpow_add✝ : ∀ (n : Nat) {x y : A}, Membership.mem I x → Membership.mem I y → E …
      dpow_mul✝ : ∀ (n : Nat) {a x : A}, Membership.mem I x → Eq (hI' n (HMul.hMul a …
      mul_dpow✝ : ∀ (m n : Nat) {x : A}, Membership.mem I x → Eq (HMul.hMul (hI' m x …
      dpow_comp✝ : ∀ (m : Nat) {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Eq ( …
      h_eq : ∀ (n : Nat) {x : A}, Membership.mem I x → Eq ({ dpow := hI, dpow_null : …
      n : Nat
      x : A
      hx : Membership.mem I x
      ⊢ Eq (hI n x) (hI' n x)
    -/
  · exact h_eq n hx
    /-
      🎉 no goals
    -/
    /-
      case neg
      A : Type u_1
      inst✝ : CommSemiring A
      I : Ideal A
      hI : Nat → A → A
      h₀ : ∀ {n : Nat} {x : A}, Not (Membership.mem I x) → Eq (hI n x) 0
      dpow_zero✝¹ : ∀ {x : A}, Membership.mem I x → Eq (hI 0 x) 1
      dpow_one✝¹ : ∀ {x : A}, Membership.mem I x → Eq (hI 1 x) x
      dpow_mem✝¹ : ∀ {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Membership.mem …
      dpow_add✝¹ : ∀ (n : Nat) {x y : A}, Membership.mem I x → Membership.mem I y →  …
      dpow_mul✝¹ : ∀ (n : Nat) {a x : A}, Membership.mem I x → Eq (hI n (HMul.hMul a …
      mul_dpow✝¹ : ∀ (m n : Nat) {x : A}, Membership.mem I x → Eq (HMul.hMul (hI m x …
      dpow_comp✝¹ : ∀ (m : Nat) {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Eq  …
      hI' : Nat → A → A
      h₀' : ∀ {n : Nat} {x : A}, Not (Membership.mem I x) → Eq (hI' n x) 0
      dpow_zero✝ : ∀ {x : A}, Membership.mem I x → Eq (hI' 0 x) 1
      dpow_one✝ : ∀ {x : A}, Membership.mem I x → Eq (hI' 1 x) x
      dpow_mem✝ : ∀ {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Membership.mem  …
      dpow_add✝ : ∀ (n : Nat) {x y : A}, Membership.mem I x → Membership.mem I y → E …
      dpow_mul✝ : ∀ (n : Nat) {a x : A}, Membership.mem I x → Eq (hI' n (HMul.hMul a …
      mul_dpow✝ : ∀ (m n : Nat) {x : A}, Membership.mem I x → Eq (HMul.hMul (hI' m x …
      dpow_comp✝ : ∀ (m : Nat) {n : Nat} {x : A}, Ne n 0 → Membership.mem I x → Eq ( …
      h_eq : ∀ (n : Nat) {x : A}, Membership.mem I x → Eq ({ dpow := hI, dpow_null : …
      n : Nat
      x : A
      hx : Not (Membership.mem I x)
      ⊢ Eq (hI n x) (hI' n x)
    -/
  · rw [h₀ hx, h₀' hx]
    /-
      🎉 no goals
    -/


theorem DividedPowers.coe_injective :
    Function.Injective (fun (h : DividedPowers I) ↦ (h : ℕ → A → A)) := fun hI hI' h ↦ by
  /-
    A : Type u_1
    inst✝ : CommSemiring A
    I : Ideal A
    hI hI' : DividedPowers I
    h : Eq ((fun h => h.dpow) hI) ((fun h => h.dpow) hI')
    ⊢ Eq hI hI'
  -/
  ext n x
  /-
    case h_eq
    A : Type u_1
    inst✝ : CommSemiring A
    I : Ideal A
    hI hI' : DividedPowers I
    h : Eq ((fun h => h.dpow) hI) ((fun h => h.dpow) hI')
    n : Nat
    x : A
    x✝ : Membership.mem I x
    ⊢ Eq (hI.dpow n x) (hI'.dpow n x)
  -/
  exact congr_fun (congr_fun h n) x
  /-
    🎉 no goals
  -/


/-- Variant of `DividedPowers.dpow_add` with a sum on `range (n + 1)` -/
theorem dpow_add' (hI : DividedPowers I) (n : ℕ) (ha : a ∈ I) (hb : b ∈ I) :
    hI.dpow n (a + b) = (range (n + 1)).sum fun k ↦ hI.dpow k a * hI.dpow (n - k) b := by
  /-
    A : Type u_1
    inst✝ : CommSemiring A
    I : Ideal A
    a b : A
    hI : DividedPowers I
    n : Nat
    ha : Membership.mem I a
    hb : Membership.mem I b
    ⊢ Eq (hI.dpow n (HAdd.hAdd a b)) ((Finset.range (HAdd.hAdd n 1)).sum fun k =>  …
  -/
  rw [hI.dpow_add n ha hb, sum_antidiagonal_eq_sum_range_succ_mk]
  /-
    🎉 no goals
  -/


/-- The exponential series of an element in the context of divided powers,
`Σ (dpow n a) X ^ n` -/
def exp (hI : DividedPowers I) (a : A) : PowerSeries A :=
  PowerSeries.mk fun n ↦ hI.dpow n a


/-- A more general of `DividedPowers.exp_add` -/
theorem exp_add' (dp : ℕ → A → A)
    (dp_add : ∀ n, dp n (a + b) = (antidiagonal n).sum fun k ↦ dp k.1 a * dp k.2 b) :
    PowerSeries.mk (fun n ↦ dp n (a + b)) =
      (PowerSeries.mk fun n ↦ dp n a) * (PowerSeries.mk fun n ↦ dp n b) := by
  /-
    A : Type u_1
    inst✝ : CommSemiring A
    a b : A
    dp : Nat → A → A
    dp_add : ∀ (n : Nat), Eq (dp n (HAdd.hAdd a b)) ((Finset.HasAntidiagonal.antid …
    ⊢ Eq (PowerSeries.mk fun n => dp n (HAdd.hAdd a b)) (HMul.hMul (PowerSeries.mk …
  -/
  ext n
  simp only [exp, PowerSeries.coeff_mk, PowerSeries.coeff_mul, dp_add n,
    sum_antidiagonal_eq_sum_range_succ_mk]


theorem exp_add (hI : DividedPowers I) (ha : a ∈ I) (hb : b ∈ I) :
    hI.exp (a + b) = hI.exp a * hI.exp b :=
  exp_add' _ (fun n ↦ hI.dpow_add n ha hb)


theorem dpow_smul (n : ℕ) (ha : a ∈ I) :
    hI.dpow n (b • a) = b ^ n • hI.dpow n a := by
  /-
    A : Type u_1
    inst✝ : CommSemiring A
    I : Ideal A
    a b : A
    hI : DividedPowers I
    n : Nat
    ha : Membership.mem I a
    ⊢ Eq (hI.dpow n (HSMul.hSMul b a)) (HSMul.hSMul (HPow.hPow b n) (hI.dpow n a))
  -/
  simp only [smul_eq_mul, hI.dpow_mul, ha]
  /-
    🎉 no goals
  -/


theorem dpow_mul_right (n : ℕ) (ha : a ∈ I) :
    hI.dpow n (a * b) = hI.dpow n a * b ^ n := by
  /-
    A : Type u_1
    inst✝ : CommSemiring A
    I : Ideal A
    a b : A
    hI : DividedPowers I
    n : Nat
    ha : Membership.mem I a
    ⊢ Eq (hI.dpow n (HMul.hMul a b)) (HMul.hMul (hI.dpow n a) (HPow.hPow b n))
  -/
  rw [mul_comm, hI.dpow_mul n ha, mul_comm]
  /-
    🎉 no goals
  -/


theorem dpow_smul_right (n : ℕ) (ha : a ∈ I) :
    hI.dpow n (a • b) = hI.dpow n a • b ^ n := by
  /-
    A : Type u_1
    inst✝ : CommSemiring A
    I : Ideal A
    a b : A
    hI : DividedPowers I
    n : Nat
    ha : Membership.mem I a
    ⊢ Eq (hI.dpow n (HSMul.hSMul a b)) (HSMul.hSMul (hI.dpow n a) (HPow.hPow b n))
  -/
  rw [smul_eq_mul, hI.dpow_mul_right n ha, smul_eq_mul]
  /-
    🎉 no goals
  -/


theorem factorial_mul_dpow_eq_pow (n : ℕ) (ha : a ∈ I) :
    (n ! : A) * hI.dpow n a = a ^ n := by
  induction n with
  | zero => rw [factorial_zero, cast_one, one_mul, pow_zero, hI.dpow_zero ha]
  | succ n ih =>
    rw [factorial_succ, mul_comm (n + 1)]
    nth_rewrite 1 [← (n + 1).choose_one_right]
    rw [← choose_symm_add, cast_mul, mul_assoc,
      ← hI.mul_dpow n 1 ha, ← mul_assoc, ih, hI.dpow_one ha, pow_succ, mul_comm]


theorem dpow_eval_zero {n : ℕ} (hn : n ≠ 0) : hI.dpow n 0 = 0 := by
  rw [← MulZeroClass.mul_zero (0 : A), hI.dpow_mul n I.zero_mem,
    zero_pow hn, zero_mul, zero_mul]


/-- If an element of a divided power ideal is killed by multiplication
by some nonzero integer `n`, then its `n`th power is zero.

Proposition 1.2.7 of [Berthelot-1974], part (i). -/
theorem nilpotent_of_mem_dpIdeal {n : ℕ} (hn : n ≠ 0) (hnI : ∀ {y} (_ : y ∈ I), n • y = 0)
    (hI : DividedPowers I) (ha : a ∈ I) : a ^ n = 0 := by
  have h_fac : (n ! : A) * hI.dpow n a =
    n • ((n - 1)! : A) * hI.dpow n a := by
    rw [nsmul_eq_mul, ← cast_mul, mul_factorial_pred (Nat.pos_of_ne_zero hn)]
  /-
    A : Type u_1
    inst✝ : CommSemiring A
    I : Ideal A
    a : A
    n : Nat
    hn : Ne n 0
    hnI : ∀ {y : A}, Membership.mem I y → Eq (HSMul.hSMul n y) 0
    hI : DividedPowers I
    ha : Membership.mem I a
    h_fac : Eq (HMul.hMul (↑n.factorial) (hI.dpow n a)) (HMul.hMul (HSMul.hSMul n  …
    ⊢ Eq (HPow.hPow a n) 0
  -/
  rw [← hI.factorial_mul_dpow_eq_pow n ha, h_fac, smul_mul_assoc]
  /-
    A : Type u_1
    inst✝ : CommSemiring A
    I : Ideal A
    a : A
    n : Nat
    hn : Ne n 0
    hnI : ∀ {y : A}, Membership.mem I y → Eq (HSMul.hSMul n y) 0
    hI : DividedPowers I
    ha : Membership.mem I a
    h_fac : Eq (HMul.hMul (↑n.factorial) (hI.dpow n a)) (HMul.hMul (HSMul.hSMul n  …
    ⊢ Eq (HSMul.hSMul n (HMul.hMul (↑(HSub.hSub n 1).factorial) (hI.dpow n a))) 0
  -/
  exact hnI (I.mul_mem_left ((n - 1)! : A) (hI.dpow_mem hn ha))
  /-
    🎉 no goals
  -/


/-- If J is another ideal of A with divided powers,
then the divided powers of I and J coincide on I • J

[Berthelot-1974], 1.6.1 (ii) -/
theorem coincide_on_smul {J : Ideal A} (hJ : DividedPowers J) {n : ℕ} (ha : a ∈ I • J) :
    hI.dpow n a = hJ.dpow n a := by
  induction ha using Submodule.smul_induction_on' generalizing n with
  | smul a ha b hb =>
    rw [Algebra.id.smul_eq_mul, hJ.dpow_mul n hb, mul_comm a b, hI.dpow_mul n ha, ←
      hJ.factorial_mul_dpow_eq_pow n hb, ← hI.factorial_mul_dpow_eq_pow n ha]
    ring
  | add x hx y hy hx' hy' =>
    rw [hI.dpow_add n (mul_le_right hx) (mul_le_right hy),
      hJ.dpow_add n (mul_le_left hx) (mul_le_left hy)]
    apply sum_congr rfl
    intro k _
    rw [hx', hy']


/-- A product of divided powers is a multinomial coefficient times the divided power

[Roby-1965], formula (III') -/
theorem prod_dpow {ι : Type*} {s : Finset ι} (n : ι → ℕ) (ha : a ∈ I) :
    (s.prod fun i ↦ hI.dpow (n i) a) = multinomial s n * hI.dpow (s.sum n) a := by
  classical
  induction s using Finset.induction with
  | empty =>
    simp only [prod_empty, multinomial_empty, cast_one, sum_empty, one_mul]
    rw [hI.dpow_zero ha]
  | insert hi hrec =>
    rw [prod_insert hi, hrec, ← mul_assoc, mul_comm (hI.dpow (n _) a),
      mul_assoc, mul_dpow _ _ _ ha, ← sum_insert hi, ← mul_assoc]
    apply congr_arg₂ _ _ rfl
    rw [multinomial_insert hi, mul_comm, cast_mul, sum_insert hi]

-- TODO : can probably be simplified using `DividedPowers.exp`


/-- Lemma towards `dpow_sum` when we only have partial information on a divided power ideal -/
theorem dpow_sum' {M : Type*} [AddCommMonoid M] {I : AddSubmonoid M} (dpow : ℕ → M → A)
    (dpow_zero : ∀ {x} (_ : x ∈ I), dpow 0 x = 1)
    (dpow_add : ∀ (n) {x y} (_ : x ∈ I) (_ : y ∈ I),
      dpow n (x + y) = (antidiagonal n).sum fun k ↦ dpow k.1 x * dpow k.2 y)
    (dpow_eval_zero : ∀ {n : ℕ} (_ : n ≠ 0), dpow n 0 = 0)
    {ι : Type*} [DecidableEq ι] {s : Finset ι} {x : ι → M} (hx : ∀ i ∈ s, x i ∈ I) (n : ℕ) :
    dpow n (s.sum x) = (s.sym n).sum fun k ↦ s.prod fun i ↦ dpow (Multiset.count i k) (x i) := by
  /-
    A : Type u_1
    inst✝² : CommSemiring A
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    I : AddSubmonoid M
    dpow : Nat → M → A
    dpow_zero : ∀ {x : M}, Membership.mem I x → Eq (dpow 0 x) 1
    dpow_add : ∀ (n : Nat) {x y : M}, Membership.mem I x → Membership.mem I y → Eq …
    dpow_eval_zero : ∀ {n : Nat}, Ne n 0 → Eq (dpow n 0) 0
    ι : Type u_3
    inst✝ : DecidableEq ι
    s : Finset ι
    x : ι → M
    hx : ∀ (i : ι), Membership.mem s i → Membership.mem I (x i)
    n : Nat
    ⊢ Eq (dpow n (s.sum x)) ((s.sym n).sum fun k => s.prod fun i => dpow (Multiset …
  -/
  simp only [sum_antidiagonal_eq_sum_range_succ_mk] at dpow_add
  induction s using Finset.induction generalizing n with
  | empty =>
    simp only [sum_empty, prod_empty, sum_const, nsmul_eq_mul, mul_one]
    by_cases hn : n = 0
    · rw [hn]
      rw [dpow_zero I.zero_mem]
      simp only [sym_zero, card_singleton, cast_one]
    · rw [dpow_eval_zero hn, eq_comm, ← cast_zero]
      apply congr_arg
      rw [card_eq_zero, sym_eq_empty]
      exact ⟨hn, rfl⟩
  | @insert a s ha ih =>
    -- This should be golfable using `Finset.symInsertEquiv`
    have hx' : ∀ i, i ∈ s → x i ∈ I := fun i hi ↦ hx i (mem_insert_of_mem hi)
    simp_rw [sum_insert ha,
      dpow_add n (hx a (mem_insert_self a s)) (I.sum_mem fun i ↦ hx' i),
      sum_range, ih hx', mul_sum, sum_sigma', eq_comm]
    apply sum_bij'
      (fun m _ ↦ m.filterNe a)
      (fun m _ ↦ m.2.fill a m.1)
      (fun m hm ↦ mem_sigma.2 ⟨mem_univ _, _⟩)
      (fun m hm ↦ by
        simp only [succ_eq_add_one, mem_sym_iff, mem_insert, Sym.mem_fill_iff]
        simp only [mem_sigma, mem_univ, mem_sym_iff, true_and] at hm
        intro b
        apply Or.imp (fun h ↦ h.2) (fun h ↦ hm b h))
      (fun m _ ↦ m.fill_filterNe a)
    · intro m hm
      simp only [mem_sigma, mem_univ, mem_sym_iff, true_and] at hm
      exact Sym.filter_ne_fill a m fun a_1 ↦ ha (hm a a_1)
    · intro m hm
      simp only [mem_sym_iff, mem_insert] at hm
      rw [prod_insert ha]
      apply congr_arg₂ _ rfl
      apply prod_congr rfl
      intro i hi
      apply congr_arg₂ _ _ rfl
      conv_lhs => rw [← m.fill_filterNe a]
      exact Sym.count_coe_fill_of_ne (ne_of_mem_of_not_mem hi ha)
    · intro m hm
      convert sym_filterNe_mem a hm
      rw [erase_insert ha]


/-- A “multinomial” theorem for divided powers — without multinomial coefficients -/
theorem dpow_sum {ι : Type*} [DecidableEq ι] {s : Finset ι} {x : ι → A}
    (hx : ∀ i ∈ s, x i ∈ I) (n : ℕ) :
    hI.dpow n (s.sum x) =
      (s.sym n).sum fun k ↦ s.prod fun i ↦ hI.dpow (Multiset.count i k) (x i) :=
  dpow_sum' hI.dpow hI.dpow_zero hI.dpow_add hI.dpow_eval_zero hx n


/-- Transfer divided powers under an equivalence -/
def ofRingEquiv (hI : DividedPowers I) : DividedPowers J where
  dpow n b := e (hI.dpow n (e.symm b))
  dpow_null {n} {x} hx := by
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      n : Nat
      x : B
      hx : Not (Membership.mem J x)
      ⊢ Eq ((fun n b => e (hI.dpow n (e.symm b))) n x) 0
    -/
    rw [EmbeddingLike.map_eq_zero_iff, hI.dpow_null]
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      n : Nat
      x : B
      hx : Not (Membership.mem J x)
      ⊢ Not (Membership.mem I (e.symm x))
    -/
    rwa [symm_apply_mem_of_equiv_iff, h]
    /-
      🎉 no goals
    -/
  dpow_zero {x} hx := by
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      x : B
      hx : Membership.mem J x
      ⊢ Eq ((fun n b => e (hI.dpow n (e.symm b))) 0 x) 1
    -/
    rw [EmbeddingLike.map_eq_one_iff, hI.dpow_zero]
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      x : B
      hx : Membership.mem J x
      ⊢ Membership.mem I (e.symm x)
    -/
    rwa [symm_apply_mem_of_equiv_iff, h]
    /-
      🎉 no goals
    -/
  dpow_one {x} hx := by
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      x : B
      hx : Membership.mem J x
      ⊢ Eq ((fun n b => e (hI.dpow n (e.symm b))) 1 x) x
    -/
    simp only
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      x : B
      hx : Membership.mem J x
      ⊢ Eq (e (hI.dpow 1 (e.symm x))) x
    -/
    rw [dpow_one, RingEquiv.apply_symm_apply]
    /-
      case x
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      x : B
      hx : Membership.mem J x
      ⊢ Membership.mem I (e.symm x)
    -/
    rwa [I.symm_apply_mem_of_equiv_iff, h]
    /-
      🎉 no goals
    -/
  dpow_mem {n} {x} hn hx := by
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      n : Nat
      x : B
      hn : Ne n 0
      hx : Membership.mem J x
      ⊢ Membership.mem J ((fun n b => e (hI.dpow n (e.symm b))) n x)
    -/
    simp only
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      n : Nat
      x : B
      hn : Ne n 0
      hx : Membership.mem J x
      ⊢ Membership.mem J (e (hI.dpow n (e.symm x)))
    -/
    rw [← h, I.apply_mem_of_equiv_iff]
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      n : Nat
      x : B
      hn : Ne n 0
      hx : Membership.mem J x
      ⊢ Membership.mem I (hI.dpow n (e.symm x))
    -/
    apply hI.dpow_mem hn
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      n : Nat
      x : B
      hn : Ne n 0
      hx : Membership.mem J x
      ⊢ Membership.mem I (e.symm x)
    -/
    rwa [I.symm_apply_mem_of_equiv_iff, h]
    /-
      🎉 no goals
    -/
  dpow_add n {x y} hx hy := by
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      n : Nat
      x y : B
      hx : Membership.mem J x
      hy : Membership.mem J y
      ⊢ Eq ((fun n b => e (hI.dpow n (e.symm b))) n (HAdd.hAdd x y)) ((Finset.HasAnt …
    -/
    simp only [map_add]
    rw [hI.dpow_add n (symm_apply_mem_of_equiv_iff.mpr (h ▸ hx))
        (symm_apply_mem_of_equiv_iff.mpr (h ▸ hy))]
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      n : Nat
      x y : B
      hx : Membership.mem J x
      hy : Membership.mem J y
      ⊢ Eq (e ((Finset.HasAntidiagonal.antidiagonal n).sum fun k => HMul.hMul (hI.dp …
    -/
    simp only [map_sum, _root_.map_mul]
    /-
      🎉 no goals
    -/
  dpow_mul n {a x} hx := by
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      n : Nat
      a x : B
      hx : Membership.mem J x
      ⊢ Eq ((fun n b => e (hI.dpow n (e.symm b))) n (HMul.hMul a x)) (HMul.hMul (HPo …
    -/
    simp only [_root_.map_mul]
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      n : Nat
      a x : B
      hx : Membership.mem J x
      ⊢ Eq (e (hI.dpow n (HMul.hMul (e.symm a) (e.symm x)))) (HMul.hMul (HPow.hPow a …
    -/
    rw [hI.dpow_mul n (symm_apply_mem_of_equiv_iff.mpr (h ▸ hx))]
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      n : Nat
      a x : B
      hx : Membership.mem J x
      ⊢ Eq (e (HMul.hMul (HPow.hPow (e.symm a) n) (hI.dpow n (e.symm x)))) (HMul.hMu …
    -/
    rw [_root_.map_mul, map_pow]
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      n : Nat
      a x : B
      hx : Membership.mem J x
      ⊢ Eq (HMul.hMul (HPow.hPow (e (e.symm a)) n) (e (hI.dpow n (e.symm x)))) (HMul …
    -/
    simp only [RingEquiv.apply_symm_apply]
    /-
      🎉 no goals
    -/
  mul_dpow m n {x} hx := by
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      m n : Nat
      x : B
      hx : Membership.mem J x
      ⊢ Eq (HMul.hMul ((fun n b => e (hI.dpow n (e.symm b))) m x) ((fun n b => e (hI …
    -/
    simp only
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      m n : Nat
      x : B
      hx : Membership.mem J x
      ⊢ Eq (HMul.hMul (e (hI.dpow m (e.symm x))) (e (hI.dpow n (e.symm x)))) (HMul.h …
    -/
    rw [← _root_.map_mul, hI.mul_dpow, _root_.map_mul]
      /-
        A : Type u_1
        B : Type u_2
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : CommSemiring B
        J : Ideal B
        e : RingEquiv A B
        h : Eq (Ideal.map e I) J
        hI : DividedPowers I
        m n : Nat
        x : B
        hx : Membership.mem J x
        ⊢ Eq (HMul.hMul (e ↑((HAdd.hAdd m n).choose m)) (e (hI.dpow (HAdd.hAdd m n) (e …
      -/
    · simp only [map_natCast]
      /-
        🎉 no goals
      -/
      /-
        case x
        A : Type u_1
        B : Type u_2
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : CommSemiring B
        J : Ideal B
        e : RingEquiv A B
        h : Eq (Ideal.map e I) J
        hI : DividedPowers I
        m n : Nat
        x : B
        hx : Membership.mem J x
        ⊢ Membership.mem I (e.symm x)
      -/
    · rwa [symm_apply_mem_of_equiv_iff, h]
      /-
        🎉 no goals
      -/
  dpow_comp m {n x} hn hx := by
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      m n : Nat
      x : B
      hn : Ne n 0
      hx : Membership.mem J x
      ⊢ Eq ((fun n b => e (hI.dpow n (e.symm b))) m ((fun n b => e (hI.dpow n (e.sym …
    -/
    simp only [RingEquiv.symm_apply_apply]
    /-
      A : Type u_1
      B : Type u_2
      inst✝¹ : CommSemiring A
      I : Ideal A
      inst✝ : CommSemiring B
      J : Ideal B
      e : RingEquiv A B
      h : Eq (Ideal.map e I) J
      hI : DividedPowers I
      m n : Nat
      x : B
      hn : Ne n 0
      hx : Membership.mem J x
      ⊢ Eq (e (hI.dpow m (hI.dpow n (e.symm x)))) (HMul.hMul (↑(m.uniformBell n)) (e …
    -/
    rw [hI.dpow_comp _ hn]
      /-
        A : Type u_1
        B : Type u_2
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : CommSemiring B
        J : Ideal B
        e : RingEquiv A B
        h : Eq (Ideal.map e I) J
        hI : DividedPowers I
        m n : Nat
        x : B
        hn : Ne n 0
        hx : Membership.mem J x
        ⊢ Eq (e (HMul.hMul (↑(m.uniformBell n)) (hI.dpow (HMul.hMul m n) (e.symm x)))) …
      -/
    · simp only [_root_.map_mul, map_natCast]
      /-
        🎉 no goals
      -/
      /-
        A : Type u_1
        B : Type u_2
        inst✝¹ : CommSemiring A
        I : Ideal A
        inst✝ : CommSemiring B
        J : Ideal B
        e : RingEquiv A B
        h : Eq (Ideal.map e I) J
        hI : DividedPowers I
        m n : Nat
        x : B
        hn : Ne n 0
        hx : Membership.mem J x
        ⊢ Membership.mem I (e.symm x)
      -/
    · rwa [symm_apply_mem_of_equiv_iff, h]
      /-
        🎉 no goals
      -/


@[simp]
theorem ofRingEquiv_dpow (hI : DividedPowers I) (n : ℕ) (b : B) :
    (ofRingEquiv h hI).dpow n b = e (hI.dpow n (e.symm b)) := rfl


theorem ofRingEquiv_dpow_apply (hI : DividedPowers I) (n : ℕ) (a : A) :
    (ofRingEquiv h hI).dpow n (e a) = e (hI.dpow n a) := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝¹ : CommSemiring A
    I : Ideal A
    inst✝ : CommSemiring B
    J : Ideal B
    e : RingEquiv A B
    h : Eq (Ideal.map e I) J
    hI : DividedPowers I
    n : Nat
    a : A
    ⊢ Eq ((DividedPowers.ofRingEquiv h hI).dpow n (e a)) (e (hI.dpow n a))
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Transfer divided powers under an equivalence (Equiv version) -/
def equiv : DividedPowers I ≃ DividedPowers J where
  toFun := ofRingEquiv h
                                                  /-
                                                    A : Type u_1
                                                    B : Type u_2
                                                    inst✝¹ : CommSemiring A
                                                    I : Ideal A
                                                    inst✝ : CommSemiring B
                                                    J : Ideal B
                                                    e : RingEquiv A B
                                                    h : Eq (Ideal.map e I) J
                                                    ⊢ Eq (Ideal.map e.symm J) I
                                                  -/
  invFun := ofRingEquiv (show map e.symm J = I by rw [← h]; exact I.map_of_equiv e)
                                                            /-
                                                              🎉 no goals
                                                            -/
                          /-
                            A : Type u_1
                            B : Type u_2
                            inst✝¹ : CommSemiring A
                            I : Ideal A
                            inst✝ : CommSemiring B
                            J : Ideal B
                            e : RingEquiv A B
                            h : Eq (Ideal.map e I) J
                            hI : DividedPowers I
                            ⊢ Eq (DividedPowers.ofRingEquiv ⋯ (DividedPowers.ofRingEquiv h hI)) hI
                          -/
  left_inv := fun hI ↦ by ext n a; simp [ofRingEquiv]
                                   /-
                                     🎉 no goals
                                   -/
                           /-
                             A : Type u_1
                             B : Type u_2
                             inst✝¹ : CommSemiring A
                             I : Ideal A
                             inst✝ : CommSemiring B
                             J : Ideal B
                             e : RingEquiv A B
                             h : Eq (Ideal.map e I) J
                             hJ : DividedPowers J
                             ⊢ Eq (DividedPowers.ofRingEquiv h (DividedPowers.ofRingEquiv ⋯ hJ)) hJ
                           -/
  right_inv := fun hJ ↦ by ext n b; simp [ofRingEquiv]
                                    /-
                                      🎉 no goals
                                    -/


theorem equiv_apply (hI : DividedPowers I) (n : ℕ) (b : B) :
    (equiv h hI).dpow n b = e (hI.dpow n (e.symm b)) := rfl


/-- Variant of `DividedPowers.equiv_apply` -/
theorem equiv_apply' (hI : DividedPowers I) (n : ℕ) (a : A) :
    (equiv h hI).dpow n (e a) = e (hI.dpow n a) :=
  ofRingEquiv_dpow_apply h hI n a


