/-- An ideal `P` of a ring `R` is prime if `P ≠ R` and `xy ∈ P → x ∈ P ∨ y ∈ P` -/
class IsPrime (I : Ideal α) : Prop where
  /-- The prime ideal is not the entire ring. -/
  ne_top' : I ≠ ⊤
  /-- If a product lies in the prime ideal, then at least one element lies in the prime ideal. -/
  mem_or_mem' : ∀ {x y : α}, x * y ∈ I → x ∈ I ∨ y ∈ I


theorem isPrime_iff {I : Ideal α} : IsPrime I ↔ I ≠ ⊤ ∧ ∀ {x y : α}, x * y ∈ I → x ∈ I ∨ y ∈ I :=
  ⟨fun h => ⟨h.1, h.2⟩, fun h => ⟨h.1, h.2⟩⟩


theorem IsPrime.ne_top {I : Ideal α} (hI : I.IsPrime) : I ≠ ⊤ :=
  hI.1


theorem IsPrime.mem_or_mem {I : Ideal α} (hI : I.IsPrime) {x y : α} : x * y ∈ I → x ∈ I ∨ y ∈ I :=
  hI.2


theorem IsPrime.mem_or_mem_of_mul_eq_zero {I : Ideal α} (hI : I.IsPrime) {x y : α} (h : x * y = 0) :
    x ∈ I ∨ y ∈ I :=
  hI.mem_or_mem (h.symm ▸ I.zero_mem)


theorem IsPrime.mem_of_pow_mem {I : Ideal α} (hI : I.IsPrime) {r : α} (n : ℕ) (H : r ^ n ∈ I) :
    r ∈ I := by
  induction n with
  | zero =>
    rw [pow_zero] at H
    exact (mt (eq_top_iff_one _).2 hI.1).elim H
  | succ n ih =>
    rw [pow_succ] at H
    exact Or.casesOn (hI.mem_or_mem H) ih id


theorem not_isPrime_iff {I : Ideal α} :
    ¬I.IsPrime ↔ I = ⊤ ∨ ∃ (x : α) (_hx : x ∉ I) (y : α) (_hy : y ∉ I), x * y ∈ I := by
  /-
    α : Type u
    inst✝ : Semiring α
    I : Ideal α
    ⊢ Iff (Not I.IsPrime) (Or (Eq I Top.top) (Exists fun x => Exists fun _hx => Ex …
  -/
  simp_rw [Ideal.isPrime_iff, not_and_or, Ne, Classical.not_not, not_forall, not_or]
  exact
    or_congr Iff.rfl
      ⟨fun ⟨x, y, hxy, hx, hy⟩ => ⟨x, hx, y, hy, hxy⟩, fun ⟨x, hx, y, hy, hxy⟩ =>
        ⟨x, y, hxy, hx, hy⟩⟩


theorem bot_prime [IsDomain α] : (⊥ : Ideal α).IsPrime :=
                                     /-
                                       α : Type u
                                       inst✝¹ : Semiring α
                                       inst✝ : IsDomain α
                                       h : Eq Bot.bot Top.top
                                       ⊢ Eq 1 0
                                     -/
  ⟨fun h => one_ne_zero (α := α) (by rwa [Ideal.eq_top_iff_one, Submodule.mem_bot] at h), fun h =>
                                     /-
                                       🎉 no goals
                                     -/
                       /-
                         α : Type u
                         inst✝¹ : Semiring α
                         inst✝ : IsDomain α
                         x✝ y✝ : α
                         h : Membership.mem Bot.bot (HMul.hMul x✝ y✝)
                         ⊢ Eq (HMul.hMul x✝ y✝) 0
                       -/
    mul_eq_zero.mp (by simpa only [Submodule.mem_bot] using h)⟩
                       /-
                         🎉 no goals
                       -/


theorem IsPrime.mul_mem_iff_mem_or_mem {I : Ideal α} (hI : I.IsPrime) :
    ∀ {x y : α}, x * y ∈ I ↔ x ∈ I ∨ y ∈ I := @fun x y =>
  ⟨hI.mem_or_mem, by
    /-
      α : Type u
      inst✝ : CommSemiring α
      I : Ideal α
      hI : I.IsPrime
      x y : α
      ⊢ Or (Membership.mem I x) (Membership.mem I y) → Membership.mem I (HMul.hMul x …
    -/
    rintro (h | h)
    /-
      case inl
      α : Type u
      inst✝ : CommSemiring α
      I : Ideal α
      hI : I.IsPrime
      x y : α
      h : Membership.mem I x
      ⊢ Membership.mem I (HMul.hMul x y)
    -/
    exacts [I.mul_mem_right y h, I.mul_mem_left x h]⟩
    /-
      🎉 no goals
    -/


theorem IsPrime.pow_mem_iff_mem {I : Ideal α} (hI : I.IsPrime) {r : α} (n : ℕ) (hn : 0 < n) :
    r ^ n ∈ I ↔ r ∈ I :=
  ⟨hI.mem_of_pow_mem n, fun hr => I.pow_mem_of_mem hr n hn⟩


theorem IsDomain.of_bot_isPrime (A : Type*) [Ring A] [hbp : (⊥ : Ideal A).IsPrime] : IsDomain A :=
  @NoZeroDivisors.to_isDomain A _
    ⟨1, 0, fun h => hbp.ne_top ((Ideal.eq_top_iff_one ⊥).mpr h)⟩ ⟨fun h => hbp.2 h⟩


theorem eq_bot_of_prime [h : I.IsPrime] : I = ⊥ :=
  or_iff_not_imp_right.mp I.eq_bot_or_top h.1


